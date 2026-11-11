"""Benchmark individual WSDP algorithms on all valid raw samples of one dataset."""

from __future__ import annotations

import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"

import csv
import multiprocessing as mp
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT_ROOT / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main" / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from wsdp import readers
from wsdp.algorithms import execute_pipeline


SEEDS = (42, 727, 914)
WORKERS = 4
TIME_FIELDS = {seed: f"time_seed_{seed}_seconds" for seed in SEEDS}
FIELDS = ("algorithm_name", "algorithm_category", *TIME_FIELDS.values(),
          "average_seconds", "dataset", "status", "error")
ALGORITHMS = {
    "denoise": ("wavelet", "butterworth", "savgol", "bandpass", "hampel"),
    "outliers": ("iqr", "z-score"),
    "normalize": ("z-score", "min-max"),
    "interpolate": ("linear", "cubic", "nearest", "decimate"),
    "calibrate": ("linear", "polynomial", "stc", "robust"),
}


def read_file(path: Path, dataset: str):
    """Match the full-test sample inclusion and frame order."""
    reader = readers.get_reader_class(dataset)()
    if not reader.sniff(str(path)):
        return []
    samples = reader.read_file(str(path))
    samples = samples if isinstance(samples, list) else [samples]
    if not samples:
        raise RuntimeError(f"读取器没有返回样本: {path}")
    result = []
    for sample in samples:
        frames = sorted(sample.frames, key=lambda frame: frame.timestamp)
        if len(frames) < 2:
            continue
        data = np.stack([frame.csi_array for frame in frames])
        if data.ndim == 2:
            data = data[..., None]
        if data.ndim != 3 or not np.isfinite(data).all():
            raise ValueError(f"原始 CSI 形状或数值无效: {path}")
        result.append(data)
    return result


def load_samples(executor, data_dir: Path, dataset: str):
    files = sorted(path for path in data_dir.rglob("*")
                   if path.is_file() and "truth" not in path.name)
    if not files:
        raise FileNotFoundError(f"未找到原始数据文件: {data_dir}")
    samples = []
    reader = partial(read_file, dataset=dataset)
    for batch in batched(files, WORKERS * 2):
        for file_samples in executor.map(reader, batch):
            samples.extend(file_samples)
    if not samples:
        raise RuntimeError(f"没有有效样本: {data_dir}")
    return samples


def batched(items, size):
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def initialize_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2)


def process_one(item, *, dataset, category, method, target_k, seed):
    index, data = item
    try:
        random.seed(seed + index)
        np.random.seed((seed + index) % (2 ** 32))
        params = {"method": method}
        if category == "interpolate":
            params["target_K"] = target_k
        execute_pipeline(data, {category: params}, dataset=dataset)
        return None
    except Exception as exc:
        return f"样本 {index}: {type(exc).__name__}: {exc}"


def measure(executor, samples, *, dataset, category, method, target_k, seed):
    worker = partial(process_one, dataset=dataset, category=category,
                     method=method, target_k=target_k, seed=seed)
    start = time.perf_counter()
    for batch in batched(enumerate(samples), WORKERS * 2):
        errors = list(executor.map(worker, batch))
        if error := next((error for error in errors if error is not None), None):
            return "", error
    return f"{time.perf_counter() - start:.6f}", ""


def save_rows(path: Path, rows: dict):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())
    temporary.replace(path)


def main(folder: str, dataset: str, target_k: int, output_folder: str | None = None):
    data_dir = PROJECT_ROOT / "datasets" / folder
    output_folder = output_folder or folder
    output = Path(__file__).resolve().parent / output_folder / f"test_run_time_{output_folder}.csv"
    rows = {}
    if output.exists():
        with output.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                rows[(row["algorithm_category"], row["algorithm_name"])] = row
    print(f"读取原始数据（不计时）: {data_dir}", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn"),
                             initializer=initialize_worker) as executor:
        samples = load_samples(executor, data_dir, dataset)
        print(f"有效样本: {len(samples)}；每个算法运行 {len(SEEDS)} 次", flush=True)
        for category, methods in ALGORITHMS.items():
            if category == "calibrate" and dataset not in {"widar", "gait"}:
                continue
            for method in methods:
                key = (category, method)
                row = rows.get(key, dict(algorithm_name=method, algorithm_category=category,
                                         dataset=output_folder))
                if row.get("status") == "ok":
                    continue
                row["error"] = ""
                for seed in SEEDS:
                    field = TIME_FIELDS[seed]
                    if row.get(field):
                        continue
                    random.seed(seed)
                    np.random.seed(seed)
                    elapsed, error = measure(executor, samples, dataset=dataset,
                                             category=category, method=method,
                                             target_k=target_k, seed=seed)
                    row[field] = elapsed
                    row["error"] = error
                    row["status"] = "failed" if error else "partial"
                    if not error and all(row.get(field) for field in TIME_FIELDS.values()):
                        row["average_seconds"] = f"{sum(float(row[field]) for field in TIME_FIELDS.values()) / len(SEEDS):.6f}"
                        row["status"] = "ok"
                    rows[key] = row
                    save_rows(output, rows)
                    print(f"{category}/{method} seed={seed}: "
                          f"{elapsed + ' s' if not error else error}", flush=True)
                    if error:
                        break
