"""Minimal one-seed Robust component ablation for Gait and Widar.

Data flow:

    WSDP readers.load_data (bad files are skipped)
    -> parse valid CSI once
    -> savgol(7, 3) + IQR(1.5) once in memory
    -> one phase condition at a time
    -> nearest15 -> z-score [amplitude, phase] -> length1500
    -> mlpmodel training on cuda:1

No per-sample prefix or processed-data cache is written to disk.  Only the
scientific outputs (checkpoint, history, predictions, status and summaries)
are retained.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
import tempfile
import time
import traceback
import types
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable, Iterator

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wsdp_mplconfig")
)

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "robust_component_results"
PROTOCOL_VERSION = "robust-components-minimal-v1"


def find_wsdp_src() -> Path:
    candidates = (
        PROJECT_ROOT
        / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
        / "src",
        PROJECT_ROOT
        / "SDP"
        / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
        / "src",
        PROJECT_ROOT / "src",
    )
    for candidate in candidates:
        if (candidate / "wsdp").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find WSDP source. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


WSDP_SRC = find_wsdp_src()
sys.path.insert(0, str(WSDP_SRC))
sys.modules.setdefault("kagglehub", types.ModuleType("kagglehub"))

from wsdp import readers  # noqa: E402
from wsdp.algorithms import execute_pipeline  # noqa: E402
from wsdp.algorithms.amplitude import normalize_amplitude  # noqa: E402
from wsdp.algorithms.phase import robust_phase_sanitization  # noqa: E402
from wsdp.core import _create_data_split, _evaluate_model  # noqa: E402
from wsdp.models import create_model  # noqa: E402
from wsdp.processors.configurable_processor import (  # noqa: E402
    _process_single_csi_configurable,
)
from wsdp.utils import load_params, train_model  # noqa: E402


PREFIX_STEPS = {
    "denoise": {"method": "savgol", "window_length": 7, "polyorder": 3},
    "outliers": {"method": "iqr", "factor": 1.5},
}
INTERPOLATION_STEP = {"method": "nearest", "target_K": 15}
PADDING_LENGTH = 1500

ALL_CONDITIONS = (
    "linear_reference",
    "robust_first50",
    "no_calibration",
    "common_only",
    "detrend_first50_only",
    "robust_shared_first50",
    "robust_window_limited",
    "robust_fullspan50",
    "detrend_fullspan50_only",
)
CORE_CONDITIONS = ALL_CONDITIONS[:8]

DEFAULT_EPOCHS = {"gait": 60, "widar": 80}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_data_path(dataset: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {path}")
        return path

    roots = (PROJECT_ROOT, PROJECT_ROOT / "SDP", PROJECT_ROOT.parent)
    relatives = {
        "gait": (
            Path("sdp_dataset") / "Gait_Dataset" / "CSI_Gait",
            Path("sdp_dataset") / "Gait_Dataset",
            Path("sdp_dataset") / "gait",
        ),
        "widar": (
            Path("sdp_dataset") / "widar_common3",
            Path("sdp_dataset") / "widar",
        ),
    }[dataset]
    candidates = [root / relative for root in roots for relative in relatives]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find {dataset} data. Pass --data-path explicitly. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_text: str) -> torch.device:
    device = torch.device(device_text)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA "
                "device(s) are visible. Do not combine default cuda:1 with "
                "CUDA_VISIBLE_DEVICES=1; that mask renumbers the selected GPU to cuda:0."
            )
    return device


def select_balanced_indices(labels: list[int], limit: int, seed: int) -> list[int]:
    if limit <= 0 or len(labels) <= limit:
        return list(range(len(labels)))
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        buckets[int(label)].append(index)
    rng = np.random.default_rng(seed)
    for indices in buckets.values():
        rng.shuffle(indices)
    selected: list[int] = []
    while len(selected) < limit:
        progressed = False
        for label in sorted(buckets):
            if buckets[label] and len(selected) < limit:
                selected.append(buckets[label].pop())
                progressed = True
        if not progressed:
            break
    return sorted(selected)


def load_valid_csi(
    data_path: Path, dataset: str, max_samples: int, seed: int
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[int], list[str], str]:
    """Read first, then parse; invalid files/samples are skipped like WSDP."""
    print(f"Reading {dataset} from {data_path}", flush=True)
    print(
        "WSDP readers.load_data will skip format mismatches and read errors.",
        flush=True,
    )
    items = readers.load_data(str(data_path), dataset)
    items.sort(key=lambda item: item.file_name, reverse=True)
    reader_valid = len(items)

    raw_data: list[np.ndarray] = []
    raw_labels: list[int] = []
    raw_groups: list[int] = []
    file_names: list[str] = []
    parse_skipped = 0

    # Pop releases each CSIData object as soon as its frames have been stacked.
    while items:
        item = items.pop()
        try:
            csi, label, group = _process_single_csi_configurable(item, dataset, {})
        except Exception as error:
            parse_skipped += 1
            print(f"Skip invalid sample {item.file_name}: {error}", flush=True)
            continue
        if csi is None or csi.ndim != 3 or not np.iscomplexobj(csi):
            parse_skipped += 1
            print(f"Skip empty/degenerate sample {item.file_name}", flush=True)
            continue
        raw_data.append(csi)
        raw_labels.append(int(label))
        raw_groups.append(int(group))
        file_names.append(item.file_name)

    if not raw_data:
        raise RuntimeError("No valid CSI samples remain after reading and parsing")

    selected = select_balanced_indices(raw_labels, max_samples, seed)
    if len(selected) != len(raw_data):
        raw_data = [raw_data[index] for index in selected]
        raw_labels = [raw_labels[index] for index in selected]
        raw_groups = [raw_groups[index] for index in selected]
        file_names = [file_names[index] for index in selected]

    unique_labels = sorted(set(raw_labels))
    unique_groups = sorted(set(raw_groups))
    label_map = {label: index for index, label in enumerate(unique_labels)}
    group_map = {group: index for index, group in enumerate(unique_groups)}
    labels = np.asarray([label_map[label] for label in raw_labels], dtype=np.int64)
    groups = np.asarray([group_map[group] for group in raw_groups], dtype=np.int64)
    manifest_id = fingerprint(
        list(zip(file_names, raw_labels, raw_groups, strict=True))
    )

    print(
        f"Reader-valid={reader_valid}, parse-skipped={parse_skipped}, "
        f"selected={len(raw_data)}",
        flush=True,
    )
    print(f"Labels: {dict(Counter(raw_labels))}", flush=True)
    print(f"Groups: {len(unique_groups)}", flush=True)
    return raw_data, labels, groups, unique_labels, file_names, manifest_id


def bounded_map(function: Any, tasks: Iterable[Any], workers: int) -> Iterator[Any]:
    """Process tasks with at most 2*workers futures retained in memory."""
    if workers == 1:
        yield from map(function, tasks)
        return
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn")
    ) as executor:
        iterator = iter(tasks)
        in_flight = set()
        for _ in range(workers * 2):
            try:
                in_flight.add(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while in_flight:
            completed, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                try:
                    in_flight.add(executor.submit(function, next(iterator)))
                except StopIteration:
                    pass


def prefix_worker(task: tuple[int, np.ndarray, str]) -> tuple[int, np.ndarray]:
    index, csi, dataset = task
    return index, execute_pipeline(csi, PREFIX_STEPS, dataset=dataset)


def build_prefix_in_memory(
    raw_data: list[np.ndarray], dataset: str, workers: int
) -> list[np.ndarray]:
    """Replace raw arrays with the shared prefix; nothing is written to disk."""
    total = len(raw_data)
    tasks = ((index, raw_data[index], dataset) for index in range(total))
    for completed, (index, prefixed) in enumerate(
        bounded_map(prefix_worker, tasks, workers), 1
    ):
        raw_data[index] = prefixed
        if completed % 100 == 0 or completed == total:
            print(f"Shared savgol+IQR prefix: {completed}/{total}", flush=True)
    gib = sum(array.nbytes for array in raw_data) / 2**30
    print(f"Shared prefix kept only in RAM: {gib:.2f} GiB", flush=True)
    return raw_data


def unwrap_phase(csi: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(csi), axis=0).astype(np.float64, copy=False)


def reconstruct(csi: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return (np.abs(csi) * np.exp(1j * phase)).astype(csi.dtype, copy=False)


def theil_sen_slope(phases: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size < 2:
        return np.zeros(phases.shape[1:], dtype=np.float64)
    left, right = np.triu_indices(indices.size, k=1)
    left_indices = indices[left]
    right_indices = indices[right]
    denominators = (right_indices - left_indices).astype(np.float64)[:, None, None]
    slopes = (phases[right_indices] - phases[left_indices]) / denominators
    return np.median(slopes, axis=0)


def apply_phase_condition(
    csi: np.ndarray, condition: str, dataset: str
) -> np.ndarray:
    if condition == "no_calibration":
        return csi
    if condition == "linear_reference":
        return execute_pipeline(
            csi, {"calibrate": {"method": "linear"}}, dataset=dataset
        )
    if condition == "robust_first50":
        return robust_phase_sanitization(csi)

    phase = unwrap_phase(csi)
    centered = phase - np.median(phase, axis=1, keepdims=True)
    fit_count = min(csi.shape[0], 50)
    first_indices = np.arange(fit_count, dtype=np.int64)
    visible_frames = min(csi.shape[0], PADDING_LENGTH)
    visible_indices = np.unique(
        np.linspace(0, visible_frames - 1, fit_count, dtype=np.int64)
    )
    times = np.arange(csi.shape[0], dtype=np.float64)[:, None, None]

    if csi.shape[0] < 3:
        if condition in {
            "common_only",
            "robust_shared_first50",
            "robust_window_limited",
            "robust_fullspan50",
        }:
            return reconstruct(csi, centered)
        if condition in {"detrend_first50_only", "detrend_fullspan50_only"}:
            return reconstruct(csi, phase)

    if condition == "common_only":
        corrected = centered
    elif condition == "detrend_first50_only":
        slope = theil_sen_slope(phase, first_indices)
        corrected = phase - times * slope[None, :, :]
    elif condition == "detrend_fullspan50_only":
        slope = theil_sen_slope(phase, visible_indices)
        corrected = phase - times * slope[None, :, :]
    elif condition == "robust_shared_first50":
        slope = theil_sen_slope(centered, first_indices)
        shared = np.median(slope, axis=0, keepdims=True)
        corrected = centered - times * shared[None, :, :]
    elif condition == "robust_window_limited":
        slope = theil_sen_slope(centered, first_indices)
        limited_times = np.minimum(times, max(fit_count - 1, 0))
        corrected = centered - limited_times * slope[None, :, :]
    elif condition == "robust_fullspan50":
        slope = theil_sen_slope(centered, visible_indices)
        corrected = centered - times * slope[None, :, :]
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return reconstruct(csi, corrected)


def resize_one(sample: np.ndarray) -> np.ndarray:
    if sample.shape[0] > PADDING_LENGTH:
        sample = sample[:PADDING_LENGTH]
    elif sample.shape[0] < PADDING_LENGTH:
        sample = np.pad(
            sample,
            ((0, PADDING_LENGTH - sample.shape[0]), (0, 0), (0, 0)),
            mode="constant",
        )
    return sample.astype(np.float32, copy=False)


def condition_worker(
    task: tuple[int, np.ndarray, str, str]
) -> tuple[int, np.ndarray]:
    index, csi, condition, dataset = task
    calibrated = apply_phase_condition(csi, condition, dataset)
    interpolated = execute_pipeline(
        calibrated,
        {"interpolate": INTERPOLATION_STEP},
        dataset=dataset,
    )
    explicit = normalize_amplitude(
        interpolated,
        method="z-score",
        return_phase_channels=True,
    )
    return index, resize_one(explicit)


def build_condition_data(
    prefix_data: list[np.ndarray], condition: str, dataset: str, workers: int
) -> np.ndarray:
    antennas = int(prefix_data[0].shape[2])
    output = np.empty(
        (len(prefix_data), PADDING_LENGTH, 15, 2 * antennas), dtype=np.float32
    )
    tasks = (
        (index, prefix_data[index], condition, dataset)
        for index in range(len(prefix_data))
    )
    for completed, (index, sample) in enumerate(
        bounded_map(condition_worker, tasks, workers), 1
    ):
        if sample.shape != output.shape[1:] or not np.all(np.isfinite(sample)):
            raise RuntimeError(f"Invalid processed sample {index}: {sample.shape}")
        output[index] = sample
        if completed % 100 == 0 or completed == len(prefix_data):
            print(f"{condition}: {completed}/{len(prefix_data)}", flush=True)
    return output


def build_split_indices(
    labels: np.ndarray, groups: np.ndarray, dataset: str, seed: int
) -> dict[str, np.ndarray]:
    payload = np.arange(len(labels), dtype=np.int64)[:, None]
    split = _create_data_split(
        payload,
        labels,
        groups,
        test_split=0.3,
        val_split=0.5,
        seed=seed,
        use_simple_split=len(np.unique(groups)) < 3,
        dataset=dataset,
        pipeline_steps={
            **PREFIX_STEPS,
            "normalize": {"method": "z-score"},
            "interpolate": INTERPOLATION_STEP,
        },
    )
    result = {
        "train": np.asarray(split[0]).reshape(-1).astype(np.int64),
        "val": np.asarray(split[1]).reshape(-1).astype(np.int64),
        "test": np.asarray(split[2]).reshape(-1).astype(np.int64),
    }
    concatenated = np.concatenate(list(result.values()))
    if len(concatenated) != len(labels) or len(np.unique(concatenated)) != len(labels):
        raise RuntimeError("Split indices overlap or do not cover every sample")
    if len(np.unique(groups)) >= 3:
        sets = {name: set(groups[index].tolist()) for name, index in result.items()}
        if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets[
            "val"
        ] & sets["test"]:
            raise RuntimeError("Group leakage detected")
    return result


def build_loaders(
    processed: np.ndarray,
    labels: np.ndarray,
    split: dict[str, np.ndarray],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = TensorDataset(torch.from_numpy(processed), torch.from_numpy(labels).long())

    def make(name: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            Subset(dataset, split[name].tolist()),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    return make("train", True), make("val", False), make("test", False)


def save_history(history: dict[str, list[Any]], path: Path) -> None:
    fields = list(history)
    length = max((len(history[field]) for field in fields), default=0)
    rows = [
        {
            field: history[field][index] if index < len(history[field]) else ""
            for field in fields
        }
        for index in range(length)
    ]
    write_csv(path, rows, fields)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_condition(
    processed: np.ndarray,
    labels: np.ndarray,
    split: dict[str, np.ndarray],
    num_classes: int,
    condition: str,
    condition_dir: Path,
    model_name: str,
    model_seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    run_id: str,
    preprocess_seconds: float,
) -> dict[str, Any]:
    run_dir = condition_dir / f"seed_{model_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_checkpoint.pth"
    history_path = run_dir / "training_history.csv"
    predictions_path = run_dir / "test_predictions.npz"
    for path in (checkpoint_path, history_path, predictions_path):
        path.unlink(missing_ok=True)

    set_seed(model_seed)
    loaders = build_loaders(processed, labels, split, batch_size)
    model = create_model(
        model_name,
        num_classes=num_classes,
        input_shape=tuple(processed.shape[1:]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    started = time.time()
    history = train_model(
        model,
        nn.CrossEntropyLoss(),
        optimizer,
        scheduler,
        loaders[0],
        loaders[1],
        epochs,
        device,
        checkpoint_path,
        PADDING_LENGTH,
    )
    if not checkpoint_path.is_file():
        raise RuntimeError("Training produced no best checkpoint")
    save_history(history, history_path)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions, y_true, test_acc = _evaluate_model(model, loaders[2], device)
    with predictions_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_indices=split["test"],
            y_true=np.asarray(y_true),
            y_pred=np.asarray(predictions),
        )
    result = {
        "status": "ok",
        "condition": condition,
        "model": model_name,
        "model_seed": model_seed,
        "epochs": epochs,
        "best_val_acc": float(checkpoint.get("best_val_acc", 0.0)) / 100.0,
        "test_acc": float(test_acc),
        "preprocess_seconds": preprocess_seconds,
        "training_seconds": time.time() - started,
        "run_id": run_id,
    }
    write_json(run_dir / "status.json", result)
    del model, loaders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def status_complete(condition_dir: Path, seed: int, run_id: str) -> bool:
    run_dir = condition_dir / f"seed_{seed}"
    status = read_json(run_dir / "status.json")
    return bool(
        status
        and status.get("status") == "ok"
        and status.get("run_id") == run_id
        and (run_dir / "best_checkpoint.pth").is_file()
        and (run_dir / "training_history.csv").is_file()
        and (run_dir / "test_predictions.npz").is_file()
    )


def rebuild_summary(dataset_dir: Path, run_id: str) -> None:
    rows: list[dict[str, Any]] = []
    for condition in ALL_CONDITIONS:
        for status_path in sorted((dataset_dir / condition).glob("seed_*/status.json")):
            row = read_json(status_path)
            if row and row.get("run_id") == run_id and row.get("status") == "ok":
                rows.append(row)
    successful = {row["condition"]: float(row["test_acc"]) for row in rows}
    linear = successful.get("linear_reference")
    robust = successful.get("robust_first50")
    for row in rows:
        accuracy = float(row["test_acc"])
        row["delta_vs_linear_pp"] = (
            "" if linear is None else 100.0 * (accuracy - linear)
        )
        row["delta_vs_robust_pp"] = (
            "" if robust is None else 100.0 * (accuracy - robust)
        )
    write_csv(
        dataset_dir / "summary.csv",
        rows,
        (
            "condition",
            "model",
            "model_seed",
            "epochs",
            "best_val_acc",
            "test_acc",
            "delta_vs_linear_pp",
            "delta_vs_robust_pp",
            "preprocess_seconds",
            "training_seconds",
            "run_id",
        ),
    )
    contrasts = {}
    specs = {
        "robust_minus_linear": ("robust_first50", "linear_reference"),
        "robust_minus_common": ("robust_first50", "common_only"),
        "shared_minus_common": ("robust_shared_first50", "common_only"),
        "window_limited_minus_robust": (
            "robust_window_limited",
            "robust_first50",
        ),
        "fullspan50_minus_robust": ("robust_fullspan50", "robust_first50"),
        "common_minus_none": ("common_only", "no_calibration"),
    }
    for name, (left, right) in specs.items():
        if left in successful and right in successful:
            contrasts[name] = 100.0 * (successful[left] - successful[right])
    write_json(
        dataset_dir / "contrasts.json",
        {"test_accuracy": successful, "delta_pp": contrasts},
    )


def resolve_conditions(args: argparse.Namespace) -> list[str]:
    if args.conditions:
        conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    else:
        conditions = list(CORE_CONDITIONS if args.suite == "core" else ALL_CONDITIONS)
    unknown = sorted(set(conditions) - set(ALL_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    return conditions


def run(args: argparse.Namespace) -> None:
    conditions = resolve_conditions(args)
    data_path = resolve_data_path(args.dataset, args.data_path)
    device = resolve_device(args.device)
    params = load_params(args.dataset)
    epochs = args.epochs or DEFAULT_EPOCHS[args.dataset]
    batch_size = args.batch_size or int(params.get("batch", 32))
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(params.get("lr", 3e-4))
    )
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(params.get("wd", 1e-3))
    )
    output_root = Path(args.output_root).expanduser().resolve()
    dataset_dir = output_root / args.dataset

    print(f"Protocol: {PROTOCOL_VERSION}")
    print(f"WSDP: {WSDP_SRC}")
    print(f"Data: {data_path}")
    print(f"Conditions: {conditions}")
    print(f"Device: {device}")
    print("Disk cache: disabled (prefix and processed data stay in RAM)")
    print(f"Output: {dataset_dir}")
    legacy_cache = SCRIPT_DIR / "robust_component_server_results" / args.dataset / "cache"
    if legacy_cache.exists():
        print(f"Legacy cache is unused and may be deleted manually: {legacy_cache}")
    if args.dry_run:
        return

    raw_data, labels, groups, unique_labels, file_names, manifest_id = load_valid_csi(
        data_path, args.dataset, args.max_samples, args.split_seed
    )
    if len(unique_labels) < 2 and not args.preprocess_only:
        raise RuntimeError(f"Training requires at least two classes: {unique_labels}")

    settings = {
        "protocol": PROTOCOL_VERSION,
        "dataset": args.dataset,
        "data_path": str(data_path),
        "sample_count": len(raw_data),
        "manifest_id": manifest_id,
        "model": args.model,
        "model_seed": args.model_seed,
        "split_seed": args.split_seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "device": str(device),
        "pipeline": {
            **PREFIX_STEPS,
            "phase": "condition",
            "interpolate": INTERPOLATION_STEP,
            "normalize": "z-score [amplitude, phase]",
            "length": PADDING_LENGTH,
        },
    }
    run_id = fingerprint(settings)
    settings["run_id"] = run_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    existing = read_json(dataset_dir / "settings.json")
    if existing and existing.get("run_id") != run_id:
        raise RuntimeError(
            f"Existing output has another configuration: {dataset_dir}. "
            "Use a new --output-root."
        )
    write_json(dataset_dir / "settings.json", settings)
    write_json(
        dataset_dir / "dataset_metadata.json",
        {
            "sample_count": len(raw_data),
            "manifest_id": manifest_id,
            "classes": unique_labels,
            "label_counts": dict(Counter(labels.tolist())),
            "group_count": len(np.unique(groups)),
            "first_files": file_names[:20],
        },
    )

    split = build_split_indices(labels, groups, args.dataset, args.split_seed)
    with (dataset_dir / "split_indices.npz").open("wb") as handle:
        np.savez_compressed(handle, **split)
    print(
        f"Split: train={len(split['train'])}, val={len(split['val'])}, "
        f"test={len(split['test'])}",
        flush=True,
    )

    prefix_data = build_prefix_in_memory(raw_data, args.dataset, args.workers)
    del raw_data
    gc.collect()

    failures: list[str] = []
    for condition in conditions:
        condition_dir = dataset_dir / condition
        if status_complete(condition_dir, args.model_seed, run_id):
            print(f"Skip completed condition: {condition}")
            continue
        print("\n" + "=" * 80)
        print(f"Condition: {args.dataset} / {condition}")
        processed: np.ndarray | None = None
        started = time.time()
        try:
            processed = build_condition_data(
                prefix_data, condition, args.dataset, args.workers
            )
            preprocess_seconds = time.time() - started
            print(
                f"Model input: {processed.shape}, {processed.nbytes / 2**30:.2f} GiB RAM"
            )
            if args.preprocess_only:
                continue
            result = train_condition(
                processed,
                labels,
                split,
                len(unique_labels),
                condition,
                condition_dir,
                args.model,
                args.model_seed,
                epochs,
                batch_size,
                learning_rate,
                weight_decay,
                device,
                run_id,
                preprocess_seconds,
            )
            print(
                f"Completed {condition}: val={result['best_val_acc']:.4f}, "
                f"test={result['test_acc']:.4f}"
            )
            rebuild_summary(dataset_dir, run_id)
        except BaseException:
            failures.append(condition)
            condition_dir.mkdir(parents=True, exist_ok=True)
            (condition_dir / "error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            print(traceback.format_exc(), file=sys.stderr)
            if not args.continue_on_error:
                raise
        finally:
            del processed
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rebuild_summary(dataset_dir, run_id)
    print(f"Finished. Summary: {dataset_dir / 'summary.csv'}")
    if failures:
        raise RuntimeError(f"Failed conditions: {failures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gait", "widar"), default="gait")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--suite", choices=("core", "full"), default="core")
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--model", default="mlpmodel")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-conditions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.max_samples < 0:
        raise ValueError("--workers must be >=1 and --max-samples must be >=0")
    if args.list_conditions:
        print("core:", ",".join(CORE_CONDITIONS))
        print("full:", ",".join(ALL_CONDITIONS))
        return
    run(args)


if __name__ == "__main__":
    main()
