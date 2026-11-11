"""独立的两阶段实验步骤；直接运行本文件即可，--check 只检查配置。

本文件运行第 1–160 种组合；full_test_widar_part2.py 运行第 161–320 种。

所有 acc/F1 均为 0–1。测试只使用最佳验证 checkpoint。
splits/ 保存两个阶段共享的原始样本清单和划分，数据变化时拒绝混用旧结果。
"""
from __future__ import annotations

import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"

import contextlib
import csv
import gc
import hashlib
import io
import json
import multiprocessing as mp
import random
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import islice
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
if not (SOURCE_ROOT / "src" / "wsdp").is_dir():
    raise FileNotFoundError(f"Local WSDP source not found: {SOURCE_ROOT}")
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from wsdp import readers
from wsdp.algorithms import apply_preset, list_presets, execute_pipeline
from wsdp.algorithms.amplitude import normalize_amplitude
from wsdp.core import _create_data_split, _create_xrf55_repetition_split
from wsdp.datasets import CSIDataset
from wsdp.models import create_model, list_models
from wsdp.processors.base_processor import _parse_file_info_from_filename, _selector
from wsdp.utils import load_params, resize_csi_to_fixed_length

DATASET_NAME = "widar"
DATA_FOLDER = "widar3.0"
MODEL_NAME = "mlpmodel"
STAGE = "full"
SEEDS = (42, 727, 914)
CUDA_DEVICE_INDEX = 2  # 对应 nvidia-smi 中的 GPU 编号。
PARALLEL_WORKERS = 4  # 22 vCPU / 110 GB 服务器的初始设置，可按实际吞吐调整。
DATA_PATH = PROJECT_ROOT / "datasets" / DATA_FOLDER
OUTPUT_ROOT = Path(__file__).resolve().parent
SPLIT_ROOT = OUTPUT_ROOT.parent / "splits"
TEST_SPLIT, VAL_SPLIT = 0.3, 0.5
PHASE_DATASET = DATASET_NAME in {"widar", "gait"}
SUMMARY_FIELDS = ["seed", "combo_name", "model", "best_val_acc",
                  "test_acc", "macro_f1", "output_dir", "status",
                  "training_seconds", "split_digest", "run_digest"]

# 沿用对应旧脚本的算法参数及组合顺序。
DENOISE_OPTIONS = [
    ("wavelet", {"method": "wavelet"}),
    ("butterworth_o5_c0.3", {"method": "butterworth", "order": 5, "cutoff": 0.3}),
    ("savgol_w7_p3", {"method": "savgol", "window_length": 7, "polyorder": 3}),
    (
        "bandpass_0.5-50",
        {
            "method": "bandpass",
            "order": 4,
            "low_freq": 0.5,
            "high_freq": 50.0,
            "fs": 1000.0,
        },
    ),
    ("hampel_w5_s3", {"method": "hampel", "window_size": 5, "n_sigma": 3.0}),
]

OUTLIER_OPTIONS = [
    ("iqr", {"method": "iqr", "factor": 1.5}),
    ("outlier_z-score", {"method": "z-score", "factor": 3.0}),
]

CALIBRATE_OPTIONS = [
    ("linear", {"method": "linear"}),
    ("polynomial_d3", {"method": "polynomial", "degree": 3}),
    ("stc", {"method": "stc"}),
    ("robust", {"method": "robust"}),
]

NORMALIZE_OPTIONS = [
    ("z-score", {"method": "z-score"}),
    ("min-max", {"method": "min-max"}),
]

INTERPOLATE_OPTIONS = [
    ("linear15", {"method": "linear", "target_K": 15}),
    ("cubic15", {"method": "cubic", "target_K": 15}),
    ("nearest15", {"method": "nearest", "target_K": 15}),
    ("decimate15", {"method": "decimate", "target_K": 15}),
]


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def digest(value):
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_experiments():
    if STAGE == "preset":
        return [(name, apply_preset(name)) for name in list_presets()]
    categories = ["denoise", "outliers"]
    options = [DENOISE_OPTIONS, OUTLIER_OPTIONS]
    if PHASE_DATASET:
        categories.append("calibrate")
        options.append(CALIBRATE_OPTIONS)
    categories += ["normalize", "interpolate"]
    options += [NORMALIZE_OPTIONS, INTERPOLATE_OPTIONS]
    combinations = [("+".join(item[0] for item in choice),
                     {key: item[1].copy() for key, item in zip(categories, choice)})
                    for choice in product(*options)]
    assert len(combinations) == (320 if PHASE_DATASET else 80)
    return combinations


def verify_or_save(path, value, *, ignore_sample_mtime=False):
    """校验后返回已保存的清单；保留旧摘要和断点续跑兼容性。"""
    def comparable(record):
        if not ignore_sample_mtime:
            return record
        # 仅忽略样本修改时间，仍严格比较顺序、路径、大小、标签、分组及排除项。
        return {**record, "samples": [
            {key: item for key, item in sample.items() if key != "mtime_ns"}
            for sample in record["samples"]
        ]}

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if comparable(saved) != comparable(value):
            raise RuntimeError(f"样本/划分已变化，请先归档旧结果和 splits 后重新运行: {path}")
        return saved
    else:
        # 独占创建，避免同时启动两个阶段时覆盖另一个阶段的清单。
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
        except FileExistsError:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if comparable(saved) != comparable(value):
                raise RuntimeError(f"共享清单不一致: {path}")
            return saved
    return value


def initialize_worker():
    """每个 CPU 进程使用两个计算线程，避免多进程与 BLAS 线程叠加。"""
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2)
    torch.set_num_threads(2)


def ordered_parallel_map(executor, function, items):
    """每批最多提交 2 × 工作进程数个任务，按输入顺序返回。"""
    iterator = iter(items)
    while batch := list(islice(iterator, PARALLEL_WORKERS * 2)):
        yield from executor.map(function, batch)


def read_raw_file(path):
    """工作进程只返回数组和元数据，不传回逐帧对象。"""
    reader = readers.get_reader_class(DATASET_NAME)()
    arrays, records, excluded = [], [], []
    relative = path.relative_to(DATA_PATH).as_posix()
    if not reader.sniff(str(path)):
        excluded.append({"file": relative, "reason": "format_mismatch"})
        return [], [], excluded
    # 匹配格式却无法读取时停止，防止两个阶段接受不同的原始样本集合。
    samples = reader.read_file(str(path))
    samples = samples if isinstance(samples, list) else [samples]
    if not samples:
        raise RuntimeError(f"读取器未返回样本: {path}")
    for index, sample in enumerate(samples):
        label, group = _selector(_parse_file_info_from_filename(sample.file_name, DATASET_NAME), DATASET_NAME)
        if len(sample.frames) < 2:
            excluded.append({"file": relative, "sample": index, "reason": "fewer_than_2_frames"})
            continue
        data = np.stack([frame.csi_array for frame in sorted(sample.frames, key=lambda f: f.timestamp)])
        if data.ndim == 2:
            data = data[..., None]
        if data.ndim != 3 or not np.isfinite(data).all():
            raise ValueError(f"原始 CSI 形状或有限值检查失败: {relative} #{index}")
        arrays.append(data)
        stat = path.stat()
        records.append({"id": f"{relative}#{index}", "label": int(label), "group": int(group),
                        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return arrays, records, excluded


def load_raw_data(executor):
    """step 1：并行读取一次，按相对路径及文件内样本编号固定顺序。"""
    print("step 1: load_raw_data", DATA_PATH, "workers=", PARALLEL_WORKERS)
    files = [path for path in sorted(DATA_PATH.rglob("*"))
             if path.is_file() and "truth" not in path.name]
    arrays, records, excluded = [], [], []
    for file_arrays, file_records, file_excluded in ordered_parallel_map(executor, read_raw_file, files):
        arrays.extend(file_arrays)
        records.extend(file_records)
        excluded.extend(file_excluded)
    if not arrays:
        raise RuntimeError(f"无有效样本: {DATA_PATH}")
    manifest = {"dataset": DATASET_NAME, "samples": records, "excluded": excluded}
    # 时间变化只放宽比较；摘要继续使用磁盘中的原清单，避免旧 split/结果失效。
    manifest = verify_or_save(SPLIT_ROOT / "samples.json", manifest, ignore_sample_mtime=True)
    classes = sorted({row["label"] for row in records})
    labels = np.asarray([classes.index(row["label"]) for row in records])
    # group 保留源码含义和原始值，尤其不能把 XRF55 repetition 随意重新编号。
    groups = np.asarray([row["group"] for row in records])
    print(f"有效样本={len(arrays)}, 排除={len(excluded)}, 原始标签={classes}")
    return arrays, labels, groups, classes, digest(manifest)


def prepare_split_indices(labels, groups, seed, sample_digest):
    """用源码划分索引；算法处理永远不能改变样本集合。"""
    split = _create_data_split(np.arange(len(labels)), labels, groups,
                              test_split=TEST_SPLIT, val_split=VAL_SPLIT, seed=seed,
                              use_simple_split=False, dataset=DATASET_NAME, pipeline_steps={})
    indices = [np.asarray(part, dtype=int) for part in split[:3]]
    if sorted(np.concatenate(indices).tolist()) != list(range(len(labels))):
        raise RuntimeError("划分存在重复或遗漏样本")
    group_sets = [set(groups[index]) for index in indices]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("训练/验证/测试之间存在 group 泄漏")
    record = {"seed": seed, "sample_digest": sample_digest,
              "train": indices[0].tolist(), "validation": indices[1].tolist(), "test": indices[2].tolist()}
    verify_or_save(SPLIT_ROOT / f"split{seed}.json", record)
    return indices, digest(record)


def process_sample(item, steps, phase_zscore=False, padding_length=None):
    """单样本算法工作函数；顺序及异常处理与串行版本一致。"""
    index, data = item
    try:
        result = execute_pipeline(data, steps, dataset=DATASET_NAME)
        if phase_zscore:
            result = normalize_amplitude(result, method="z-score", return_phase_channels=True)
        if result.ndim != 3 or not result.size or not np.isfinite(result).all():
            raise ValueError("算法输出形状或有限值检查失败")
        if padding_length is not None:
            result = resize_csi_to_fixed_length([result], target_length=padding_length)[0]
        return result
    except Exception as exc:
        raise RuntimeError(f"样本索引 {index} 预处理失败，整个组合失败，禁止丢弃样本") from exc


def process_data(raw, pipeline_steps, padding_length, labels, groups, classes, executor, summary_log=None):
    """step 2：每个预设/组合仅处理一次，三个种子复用；不允许丢样本。

    每个组合独立执行完整预处理，不保留公共前缀缓存以降低内存占用。
    z-score 在插值后形成有符号幅度+相位，顺序与源码处理器一致。
    """
    def report(*args):
        """只将本步骤的统计信息写入日志，不复制算法内部逐样本提示。"""
        print(*args)
        if summary_log is not None:
            print(*args, file=summary_log)

    report("step 2: process_data", json_text(pipeline_steps))
    report(f"待处理样本数: {len(raw)}; workers={PARALLEL_WORKERS}")
    report("原始样本形状及数量 (T, F, A):", dict(Counter(tuple(item.shape) for item in raw)))
    report(f"类别数: {len(classes)}; 原始标签: {classes}")
    report("标签映射 (原始标签 -> 训练标签):", dict(zip(classes, range(len(classes)))))
    report("原始标签分布:", {classes[label]: count for label, count in sorted(Counter(labels.tolist()).items())})
    report(f"分组数: {len(set(groups.tolist()))}; 原始分组分布:", dict(sorted(Counter(groups.tolist()).items())))
    steps = dict(pipeline_steps)
    method = steps.get("normalize", {}).get("method")
    phase_zscore = PHASE_DATASET and method == "z-score"
    if phase_zscore or (DATASET_NAME == "xrf55" and method in {"z-score", "min-max"}):
        steps.pop("normalize")
    worker = partial(process_sample, steps=steps, phase_zscore=phase_zscore,
                     padding_length=padding_length)
    processed = list(ordered_parallel_map(executor, worker, enumerate(raw)))
    processed_data = np.stack(processed)
    report(f"处理完成样本数: {len(processed_data)}")
    report(f"处理后数据形状 (N, T, F, A): {processed_data.shape}; dtype={processed_data.dtype}")
    report(f"处理后单样本形状: {processed_data.shape[1:]}; padding_length={padding_length}")
    return processed_data


def split_data(data, labels, groups, indices, pipeline_steps, split_digest):
    """step 3：使用共享索引；XRF55 使用源码的训练集归一化。"""
    print("step 3: split_data; split_digest=", split_digest)
    for name, index in zip(("train", "validation", "test"), indices):
        print(f"{name}: n={len(index)}, groups={sorted(set(groups[index].tolist()))}, labels={dict(Counter(labels[index].tolist()))}")
    if DATASET_NAME == "xrf55":
        return _create_xrf55_repetition_split(data, labels, groups, pipeline_steps=pipeline_steps)
    return tuple(data[index] for index in indices) + tuple(labels[index] for index in indices)


def build_loaders(split, steps, batch_size, seed):
    """step 4：每个模型重建随机生成器，保证模型顺序不影响 shuffle。"""
    print("step 4: build_loaders; batch=", batch_size)
    return tuple(DataLoader(CSIDataset(split[i], split[i + 3], dataset_name=DATASET_NAME, pipeline_steps=steps),
                            batch_size=batch_size, shuffle=i == 0, num_workers=0,
                            generator=torch.Generator().manual_seed(seed)) for i in range(3))


def create_registered_model(name, classes, input_shape):
    print("step 5: create_registered_model", name, input_shape)
    device = torch.device(f"cuda:{CUDA_DEVICE_INDEX}")
    model = create_model(name, num_classes=classes, input_shape=input_shape).to(device)
    print("device=", device, "parameters=", sum(p.numel() for p in model.parameters()))
    return model, device


def train_registered_model(model, device, loaders, params, output):
    """step 6：沿用 AdamW/plateau；首轮即保存，兼容验证 acc 始终为零。"""
    print("step 6: train", json_text(params))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    history = {"train_loss": [], "val_loss": []}
    best = -1.0
    checkpoint = output / "best_checkpoint.pth"
    start = time.perf_counter()
    try:
        for epoch in range(params["num_epochs"]):
            metrics = []
            for training, loader in zip((True, False), loaders[:2]):
                model.train(training)
                total_loss = correct = count = 0
                with torch.set_grad_enabled(training):
                    for data, labels in loader:
                        data, labels = data.to(device), labels.to(device)
                        if training:
                            optimizer.zero_grad()
                        predictions = model(data)
                        loss = criterion(predictions, labels)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("训练/验证 loss 非有限值")
                        if training:
                            loss.backward()
                            optimizer.step()
                        total_loss += loss.item() * len(labels)
                        correct += (predictions.argmax(1) == labels).sum().item()
                        count += len(labels)
                metrics.append((total_loss / count, correct / count))
            (train_loss, train_acc), (val_loss, val_acc) = metrics
            scheduler.step(val_loss)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(f"Epoch {epoch + 1}/{params['num_epochs']} train_loss={train_loss:.6f} train_acc={train_acc:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.6f}")
            if val_acc > best:
                best = val_acc
                torch.save({"model_state_dict": model.state_dict(), "best_val_acc": best,
                            "epoch": epoch + 1, "padding_length": params["padding_length"]}, checkpoint)
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        print(f"训练耗时（从开始训练到结束，含验证和 checkpoint 保存）: {elapsed:.3f} 秒")
        if history["train_loss"]:
            plt.figure(figsize=(8, 5))
            for key, values in history.items():
                plt.plot(range(1, len(values) + 1), values, label=key)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output / "loss_curve.png", dpi=150)
            plt.close()
    return checkpoint, elapsed


def evaluate_checkpoint(model, device, loader, checkpoint, classes):
    """step 7：最佳验证 checkpoint 测试一次，acc/F1 来自同一权重。"""
    print("step 7: evaluate_checkpoint")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    truth, predicted = [], []
    with torch.no_grad():
        for data, labels in loader:
            predicted.extend(model(data.to(device)).argmax(1).cpu().tolist())
            truth.extend(labels.tolist())
    metrics = (saved["best_val_acc"], accuracy_score(truth, predicted),
               f1_score(truth, predicted, labels=list(range(classes)), average="macro", zero_division=0))
    print("best_val_acc, test_acc, macro_f1 =", metrics)
    return metrics


def run_model(seed, name, steps, model_name, split, pipeline_log, params, classes, split_digest, run_digest, error):
    output = OUTPUT_ROOT / f"result{seed}" / f"{seed}+{name}+{model_name}"
    output.mkdir(parents=True, exist_ok=True)
    row = dict.fromkeys(SUMMARY_FIELDS, "")
    row.update(seed=seed, combo_name=name,
               model=model_name, output_dir=str(output),
               split_digest=split_digest, run_digest=run_digest, status="failed")
    with (output / "train_process.txt").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f"seed={seed}; {STAGE}={name}; model={model_name}")
            print("具体算法:", json_text(steps))
            print("训练配置:", json_text(params))
            print("样本/划分清单:", SPLIT_ROOT, "label_mapping=", classes)
            print(pipeline_log, end="")
            try:
                if error:
                    raise RuntimeError(error)
                set_seed(seed)
                loaders = build_loaders(split, steps, params["batch"], seed)
                model, device = create_registered_model(model_name, len(classes), tuple(loaders[0].dataset.data_list.shape[1:]))
                checkpoint, elapsed = train_registered_model(model, device, loaders, params, output)
                val, test, f1 = evaluate_checkpoint(model, device, loaders[2], checkpoint, len(classes))
                row.update(best_val_acc=val, test_acc=test, macro_f1=f1, training_seconds=elapsed, status="ok")
            except Exception:
                traceback.print_exc()
    return row


def save_summary(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())
    temp.replace(path)


def read_summary(path):
    """读取当前阶段的结果，用于断点续跑。"""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_experiments(executor, start=0, stop=160):
    params = load_params(DATASET_NAME)
    if DATASET_NAME == "xrf55":
        params["num_epochs"] = 50
    experiments = build_experiments()[start:stop]
    print(f"组合范围: 第 {start + 1}–{stop} 种（沿用完整列表顺序）", flush=True)
    models = list(list_models()) if STAGE == "preset" else [MODEL_NAME]
    print(DATASET_NAME, STAGE, "seeds=", SEEDS, "pipelines=", len(experiments), "models=", models, "params=", params)
    if "--check" in sys.argv:
        return
    # step 1：读取一次，并在任何算法执行前固定三份划分。
    raw, labels, groups, classes, sample_digest = load_raw_data(executor)
    splits = {seed: prepare_split_indices(labels, groups, seed, sample_digest) for seed in SEEDS}
    summaries, paths = {}, {}
    for seed in SEEDS:
        directory = OUTPUT_ROOT / f"result{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        paths[seed] = directory / f"{DATA_FOLDER}_{STAGE}_{start + 1}_{stop}_summary.csv"
        rows = {}
        if paths[seed].exists():
            rows = {(row["combo_name"], row["model"]): row
                    for row in read_summary(paths[seed])}
        # 旧服务器保存的是绝对路径；确认当前目录的文件齐全后更新汇总表。
        relocated = False
        for row in rows.values():
            current_output = directory / f"{seed}+{row['combo_name']}+{row['model']}"
            if (row.get("status") == "ok" and row.get("output_dir") != str(current_output)
                    and all((current_output / filename).is_file()
                            for filename in ("best_checkpoint.pth", "loss_curve.png", "train_process.txt"))):
                row["output_dir"] = str(current_output)
                relocated = True
        if relocated:
            save_summary(paths[seed], rows)
        summaries[seed] = rows
    for name, steps in experiments:
        run_digest = digest({"params": params, "steps": steps, "sample_digest": sample_digest})
        pending = {}
        for seed in SEEDS:
            pending[seed] = []
            for model in models:
                old = summaries[seed].get((name, model), {})
                if old.get("status") == "failed":
                    continue
                complete = (old.get("status") == "ok" and old.get("run_digest") == run_digest
                            and old.get("split_digest") == splits[seed][1]
                            and all((Path(old.get("output_dir", "")) / filename).is_file()
                                    for filename in ("best_checkpoint.pth", "loss_curve.png", "train_process.txt")))
                if not complete:
                    pending[seed].append(model)
        if not any(pending.values()):
            continue
        error, data = "", None
        process_capture = io.StringIO()
        try:
            set_seed(SEEDS[0])
            # step 2：处理一次，仅复制统计信息供所有种子/模型的日志复用。
            data = process_data(raw, steps, params["padding_length"],
                                labels, groups, classes, executor, summary_log=process_capture)
        except Exception:
            error = traceback.format_exc()
        for seed in SEEDS:
            if not pending[seed]:
                continue
            split, split_error = None, error
            capture = io.StringIO()
            try:
                with contextlib.redirect_stdout(capture):
                    if not split_error:
                        # step 3：同一预设同一种子仅划分一次，日志复制到每个模型。
                        split = split_data(data, labels, groups, splits[seed][0], steps, splits[seed][1])
            except Exception:
                split_error = traceback.format_exc()
            for model in pending[seed]:
                # step 4–7：DataLoader → 模型 → 训练 → 最佳 checkpoint 测试。
                row = run_model(seed, name, steps, model, split, process_capture.getvalue() + capture.getvalue(), params, classes,
                                splits[seed][1], run_digest, split_error)
                summaries[seed][(name, model)] = row
                save_summary(paths[seed], summaries[seed])
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del split
        del data
    print("实验结束；失败项已记录，正常续跑会跳过。")


def main(start=0, stop=160):
    # spawn 在 Windows/Linux 上使用相同的启动方式，工作进程不继承主进程 CUDA 状态。
    # 读取和全部预处理共用此池；退出脚本时统一关闭。
    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS,
                             mp_context=mp.get_context("spawn"),
                             initializer=initialize_worker) as executor:
        run_experiments(executor, start, stop)


if __name__ == "__main__":
    main()
