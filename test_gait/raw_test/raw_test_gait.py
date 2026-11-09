"""原始数据全模型基线；直接运行，--check 仅检查配置。

不执行去噪、异常值处理、相位校准、归一化或子载波插值。
只保留读取/排序/张量堆叠、时间轴截断或补零及源码 CSIDataset 的实数输入表示：
Elder/XRF55 为幅度，Widar/Gait 为幅度与相位；保留原始子载波数。
共享前两个阶段的样本清单、label/group 和三份划分，每个文件独立运行。
acc/F1 范围为 0–1，测试只使用最佳验证 checkpoint。
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
from wsdp.core import _create_data_split
from wsdp.datasets import CSIDataset
from wsdp.models import create_model, list_models
from wsdp.processors.base_processor import _parse_file_info_from_filename, _selector
from wsdp.utils import load_params, resize_csi_to_fixed_length

DATASET_NAME = "gait"
DATA_FOLDER = "gait"
SEEDS = (42, 727, 914)
CUDA_DEVICE_INDEX = 2  # 对应 nvidia-smi 中的 GPU 编号。
PARALLEL_WORKERS = 4  # 22 vCPU / 110 GB 服务器的初始设置，可按实际吞吐调整。
DATA_PATH = PROJECT_ROOT / "datasets" / DATA_FOLDER
OUTPUT_ROOT = Path(__file__).resolve().parent
SPLIT_ROOT = OUTPUT_ROOT.parent / "splits"
TEST_SPLIT, VAL_SPLIT = 0.3, 0.5
PHASE_DATASET = DATASET_NAME in {"widar", "gait"}
SUMMARY_FIELDS = ["seed", "model", "best_val_acc",
                  "test_acc", "macro_f1", "output_dir", "status",
                  "training_seconds", "split_digest", "run_digest"]

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


def verify_or_save(path, value, *, ignore_sample_mtime=False):
    """raw 与前两个阶段共享同一份清单；不悄悄覆盖旧实验的划分。"""
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
    # 匹配格式却无法读取时停止，防止各阶段接受不同的原始样本集合。
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


def prepare_raw_inputs(raw, padding_length, labels, groups, classes, summary_log=None):
    """step 2：只截断/补零以组成 batch，不调用任何信号预处理算法。"""
    def report(*args):
        """只将本步骤的统计信息写入日志，不复制算法内部逐样本提示。"""
        print(*args)
        if summary_log is not None:
            print(*args, file=summary_log)

    report("step 2: prepare_raw_inputs; preprocessing={}")
    report(f"样本数: {len(raw)}")
    report("原始样本形状及数量 (T, F, A):", dict(Counter(tuple(item.shape) for item in raw)))
    report(f"类别数: {len(classes)}; 原始标签: {classes}")
    report("标签映射 (原始标签 -> 训练标签):", dict(zip(classes, range(len(classes)))))
    report("原始标签分布:", {classes[label]: count for label, count in sorted(Counter(labels.tolist()).items())})
    report(f"分组数: {len(set(groups.tolist()))}; 原始分组分布:", dict(sorted(Counter(groups.tolist()).items())))
    report(f"仅做时间轴截断/补零: padding_length={padding_length}; 保留原始子载波和天线维")
    report("模型输入表示:", "幅度+相位实数通道" if PHASE_DATASET else "幅度实数通道")
    data = np.stack(resize_csi_to_fixed_length(raw, target_length=padding_length))
    report(f"长度对齐完成: 样本数={len(data)}, 形状 (N, T, F, A)={data.shape}, dtype={data.dtype}")
    return data


def split_data(data, labels, groups, indices, split_digest):
    """step 3：直接使用共享划分索引，包含 XRF55 固定 trial 划分；不归一化。"""
    print("step 3: split_data; split_digest=", split_digest)
    for name, index in zip(("train", "validation", "test"), indices):
        print(f"{name}: n={len(index)}, groups={sorted(set(groups[index].tolist()))}, labels={dict(Counter(labels[index].tolist()))}")
    return tuple(data[index] for index in indices) + tuple(labels[index] for index in indices)


def build_loaders(split, batch_size, seed):
    """step 4：每个模型重建随机生成器，保证模型顺序不影响 shuffle。"""
    print("step 4: build_loaders; batch=", batch_size)
    return tuple(DataLoader(CSIDataset(split[i], split[i + 3], dataset_name=DATASET_NAME, pipeline_steps={}),
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


def run_model(seed, model_name, split, pipeline_log, params, classes, split_digest, run_digest, error):
    output = OUTPUT_ROOT / f"result{seed}" / f"{seed}+raw+{model_name}"
    output.mkdir(parents=True, exist_ok=True)
    row = dict.fromkeys(SUMMARY_FIELDS, "")
    row.update(seed=seed, model=model_name, output_dir=str(output),
               split_digest=split_digest, run_digest=run_digest, status="failed")
    with (output / "train_process.txt").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f"seed={seed}; run=raw; model={model_name}")
            print("预处理算法: 无；仅做固定长度与模型输入表示适配")
            print("训练配置:", json_text(params))
            print("样本/划分清单:", SPLIT_ROOT, "label_mapping=", classes)
            print(pipeline_log, end="")
            try:
                if error:
                    raise RuntimeError(error)
                set_seed(seed)
                loaders = build_loaders(split, params["batch"], seed)
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


def enabled_models():
    """读取本目录的跳过说明；只过滤待运行模型，不删除历史结果。"""
    policy = json.loads((OUTPUT_ROOT / "model_exclusions.json").read_text(encoding="utf-8"))
    models = []
    for name in list_models():
        rule = policy.get(name.lower(), {})
        if rule.get("skip", False):
            print(f"跳过模型 {name}: {rule['reason']}", flush=True)
        else:
            models.append(name)
    return models


def run_experiments(executor):
    params = load_params(DATASET_NAME)
    if DATASET_NAME == "xrf55":
        params["num_epochs"] = 50
    models = enabled_models()
    print(DATASET_NAME, "raw", "seeds=", SEEDS, "models=", models, "params=", params)
    if "--check" in sys.argv:
        return
    # step 1：读取全量原始数据一次，使用与前两个阶段完全相同的样本清单。
    raw, labels, groups, classes, sample_digest = load_raw_data(executor)
    splits = {seed: prepare_split_indices(labels, groups, seed, sample_digest) for seed in SEEDS}
    run_digest = digest({"run": "raw", "params": params, "steps": {}, "sample_digest": sample_digest})
    data, preparation_error = None, ""
    preparation_log = io.StringIO()
    prepared = False
    for seed in SEEDS:
        directory = OUTPUT_ROOT / f"result{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        summary = directory / f"{DATA_FOLDER}_raw_summary.csv"
        rows = {}
        if summary.exists():
            rows = {row["model"]: row for row in read_summary(summary)}
        # 旧服务器保存的是绝对路径；确认当前目录的文件齐全后更新汇总表。
        relocated = False
        for row in rows.values():
            current_output = directory / f"{seed}+raw+{row['model']}"
            if (row.get("status") == "ok" and row.get("output_dir") != str(current_output)
                    and all((current_output / filename).is_file()
                            for filename in ("best_checkpoint.pth", "loss_curve.png", "train_process.txt"))):
                row["output_dir"] = str(current_output)
                relocated = True
        if relocated:
            save_summary(summary, rows)
        pending = []
        for model in models:
            old = rows.get(model, {})
            complete = (old.get("status") == "ok" and old.get("run_digest") == run_digest
                        and old.get("split_digest") == splits[seed][1]
                        and all((Path(old.get("output_dir", "")) / filename).is_file()
                                for filename in ("best_checkpoint.pth", "loss_curve.png", "train_process.txt")))
            if not complete:
                pending.append(model)
        if not pending:
            continue
        if not prepared:
            # step 2：仅输入长度适配一次，三个种子、全部模型共用。
            try:
                data = prepare_raw_inputs(raw, params["padding_length"], labels, groups, classes,
                                          summary_log=preparation_log)
            except Exception:
                preparation_error = traceback.format_exc()
            prepared = True
        split, error = None, preparation_error
        split_log = io.StringIO()
        try:
            with contextlib.redirect_stdout(split_log):
                if not error:
                    # step 3：按固定索引划分，不拟合任何归一化统计量。
                    split = split_data(data, labels, groups, splits[seed][0], splits[seed][1])
        except Exception:
            error = traceback.format_exc()
        for model in pending:
            # step 4–7：DataLoader → 模型 → 训练 → 最佳验证 checkpoint 测试。
            rows[model] = run_model(seed, model, split, preparation_log.getvalue() + split_log.getvalue(),
                                    params, classes, splits[seed][1], run_digest, error)
            save_summary(summary, rows)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del split
    print("raw 全模型实验结束；失败项已记录，下次运行会重试。")


def main():
    # spawn 在 Windows/Linux 上使用相同的启动方式，工作进程不继承主进程 CUDA 状态。
    # 读取和全部预处理共用此池；退出脚本时统一关闭。
    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS,
                             mp_context=mp.get_context("spawn"),
                             initializer=initialize_worker) as executor:
        run_experiments(executor)


if __name__ == "__main__":
    main()
