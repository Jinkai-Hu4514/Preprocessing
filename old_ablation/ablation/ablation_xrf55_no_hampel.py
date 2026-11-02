"""XRF55 不使用 Hampel 的严格对照实验。

作用
----
本脚本补充 ``ablation_xrf55_hampel.py`` 中缺失的 ``no_hampel`` 基线，
用于回答：在其余处理完全相同的条件下，XRF55 是否真的需要 Hampel。

固定处理流程为：

    不使用 Hampel
    -> IQR(factor=1.5)
    -> 仅用训练 repetition 拟合 z-score
    -> cubic15
    -> ResNet1D

数据范围和划分与现有 XRF55 Hampel 窗口实验保持一致：

* 默认只使用用户 1--3；
* repetition 1--12 为训练集，13--16 为验证集，17--20 为测试集；
* 默认模型随机种子和划分随机种子均为 42；
* 默认训练 50 epoch，并使用与现有实验相同的 WSDP 参数。

输出
----
结果写入本文件同目录下的 ``ablation_xrf55_no_hampel_result``：

* ``experiment_settings.json``：完整实验条件；
* ``dataset_metadata.json``：数据规模和帧数检查；
* ``preprocessing_metadata.json``：确认流水线中没有 Hampel；
* ``hampel_stage_identity.json``：跳过 Hampel 时信号恒等映射的定义；
* ``training_summary.csv``：逐随机种子验证/测试准确率；
* ``training_aggregate.csv``：多随机种子均值和标准差；
* ``comparison_with_default_hampel.csv``：与现有默认 11 帧结果对比；
* ``figures/``：准确率对比科研图；
* ``runs/no_hampel/seed_*/``：日志、checkpoint、训练曲线和混淆矩阵。

本文件是独立实验入口，不导入其他消融实验脚本；只复用项目中的 WSDP
源码接口。禁用 Hampel 的正确实现是从 pipeline 中省略 ``denoise``，
而不是传入非法的 ``half_window=0``。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import os
import random
import re
import sys
import tempfile
import time
import traceback
import types
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterable


# 必须在导入 PyTorch/Matplotlib 之前设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "wsdp_xrf55_no_hampel_mpl"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 路径与 WSDP 源码导入
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULT_ROOT = SCRIPT_DIR / "ablation_xrf55_no_hampel_result"
FIGURE_DIR = RESULT_ROOT / "figures"
RUNS_DIR = RESULT_ROOT / "runs"

EXISTING_HAMPEL_RESULT = SCRIPT_DIR / "ablation_xrf55_hampel_result"
EXISTING_HAMPEL_SUMMARY = EXISTING_HAMPEL_RESULT / "training_summary.csv"

WSDP_SRC_CANDIDATES = (
    PROJECT_ROOT
    / "SDP"
    / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
    / "src",
    PROJECT_ROOT
    / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
    / "src",
)
WSDP_SRC = next(
    (candidate for candidate in WSDP_SRC_CANDIDATES if (candidate / "wsdp").is_dir()),
    None,
)
if WSDP_SRC is None:
    raise FileNotFoundError(
        "找不到本地 WSDP 源码目录，检查过："
        + ", ".join(str(path) for path in WSDP_SRC_CANDIDATES)
    )

sys.path.insert(0, str(WSDP_SRC))
sys.modules.setdefault("kagglehub", types.ModuleType("kagglehub"))

from wsdp import readers  # noqa: E402
from wsdp.algorithms import execute_pipeline  # noqa: E402
from wsdp.core import _create_data_split, _evaluate_model  # noqa: E402
from wsdp.datasets import CSIDataset  # noqa: E402
from wsdp.models import create_model  # noqa: E402
from wsdp.processors.base_processor import (  # noqa: E402
    _parse_file_info_from_filename,
    _selector,
)
from wsdp.utils import (  # noqa: E402
    load_params,
    resize_csi_to_fixed_length,
    train_model,
)


# ---------------------------------------------------------------------------
# 固定实验参数
# ---------------------------------------------------------------------------

DATASET_NAME = "xrf55"
MODEL_NAME = "resnet1d"
IQR_FACTOR = 1.5
TARGET_SUBCARRIERS = 15
PADDING_LENGTH = 1000
NOMINAL_FS_HZ = 200.0
ACTION_DURATION_SECONDS = 5.0
EXPECTED_FRAMES = int(NOMINAL_FS_HZ * ACTION_DURATION_SECONDS)
DEFAULT_MODEL_SEEDS = (42,)
DEFAULT_SPLIT_SEED = 42

SETTINGS_PATH = RESULT_ROOT / "experiment_settings.json"
DATASET_METADATA_PATH = RESULT_ROOT / "dataset_metadata.json"
PREPROCESSING_METADATA_PATH = RESULT_ROOT / "preprocessing_metadata.json"
IDENTITY_PATH = RESULT_ROOT / "hampel_stage_identity.json"
TRAINING_SUMMARY_PATH = RESULT_ROOT / "training_summary.csv"
TRAINING_AGGREGATE_PATH = RESULT_ROOT / "training_aggregate.csv"
COMPARISON_PATH = RESULT_ROOT / "comparison_with_default_hampel.csv"

TRAINING_FIELDS = [
    "condition",
    "hampel_enabled",
    "iqr_factor",
    "normalization",
    "interpolation",
    "model",
    "model_seed",
    "split_seed",
    "epochs",
    "status",
    "best_val_acc",
    "test_acc",
    "train_size",
    "val_size",
    "test_size",
    "input_shape",
    "checkpoint",
    "output_dir",
    "duration_sec",
    "error",
]

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "black": "#000000",
}


class Tee:
    """同时把训练输出写到终端和日志。"""

    def __init__(self, *files):
        self.files = files

    def write(self, data: str) -> None:
        for file in self.files:
            file.write(data)

    def flush(self) -> None:
        for file in self.files:
            file.flush()


def configure_publication_style() -> None:
    """设置适合论文的统一绘图样式。"""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.linewidth": 0.8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.6,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def set_seed(seed: int) -> None:
    """固定模型初始化、DataLoader 和 CUDA 随机性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_unique_positive_ints(values: Iterable[int], name: str) -> list[int]:
    result = sorted(set(int(value) for value in values))
    if not result or result[0] < 1:
        raise ValueError(f"{name} 必须包含至少一个正整数")
    return result


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------

def resolve_data_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    candidates = (
        PROJECT_ROOT / "sdp_dataset" / "xrf55" / "wifi",
        PROJECT_ROOT / "sdp_dataset" / "xrf55",
        PROJECT_ROOT.parent / "sdp_dataset" / "xrf55" / "wifi",
        PROJECT_ROOT.parent / "sdp_dataset" / "xrf55",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def is_selected_user_file(file_path: Path, user_limit: int) -> bool:
    """XRF55 文件名格式为 user_action_repetition。"""
    match = re.search(r"(\d+)_(\d+)_(\d+)", file_path.stem)
    return bool(match and int(match.group(1)) <= user_limit)


def load_raw_records(data_path: Path, user_limit: int) -> list:
    """使用 WSDP XRF55 reader 读取前 N 个用户。"""
    if not data_path.is_dir():
        raise FileNotFoundError(f"找不到 XRF55 数据目录: {data_path}")

    reader_class = readers.get_reader_class(DATASET_NAME)
    reader = reader_class()
    files = [
        path
        for path in sorted(data_path.rglob("*"))
        if path.is_file()
        and "truth" not in path.name.lower()
        and is_selected_user_file(path, user_limit)
    ]
    if not files:
        raise RuntimeError(f"没有找到前 {user_limit} 个用户的 XRF55 文件")

    records = []
    skipped = 0
    for index, file_path in enumerate(files, start=1):
        if not reader.sniff(str(file_path)):
            skipped += 1
            continue
        loaded = reader.read_file(str(file_path))
        if isinstance(loaded, list):
            records.extend(loaded)
        else:
            records.append(loaded)
        if index % 100 == 0 or index == len(files):
            print(f"读取 XRF55 文件: {index}/{len(files)}")

    if not records:
        raise RuntimeError(f"reader 未能从 {data_path} 读取有效样本")
    print(f"有效样本数: {len(records)}；跳过文件数: {skipped}")
    return records


def dataset_metadata(records: list, data_path: Path, user_limit: int) -> dict:
    lengths = np.asarray([len(record.frames) for record in records], dtype=int)
    scalar_values = 0
    channel_shapes: set[tuple[int, ...]] = set()
    for record in records:
        if not record.frames:
            continue
        frame_shape = tuple(np.asarray(record.frames[0].csi_array).shape)
        channel_shapes.add(frame_shape)
        scalar_values += len(record.frames) * int(np.prod(frame_shape))

    return {
        "data_path": str(data_path),
        "samples": len(records),
        "user_limit": user_limit,
        "frame_count_min": int(np.min(lengths)),
        "frame_count_median": float(np.median(lengths)),
        "frame_count_mean": float(np.mean(lengths)),
        "frame_count_max": int(np.max(lengths)),
        "samples_with_expected_1000_frames": int(
            np.count_nonzero(lengths == EXPECTED_FRAMES)
        ),
        "fraction_with_expected_1000_frames": float(
            np.mean(lengths == EXPECTED_FRAMES)
        ),
        "frame_channel_shapes": [list(shape) for shape in sorted(channel_shapes)],
        "total_raw_scalar_values": int(scalar_values),
        "nominal_fs_hz": NOMINAL_FS_HZ,
        "nominal_action_duration_seconds": ACTION_DURATION_SECONDS,
    }


# ---------------------------------------------------------------------------
# 严格 no-Hampel 预处理
# ---------------------------------------------------------------------------

def build_pipeline_steps() -> dict[str, dict[str, Any]]:
    """返回与现有实验同口径、但不包含 denoise/Hampel 的流水线。"""
    return {
        "outliers": {
            "method": "iqr",
            "factor": IQR_FACTOR,
        },
        "normalize": {
            "method": "z-score",
        },
        "interpolate": {
            "method": "cubic",
            "target_K": TARGET_SUBCARRIERS,
        },
    }


def process_record_worker(record, pipeline_steps: dict):
    """处理一个样本；z-score 留到 repetition 划分后拟合。"""
    parsed = _parse_file_info_from_filename(record.file_name, DATASET_NAME)
    if parsed is None:
        return None, None, None
    label, group = _selector(parsed, DATASET_NAME)
    frames = sorted(record.frames, key=lambda frame: frame.timestamp)
    if not frames:
        return None, None, None
    csi = np.stack([frame.csi_array for frame in frames], axis=0)
    if csi.ndim == 2:
        csi = np.expand_dims(csi, -1)
    if csi.ndim != 3 or csi.shape[0] < 2:
        return None, None, None

    # 与 XRF55 现有实验相同：normalize 只能在固定 split 之后用训练集拟合。
    preprocessing_steps = {
        key: value
        for key, value in pipeline_steps.items()
        if key != "normalize"
    }
    if "denoise" in preprocessing_steps:
        raise AssertionError("no_hampel 基线不得包含 denoise 步骤")
    processed = execute_pipeline(
        csi,
        preprocessing_steps,
        dataset=DATASET_NAME,
    )
    return processed, label, group


def preprocess_records(
    records: list,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, dict]:
    """执行 IQR+cubic15，并构造标签和 repetition。"""
    pipeline_steps = build_pipeline_steps()
    worker = partial(process_record_worker, pipeline_steps=pipeline_steps)
    arrays: list[np.ndarray] = []
    raw_labels: list[Any] = []
    raw_groups: list[Any] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, (array, label, group) in enumerate(
            executor.map(worker, records, chunksize=1),
            start=1,
        ):
            if array is not None:
                arrays.append(array)
                raw_labels.append(label)
                raw_groups.append(group)
            if index % 100 == 0 or index == len(records):
                print(f"预处理 no_hampel: {index}/{len(records)}")

    if not arrays:
        raise RuntimeError("no_hampel 没有产生有效样本")

    resized = resize_csi_to_fixed_length(
        arrays,
        target_length=PADDING_LENGTH,
    )
    unique_labels = sorted(set(raw_labels))
    unique_groups = sorted(set(raw_groups))
    label_map = {label: index for index, label in enumerate(unique_labels)}
    group_map = {group: index for index, group in enumerate(unique_groups)}

    processed = np.asarray(resized)
    labels = np.asarray(
        [label_map[label] for label in raw_labels],
        dtype=np.int64,
    )
    groups = np.asarray(
        [group_map[group] for group in raw_groups],
        dtype=np.int64,
    )
    metadata = {
        "condition": "no_hampel",
        "hampel_enabled": False,
        "samples": len(processed),
        "sample_shape": list(processed[0].shape),
        "classes": len(unique_labels),
        "raw_labels": [str(value) for value in unique_labels],
        "repetition_groups": len(unique_groups),
        "pipeline_steps": pipeline_steps,
        "assertion": (
            "pipeline_steps has no denoise key; IQR, train-split z-score and "
            "cubic15 remain enabled"
        ),
    }
    return processed, labels, groups, unique_labels, metadata


def split_xrf55(
    processed: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    pipeline_steps: dict,
    split_seed: int,
):
    """固定使用 XRF55 repetition 12/4/4 划分和训练集 z-score。"""
    return _create_data_split(
        processed,
        labels,
        groups,
        test_split=0.3,
        val_split=0.5,
        seed=split_seed,
        use_simple_split=len(set(groups.tolist())) < 3,
        dataset=DATASET_NAME,
        pipeline_steps=pipeline_steps,
    )


def build_loaders(
    split,
    pipeline_steps: dict,
    batch_size: int,
    model_seed: int,
):
    train_data, val_data, test_data, train_y, val_y, test_y = split

    def make_loader(data, labels, shuffle: bool):
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(model_seed)
        return DataLoader(
            CSIDataset(
                data,
                labels,
                dataset_name=DATASET_NAME,
                pipeline_steps=pipeline_steps,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            generator=generator,
        )

    return (
        make_loader(train_data, train_y, True),
        make_loader(val_data, val_y, False),
        make_loader(test_data, test_y, False),
    )


# ---------------------------------------------------------------------------
# 训练、评估与绘图
# ---------------------------------------------------------------------------

def save_training_history(history: dict, path: Path) -> None:
    """保存历史；显式使用1开始的epoch，避免源码同名字段覆盖。"""
    keys = [key for key in history if key != "epoch"]
    row_count = max((len(history[key]) for key in keys), default=0)
    rows = []
    for index in range(row_count):
        row = {"epoch": index + 1}
        for key in keys:
            row[key] = history[key][index] if index < len(history[key]) else ""
        rows.append(row)
    write_csv(path, rows)


def plot_training_loss(history: dict, output_dir: Path, model_seed: int) -> None:
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if not train_loss or not val_loss:
        return
    epoch_count = min(len(train_loss), len(val_loss))
    epochs = np.arange(1, epoch_count + 1)
    fig, ax = plt.subplots(figsize=(5.4, 3.5), constrained_layout=True)
    ax.plot(epochs, train_loss[:epoch_count], color=OKABE_ITO["blue"], label="Train")
    ax.plot(
        epochs,
        val_loss[:epoch_count],
        color=OKABE_ITO["orange"],
        label="Validation",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"XRF55 without Hampel, model seed={model_seed}")
    ax.legend(frameon=False)
    fig.savefig(output_dir / "loss_curve.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "loss_curve.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_normalized_confusion_matrix(
    predictions: list,
    targets: list,
    class_count: int,
    output_dir: Path,
) -> None:
    matrix = np.zeros((class_count, class_count), dtype=np.float64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix),
        where=row_sums > 0,
    )

    fig, ax = plt.subplots(figsize=(5.8, 5.0), constrained_layout=True)
    image = ax.imshow(
        normalized,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("XRF55 without Hampel")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized proportion")
    fig.savefig(output_dir / "confusion_matrix_normalized.png", dpi=600)
    fig.savefig(output_dir / "confusion_matrix_normalized.pdf")
    plt.close(fig)


def train_one_seed(
    split,
    unique_labels: list,
    pipeline_steps: dict,
    model_seed: int,
    split_seed: int,
    epochs: int,
    params: dict,
) -> dict[str, Any]:
    output_dir = RUNS_DIR / "no_hampel" / f"seed_{model_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_process.txt"
    start_time = time.time()

    row = {
        "condition": "no_hampel",
        "hampel_enabled": False,
        "iqr_factor": IQR_FACTOR,
        "normalization": "train-split z-score",
        "interpolation": f"cubic{TARGET_SUBCARRIERS}",
        "model": MODEL_NAME,
        "model_seed": model_seed,
        "split_seed": split_seed,
        "epochs": epochs,
        "status": "failed",
        "best_val_acc": "",
        "test_acc": "",
        "train_size": "",
        "val_size": "",
        "test_size": "",
        "input_shape": "",
        "checkpoint": "",
        "output_dir": str(output_dir),
        "duration_sec": "",
        "error": "",
    }

    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
            try:
                set_seed(model_seed)
                batch_size = int(params.get("batch", 32))
                loaders = build_loaders(
                    split,
                    pipeline_steps,
                    batch_size,
                    model_seed,
                )
                input_shape = tuple(loaders[0].dataset.data_list.shape[1:])
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = create_model(
                    MODEL_NAME,
                    num_classes=len(unique_labels),
                    input_shape=input_shape,
                ).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(params.get("lr", 3e-4)),
                    weight_decay=float(params.get("wd", 1e-3)),
                )
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.1,
                    patience=5,
                )
                checkpoint = output_dir / "best_checkpoint.pth"

                print("=" * 80)
                print("XRF55 no-Hampel 严格对照")
                print(f"model_seed={model_seed}; split_seed={split_seed}")
                print(f"input_shape={input_shape}; device={device}")
                print(json.dumps(pipeline_steps, ensure_ascii=False, indent=2))
                if "denoise" in pipeline_steps:
                    raise AssertionError("no_hampel pipeline 意外包含 denoise")
                print("=" * 80)

                history = train_model(
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    loaders[0],
                    loaders[1],
                    epochs,
                    device,
                    checkpoint,
                    PADDING_LENGTH,
                )
                save_training_history(
                    history,
                    output_dir / "training_history.csv",
                )
                plot_training_loss(history, output_dir, model_seed)
                if not checkpoint.exists():
                    raise RuntimeError(f"没有生成最佳 checkpoint: {checkpoint}")

                saved = torch.load(checkpoint, map_location=device)
                model.load_state_dict(saved["model_state_dict"])
                predictions, targets, test_acc = _evaluate_model(
                    model,
                    loaders[2],
                    device,
                )
                best_val_acc = float(saved.get("best_val_acc", 0.0)) / 100.0
                plot_normalized_confusion_matrix(
                    predictions,
                    targets,
                    len(unique_labels),
                    output_dir,
                )

                row.update(
                    {
                        "status": "ok",
                        "best_val_acc": best_val_acc,
                        "test_acc": float(test_acc),
                        "train_size": len(loaders[0].dataset),
                        "val_size": len(loaders[1].dataset),
                        "test_size": len(loaders[2].dataset),
                        "input_shape": json.dumps(input_shape),
                        "checkpoint": str(checkpoint),
                    }
                )
                print(
                    f"完成: best_val_acc={best_val_acc:.4f}, "
                    f"test_acc={float(test_acc):.4f}"
                )
                del model, loaders
            except Exception:
                row["error"] = traceback.format_exc()
                print(row["error"])

    row["duration_sec"] = f"{time.time() - start_time:.2f}"
    clear_cuda_cache()
    return row


def successful_no_hampel_rows() -> list[dict[str, str]]:
    """同一seed若重复运行，保留最后一条成功记录。"""
    rows: dict[int, dict[str, str]] = {}
    for row in read_csv_rows(TRAINING_SUMMARY_PATH):
        if row.get("status") != "ok":
            continue
        rows[int(row["model_seed"])] = row
    return [rows[key] for key in sorted(rows)]


def write_training_aggregate() -> list[dict[str, Any]]:
    rows = successful_no_hampel_rows()
    if not rows:
        return []
    test_scores = np.asarray([float(row["test_acc"]) for row in rows], dtype=float)
    val_scores = np.asarray(
        [float(row["best_val_acc"]) for row in rows],
        dtype=float,
    )
    aggregate = [
        {
            "condition": "no_hampel",
            "hampel_enabled": False,
            "seed_count": len(rows),
            "model_seeds": ",".join(row["model_seed"] for row in rows),
            "best_val_acc_mean": float(np.mean(val_scores)),
            "best_val_acc_std": (
                float(np.std(val_scores, ddof=1)) if len(rows) > 1 else 0.0
            ),
            "test_acc_mean": float(np.mean(test_scores)),
            "test_acc_std": (
                float(np.std(test_scores, ddof=1)) if len(rows) > 1 else 0.0
            ),
        }
    ]
    write_csv(TRAINING_AGGREGATE_PATH, aggregate)
    return aggregate


def latest_default_hampel_rows() -> dict[int, dict[str, str]]:
    """读取现有默认 half_window=5（完整11帧）的最后成功结果。"""
    rows: dict[int, dict[str, str]] = {}
    for row in read_csv_rows(EXISTING_HAMPEL_SUMMARY):
        if row.get("status") != "ok":
            continue
        if int(row.get("half_window", -1)) != 5:
            continue
        rows[int(row["model_seed"])] = row
    return rows


def write_comparison_and_plot() -> list[dict[str, Any]]:
    """按相同seed比较 no-Hampel 与默认11帧Hampel。"""
    no_hampel = {
        int(row["model_seed"]): row
        for row in successful_no_hampel_rows()
    }
    default_hampel = latest_default_hampel_rows()
    common_seeds = sorted(set(no_hampel) & set(default_hampel))
    comparison_rows: list[dict[str, Any]] = []
    for seed in common_seeds:
        no_score = float(no_hampel[seed]["test_acc"])
        hampel_score = float(default_hampel[seed]["test_acc"])
        comparison_rows.append(
            {
                "model_seed": seed,
                "no_hampel_test_acc": no_score,
                "default_hampel_11_frames_test_acc": hampel_score,
                "no_hampel_minus_hampel_percentage_points": (
                    100.0 * (no_score - hampel_score)
                ),
                "same_split_seed": (
                    int(no_hampel[seed]["split_seed"])
                    == int(default_hampel[seed]["split_seed"])
                ),
                "comparison_note": (
                    "Both pipelines keep IQR1.5, train-split z-score, "
                    "cubic15 and ResNet1D; only Hampel presence differs."
                ),
            }
        )

    if comparison_rows:
        write_csv(COMPARISON_PATH, comparison_rows)

        no_scores = np.asarray(
            [row["no_hampel_test_acc"] for row in comparison_rows],
            dtype=float,
        )
        hampel_scores = np.asarray(
            [
                row["default_hampel_11_frames_test_acc"]
                for row in comparison_rows
            ],
            dtype=float,
        )
        means = 100.0 * np.asarray(
            [np.mean(no_scores), np.mean(hampel_scores)],
            dtype=float,
        )
        stds = 100.0 * np.asarray(
            [
                np.std(no_scores, ddof=1) if len(no_scores) > 1 else 0.0,
                np.std(hampel_scores, ddof=1) if len(hampel_scores) > 1 else 0.0,
            ],
            dtype=float,
        )

        fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
        x = np.arange(2)
        bars = ax.bar(
            x,
            means,
            yerr=stds if len(comparison_rows) > 1 else None,
            capsize=4,
            width=0.62,
            color=[OKABE_ITO["blue"], OKABE_ITO["orange"]],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["No Hampel", "Hampel\n11 frames"])
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("XRF55: effect of enabling Hampel")
        lower = max(0.0, float(np.min(means) - 8.0))
        upper = min(100.0, float(np.max(means) + 6.0))
        ax.set_ylim(lower, upper)
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.4,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        note = (
            f"Matched model seeds: {len(comparison_rows)}"
            if len(comparison_rows) > 1
            else f"Single matched seed: {common_seeds[0]}"
        )
        ax.text(
            0.02,
            0.02,
            note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
        )
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            FIGURE_DIR / "figure_1_no_hampel_vs_default.png",
            dpi=600,
            bbox_inches="tight",
        )
        fig.savefig(
            FIGURE_DIR / "figure_1_no_hampel_vs_default.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    return comparison_rows


# ---------------------------------------------------------------------------
# CLI 与主流程
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        help="XRF55数据目录；默认自动查找 sdp_dataset/xrf55[/wifi]",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_MODEL_SEEDS),
        help="模型随机种子；默认42，可传入多个seed",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--user-limit", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使相同seed已有成功结果也重新训练",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="只根据已有结果重新生成汇总和对比图",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_publication_style()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    seeds = parse_unique_positive_ints(args.seeds, "--seeds")
    if args.epochs < 1:
        raise ValueError("--epochs 必须 >= 1")
    if args.user_limit < 1:
        raise ValueError("--user-limit 必须 >= 1")
    if args.workers < 1:
        raise ValueError("--workers 必须 >= 1")

    pipeline_steps = build_pipeline_steps()
    if "denoise" in pipeline_steps:
        raise AssertionError("no_hampel pipeline 不得包含 denoise")

    write_json(
        SETTINGS_PATH,
        {
            "research_question": (
                "Does XRF55 benefit from Hampel when all downstream "
                "preprocessing and the model are held fixed?"
            ),
            "condition": "no_hampel",
            "single_changed_factor": (
                "Hampel disabled versus the existing default 11-frame Hampel"
            ),
            "hampel_enabled": False,
            "fixed_pipeline": {
                "outliers": {"method": "iqr", "factor": IQR_FACTOR},
                "normalize": {
                    "method": "z-score",
                    "fit": "training repetitions only",
                },
                "interpolate": {
                    "method": "cubic",
                    "target_K": TARGET_SUBCARRIERS,
                },
                "model": MODEL_NAME,
            },
            "dataset_scope": {
                "users": f"1-{args.user_limit}",
                "split": {
                    "train_repetitions": "1-12",
                    "validation_repetitions": "13-16",
                    "test_repetitions": "17-20",
                },
            },
            "model_seeds": seeds,
            "split_seed": args.split_seed,
            "epochs": args.epochs,
            "physical_gpu": 1,
            "result_root": str(RESULT_ROOT),
            "comparison_source": str(EXISTING_HAMPEL_SUMMARY),
        },
    )

    if args.plot_only:
        write_training_aggregate()
        comparison = write_comparison_and_plot()
        if not comparison:
            print("没有找到相同seed的no-Hampel和默认11帧Hampel成功结果")
        else:
            print(f"已生成对比结果: {COMPARISON_PATH}")
        return

    data_path = resolve_data_path(args.data_path)
    records = load_raw_records(data_path, args.user_limit)
    metadata = dataset_metadata(records, data_path, args.user_limit)
    write_json(DATASET_METADATA_PATH, metadata)
    write_json(
        IDENTITY_PATH,
        {
            "stage": "Hampel only",
            "hampel_enabled": False,
            "input_scalar_values": metadata["total_raw_scalar_values"],
            "output_scalar_values": metadata["total_raw_scalar_values"],
            "replaced_values": 0,
            "replacement_rate": 0.0,
            "waveform_correlation": 1.0,
            "total_variation_retention": 1.0,
            "dynamic_peak_retention": 1.0,
            "important_note": (
                "These identity metrics describe the skipped Hampel stage "
                "only. The following IQR step can still change CSI values."
            ),
        },
    )

    existing = {
        int(row["model_seed"])
        for row in successful_no_hampel_rows()
    }
    pending_seeds = seeds if args.force else [seed for seed in seeds if seed not in existing]
    if not pending_seeds:
        print("所有请求seed均已有成功结果，跳过训练")
        write_training_aggregate()
        write_comparison_and_plot()
        return

    print("\n" + "=" * 80)
    print(f"准备no-Hampel基线，待训练seed: {pending_seeds}")
    set_seed(args.split_seed)
    processed, labels, groups, unique_labels, preprocessing_metadata = (
        preprocess_records(records, args.workers)
    )
    write_json(PREPROCESSING_METADATA_PATH, preprocessing_metadata)
    split = split_xrf55(
        processed,
        labels,
        groups,
        pipeline_steps,
        args.split_seed,
    )
    del processed, labels, groups, records
    gc.collect()

    params = load_params(DATASET_NAME)
    for model_seed in pending_seeds:
        row = train_one_seed(
            split,
            unique_labels,
            pipeline_steps,
            model_seed,
            args.split_seed,
            args.epochs,
            params,
        )
        append_csv(TRAINING_SUMMARY_PATH, row, TRAINING_FIELDS)
        write_training_aggregate()
        write_comparison_and_plot()

    del split
    clear_cuda_cache()
    aggregate = write_training_aggregate()
    comparison = write_comparison_and_plot()

    print("\n实验完成")
    print(f"逐seed结果: {TRAINING_SUMMARY_PATH}")
    print(f"汇总结果: {TRAINING_AGGREGATE_PATH}")
    if comparison:
        print(f"与默认11帧Hampel对比: {COMPARISON_PATH}")
    else:
        print("未找到相同seed的默认11帧Hampel结果，暂未生成直接对比")
    if aggregate:
        print(
            "no-Hampel test accuracy mean="
            f"{100.0 * float(aggregate[0]['test_acc_mean']):.2f}%"
        )


if __name__ == "__main__":
    main()
