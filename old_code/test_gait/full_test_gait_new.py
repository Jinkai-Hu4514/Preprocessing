"""Gait 320 组 pipeline 公共前缀复用版全量实验脚本。

固定模型：resnet2d

组合空间：
denoise(5) -> outliers(2) -> calibrate(4) -> normalize(2) -> interpolate(4)
= 320 组 pipeline。

优化方式：
- 原始数据只读取并解析一次；
- denoise -> outliers -> calibrate 的 40 个公共前缀各处理一次；
- Hampel 去噪结果在其 8 个 outliers -> calibrate 前缀之间复用；
- 每个公共前缀派生 8 个最终组合；
- Gait + z-score 保持源码真实顺序：prefix -> interpolate -> z-score；
- min-max 保持普通 pipeline 顺序：prefix -> min-max -> interpolate。

这个文件不依赖 pipline_gait_steps.py 或 full_test_gait.py，直接调用 WSDP 源码。
"""

from __future__ import annotations

import contextlib
import csv
import gc
import json
import os
import random
import sys
import time
import traceback
import types
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product
from multiprocessing import get_context
from pathlib import Path

# ==================== 环境设置 ====================
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
    "GAIT_CUDA_VISIBLE_DEVICES",
    "0",
)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wsdp_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


# ==================== 路径设置 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WSDP_SRC = (
    PROJECT_ROOT
    / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
    / "src"
)
DATA_ROOT = PROJECT_ROOT.parent / "sdp_dataset"

if not (WSDP_SRC / "wsdp").is_dir():
    raise FileNotFoundError(f"找不到本地 WSDP 源码目录: {WSDP_SRC}")

sys.path.insert(0, str(WSDP_SRC))
sys.modules.setdefault("kagglehub", types.ModuleType("kagglehub"))


# ==================== WSDP 源码模块 ====================
from wsdp import readers
from wsdp.algorithms import execute_pipeline
from wsdp.algorithms.amplitude import normalize_amplitude
from wsdp.core import _create_data_split, _evaluate_model
from wsdp.dataset_policy import real_if_negligible_imaginary
from wsdp.datasets import CSIDataset
from wsdp.models import create_model
from wsdp.processors.base_processor import (
    _parse_file_info_from_filename,
    _selector,
)
from wsdp.utils import load_params, resize_csi_to_fixed_length, train_model


# ==================== 数据集与实验设置 ====================
DATASET_NAME = "gait"
DATA_PATH = DATA_ROOT / "Gait_Dataset" / "CSI_Gait"

RUN_NAME = os.environ.get("GAIT_RUN_NAME", "gait_320_pipeline_optimized")
MODEL_NAME = "mlpmodel"

BATCH_SIZE = None
LEARNING_RATE = None
WEIGHT_DECAY = None
NUM_EPOCHS = 60
PADDING_LENGTH = 1500
TEST_SPLIT = 0.3
VAL_SPLIT = 0.5
SEED = 42
COMBO_START = int(os.environ.get("GAIT_COMBO_START", "258"))
COMBO_END = int(os.environ.get("GAIT_COMBO_END", "289"))
PROCESS_WORKERS = int(
    os.environ.get("GAIT_PROCESS_WORKERS", str(min(16, os.cpu_count() or 1)))
)
PROCESS_CHUNKSIZE = int(os.environ.get("GAIT_PROCESS_CHUNKSIZE", "16"))

if not 1 <= COMBO_START <= COMBO_END <= 320:
    raise ValueError("GAIT_COMBO_START and GAIT_COMBO_END must define a range within 1..320")
if PROCESS_WORKERS < 1:
    raise ValueError("GAIT_PROCESS_WORKERS must be at least 1")
if PROCESS_CHUNKSIZE < 1:
    raise ValueError("GAIT_PROCESS_CHUNKSIZE must be at least 1")

RESULT_DIR_NAME = os.environ.get("GAIT_RESULT_DIR_NAME", "full_tests_new")
RESULT_DIR = Path(__file__).resolve().parent / "result" / RESULT_DIR_NAME
SUMMARY_PATH = RESULT_DIR / f"{RUN_NAME}_{MODEL_NAME}_summary.csv"

SUMMARY_FIELDS = [
    "combo_index",
    "combo_id",
    "combo_name",
    "model",
    "status",
    "best_val_acc",
    "test_acc",
    "denoise",
    "outliers",
    "calibrate",
    "normalize",
    "interpolate",
    "pipeline_steps",
    "output_dir",
    "duration_sec",
    "error",
]


# ==================== 320 组算法组合配置 ====================
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

# Gait 原始子载波维通常是 30，四种方法统一输出 15 个子载波。
INTERPOLATE_OPTIONS = [
    ("linear15", {"method": "linear", "target_K": 15}),
    ("cubic15", {"method": "cubic", "target_K": 15}),
    ("nearest15", {"method": "nearest", "target_K": 15}),
    ("decimate15", {"method": "decimate", "target_K": 15}),
]


class Tee:
    """同时把 stdout 写到终端和 train_process.txt。"""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)

    def flush(self):
        for file in self.files:
            file.flush()


def set_seed(seed: int) -> None:
    """固定随机性，方便复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pipeline_combinations() -> list[dict]:
    """按旧脚本完全相同的顺序生成 320 组组合。"""
    combinations = []

    for combo_index, (denoise, outliers, calibrate, normalize, interpolate) in enumerate(
        product(
            DENOISE_OPTIONS,
            OUTLIER_OPTIONS,
            CALIBRATE_OPTIONS,
            NORMALIZE_OPTIONS,
            INTERPOLATE_OPTIONS,
        ),
        start=1,
    ):
        denoise_name, denoise_cfg = denoise
        outliers_name, outliers_cfg = outliers
        calibrate_name, calibrate_cfg = calibrate
        normalize_name, normalize_cfg = normalize
        interpolate_name, interpolate_cfg = interpolate

        pipeline_steps = {
            "denoise": denoise_cfg.copy(),
            "outliers": outliers_cfg.copy(),
            "calibrate": calibrate_cfg.copy(),
            "normalize": normalize_cfg.copy(),
            "interpolate": interpolate_cfg.copy(),
        }
        combo_id = f"gait_{combo_index:03d}"
        combo_name = "+".join(
            [
                denoise_name,
                outliers_name,
                calibrate_name,
                normalize_name,
                interpolate_name,
            ]
        )

        combinations.append(
            {
                "combo_index": combo_index,
                "prefix_index": (combo_index - 1) // 8 + 1,
                "combo_id": combo_id,
                "combo_name": combo_name,
                "denoise": denoise_name,
                "outliers": outliers_name,
                "calibrate": calibrate_name,
                "normalize": normalize_name,
                "interpolate": interpolate_name,
                "pipeline_steps": pipeline_steps,
            }
        )

    return combinations


def build_prefix_groups(combinations: list[dict]) -> list[list[dict]]:
    """将连续的 320 组按相同前三步划分成 40 组，每组 8 个组合。"""
    groups = [combinations[index : index + 8] for index in range(0, len(combinations), 8)]
    if len(groups) != 40 or any(len(group) != 8 for group in groups):
        raise RuntimeError("Gait 组合没有正确划分成 40 个公共前缀 × 8 个分支")
    return groups


def is_selected_combo(combo: dict) -> bool:
    """Return whether a combination belongs to this process's assigned range."""
    return COMBO_START <= combo["combo_index"] <= COMBO_END


def prefix_steps_for(combo: dict) -> dict:
    steps = combo["pipeline_steps"]
    return {
        "denoise": steps["denoise"].copy(),
        "outliers": steps["outliers"].copy(),
        "calibrate": steps["calibrate"].copy(),
    }


def is_hampel_prefix(prefix_combos: list[dict]) -> bool:
    """Return whether a shared prefix starts with the Hampel denoiser."""
    return prefix_combos[0]["pipeline_steps"]["denoise"].get("method") == "hampel"


def post_hampel_prefix_steps_for(combo: dict) -> dict:
    """Build the per-prefix work that follows the shared Hampel cache."""
    steps = combo["pipeline_steps"]
    return {
        "outliers": steps["outliers"].copy(),
        "calibrate": steps["calibrate"].copy(),
    }


def actual_flow_for(combo: dict) -> str:
    prefix = f"{combo['denoise']} -> {combo['outliers']} -> {combo['calibrate']}"
    if combo["normalize"] == "z-score":
        return f"{prefix} -> {combo['interpolate']} -> z-score -> 训练"
    return f"{prefix} -> min-max -> {combo['interpolate']} -> 训练"


def _parse_single_raw_sample(csi_data):
    """只做一次文件信息解析、帧排序和 (T,F,A) 堆叠。"""
    parsed = _parse_file_info_from_filename(csi_data.file_name, DATASET_NAME)
    if parsed is None:
        return None, None, None

    label, group = _selector(parsed, DATASET_NAME)
    sorted_frames = sorted(csi_data.frames, key=lambda frame: frame.timestamp)
    frame_tensors = [frame.csi_array for frame in sorted_frames]
    if not frame_tensors:
        return None, None, None

    whole_csi = np.stack(frame_tensors, axis=0)
    if whole_csi.ndim == 2:
        whole_csi = np.expand_dims(whole_csi, -1)
    elif whole_csi.ndim == 1:
        return None, None, None
    if whole_csi.shape[0] < 2:
        return None, None, None

    return whole_csi, label, group


def _vectorized_hampel_real(csi: np.ndarray, window_size: int, n_sigma: float) -> np.ndarray:
    """Apply WSDP's Hampel rule to all trailing channels at once."""
    frame_count = csi.shape[0]
    result = csi.copy()

    if frame_count > 2 * window_size:
        windows = np.lib.stride_tricks.sliding_window_view(
            csi,
            window_shape=2 * window_size + 1,
            axis=0,
        )
        medians = np.median(windows, axis=-1)
        median_absolute_deviations = np.median(
            np.abs(windows - medians[..., None]),
            axis=-1,
        )
        centers = csi[window_size:-window_size]
        thresholds = n_sigma * 1.4826 * median_absolute_deviations
        result[window_size:-window_size] = np.where(
            np.abs(centers - medians) > thresholds,
            medians,
            centers,
        )
        edge_indices = list(range(window_size))
        edge_indices.extend(range(frame_count - window_size, frame_count))
    else:
        edge_indices = range(frame_count)

    # WSDP truncates the sliding window at both ends. Only these few edge
    # positions need a small loop; all channels remain vectorized.
    for index in edge_indices:
        lower = max(0, index - window_size)
        upper = min(frame_count, index + window_size + 1)
        window = csi[lower:upper]
        median = np.median(window, axis=0)
        median_absolute_deviation = np.median(np.abs(window - median), axis=0)
        threshold = n_sigma * 1.4826 * median_absolute_deviation
        result[index] = np.where(
            np.abs(csi[index] - median) > threshold,
            median,
            csi[index],
        )

    return result


def vectorized_hampel_filter(csi, window_size=5, n_sigma=3.0):
    """Exact batched equivalent of WSDP's scalar Hampel implementation."""
    if csi.size == 0:
        return csi.copy()
    if csi.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {csi.shape}")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if n_sigma <= 0:
        raise ValueError(f"n_sigma must be > 0, got {n_sigma}")
    if csi.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {csi.shape}")

    if csi.shape[0] < 3:
        return csi.copy()

    result = np.empty_like(csi)
    if np.iscomplexobj(csi):
        result.real = _vectorized_hampel_real(csi.real, window_size, n_sigma)
        result.imag = _vectorized_hampel_real(csi.imag, window_size, n_sigma)
    else:
        result[...] = _vectorized_hampel_real(csi, window_size, n_sigma)
    return result


def _execute_sample(csi, pipeline_steps: dict, return_phase_channels: bool):
    """子进程中处理一个已解析的 CSI 样本。"""
    denoise = pipeline_steps.get("denoise")
    if denoise and denoise.get("method") == "hampel":
        hampel_params = denoise.copy()
        hampel_params.pop("method")
        result = vectorized_hampel_filter(csi, **hampel_params)
        result = real_if_negligible_imaginary(result, DATASET_NAME)
        remaining_steps = {
            category: config
            for category, config in pipeline_steps.items()
            if category != "denoise"
        }
        if remaining_steps:
            result = execute_pipeline(result, remaining_steps, dataset=DATASET_NAME)
    else:
        result = execute_pipeline(csi, pipeline_steps, dataset=DATASET_NAME)
    if return_phase_channels:
        result = normalize_amplitude(
            result,
            method="z-score",
            return_phase_channels=True,
        )
    return result


def load_raw_data():
    print("step 1: load_raw_data")
    data = readers.load_data(str(DATA_PATH), DATASET_NAME)
    print(f"加载样本数: {len(data)}")
    return data


def parse_raw_data_once(executor: ProcessPoolExecutor, csi_data_list):
    """把原始 CSIData 解析成数组一次，后续 40 个公共前缀复用。"""
    print("step 2: parse_raw_data_once")
    raw_arrays, raw_labels, raw_groups = [], [], []

    for csi, label, group in executor.map(
        _parse_single_raw_sample,
        csi_data_list,
        chunksize=PROCESS_CHUNKSIZE,
    ):
        if csi is not None:
            raw_arrays.append(csi)
            raw_labels.append(label)
            raw_groups.append(group)

    if not raw_arrays:
        raise RuntimeError("Gait 原始数据解析后没有有效样本")

    unique_labels = sorted(set(raw_labels))
    unique_groups = sorted(set(raw_groups))
    label_map = {label: index for index, label in enumerate(unique_labels)}
    group_map = {group: index for index, group in enumerate(unique_groups)}

    labels = np.asarray([label_map[label] for label in raw_labels])
    groups = np.asarray([group_map[group] for group in raw_groups])

    print(f"有效样本数: {len(raw_arrays)}")
    print(f"标签分布: {dict(Counter(raw_labels))}")
    print(f"分组分布: {dict(Counter(raw_groups))}")
    print(f"首个原始样本形状: {raw_arrays[0].shape}")

    return raw_arrays, raw_labels, raw_groups, labels, groups, unique_labels


def parallel_execute(
    executor: ProcessPoolExecutor,
    data_list,
    pipeline_steps: dict,
    *,
    return_phase_channels: bool = False,
) -> list[np.ndarray]:
    """使用常驻进程池并行执行一段 pipeline。输入缓存不会被原地修改。"""
    worker = partial(
        _execute_sample,
        pipeline_steps=pipeline_steps,
        return_phase_channels=return_phase_channels,
    )
    return list(executor.map(worker, data_list, chunksize=PROCESS_CHUNKSIZE))


def prepare_model_data(
    final_data,
    raw_labels,
    raw_groups,
    labels,
    groups,
    unique_labels,
) -> np.ndarray:
    """最终算法完成后统一补齐长度，并输出与旧脚本一致的处理摘要。"""
    processed_data = np.asarray(
        resize_csi_to_fixed_length(final_data, target_length=PADDING_LENGTH)
    )
    summary_lines = [
        f"   处理完成: {len(processed_data)} 个样本",
        f"   标签分布: {dict(Counter(raw_labels))}",
        f"   分组分布: {dict(Counter(raw_groups))}",
        f"   样本形状: {processed_data[0].shape}",
        f"   类别数: {len(unique_labels)} (原始标签: {unique_labels})",
    ]
    for line in summary_lines:
        print(line)

    if not (len(processed_data) == len(labels) == len(groups)):
        raise RuntimeError("处理后的数据、标签和分组数量不一致")
    return processed_data


def split_data(processed_data, labels, groups, pipeline_steps):
    print("step 3: split_data")
    split = _create_data_split(
        processed_data,
        labels,
        groups,
        test_split=TEST_SPLIT,
        val_split=VAL_SPLIT,
        seed=SEED,
        use_simple_split=len(set(groups.tolist())) < 3,
        dataset=DATASET_NAME,
        pipeline_steps=pipeline_steps,
    )
    print(f"训练集 {len(split[0])} | 验证集 {len(split[1])} | 测试集 {len(split[2])}")
    return split


def build_loaders(split, pipeline_steps, batch_size: int):
    print("step 4: 构造 DataLoader")
    train_data, val_data, test_data, train_labels, val_labels, test_labels = split

    def make_loader(data, current_labels, shuffle: bool):
        return DataLoader(
            CSIDataset(
                data,
                current_labels,
                dataset_name=DATASET_NAME,
                pipeline_steps=pipeline_steps,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    return (
        make_loader(train_data, train_labels, shuffle=True),
        make_loader(val_data, val_labels, shuffle=False),
        make_loader(test_data, test_labels, shuffle=False),
    )


def create_registered_model(num_classes: int, input_shape):
    print("step 5: create_registered_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        MODEL_NAME,
        num_classes=num_classes,
        input_shape=input_shape,
    ).to(device)
    print(f"模型参数量 {sum(parameter.numel() for parameter in model.parameters())}")
    print(f"训练设备: {device}")
    return model, device


def save_loss_curve(history: dict, output_dir: Path, combo_id: str) -> None:
    if not history.get("train_loss") or not history.get("val_loss"):
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{combo_id}+{MODEL_NAME} Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def train_registered_model(model, device, loaders, params: dict, output_dir: Path, combo_id: str):
    print(f"step 6: 开始训练 {NUM_EPOCHS} 轮")
    train_loader, val_loader, _ = loaders
    criterion = nn.CrossEntropyLoss()
    lr = LEARNING_RATE if LEARNING_RATE is not None else params.get("lr", 3e-4)
    wd = WEIGHT_DECAY if WEIGHT_DECAY is not None else params.get("wd", 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=5,
    )
    checkpoint_path = output_dir / "best_checkpoint.pth"
    history = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        NUM_EPOCHS,
        device,
        checkpoint_path,
        PADDING_LENGTH,
    )
    save_loss_curve(history, output_dir, combo_id)
    print(f"训练完成，最佳模型保存至: {checkpoint_path}")
    return checkpoint_path


def evaluate_checkpoint(model, device, test_loader, checkpoint_path: Path):
    print("step 7: 测试集评估")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, test_acc = _evaluate_model(model, test_loader, device)
    val_acc = checkpoint.get("best_val_acc", 0.0) / 100.0
    print(f"最佳验证准确率: {val_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    return val_acc, test_acc


def load_ok_records() -> set[str]:
    """断点续跑只跳过 status=ok 的当前模型记录；failed 会在下次重试。"""
    if not SUMMARY_PATH.exists():
        return set()

    with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as file:
        return {
            row["combo_id"]
            for row in csv.DictReader(file)
            if row.get("combo_id")
            and row.get("model") == MODEL_NAME
            and row.get("status") == "ok"
        }


def append_summary(row: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_PATH.exists() or SUMMARY_PATH.stat().st_size == 0
    with SUMMARY_PATH.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def output_dir_for(combo: dict) -> Path:
    return RESULT_DIR / f"{combo['combo_id']}+{combo['combo_name']}+{MODEL_NAME}"


def pipeline_steps_to_json(pipeline_steps: dict) -> str:
    return json.dumps(pipeline_steps, ensure_ascii=False, sort_keys=True)


def result_row(
    combo: dict,
    *,
    status: str,
    output_dir: Path,
    duration_sec: float,
    best_val_acc="",
    test_acc="",
    error: str = "",
) -> dict:
    return {
        "combo_index": combo["combo_index"],
        "combo_id": combo["combo_id"],
        "combo_name": combo["combo_name"],
        "model": MODEL_NAME,
        "status": status,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "denoise": combo["denoise"],
        "outliers": combo["outliers"],
        "calibrate": combo["calibrate"],
        "normalize": combo["normalize"],
        "interpolate": combo["interpolate"],
        "pipeline_steps": pipeline_steps_to_json(combo["pipeline_steps"]),
        "output_dir": str(output_dir),
        "duration_sec": f"{duration_sec:.2f}",
        "error": error,
    }


def print_combo_header(combo: dict, combo_total: int, output_dir: Path) -> None:
    print("\n" + "=" * 80)
    print(
        f"当前算法组合: {combo['combo_index']}/{combo_total}"
        f" | 公共前缀: {combo['prefix_index']}/40"
        f" | combo_id={combo['combo_id']}"
        f" | model={MODEL_NAME}"
    )
    print(f"combo_name: {combo['combo_name']}")
    print(f"实际执行流程: {actual_flow_for(combo)}")
    print("公共前缀步骤（本组8个组合只计算一次）:")
    print(json.dumps(prefix_steps_for(combo), ensure_ascii=False, indent=2))
    print(
        "当前分支复用的缓存: "
        + (
            "公共前缀缓存；随后执行当前 interpolate 和特殊 z-score"
            if combo["normalize"] == "z-score"
            else "公共前缀生成的共享 min-max 缓存；随后只执行当前 interpolate"
        )
    )
    print(f"保存目录: {output_dir}")
    print("当前组合的完整 pipeline_steps:")
    print(json.dumps(combo["pipeline_steps"], ensure_ascii=False, indent=2))
    print("=" * 80)


def run_one_combo(
    combo: dict,
    combo_total: int,
    executor: ProcessPoolExecutor,
    branch_source,
    raw_labels,
    raw_groups,
    labels,
    groups,
    unique_labels,
    params: dict,
) -> dict:
    """从公共前缀缓存生成最终分支，然后按旧 step 流程训练和评估。"""
    output_dir = output_dir_for(combo)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    final_data = None
    processed_data = None
    split = None
    loaders = None
    model = None

    with (output_dir / "train_process.txt").open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
            try:
                print_combo_header(combo, combo_total, output_dir)
                set_seed(SEED)

                interpolation_step = {
                    "interpolate": combo["pipeline_steps"]["interpolate"].copy()
                }
                final_data = parallel_execute(
                    executor,
                    branch_source,
                    interpolation_step,
                    return_phase_channels=combo["normalize"] == "z-score",
                )
                print(
                    "分支处理: "
                    + (
                        "interpolate -> z-score -> 实数[归一化幅度, 相位]"
                        if combo["normalize"] == "z-score"
                        else "已缓存min-max -> interpolate"
                    )
                )

                processed_data = prepare_model_data(
                    final_data,
                    raw_labels,
                    raw_groups,
                    labels,
                    groups,
                    unique_labels,
                )
                del final_data

                split = split_data(
                    processed_data,
                    labels,
                    groups,
                    combo["pipeline_steps"],
                )
                processed_data = None
                gc.collect()
                batch_size = (
                    BATCH_SIZE if BATCH_SIZE is not None else params.get("batch", 32)
                )
                loaders = build_loaders(split, combo["pipeline_steps"], batch_size)
                split = None
                gc.collect()
                input_shape = tuple(loaders[0].dataset.data_list.shape[1:])
                print(f"模型实际输入形状: {input_shape}")

                model, device = create_registered_model(len(unique_labels), input_shape)
                checkpoint_path = train_registered_model(
                    model,
                    device,
                    loaders,
                    params,
                    output_dir,
                    combo["combo_id"],
                )
                val_acc, test_acc = evaluate_checkpoint(
                    model,
                    device,
                    loaders[2],
                    checkpoint_path,
                )
                duration_sec = time.time() - start_time
                print("\n" + "=" * 80)
                print(
                    f"组合完成: {combo['combo_index']}/{combo_total}"
                    f" | combo_id={combo['combo_id']}"
                    f" | best_val_acc={val_acc:.4f}"
                    f" | test_acc={test_acc:.4f}"
                    f" | duration={duration_sec:.2f}s"
                )
                print("=" * 80)
                return result_row(
                    combo,
                    status="ok",
                    output_dir=output_dir,
                    duration_sec=duration_sec,
                    best_val_acc=val_acc,
                    test_acc=test_acc,
                )
            except Exception:
                duration_sec = time.time() - start_time
                traceback_text = traceback.format_exc()
                print("\n组合失败，完整错误如下：")
                print(traceback_text)
                return result_row(
                    combo,
                    status="failed",
                    output_dir=output_dir,
                    duration_sec=duration_sec,
                    error=traceback_text.splitlines()[-1],
                )
            finally:
                # 320 次训练逐次释放分支数组、Dataset/DataLoader 和 GPU 模型，
                # 公共前缀缓存由 main() 在当前 8 个组合结束后单独释放。
                final_data = None
                processed_data = None
                split = None
                loaders = None
                model = None
                plt.close("all")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def record_shared_stage_failure(
    combo: dict,
    combo_total: int,
    stage_name: str,
    traceback_text: str,
    start_time: float,
) -> dict:
    """公共前缀或共享 min-max 失败时，为每个受影响组合留下完整记录。"""
    output_dir = output_dir_for(combo)
    output_dir.mkdir(parents=True, exist_ok=True)
    duration_sec = time.time() - start_time

    with (output_dir / "train_process.txt").open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
            print_combo_header(combo, combo_total, output_dir)
            print(f"{stage_name}失败，当前组合无法继续。完整错误如下：")
            print(traceback_text)

    return result_row(
        combo,
        status="failed",
        output_dir=output_dir,
        duration_sec=duration_sec,
        error=traceback_text.splitlines()[-1],
    )


def clear_runtime_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 Gait 数据目录: {DATA_PATH}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    combinations = build_pipeline_combinations()
    prefix_groups = build_prefix_groups(combinations)
    combo_total = len(combinations)
    ok_records = load_ok_records()
    params = load_params(DATASET_NAME)

    print(f"Gait 全量 pipeline 组合数: {combo_total}")
    print(f"本进程组合范围: gait_{COMBO_START:03d} - gait_{COMBO_END:03d}")
    print(f"公共前缀数: {len(prefix_groups)}，每个前缀最多派生 8 个组合")
    print(f"固定模型: {MODEL_NAME}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"结果目录: {RESULT_DIR}")
    print(f"summary: {SUMMARY_PATH}")
    print(f"断点续跑已完成(status=ok): {len(ok_records)}")
    print(
        f"预处理并行配置: workers={PROCESS_WORKERS}, "
        f"chunksize={PROCESS_CHUNKSIZE}"
    )

    # 常驻进程池只创建一次；原始数据也只读取、解析一次。
    # 显式使用 spawn：进程池在第一次提交任务时才真正启动，此时原始数据可能
    # 已占用大量内存。spawn 避免 Linux fork 继承整份父进程地址空间。
    with ProcessPoolExecutor(
        max_workers=PROCESS_WORKERS,
        mp_context=get_context("spawn"),
    ) as executor:
        csi_data_list = load_raw_data()
        (
            raw_arrays,
            raw_labels,
            raw_groups,
            labels,
            groups,
            unique_labels,
        ) = parse_raw_data_once(executor, csi_data_list)
        del csi_data_list
        gc.collect()

        hampel_data = None
        for prefix_index, prefix_combos in enumerate(prefix_groups, start=1):
            pending = [
                combo
                for combo in prefix_combos
                if is_selected_combo(combo) and combo["combo_id"] not in ok_records
            ]
            if not pending:
                print(
                    f"\n跳过公共前缀 {prefix_index}/40："
                    "没有当前范围内的待运行组合"
                )
                continue

            prefix_steps = prefix_steps_for(prefix_combos[0])
            hampel_prefix = is_hampel_prefix(prefix_combos)
            prefix_name = "+".join(
                [
                    prefix_combos[0]["denoise"],
                    prefix_combos[0]["outliers"],
                    prefix_combos[0]["calibrate"],
                ]
            )
            prefix_start = time.time()
            print("\n" + "#" * 80)
            print(f"处理公共前缀 {prefix_index}/40: {prefix_name}")
            print(f"本前缀待运行组合: {len(pending)}/8")
            print(json.dumps(prefix_steps, ensure_ascii=False, indent=2))
            print("#" * 80)

            try:
                if hampel_prefix:
                    if hampel_data is None:
                        hampel_start = time.time()
                        shared_hampel_steps = {
                            "denoise": prefix_steps["denoise"].copy()
                        }
                        print("计算一次共享 Hampel 去噪缓存，供后续 8 个公共前缀复用")
                        hampel_data = parallel_execute(
                            executor,
                            raw_arrays,
                            shared_hampel_steps,
                        )
                        print(
                            "共享 Hampel 去噪缓存完成，"
                            f"耗时 {time.time() - hampel_start:.2f}s"
                        )
                        if all(
                            is_hampel_prefix(group)
                            for group in prefix_groups[prefix_index - 1 :]
                        ):
                            # Hampel 位于当前组合定义的末尾，后续不再需要原始 CSI。
                            raw_arrays = None
                            gc.collect()
                            print("原始 CSI 缓存已释放，保留共享 Hampel 缓存")

                    prefix_data = parallel_execute(
                        executor,
                        hampel_data,
                        post_hampel_prefix_steps_for(prefix_combos[0]),
                    )
                else:
                    prefix_data = parallel_execute(executor, raw_arrays, prefix_steps)
                print(
                    f"公共前缀 {prefix_index}/40 处理完成，"
                    f"耗时 {time.time() - prefix_start:.2f}s"
                )
            except Exception:
                traceback_text = traceback.format_exc()
                print(f"公共前缀 {prefix_index}/40 失败：\n{traceback_text}")
                for combo in pending:
                    row = record_shared_stage_failure(
                        combo,
                        combo_total,
                        "公共前缀处理",
                        traceback_text,
                        prefix_start,
                    )
                    append_summary(row)
                clear_runtime_cache()
                continue

            minmax_data = None
            try:
                # combo 顺序保持 gait_001 ... gait_320，不因缓存优化而改变。
                for combo in prefix_combos:
                    if not is_selected_combo(combo):
                        print(
                            f"跳过范围外组合: {combo['combo_index']}/{combo_total}"
                            f" | {combo['combo_id']}"
                        )
                        continue
                    if combo["combo_id"] in ok_records:
                        print(
                            f"跳过已完成组合: {combo['combo_index']}/{combo_total}"
                            f" | {combo['combo_id']}"
                        )
                        continue

                    if combo["normalize"] == "min-max" and minmax_data is None:
                        minmax_start = time.time()
                        print(f"公共前缀 {prefix_index}/40: 计算一次共享 min-max 缓存")
                        try:
                            minmax_data = parallel_execute(
                                executor,
                                prefix_data,
                                {"normalize": {"method": "min-max"}},
                            )
                            print(
                                "共享 min-max 缓存完成，"
                                f"耗时 {time.time() - minmax_start:.2f}s"
                            )
                            # 四个 z-score 分支在组合顺序中均位于 min-max 之前；
                            # 到这里后续只依赖 minmax_data，可提前释放公共复数缓存。
                            prefix_data = None
                            gc.collect()
                            print("z-score 分支已结束，公共前缀缓存已提前释放")
                        except Exception:
                            traceback_text = traceback.format_exc()
                            print(f"共享 min-max 处理失败：\n{traceback_text}")
                            affected = [
                                item
                                for item in prefix_combos
                                if is_selected_combo(item)
                                and item["normalize"] == "min-max"
                                and item["combo_id"] not in ok_records
                            ]
                            for item in affected:
                                row = record_shared_stage_failure(
                                    item,
                                    combo_total,
                                    "共享 min-max 处理",
                                    traceback_text,
                                    minmax_start,
                                )
                                append_summary(row)
                            break

                    print(
                        f"\n开始组合: {combo['combo_index']}/{combo_total}"
                        f" | combo_id={combo['combo_id']}"
                        f" | combo_name={combo['combo_name']}"
                    )
                    branch_source = (
                        prefix_data if combo["normalize"] == "z-score" else minmax_data
                    )
                    try:
                        row = run_one_combo(
                            combo,
                            combo_total,
                            executor,
                            branch_source,
                            raw_labels,
                            raw_groups,
                            labels,
                            groups,
                            unique_labels,
                            params,
                        )
                    finally:
                        # 避免循环变量继续持有上一分支的大型公共缓存。
                        branch_source = None
                        clear_runtime_cache()

                    append_summary(row)
                    if row["status"] == "ok":
                        ok_records.add(combo["combo_id"])
            finally:
                del prefix_data
                if minmax_data is not None:
                    del minmax_data
                clear_runtime_cache()
                print(f"公共前缀 {prefix_index}/40 的缓存已释放")

        if hampel_data is not None:
            del hampel_data
            clear_runtime_cache()
            print("共享 Hampel 去噪缓存已释放")

    print(f"\n本轮执行结束，汇总已保存到: {SUMMARY_PATH}")
    print("下次启动时只跳过 summary 中 status=ok 的组合，failed 会重新运行。")


if __name__ == "__main__":
    main()
