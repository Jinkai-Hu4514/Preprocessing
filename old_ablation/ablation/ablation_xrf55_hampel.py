"""XRF55 Hampel 半窗口长度的单变量消融实验。

研究问题
--------
XRF55 的标称采样率为 200 packets/s，每个 5 秒动作约有 1000 个 CSI
时间点。WSDP 中 ``window_size`` 表示 Hampel 的半窗口，因此默认值 5
对应 11 帧，即约 50--55 ms。本脚本检验：默认短窗口是否主要替换孤立的
包级尖刺，同时保留连续动作变化，因此没有像低采样率数据那样严重损伤识别。

严格控制
--------
所有实验只改变 Hampel 的 ``window_size``。其余条件固定为：

    Hampel(window_size=变量, n_sigma=3)
    -> IQR(factor=1.5)
    -> train-split z-score
    -> cubic15
    -> ResNet1D

数据范围和划分与现有 XRF55 实验一致：

* 只使用前 3 个用户；
* repetition 1--12 为训练集，13--16 为验证集，17--20 为测试集；
* z-score 只使用训练集统计量；
* 为控制运行时间，默认只运行模型随机种子 42，数据划分保持不变。

输出
----
结果写入本文件同目录下的 ``ablation_xrf55_hampel_result``：

* ``signal_diagnostics.csv``：替换率、连续替换长度、波形相关性、
  总变化量和动态峰值保留率；
* ``training_summary.csv``：每个半窗口、每个模型 seed 的训练结果；
* ``training_aggregate.csv``：按窗口汇总单 seed 结果（仍兼容手动多 seed）；
* ``figures/*.png`` 与 ``figures/*.pdf``：600 dpi 位图和矢量科研图；
* ``runs/``：训练日志、最佳 checkpoint、训练历史和混淆矩阵。

本文件是独立实验入口，不导入项目中的其他实验脚本或公共消融模块；
只复用本仓库 WSDP 源码提供的数据读取、算法、划分、模型与训练接口。
物理 GPU 1 会映射为程序内部的逻辑 ``cuda:0``。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import traceback
import types
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterable


# 必须在导入 PyTorch/Matplotlib 之前设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "wsdp_xrf55_hampel_mpl"),
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
RESULT_ROOT = SCRIPT_DIR / "ablation_xrf55_hampel_result"
FIGURE_DIR = RESULT_ROOT / "figures"
RUNS_DIR = RESULT_ROOT / "runs"

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
from wsdp.algorithms.amplitude import hampel_filter  # noqa: E402
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
# 固定实验设置
# ---------------------------------------------------------------------------

DATASET_NAME = "xrf55"
MODEL_NAME = "resnet1d"
NOMINAL_FS_HZ = 200.0
ACTION_DURATION_SECONDS = 5.0
EXPECTED_FRAMES = int(NOMINAL_FS_HZ * ACTION_DURATION_SECONDS)
N_SIGMA = 3.0
IQR_FACTOR = 1.5
TARGET_SUBCARRIERS = 15
PADDING_LENGTH = 1000
DEFAULT_WINDOWS = (1, 2, 5, 10, 25, 50)
DEFAULT_MODEL_SEEDS = (42,)
DEFAULT_SPLIT_SEED = 42

SIGNAL_DIAGNOSTIC_PATH = RESULT_ROOT / "signal_diagnostics.csv"
SAMPLE_DIAGNOSTIC_PATH = RESULT_ROOT / "signal_diagnostics_per_sample.csv"
RUN_LENGTH_PATH = RESULT_ROOT / "replacement_run_length_histograms.json"
TRAINING_SUMMARY_PATH = RESULT_ROOT / "training_summary.csv"
TRAINING_AGGREGATE_PATH = RESULT_ROOT / "training_aggregate.csv"
SETTINGS_PATH = RESULT_ROOT / "experiment_settings.json"
DATASET_METADATA_PATH = RESULT_ROOT / "dataset_metadata.json"

TRAINING_FIELDS = [
    "half_window",
    "full_window_frames",
    "nominal_window_duration_ms",
    "first_to_last_span_ms",
    "n_sigma",
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
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}


class Tee:
    """同时把训练输出写到终端和日志文件。"""

    def __init__(self, *files):
        self.files = files

    def write(self, data: str) -> None:
        for file in self.files:
            file.write(data)

    def flush(self) -> None:
        for file in self.files:
            file.flush()


def configure_publication_style() -> None:
    """设置适合论文的统一绘图风格。"""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.25,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_publication_figure(fig: plt.Figure, stem: str) -> None:
    """同时保存 600 dpi PNG 和可编辑 PDF。"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def set_seed(seed: int) -> None:
    """固定模型初始化、DataLoader 和 CUDA 的随机性。"""
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


def append_csv(
    path: Path,
    row: dict[str, Any],
    fieldnames: list[str],
) -> None:
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


def full_window_frames(half_window: int) -> int:
    return 2 * half_window + 1


def nominal_window_duration_ms(half_window: int) -> float:
    """11 个采样槽按 200 Hz 计为 55 ms。"""
    return 1000.0 * full_window_frames(half_window) / NOMINAL_FS_HZ


def first_to_last_span_ms(half_window: int) -> float:
    """首尾样本之间只有 2*half_window 个采样间隔。"""
    return 1000.0 * (2 * half_window) / NOMINAL_FS_HZ


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
        raise RuntimeError(f"没有找到前 {user_limit} 个用户的 XRF55 文件: {data_path}")

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


def record_to_real_array(record) -> np.ndarray:
    """将一个记录转换为 XRF55 的实数 (T,F,A) CSI。"""
    frames = sorted(record.frames, key=lambda frame: frame.timestamp)
    if not frames:
        raise ValueError(f"样本没有帧: {record.file_name}")
    csi = np.stack([frame.csi_array for frame in frames], axis=0)
    if csi.ndim == 2:
        csi = np.expand_dims(csi, -1)
    if csi.ndim != 3:
        raise ValueError(f"期望 (T,F,A)，实际为 {csi.shape}: {record.file_name}")
    if np.iscomplexobj(csi):
        max_imag = float(np.max(np.abs(np.imag(csi))))
        if max_imag > 1e-10:
            raise ValueError(
                "本实验针对 XRF55 实数幅度输入，但检测到非零虚部："
                f"{record.file_name}, max|imag|={max_imag}"
            )
        csi = np.real(csi)
    return np.asarray(csi)


# ---------------------------------------------------------------------------
# 信号层诊断
# ---------------------------------------------------------------------------

def contiguous_true_run_lengths(mask: np.ndarray) -> list[int]:
    """统计每个子载波-天线时间序列中连续被替换的长度。"""
    if mask.ndim != 3:
        raise ValueError(f"期望三维替换掩码，实际为 {mask.shape}")
    time_count = mask.shape[0]
    flattened = mask.reshape(time_count, -1)
    lengths: list[int] = []
    for channel in range(flattened.shape[1]):
        vector = flattened[:, channel].astype(np.int8, copy=False)
        padded = np.pad(vector, (1, 1), mode="constant")
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        lengths.extend((ends - starts).astype(int).tolist())
    return lengths


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    x -= np.mean(x)
    y -= np.mean(y)
    denominator = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    if denominator <= 1e-12:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.dot(x, y) / denominator)


def analyse_one_sample(raw: np.ndarray, half_window: int) -> tuple[dict, Counter]:
    """调用真实 WSDP Hampel，并计算单样本信号损伤指标。"""
    filtered = hampel_filter(
        raw,
        window_size=half_window,
        n_sigma=N_SIGMA,
    )
    changed = ~np.isclose(filtered, raw, rtol=1e-10, atol=1e-12, equal_nan=True)
    channel_count = int(np.prod(raw.shape[1:]))
    changed_per_frame = changed.reshape(raw.shape[0], -1).sum(axis=1)
    dense_threshold = max(1, math.ceil(0.01 * channel_count))

    raw_tv = float(np.abs(np.diff(raw, axis=0)).sum())
    filtered_tv = float(np.abs(np.diff(filtered, axis=0)).sum())

    raw_centered = raw - np.median(raw, axis=0, keepdims=True)
    filtered_centered = filtered - np.median(filtered, axis=0, keepdims=True)
    raw_dynamic_peak = float(np.max(np.abs(raw_centered), axis=0).sum())
    filtered_dynamic_peak = float(
        np.max(np.abs(filtered_centered), axis=0).sum()
    )

    run_lengths = contiguous_true_run_lengths(changed)
    run_histogram = Counter(run_lengths)
    raw_scale = max(float(np.std(raw)), 1e-12)
    row = {
        "frames": int(raw.shape[0]),
        "values": int(raw.size),
        "replaced_values": int(changed.sum()),
        "frames_with_any_replacement": int(np.count_nonzero(changed_per_frame)),
        "frames_with_at_least_1pct_channels_replaced": int(
            np.count_nonzero(changed_per_frame >= dense_threshold)
        ),
        "absolute_change_sum": float(np.abs(filtered - raw).sum()),
        "raw_total_variation": raw_tv,
        "filtered_total_variation": filtered_tv,
        "raw_dynamic_peak_sum": raw_dynamic_peak,
        "filtered_dynamic_peak_sum": filtered_dynamic_peak,
        "waveform_correlation": safe_correlation(raw, filtered),
        "normalized_mae": float(np.mean(np.abs(filtered - raw)) / raw_scale),
        "replacement_run_count": len(run_lengths),
    }
    return row, run_histogram


def histogram_quantile(histogram: Counter, quantile: float) -> float:
    total = sum(histogram.values())
    if total == 0:
        return 0.0
    target = quantile * total
    cumulative = 0
    for length, count in sorted(histogram.items()):
        cumulative += count
        if cumulative >= target:
            return float(length)
    return float(max(histogram))


def aggregate_signal_rows(
    half_window: int,
    sample_rows: list[dict],
    run_histogram: Counter,
) -> dict[str, Any]:
    values = sum(int(row["values"]) for row in sample_rows)
    frames = sum(int(row["frames"]) for row in sample_rows)
    replaced = sum(int(row["replaced_values"]) for row in sample_rows)
    run_count = sum(run_histogram.values())
    changed_values_in_short_runs = sum(
        length * count
        for length, count in run_histogram.items()
        if length <= 3
    )
    changed_values_in_all_runs = sum(
        length * count for length, count in run_histogram.items()
    )
    raw_tv = sum(float(row["raw_total_variation"]) for row in sample_rows)
    filtered_tv = sum(
        float(row["filtered_total_variation"]) for row in sample_rows
    )
    raw_peak = sum(float(row["raw_dynamic_peak_sum"]) for row in sample_rows)
    filtered_peak = sum(
        float(row["filtered_dynamic_peak_sum"]) for row in sample_rows
    )

    return {
        "half_window": half_window,
        "full_window_frames": full_window_frames(half_window),
        "nominal_window_duration_ms": nominal_window_duration_ms(half_window),
        "first_to_last_span_ms": first_to_last_span_ms(half_window),
        "diagnostic_samples": len(sample_rows),
        "frames": frames,
        "values": values,
        "replaced_values": replaced,
        "replacement_rate": replaced / max(values, 1),
        "frame_any_replacement_rate": sum(
            int(row["frames_with_any_replacement"]) for row in sample_rows
        )
        / max(frames, 1),
        "frame_1pct_channels_replaced_rate": sum(
            int(row["frames_with_at_least_1pct_channels_replaced"])
            for row in sample_rows
        )
        / max(frames, 1),
        "mean_normalized_mae": float(
            np.mean([float(row["normalized_mae"]) for row in sample_rows])
        ),
        "mean_waveform_correlation": float(
            np.mean([float(row["waveform_correlation"]) for row in sample_rows])
        ),
        "total_variation_retention": filtered_tv / max(raw_tv, 1e-12),
        "dynamic_peak_retention": filtered_peak / max(raw_peak, 1e-12),
        "replacement_run_count": run_count,
        "single_frame_run_fraction": (
            run_histogram.get(1, 0) / run_count if run_count else 0.0
        ),
        "runs_le3_fraction": (
            sum(count for length, count in run_histogram.items() if length <= 3)
            / run_count
            if run_count
            else 0.0
        ),
        "changed_values_in_runs_le3_fraction": (
            changed_values_in_short_runs / changed_values_in_all_runs
            if changed_values_in_all_runs
            else 0.0
        ),
        "run_length_median": histogram_quantile(run_histogram, 0.5),
        "run_length_p90": histogram_quantile(run_histogram, 0.9),
        "run_length_max": max(run_histogram, default=0),
    }


def run_signal_diagnostics(
    records: list,
    windows: list[int],
    diagnostic_samples: int,
) -> tuple[list[dict[str, Any]], dict[int, Counter]]:
    """对真实 XRF55 进行 Hampel 替换形态和信号保留诊断。"""
    selected_records = records[: min(diagnostic_samples, len(records))]
    if not selected_records:
        raise RuntimeError("没有可用于信号诊断的样本")

    aggregate_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    histograms: dict[int, Counter] = {}

    for half_window in windows:
        print(
            f"信号诊断: half_window={half_window}, "
            f"full_window={full_window_frames(half_window)}"
        )
        window_rows: list[dict[str, Any]] = []
        histogram = Counter()
        for sample_index, record in enumerate(selected_records, start=1):
            raw = record_to_real_array(record)
            sample_metrics, sample_histogram = analyse_one_sample(raw, half_window)
            window_rows.append(sample_metrics)
            histogram.update(sample_histogram)
            per_sample_rows.append(
                {
                    "half_window": half_window,
                    "full_window_frames": full_window_frames(half_window),
                    "nominal_window_duration_ms": nominal_window_duration_ms(
                        half_window
                    ),
                    "sample_index": sample_index,
                    "file_name": str(record.file_name),
                    **sample_metrics,
                    "replacement_rate": sample_metrics["replaced_values"]
                    / max(sample_metrics["values"], 1),
                    "total_variation_retention": sample_metrics[
                        "filtered_total_variation"
                    ]
                    / max(sample_metrics["raw_total_variation"], 1e-12),
                    "dynamic_peak_retention": sample_metrics[
                        "filtered_dynamic_peak_sum"
                    ]
                    / max(sample_metrics["raw_dynamic_peak_sum"], 1e-12),
                }
            )

        aggregate_rows.append(
            aggregate_signal_rows(half_window, window_rows, histogram)
        )
        histograms[half_window] = histogram

    write_csv(SIGNAL_DIAGNOSTIC_PATH, aggregate_rows)
    write_csv(SAMPLE_DIAGNOSTIC_PATH, per_sample_rows)
    write_json(
        RUN_LENGTH_PATH,
        {
            str(window): {
                str(length): int(count)
                for length, count in sorted(histogram.items())
            }
            for window, histogram in histograms.items()
        },
    )
    return aggregate_rows, histograms


def select_representative_signal(
    records: list,
    default_half_window: int,
    search_samples: int,
) -> tuple[np.ndarray, int, int, str]:
    """选择默认窗口改变量最大的真实通道用于可视化。"""
    best_score = -1.0
    best: tuple[np.ndarray, int, int, str] | None = None
    for record in records[: min(search_samples, len(records))]:
        raw = record_to_real_array(record)
        filtered = hampel_filter(
            raw,
            window_size=default_half_window,
            n_sigma=N_SIGMA,
        )
        change_by_channel = np.abs(filtered - raw).sum(axis=0)
        flat_index = int(np.argmax(change_by_channel))
        frequency_index, antenna_index = np.unravel_index(
            flat_index,
            change_by_channel.shape,
        )
        score = float(change_by_channel[frequency_index, antenna_index])
        if score > best_score:
            best_score = score
            best = (
                raw,
                int(frequency_index),
                int(antenna_index),
                str(record.file_name),
            )

    if best is None:
        raise RuntimeError("无法选择代表性 XRF55 信号")
    return best


# ---------------------------------------------------------------------------
# 科研绘图
# ---------------------------------------------------------------------------

def add_duration_secondary_axis(ax: plt.Axes) -> None:
    """在帧数横轴上方显示标称物理窗口长度。"""
    forward = lambda frames: 1000.0 * frames / NOMINAL_FS_HZ
    inverse = lambda milliseconds: milliseconds * NOMINAL_FS_HZ / 1000.0
    secondary = ax.secondary_xaxis("top", functions=(forward, inverse))
    secondary.set_xlabel("Nominal duration (ms)")


def plot_signal_metrics(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: int(row["half_window"]))
    x = np.asarray([float(row["full_window_frames"]) for row in rows])
    metrics = (
        ("replacement_rate", "Replaced values (%)", 100.0),
        ("total_variation_retention", "Total variation retained (%)", 100.0),
        ("dynamic_peak_retention", "Dynamic peak retained (%)", 100.0),
        ("single_frame_run_fraction", "Single-frame runs (%)", 100.0),
    )
    colors = (
        OKABE_ITO["blue"],
        OKABE_ITO["green"],
        OKABE_ITO["orange"],
        OKABE_ITO["vermillion"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for ax, (key, ylabel, scale), color in zip(axes.flat, metrics, colors):
        y = np.asarray([float(row[key]) * scale for row in rows])
        ax.plot(x, y, marker="o", color=color)
        ax.set_xlabel("Full Hampel window (frames)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.grid(True)
    add_duration_secondary_axis(axes[0, 0])
    add_duration_secondary_axis(axes[0, 1])
    save_publication_figure(fig, "figure_1_signal_metrics_vs_window")


def plot_run_length_histograms(
    histograms: dict[int, Counter],
    max_display_length: int = 20,
) -> None:
    if not histograms:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["sky"],
        OKABE_ITO["green"],
        OKABE_ITO["orange"],
        OKABE_ITO["vermillion"],
        OKABE_ITO["purple"],
    ]
    x = np.arange(1, max_display_length + 1)
    for color, (window, histogram) in zip(colors, sorted(histograms.items())):
        counts = np.zeros(max_display_length, dtype=float)
        for length, count in histogram.items():
            index = min(int(length), max_display_length) - 1
            counts[index] += count
        if counts.sum() == 0:
            continue
        probability = counts / counts.sum()
        ax.plot(
            x,
            probability,
            marker="o",
            color=color,
            label=(
                f"{full_window_frames(window)} frames "
                f"({nominal_window_duration_ms(window):.0f} ms)"
            ),
        )
    labels = [str(value) for value in x]
    labels[-1] = f"≥{max_display_length}"
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Consecutive replaced samples")
    ax.set_ylabel("Fraction of replacement runs")
    ax.set_yscale("log")
    ax.grid(True, which="both")
    ax.legend(frameon=False, ncol=2)
    save_publication_figure(fig, "figure_2_replacement_run_lengths")


def plot_representative_waveforms(
    records: list,
    windows: list[int],
    diagnostic_samples: int,
) -> None:
    if not records or not windows:
        return
    default_window = 5 if 5 in windows else min(windows, key=lambda value: abs(value - 5))
    raw, frequency_index, antenna_index, file_name = select_representative_signal(
        records,
        default_window,
        diagnostic_samples,
    )
    # 展示全部待比较窗口；默认设置下为 3×2 的六窗口对比图。
    selected_windows = sorted(set(windows))
    time_seconds = np.arange(raw.shape[0], dtype=float) / NOMINAL_FS_HZ
    raw_signal = raw[:, frequency_index, antenna_index].astype(float)
    center = float(np.median(raw_signal))
    scale = max(float(np.std(raw_signal)), 1e-12)
    normalized_raw = (raw_signal - center) / scale

    ncols = 2 if len(selected_windows) > 1 else 1
    nrows = math.ceil(len(selected_windows) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2, 2.0 * nrows + 0.6),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=False,
    )
    axes_flat = axes.ravel()
    normalized_filtered_by_window: dict[int, np.ndarray] = {}
    changed_by_window: dict[int, np.ndarray] = {}

    for ax, window in zip(axes_flat, selected_windows):
        filtered = hampel_filter(raw, window_size=window, n_sigma=N_SIGMA)
        filtered_signal = filtered[:, frequency_index, antenna_index].astype(float)
        normalized_filtered = (filtered_signal - center) / scale
        changed = ~np.isclose(
            filtered_signal,
            raw_signal,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )
        normalized_filtered_by_window[window] = normalized_filtered
        changed_by_window[window] = changed
        ax.plot(
            time_seconds,
            normalized_raw,
            color=OKABE_ITO["black"],
            alpha=0.55,
            label="Raw",
        )
        ax.plot(
            time_seconds,
            normalized_filtered,
            color=OKABE_ITO["blue"],
            label="Hampel",
        )
        if np.any(changed):
            ax.scatter(
                time_seconds[changed],
                normalized_raw[changed],
                s=10,
                facecolors="none",
                edgecolors=OKABE_ITO["vermillion"],
                linewidths=0.8,
                label="Replaced",
                zorder=3,
            )
        is_default = window == default_window
        ax.set_title(
            (
                f"{full_window_frames(window)} frames, "
                f"{nominal_window_duration_ms(window):.0f} ms"
                + (" (default)" if is_default else "")
            ),
            loc="left",
            pad=4,
            fontweight="bold" if is_default else "normal",
        )
        if is_default:
            ax.set_facecolor("#FFF7DE")
            for spine in ax.spines.values():
                spine.set_color(OKABE_ITO["orange"])
                spine.set_linewidth(1.4)
        ax.grid(True)

    for ax in axes_flat[len(selected_windows) :]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle(
        (
            "Hampel window comparison on a representative XRF55 channel: "
            f"F={frequency_index}, A={antenna_index}"
        ),
        y=0.990,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=3,
        frameon=False,
    )
    fig.supxlabel("Time (s)", y=0.015)
    fig.supylabel("Normalized amplitude", x=0.015)
    # 为标题、全局图例及共享坐标标题预留空间。
    fig.tight_layout(rect=(0.035, 0.035, 1.0, 0.910))
    save_publication_figure(fig, "figure_3_representative_waveforms")

    # 单独放大默认窗口中处理前后差异最大的尖刺附近，便于观察替换范围。
    normalized_default = normalized_filtered_by_window[default_window]
    changed_default = changed_by_window[default_window]
    absolute_difference = np.abs(normalized_default - normalized_raw)
    finite_difference = np.where(
        np.isfinite(absolute_difference),
        absolute_difference,
        -np.inf,
    )
    focus_index = (
        int(np.argmax(finite_difference))
        if np.any(np.isfinite(absolute_difference))
        else raw_signal.shape[0] // 2
    )
    zoom_frames = min(raw_signal.shape[0], int(round(0.5 * NOMINAL_FS_HZ)))
    zoom_start = max(0, focus_index - zoom_frames // 2)
    zoom_end = min(raw_signal.shape[0], zoom_start + zoom_frames)
    zoom_start = max(0, zoom_end - zoom_frames)
    zoom_slice = slice(zoom_start, zoom_end)
    zoom_changed = changed_default[zoom_slice]

    zoom_fig, zoom_ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    zoom_ax.plot(
        time_seconds[zoom_slice],
        normalized_raw[zoom_slice],
        color=OKABE_ITO["black"],
        alpha=0.60,
        label="Raw",
    )
    zoom_ax.plot(
        time_seconds[zoom_slice],
        normalized_default[zoom_slice],
        color=OKABE_ITO["blue"],
        label="Hampel",
    )
    if np.any(zoom_changed):
        zoom_time = time_seconds[zoom_slice][zoom_changed]
        zoom_raw = normalized_raw[zoom_slice][zoom_changed]
        zoom_filtered = normalized_default[zoom_slice][zoom_changed]
        zoom_ax.vlines(
            zoom_time,
            zoom_filtered,
            zoom_raw,
            color=OKABE_ITO["vermillion"],
            alpha=0.35,
            linewidth=0.8,
        )
        zoom_ax.scatter(
            zoom_time,
            zoom_raw,
            s=24,
            facecolors="none",
            edgecolors=OKABE_ITO["vermillion"],
            linewidths=1.0,
            label="Replaced",
            zorder=3,
        )
    zoom_ax.set_title(
        (
            f"Default Hampel window: {full_window_frames(default_window)} frames, "
            f"{nominal_window_duration_ms(default_window):.0f} ms"
        )
    )
    zoom_ax.set_xlabel("Time (s)")
    zoom_ax.set_ylabel("Normalized amplitude")
    zoom_ax.set_facecolor("#FFF7DE")
    for spine in zoom_ax.spines.values():
        spine.set_color(OKABE_ITO["orange"])
        spine.set_linewidth(1.2)
    zoom_ax.grid(True)
    zoom_ax.legend(frameon=False, ncol=3)
    save_publication_figure(zoom_fig, "figure_3b_default_window_zoom")

    write_json(
        RESULT_ROOT / "representative_waveform_metadata.json",
        {
            "file_name": file_name,
            "frequency_index": frequency_index,
            "antenna_index": antenna_index,
            "selected_half_windows": selected_windows,
            "default_zoom_half_window": default_window,
            "default_zoom_frame_range": [zoom_start, zoom_end],
            "default_zoom_time_seconds": [
                float(time_seconds[zoom_start]),
                float(time_seconds[zoom_end - 1]),
            ],
            "selection_rule": (
                "channel with the largest absolute Hampel change under "
                f"half_window={default_window} among diagnostic samples"
            ),
        },
    )


def plot_training_loss(
    history: dict,
    output_dir: Path,
    half_window: int,
    model_seed: int,
) -> None:
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if not train_loss or not val_loss:
        return
    epochs = np.arange(1, min(len(train_loss), len(val_loss)) + 1)
    fig, ax = plt.subplots(figsize=(5.4, 3.5), constrained_layout=True)
    ax.plot(
        epochs,
        train_loss[: len(epochs)],
        color=OKABE_ITO["blue"],
        label="Train",
    )
    ax.plot(
        epochs,
        val_loss[: len(epochs)],
        color=OKABE_ITO["orange"],
        label="Validation",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(
        f"Hampel half-window={half_window}, model seed={model_seed}"
    )
    ax.grid(True)
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
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized proportion")
    fig.savefig(output_dir / "confusion_matrix_normalized.png", dpi=600)
    fig.savefig(output_dir / "confusion_matrix_normalized.pdf")
    plt.close(fig)


def plot_accuracy_results(
    aggregate_rows: list[dict[str, Any]],
    successful_rows: list[dict[str, Any]],
) -> None:
    if not aggregate_rows:
        return
    aggregate_rows = sorted(
        aggregate_rows,
        key=lambda row: int(row["half_window"]),
    )
    x = np.asarray(
        [float(row["full_window_frames"]) for row in aggregate_rows]
    )
    means = 100.0 * np.asarray(
        [float(row["mean_test_acc"]) for row in aggregate_rows]
    )
    stds = 100.0 * np.asarray(
        [float(row["std_test_acc"]) for row in aggregate_rows]
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.1), constrained_layout=True)
    ax.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        capsize=3,
        color=OKABE_ITO["blue"],
        ecolor=OKABE_ITO["blue"],
        label="Mean ± SD",
        zorder=3,
    )

    grouped: dict[int, list[float]] = defaultdict(list)
    for row in successful_rows:
        grouped[int(row["half_window"])].append(100.0 * float(row["test_acc"]))
    for half_window, values in sorted(grouped.items()):
        full_frames = full_window_frames(half_window)
        offsets = np.linspace(-0.35, 0.35, len(values)) if len(values) > 1 else [0.0]
        ax.scatter(
            full_frames + np.asarray(offsets),
            values,
            color=OKABE_ITO["orange"],
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
            label="Run seed" if half_window == min(grouped) else None,
        )
    ax.set_xlabel("Full Hampel window (frames)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(x)
    ax.grid(True)
    ax.legend(frameon=False)
    add_duration_secondary_axis(ax)
    save_publication_figure(fig, "figure_4_test_accuracy_vs_window")


def plot_accuracy_signal_relationship(
    aggregate_rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
) -> None:
    if not aggregate_rows or not signal_rows:
        return
    signal_by_window = {
        int(row["half_window"]): row for row in signal_rows
    }
    joined = [
        row
        for row in aggregate_rows
        if int(row["half_window"]) in signal_by_window
    ]
    if not joined:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    relationships = (
        ("total_variation_retention", "Total variation retained (%)"),
        ("dynamic_peak_retention", "Dynamic peak retained (%)"),
    )
    for ax, (key, xlabel) in zip(axes, relationships):
        for row in joined:
            window = int(row["half_window"])
            signal_row = signal_by_window[window]
            x_value = 100.0 * float(signal_row[key])
            y_value = 100.0 * float(row["mean_test_acc"])
            ax.scatter(
                x_value,
                y_value,
                color=OKABE_ITO["blue"],
                edgecolors="white",
                linewidths=0.6,
                s=38,
            )
            ax.annotate(
                f"{full_window_frames(window)}f",
                (x_value, y_value),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean test accuracy (%)")
        ax.grid(True)
    save_publication_figure(fig, "figure_5_accuracy_vs_signal_retention")


# ---------------------------------------------------------------------------
# 固定单变量预处理
# ---------------------------------------------------------------------------

def build_pipeline_steps(half_window: int) -> dict[str, dict[str, Any]]:
    """除 Hampel 半窗口外，所有参数保持完全一致。"""
    return {
        "denoise": {
            "method": "hampel",
            "window_size": int(half_window),
            "n_sigma": N_SIGMA,
        },
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
    """单样本处理；z-score 留到固定 repetition 划分之后执行。"""
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

    # 与 ConfigurableProcessor 的 XRF55 逻辑一致：normalize 在 split 后
    # 用训练 repetition 的统计量执行，防止验证/测试集泄漏。
    preprocessing_steps = {
        key: value
        for key, value in pipeline_steps.items()
        if key != "normalize"
    }
    processed = execute_pipeline(
        csi,
        preprocessing_steps,
        dataset=DATASET_NAME,
    )
    return processed, label, group


def preprocess_window(
    records: list,
    half_window: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, dict]:
    """执行 Hampel变量+IQR+cubic15，并构造标签和 repetition。"""
    pipeline_steps = build_pipeline_steps(half_window)
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
                print(
                    f"预处理 half_window={half_window}: "
                    f"{index}/{len(records)}"
                )

    if not arrays:
        raise RuntimeError(f"half_window={half_window} 没有产生有效样本")

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
        "samples": len(processed),
        "sample_shape": list(processed[0].shape),
        "classes": len(unique_labels),
        "raw_labels": [str(value) for value in unique_labels],
        "repetition_groups": len(unique_groups),
        "pipeline_steps": pipeline_steps,
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


def save_training_history(history: dict, path: Path) -> None:
    keys = list(history)
    row_count = max((len(history[key]) for key in keys), default=0)
    rows = []
    for index in range(row_count):
        rows.append(
            {
                "epoch": index + 1,
                **{
                    key: history[key][index] if index < len(history[key]) else ""
                    for key in keys
                },
            }
        )
    write_csv(path, rows)


def train_one_seed(
    split,
    unique_labels: list,
    pipeline_steps: dict,
    half_window: int,
    model_seed: int,
    split_seed: int,
    epochs: int,
    params: dict,
) -> dict[str, Any]:
    """在固定数据上只改变模型 seed，训练并评估 ResNet1D。"""
    output_dir = RUNS_DIR / f"half_window_{half_window}" / f"seed_{model_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_process.txt"
    start_time = time.time()

    row = {
        "half_window": half_window,
        "full_window_frames": full_window_frames(half_window),
        "nominal_window_duration_ms": nominal_window_duration_ms(half_window),
        "first_to_last_span_ms": first_to_last_span_ms(half_window),
        "n_sigma": N_SIGMA,
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
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
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
                print("XRF55 Hampel 半窗口单变量消融")
                print(f"half_window={half_window}")
                print(f"full_window={full_window_frames(half_window)} frames")
                print(
                    "nominal_duration="
                    f"{nominal_window_duration_ms(half_window):.1f} ms"
                )
                print(f"model_seed={model_seed}; split_seed={split_seed}")
                print(f"input_shape={input_shape}; device={device}")
                print(json.dumps(pipeline_steps, ensure_ascii=False, indent=2))
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
                plot_training_loss(
                    history,
                    output_dir,
                    half_window,
                    model_seed,
                )
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
                    f"test_acc={test_acc:.4f}"
                )
                del model, loaders
            except Exception:
                row["error"] = traceback.format_exc()
                print(row["error"])

    row["duration_sec"] = f"{time.time() - start_time:.2f}"
    clear_cuda_cache()
    return row


# ---------------------------------------------------------------------------
# 按窗口汇总与断点续跑（默认单 seed，兼容手动多 seed）
# ---------------------------------------------------------------------------

def completed_window_seeds() -> set[tuple[int, int]]:
    return {
        (int(row["half_window"]), int(row["model_seed"]))
        for row in read_csv_rows(TRAINING_SUMMARY_PATH)
        if row.get("status") == "ok"
    }


def successful_training_rows() -> list[dict[str, Any]]:
    """同一窗口/seed 若曾失败后重跑，保留最后一条成功记录。"""
    successful: dict[tuple[int, int], dict[str, Any]] = {}
    for row in read_csv_rows(TRAINING_SUMMARY_PATH):
        if row.get("status") != "ok":
            continue
        successful[(int(row["half_window"]), int(row["model_seed"]))] = row
    return list(successful.values())


def write_training_aggregate() -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in successful_training_rows():
        grouped[int(row["half_window"])].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for half_window, rows in sorted(grouped.items()):
        test_scores = np.asarray(
            [float(row["test_acc"]) for row in rows],
            dtype=float,
        )
        val_scores = np.asarray(
            [float(row["best_val_acc"]) for row in rows],
            dtype=float,
        )
        aggregate_rows.append(
            {
                "half_window": half_window,
                "full_window_frames": full_window_frames(half_window),
                "nominal_window_duration_ms": nominal_window_duration_ms(
                    half_window
                ),
                "first_to_last_span_ms": first_to_last_span_ms(half_window),
                "n_seeds": len(rows),
                "model_seeds": ",".join(
                    str(int(row["model_seed"]))
                    for row in sorted(rows, key=lambda item: int(item["model_seed"]))
                ),
                "mean_best_val_acc": float(np.mean(val_scores)),
                "std_best_val_acc": float(np.std(val_scores, ddof=1))
                if len(val_scores) > 1
                else 0.0,
                "mean_test_acc": float(np.mean(test_scores)),
                "std_test_acc": float(np.std(test_scores, ddof=1))
                if len(test_scores) > 1
                else 0.0,
                "min_test_acc": float(np.min(test_scores)),
                "max_test_acc": float(np.max(test_scores)),
            }
        )
    write_csv(TRAINING_AGGREGATE_PATH, aggregate_rows)
    return aggregate_rows


def load_signal_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv_rows(SIGNAL_DIAGNOSTIC_PATH):
        rows.append(row)
    return rows


def load_run_histograms() -> dict[int, Counter]:
    if not RUN_LENGTH_PATH.exists():
        return {}
    with RUN_LENGTH_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        int(window): Counter(
            {int(length): int(count) for length, count in histogram.items()}
        )
        for window, histogram in payload.items()
    }


def regenerate_summary_figures() -> None:
    """根据已有 CSV/JSON 重画汇总图，便于论文后期调整。"""
    signal_rows = load_signal_rows()
    histograms = load_run_histograms()
    aggregate_rows = write_training_aggregate()
    successful_rows = successful_training_rows()
    plot_signal_metrics(signal_rows)
    plot_run_length_histograms(histograms)
    plot_accuracy_results(aggregate_rows, successful_rows)
    plot_accuracy_signal_relationship(aggregate_rows, signal_rows)


# ---------------------------------------------------------------------------
# CLI 与主流程
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        help="XRF55 数据目录；默认自动查找 sdp_dataset/xrf55[/wifi]",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=list(DEFAULT_WINDOWS),
        help=(
            "Hampel 半窗口；默认 1 2 5 10 25 50，"
            "对应完整窗口 3 5 11 21 51 101 帧"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_MODEL_SEEDS),
        help="模型随机种子；默认只运行 42，手动传入多个值时仍可做多 seed",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--user-limit", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--diagnostic-samples", type=int, default=30)
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="只生成信号指标和图，不训练模型",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="复用已有诊断结果，只执行训练",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="不读取数据和训练，只根据已有结果重新生成汇总图",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_publication_style()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    windows = parse_unique_positive_ints(args.windows, "--windows")
    seeds = parse_unique_positive_ints(args.seeds, "--seeds")
    if args.epochs < 1:
        raise ValueError("--epochs 必须 >= 1")
    if args.user_limit < 1:
        raise ValueError("--user-limit 必须 >= 1")
    if args.workers < 1:
        raise ValueError("--workers 必须 >= 1")
    if args.diagnostic_samples < 1:
        raise ValueError("--diagnostic-samples 必须 >= 1")

    write_json(
        SETTINGS_PATH,
        {
            "research_question": (
                "Does XRF55 Hampel remain safe because its default 11-frame "
                "window covers only about 50-55 ms at 200 Hz?"
            ),
            "single_changed_factor": "Hampel half-window length",
            "half_windows": windows,
            "full_windows": [full_window_frames(value) for value in windows],
            "nominal_window_duration_ms": [
                nominal_window_duration_ms(value) for value in windows
            ],
            "first_to_last_span_ms": [
                first_to_last_span_ms(value) for value in windows
            ],
            "fixed_pipeline": {
                "n_sigma": N_SIGMA,
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
                "nominal_fs_hz": NOMINAL_FS_HZ,
                "nominal_action_duration_seconds": ACTION_DURATION_SECONDS,
                "expected_frames": EXPECTED_FRAMES,
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
        },
    )

    if args.plot_only:
        regenerate_summary_figures()
        print(f"已根据现有结果重新生成图表: {FIGURE_DIR}")
        return

    data_path = resolve_data_path(args.data_path)
    records = load_raw_records(data_path, args.user_limit)
    lengths = [len(record.frames) for record in records]
    dataset_metadata = {
        "data_path": str(data_path),
        "samples": len(records),
        "user_limit": args.user_limit,
        "frame_count_min": int(min(lengths)),
        "frame_count_median": float(np.median(lengths)),
        "frame_count_mean": float(np.mean(lengths)),
        "frame_count_max": int(max(lengths)),
        "samples_with_expected_1000_frames": int(
            np.count_nonzero(np.asarray(lengths) == EXPECTED_FRAMES)
        ),
        "fraction_with_expected_1000_frames": float(
            np.mean(np.asarray(lengths) == EXPECTED_FRAMES)
        ),
        "nominal_fs_hz": NOMINAL_FS_HZ,
        "nominal_action_duration_seconds": ACTION_DURATION_SECONDS,
    }
    write_json(DATASET_METADATA_PATH, dataset_metadata)

    if not args.skip_diagnostics:
        signal_rows, histograms = run_signal_diagnostics(
            records,
            windows,
            args.diagnostic_samples,
        )
        plot_signal_metrics(signal_rows)
        plot_run_length_histograms(histograms)
        plot_representative_waveforms(
            records,
            windows,
            args.diagnostic_samples,
        )
        print(f"信号诊断完成: {SIGNAL_DIAGNOSTIC_PATH}")

    if args.diagnostics_only:
        print(f"诊断模式完成，图表目录: {FIGURE_DIR}")
        return

    params = load_params(DATASET_NAME)
    completed = completed_window_seeds()
    print(f"已完成训练: {len(completed)}/{len(windows) * len(seeds)}")

    for half_window in windows:
        pending_seeds = [
            seed
            for seed in seeds
            if (half_window, seed) not in completed
        ]
        if not pending_seeds:
            print(f"跳过已完成窗口: half_window={half_window}")
            continue

        print("\n" + "=" * 80)
        print(
            f"准备窗口 half_window={half_window}, "
            f"full_window={full_window_frames(half_window)}, "
            f"pending_seeds={pending_seeds}"
        )
        set_seed(args.split_seed)
        processed, labels, groups, unique_labels, metadata = preprocess_window(
            records,
            half_window,
            args.workers,
        )
        write_json(
            RESULT_ROOT / f"preprocessing_metadata_half_window_{half_window}.json",
            metadata,
        )
        pipeline_steps = build_pipeline_steps(half_window)
        split = split_xrf55(
            processed,
            labels,
            groups,
            pipeline_steps,
            args.split_seed,
        )
        del processed, labels, groups
        gc.collect()

        for model_seed in pending_seeds:
            row = train_one_seed(
                split,
                unique_labels,
                pipeline_steps,
                half_window,
                model_seed,
                args.split_seed,
                args.epochs,
                params,
            )
            append_csv(
                TRAINING_SUMMARY_PATH,
                row,
                TRAINING_FIELDS,
            )
            if row["status"] == "ok":
                completed.add((half_window, model_seed))

            aggregate_rows = write_training_aggregate()
            plot_accuracy_results(
                aggregate_rows,
                successful_training_rows(),
            )
            plot_accuracy_signal_relationship(
                aggregate_rows,
                load_signal_rows(),
            )

        del split
        clear_cuda_cache()

    regenerate_summary_figures()
    print("\n实验完成")
    print(f"逐 seed 结果: {TRAINING_SUMMARY_PATH}")
    print(f"按窗口汇总: {TRAINING_AGGREGATE_PATH}")
    print(f"科研图表: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
