"""ElderAL Hampel 时间窗口、信号损伤与分类性能验证实验。

研究问题
--------
ElderAL 每个动作片段保存的 CSI 帧数较少，而且不同片段的帧数不同。
不能把“保存帧数/总时长”直接称为设备采样率，也不能据此断言设备会自适应
改变采样率。本脚本使用 CSV 中真实保存的逐帧 ``timestamp``，依次验证：

1. 每段帧数不同主要来自采集时长不同，还是来自有效帧缺失/间隔不均匀；
2. Hampel 默认半窗口 5（完整窗口 11 帧）在真实数据中到底覆盖多少秒，
   以及占单段动作时长和帧数的多大比例；
3. Hampel 是否把真实动作峰值、快速变化和连续动作区域当成异常值替换；
4. 缩小 Hampel 窗口后，信号保留和 CSI-Time 分类准确率是否同时恢复。

固定训练链路
------------
训练比较一个不使用 Hampel 的参考组，以及多个 Hampel 半窗口。除去是否使用
Hampel 和半窗口大小外，其余条件保持一致：

    [no Hampel 或 Hampel(window_size=变量, n_sigma=3)]
    -> IQR(factor=1.5)
    -> min-max
    -> linear64
    -> padding/truncation 到 80 帧
    -> CSI-Time

默认比较完整窗口 3、5、7、11 帧，只运行一个随机种子 42。所有条件使用相同
的 ElderAL position group 划分。训练固定使用物理 GPU 0；设置
``CUDA_VISIBLE_DEVICES=0`` 后，程序内部设备显示为 ``cuda:0``。

输出
----
结果写入本文件同目录下的 ``ablation_elder_hampel_result``：

* ``timing_per_sample.csv``：每段帧数、采集时长、有效帧频率和11帧跨度；
* ``frame_intervals_seconds.csv``：真实相邻帧时间间隔；
* ``window_span_per_sample.csv``：不同 Hampel 窗口的真实秒数和片段占比；
* ``timing_report.json``：帧数差异原因的判定指标与数据集汇总；
* ``signal_diagnostics.csv``：替换率、MAD=0、峰值和变化量保留率；
* ``signal_diagnostics_per_sample.csv``：逐样本信号诊断；
* ``replacement_run_length_histograms.json``：连续替换长度分布；
* ``training_summary.csv``：单 seed 的验证/测试准确率；
* ``figures/*.png`` 与 ``figures/*.pdf``：600 dpi 位图和可编辑矢量图；
* ``runs/``：训练日志、checkpoint、训练历史、loss 和混淆矩阵。

本文件是独立实验入口，不导入项目中的其他实验脚本或公共消融模块；只导入
本仓库 WSDP 源码提供的数据读取、算法、划分、模型和训练接口。
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
import sys
import tempfile
import time
import traceback
import types
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterable


# 必须在导入 PyTorch 和 Matplotlib 之前固定物理 GPU 0。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "wsdp_elder_hampel_mpl"),
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
RESULT_ROOT = SCRIPT_DIR / "ablation_elder_hampel_result"
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
# 固定实验设置与结果文件
# ---------------------------------------------------------------------------

DATASET_NAME = "elderAL"
MODEL_NAME = "csitime"
ACTION_DURATION_REFERENCE_SECONDS = 5.0
N_SIGMA = 3.0
IQR_FACTOR = 1.5
TARGET_SUBCARRIERS = 64
PADDING_LENGTH = 80
TEST_SPLIT = 0.30
VAL_SPLIT = 0.50
DEFAULT_WINDOWS = (1, 2, 3, 5)
DEFAULT_SEED = 42
DEFAULT_DIAGNOSTIC_SAMPLES = 162

TIMING_SAMPLE_PATH = RESULT_ROOT / "timing_per_sample.csv"
FRAME_INTERVAL_PATH = RESULT_ROOT / "frame_intervals_seconds.csv"
WINDOW_SPAN_SAMPLE_PATH = RESULT_ROOT / "window_span_per_sample.csv"
TIMING_REPORT_PATH = RESULT_ROOT / "timing_report.json"
SIGNAL_DIAGNOSTIC_PATH = RESULT_ROOT / "signal_diagnostics.csv"
SIGNAL_SAMPLE_PATH = RESULT_ROOT / "signal_diagnostics_per_sample.csv"
RUN_LENGTH_PATH = RESULT_ROOT / "replacement_run_length_histograms.json"
TRAINING_SUMMARY_PATH = RESULT_ROOT / "training_summary.csv"
SETTINGS_PATH = RESULT_ROOT / "experiment_settings.json"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#7F7F7F",
}

TRAINING_FIELDS = [
    "condition",
    "half_window",
    "full_window_frames",
    "n_sigma",
    "model",
    "seed",
    "epochs",
    "status",
    "best_val_acc",
    "test_acc",
    "train_size",
    "val_size",
    "test_size",
    "input_shape",
    "test_groups",
    "checkpoint",
    "output_dir",
    "duration_sec",
    "error",
]


class Tee:
    """同时把训练输出写入终端和日志文件。"""

    def __init__(self, *files):
        self.files = files

    def write(self, data: str) -> None:
        for file in self.files:
            file.write(data)

    def flush(self) -> None:
        for file in self.files:
            file.flush()


def configure_publication_style() -> None:
    """使用与 XRF55 消融一致的论文绘图风格。"""
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


def save_publication_figure(
    fig: plt.Figure,
    stem: str,
    directory: Path | None = None,
) -> None:
    """同时保存 600 dpi PNG 和可编辑 PDF。"""
    target = FIGURE_DIR if directory is None else directory
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(target / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def set_seed(seed: int) -> None:
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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(
    path: Path,
    row: dict[str, Any],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_positive_windows(values: Iterable[int]) -> list[int]:
    windows = sorted(set(int(value) for value in values))
    if not windows or windows[0] < 1:
        raise ValueError("--windows 必须包含至少一个正整数")
    return windows


def full_window_frames(half_window: int) -> int:
    return 2 * half_window + 1


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if abs(denominator) > 1e-12 else default


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size < 2:
        return 0.0
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    if denominator <= 1e-12:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.dot(x, y) / denominator)


def coefficient_of_variation(values: Iterable[float]) -> float:
    data = np.asarray([float(value) for value in values], dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size < 2:
        return 0.0
    mean = float(np.mean(data))
    return float(np.std(data, ddof=1) / abs(mean)) if abs(mean) > 1e-12 else 0.0


def percentile(values: Iterable[float], q: float, default: float = 0.0) -> float:
    data = np.asarray(list(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    return float(np.percentile(data, q)) if data.size else default


# ---------------------------------------------------------------------------
# 数据读取、时间戳和样本元数据
# ---------------------------------------------------------------------------

def resolve_data_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = (
        PROJECT_ROOT / "sdp_dataset" / "elderAL",
        PROJECT_ROOT.parent / "sdp_dataset" / "elderAL",
        PROJECT_ROOT / "SDP" / "sdp_dataset" / "elderAL",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def load_raw_records(data_path: Path) -> list:
    if not data_path.is_dir():
        raise FileNotFoundError(f"找不到 ElderAL 数据目录: {data_path}")
    print(f"读取 ElderAL: {data_path}")
    records = readers.load_data(str(data_path), DATASET_NAME)
    if not records:
        raise RuntimeError(f"没有从 {data_path} 读取到 ElderAL 样本")
    valid_records = [record for record in records if getattr(record, "frames", None)]
    skipped = len(records) - len(valid_records)
    if not valid_records:
        raise RuntimeError(f"{data_path} 中的 ElderAL 记录均不含有效帧")
    print(f"ElderAL 有效记录数: {len(valid_records)}；空记录: {skipped}")
    return valid_records


def record_metadata(record) -> dict[str, Any]:
    parsed = _parse_file_info_from_filename(record.file_name, DATASET_NAME)
    if parsed is None:
        return {"user_id": "", "position_id": "", "action_id": ""}
    return {
        "user_id": int(parsed[0]),
        "position_id": int(parsed[1]),
        "action_id": int(parsed[2]),
    }


def record_to_array_and_timestamps(record) -> tuple[np.ndarray, np.ndarray]:
    frames = sorted(record.frames, key=lambda frame: float(frame.timestamp))
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
                "ElderAL 应为实数幅度，但检测到非零虚部："
                f"{record.file_name}, max|imag|={max_imag}"
            )
        csi = np.real(csi)
    timestamps = np.asarray([float(frame.timestamp) for frame in frames], dtype=float)
    return np.asarray(csi), timestamps


TIMESTAMP_FACTORS = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}


def infer_timestamp_scale(
    records: list,
    requested_unit: str,
) -> tuple[str, float, dict[str, Any]]:
    """推断时间戳单位；自动模式仅用于单位换算，不用于改变采样率。"""
    durations_raw: list[float] = []
    positive_intervals_raw: list[float] = []
    for record in records:
        _, timestamps = record_to_array_and_timestamps(record)
        if timestamps.size >= 2:
            duration = float(timestamps[-1] - timestamps[0])
            if duration > 0:
                durations_raw.append(duration)
            diffs = np.diff(timestamps)
            positive_intervals_raw.extend(diffs[diffs > 0].astype(float).tolist())

    if not durations_raw:
        raise RuntimeError("ElderAL 时间戳没有正的时间跨度，无法执行时间诊断")

    median_raw_duration = float(np.median(durations_raw))
    if requested_unit != "auto":
        unit = requested_unit
        factor = TIMESTAMP_FACTORS[unit]
        method = "explicit_cli"
    else:
        candidates: list[tuple[float, str, float, float]] = []
        for unit_name, candidate_factor in TIMESTAMP_FACTORS.items():
            duration_seconds = median_raw_duration * candidate_factor
            if 0.02 <= duration_seconds <= 600.0:
                score = abs(
                    math.log(
                        max(duration_seconds, 1e-12)
                        / ACTION_DURATION_REFERENCE_SECONDS
                    )
                )
            else:
                score = 100.0 + abs(math.log10(max(duration_seconds, 1e-12)))
            candidates.append((score, unit_name, candidate_factor, duration_seconds))
        _, unit, factor, _ = min(candidates, key=lambda item: item[0])
        method = "auto_unit_closest_to_5s_protocol_reference"

    report = {
        "requested_unit": requested_unit,
        "selected_unit": unit,
        "seconds_per_timestamp_unit": factor,
        "inference_method": method,
        "reference_duration_seconds": ACTION_DURATION_REFERENCE_SECONDS,
        "median_raw_duration": median_raw_duration,
        "median_duration_seconds_after_conversion": median_raw_duration * factor,
        "median_positive_interval_raw": percentile(positive_intervals_raw, 50),
        "median_positive_interval_seconds": (
            percentile(positive_intervals_raw, 50) * factor
        ),
        "warning": (
            "auto 只推断 timestamp 的量纲，不代表设备采样率发生变化；"
            "若原始数据说明给出明确单位，请使用 --timestamp-unit 覆盖。"
        ),
    }
    return unit, factor, report


def timestamps_in_seconds(raw_timestamps: np.ndarray, factor: float) -> np.ndarray:
    if raw_timestamps.size == 0:
        return raw_timestamps.astype(float)
    return (raw_timestamps.astype(float) - float(raw_timestamps[0])) * factor


def window_spans_seconds(
    timestamps_seconds: np.ndarray,
    half_window: int,
) -> np.ndarray:
    """逐中心点计算源码截断窗口首尾时间差。"""
    count = len(timestamps_seconds)
    spans = np.zeros(count, dtype=float)
    for index in range(count):
        lo = max(0, index - half_window)
        hi = min(count - 1, index + half_window)
        spans[index] = max(
            0.0,
            float(timestamps_seconds[hi] - timestamps_seconds[lo]),
        )
    return spans


def diagnose_frame_count_cause(
    timing_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [
        row
        for row in timing_rows
        if finite_float(row["duration_seconds"]) > 0
        and int(row["frames"]) >= 2
    ]
    frames = np.asarray([int(row["frames"]) for row in valid], dtype=float)
    durations = np.asarray(
        [finite_float(row["duration_seconds"]) for row in valid],
        dtype=float,
    )
    median_intervals = np.asarray(
        [finite_float(row["median_interval_seconds"]) for row in valid],
        dtype=float,
    )
    median_intervals = median_intervals[median_intervals > 0]

    duration_cv = coefficient_of_variation(durations)
    frame_count_cv = coefficient_of_variation(frames)
    interval_cv = coefficient_of_variation(median_intervals)
    count_duration_corr = safe_correlation(frames, durations)

    if duration_cv <= 0.15 and frame_count_cv > 0.15:
        conclusion = "duration_roughly_fixed_but_retained_frame_count_varies"
        chinese = (
            "多数片段采集时长接近，但保存帧数变化明显；更支持有效帧缺失、"
            "接收间隔不均匀或数据筛选，而不是采集时长不同。"
        )
    elif duration_cv > 0.15 and count_duration_corr >= 0.80 and interval_cv <= 0.25:
        conclusion = "variable_collection_duration_supported"
        chinese = (
            "片段时长变化明显，帧数与时长高度相关，且典型帧间隔较稳定；"
            "支持“固定采集配置下，每次实际采集时长不同”。"
        )
    else:
        conclusion = "mixed_or_inconclusive"
        chinese = (
            "采集时长、帧间隔和保存帧数之间不是单一关系；当前更像多种因素"
            "共同作用，不能只归因于采集时长或采样率。"
        )

    return {
        "valid_samples": len(valid),
        "duration_cv": duration_cv,
        "frame_count_cv": frame_count_cv,
        "median_interval_across_samples_cv": interval_cv,
        "frame_count_duration_correlation": count_duration_corr,
        "decision": conclusion,
        "decision_in_chinese": chinese,
        "thresholds": {
            "roughly_fixed_duration_cv_max": 0.15,
            "variable_duration_correlation_min": 0.80,
            "stable_interval_cv_max": 0.25,
        },
    }


def run_timing_audit(
    records: list,
    windows: list[int],
    timestamp_unit: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """对全部样本执行时间戳、帧数和真实 Hampel 窗口跨度诊断。"""
    selected_unit, factor, unit_report = infer_timestamp_scale(
        records,
        timestamp_unit,
    )
    timing_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    default_window = 5 if 5 in windows else max(windows)

    for sample_index, record in enumerate(records, start=1):
        raw, raw_timestamps = record_to_array_and_timestamps(record)
        timestamps = timestamps_in_seconds(raw_timestamps, factor)
        metadata = record_metadata(record)
        diffs = np.diff(timestamps)
        positive_diffs = diffs[diffs > 0]
        duration = float(timestamps[-1]) if timestamps.size >= 2 else 0.0
        frames = int(raw.shape[0])
        duplicate_intervals = int(np.count_nonzero(diffs == 0))

        for interval_index, value in enumerate(diffs, start=1):
            interval_rows.append(
                {
                    "sample_index": sample_index,
                    "file_name": str(record.file_name),
                    **metadata,
                    "interval_index": interval_index,
                    "interval_seconds": float(value),
                    "is_positive": int(value > 0),
                    "is_duplicate_timestamp": int(value == 0),
                }
            )

        default_spans = window_spans_seconds(timestamps, default_window)
        timing_rows.append(
            {
                "sample_index": sample_index,
                "file_name": str(record.file_name),
                **metadata,
                "frames": frames,
                "duration_seconds": duration,
                "median_interval_seconds": percentile(positive_diffs, 50),
                "p90_interval_seconds": percentile(positive_diffs, 90),
                "max_interval_seconds": percentile(positive_diffs, 100),
                "duplicate_timestamp_intervals": duplicate_intervals,
                "valid_frame_rate_hz": safe_ratio(frames - 1, duration),
                "default_half_window": default_window,
                "default_full_window_frames": full_window_frames(default_window),
                "default_window_median_span_seconds": percentile(default_spans, 50),
                "default_window_p90_span_seconds": percentile(default_spans, 90),
                "default_window_median_fraction_of_duration": safe_ratio(
                    percentile(default_spans, 50),
                    duration,
                ),
                "default_window_frame_fraction": min(
                    full_window_frames(default_window),
                    frames,
                )
                / max(frames, 1),
                "sample_not_longer_than_default_window": int(
                    frames <= full_window_frames(default_window)
                ),
            }
        )

        for half_window in windows:
            spans = window_spans_seconds(timestamps, half_window)
            window_rows.append(
                {
                    "sample_index": sample_index,
                    "file_name": str(record.file_name),
                    **metadata,
                    "half_window": half_window,
                    "full_window_frames": full_window_frames(half_window),
                    "frames": frames,
                    "duration_seconds": duration,
                    "median_span_seconds": percentile(spans, 50),
                    "p25_span_seconds": percentile(spans, 25),
                    "p75_span_seconds": percentile(spans, 75),
                    "p90_span_seconds": percentile(spans, 90),
                    "max_span_seconds": percentile(spans, 100),
                    "median_fraction_of_sample_duration": safe_ratio(
                        percentile(spans, 50),
                        duration,
                    ),
                    "nominal_frame_fraction_of_sample": min(
                        full_window_frames(half_window),
                        frames,
                    )
                    / max(frames, 1),
                }
            )

        if sample_index % 200 == 0 or sample_index == len(records):
            print(f"时间诊断: {sample_index}/{len(records)}")

    cause_report = diagnose_frame_count_cause(timing_rows)
    total_duration = sum(finite_float(row["duration_seconds"]) for row in timing_rows)
    total_intervals = sum(max(int(row["frames"]) - 1, 0) for row in timing_rows)
    durations = [finite_float(row["duration_seconds"]) for row in timing_rows]
    frame_counts = [int(row["frames"]) for row in timing_rows]
    positive_intervals = [
        finite_float(row["interval_seconds"])
        for row in interval_rows
        if int(row["is_positive"]) == 1
    ]
    default_spans = [
        finite_float(row["default_window_median_span_seconds"])
        for row in timing_rows
    ]
    default_fractions = [
        finite_float(row["default_window_median_fraction_of_duration"])
        for row in timing_rows
        if finite_float(row["duration_seconds"]) > 0
    ]

    report = {
        "dataset": DATASET_NAME,
        "actual_records": len(records),
        "total_retained_frames": int(sum(frame_counts)),
        "timestamp_unit": selected_unit,
        "timestamp_unit_report": unit_report,
        "frame_count": {
            "min": int(min(frame_counts)),
            "median": percentile(frame_counts, 50),
            "mean": float(np.mean(frame_counts)),
            "p90": percentile(frame_counts, 90),
            "max": int(max(frame_counts)),
        },
        "duration_seconds": {
            "min": min(durations),
            "median": percentile(durations, 50),
            "mean": float(np.mean(durations)),
            "p90": percentile(durations, 90),
            "max": max(durations),
        },
        "positive_frame_interval_seconds": {
            "median": percentile(positive_intervals, 50),
            "p90": percentile(positive_intervals, 90),
            "p99": percentile(positive_intervals, 99),
            "max": percentile(positive_intervals, 100),
        },
        "aggregate_retained_frame_frequency_hz": safe_ratio(
            total_intervals,
            total_duration,
        ),
        "default_window": {
            "half_window": default_window,
            "full_window_frames": full_window_frames(default_window),
            "median_of_sample_median_span_seconds": percentile(default_spans, 50),
            "p90_of_sample_median_span_seconds": percentile(default_spans, 90),
            "median_fraction_of_sample_duration": percentile(default_fractions, 50),
            "samples_with_frames_le_full_window": sum(
                int(row["sample_not_longer_than_default_window"])
                for row in timing_rows
            ),
            "samples_with_frames_le_full_window_fraction": safe_ratio(
                sum(
                    int(row["sample_not_longer_than_default_window"])
                    for row in timing_rows
                ),
                len(timing_rows),
            ),
        },
        "frame_count_cause_diagnostic": cause_report,
        "terminology": {
            "aggregate_retained_frame_frequency_hz": (
                "数据集中保存下来的有效帧频率，不等于设备配置采样率"
            ),
            "window_span": (
                "用真实 timestamp 计算窗口第一帧到最后一帧的时间差"
            ),
        },
    }
    write_csv(TIMING_SAMPLE_PATH, timing_rows)
    write_csv(FRAME_INTERVAL_PATH, interval_rows)
    write_csv(WINDOW_SPAN_SAMPLE_PATH, window_rows)
    write_json(TIMING_REPORT_PATH, report)
    return timing_rows, interval_rows, window_rows, report


# ---------------------------------------------------------------------------
# Hampel 信号损伤诊断
# ---------------------------------------------------------------------------

def hampel_with_diagnostics(
    raw: np.ndarray,
    half_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """与源码等价的向量化诊断版，额外返回替换和 MAD=0 掩码。"""
    source = np.asarray(raw)
    filtered = source.copy()
    changed = np.zeros(source.shape, dtype=bool)
    mad_zero = np.zeros(source.shape, dtype=bool)
    for time_index in range(source.shape[0]):
        lo = max(0, time_index - half_window)
        hi = min(source.shape[0], time_index + half_window + 1)
        local = source[lo:hi]
        median = np.median(local, axis=0)
        mad = np.median(np.abs(local - median), axis=0)
        threshold = N_SIGMA * 1.4826 * mad
        replace = np.abs(source[time_index] - median) > threshold
        filtered[time_index][replace] = median[replace]
        changed[time_index] = replace
        mad_zero[time_index] = mad == 0
    return filtered, changed, mad_zero


def replacement_run_statistics(
    changed: np.ndarray,
    timestamps_seconds: np.ndarray,
) -> tuple[Counter, list[float]]:
    """逐子载波-链路统计连续替换段及其近似时间支撑。"""
    flattened = changed.reshape(changed.shape[0], -1)
    active_channels = np.flatnonzero(np.any(flattened, axis=0))
    frame_histogram: Counter = Counter()
    durations: list[float] = []
    positive_diffs = np.diff(timestamps_seconds)
    positive_diffs = positive_diffs[positive_diffs > 0]
    typical_interval = percentile(positive_diffs, 50)

    for channel in active_channels:
        vector = flattened[:, channel].astype(np.int8, copy=False)
        padded = np.pad(vector, (1, 1), mode="constant")
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        for start, end in zip(starts, ends):
            length = int(end - start)
            frame_histogram[length] += 1
            if timestamps_seconds.size:
                last = max(start, end - 1)
                duration = float(timestamps_seconds[last] - timestamps_seconds[start])
                duration += typical_interval
                durations.append(max(duration, 0.0))
    return frame_histogram, durations


def analyse_signal_sample(
    raw: np.ndarray,
    timestamps_seconds: np.ndarray,
    half_window: int,
) -> tuple[dict[str, Any], Counter, list[float], np.ndarray]:
    filtered, changed, mad_zero = hampel_with_diagnostics(raw, half_window)
    channel_count = int(np.prod(raw.shape[1:]))
    changed_per_frame = changed.reshape(raw.shape[0], -1).sum(axis=1)
    dense_threshold = max(1, math.ceil(0.01 * channel_count))

    raw_tv = float(np.abs(np.diff(raw, axis=0)).sum())
    filtered_tv = float(np.abs(np.diff(filtered, axis=0)).sum())
    raw_centered = raw - np.median(raw, axis=0, keepdims=True)
    filtered_centered = filtered - np.median(filtered, axis=0, keepdims=True)
    raw_peak = float(np.max(np.abs(raw_centered), axis=0).sum())
    filtered_peak = float(np.max(np.abs(filtered_centered), axis=0).sum())

    motion = np.zeros(raw.shape[0], dtype=float)
    if raw.shape[0] >= 2:
        transition_motion = np.mean(np.abs(np.diff(raw, axis=0)), axis=(1, 2))
        motion[1:] = transition_motion
        motion[:-1] = np.maximum(motion[:-1], transition_motion)
    positive_motion = motion[motion > 0]
    motion_threshold = percentile(positive_motion, 90, default=float("inf"))
    high_motion_frames = motion >= motion_threshold
    high_motion_values = int(high_motion_frames.sum()) * channel_count
    high_motion_replacements = int(changed[high_motion_frames].sum())
    overall_replacement_rate = safe_ratio(int(changed.sum()), int(raw.size))
    high_motion_replacement_rate = safe_ratio(
        high_motion_replacements,
        high_motion_values,
    )

    run_histogram, run_durations = replacement_run_statistics(
        changed,
        timestamps_seconds,
    )
    spans = window_spans_seconds(timestamps_seconds, half_window)
    duration = float(timestamps_seconds[-1]) if timestamps_seconds.size >= 2 else 0.0
    raw_scale = max(float(np.std(raw)), 1e-12)
    metrics = {
        "frames": int(raw.shape[0]),
        "values": int(raw.size),
        "replaced_values": int(changed.sum()),
        "mad_zero_values": int(mad_zero.sum()),
        "zero_mad_replacements": int(np.logical_and(changed, mad_zero).sum()),
        "frames_with_any_replacement": int(np.count_nonzero(changed_per_frame)),
        "frames_with_at_least_1pct_channels_replaced": int(
            np.count_nonzero(changed_per_frame >= dense_threshold)
        ),
        "absolute_change_sum": float(np.abs(filtered - raw).sum()),
        "raw_total_variation": raw_tv,
        "filtered_total_variation": filtered_tv,
        "raw_dynamic_peak_sum": raw_peak,
        "filtered_dynamic_peak_sum": filtered_peak,
        "waveform_correlation": safe_correlation(raw, filtered),
        "normalized_mae": float(np.mean(np.abs(filtered - raw)) / raw_scale),
        "high_motion_values": high_motion_values,
        "high_motion_replacements": high_motion_replacements,
        "high_motion_replacement_rate": high_motion_replacement_rate,
        "high_motion_replacement_enrichment": safe_ratio(
            high_motion_replacement_rate,
            overall_replacement_rate,
        ),
        "median_window_span_seconds": percentile(spans, 50),
        "p90_window_span_seconds": percentile(spans, 90),
        "median_window_fraction_of_sample": safe_ratio(
            percentile(spans, 50),
            duration,
        ),
        "replacement_run_count": int(sum(run_histogram.values())),
    }
    return metrics, run_histogram, run_durations, filtered


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


def aggregate_signal_metrics(
    half_window: int,
    rows: list[dict[str, Any]],
    histogram: Counter,
    run_durations: list[float],
) -> dict[str, Any]:
    values = sum(int(row["values"]) for row in rows)
    frames = sum(int(row["frames"]) for row in rows)
    replaced = sum(int(row["replaced_values"]) for row in rows)
    mad_zero = sum(int(row["mad_zero_values"]) for row in rows)
    zero_mad_replacements = sum(int(row["zero_mad_replacements"]) for row in rows)
    raw_tv = sum(finite_float(row["raw_total_variation"]) for row in rows)
    filtered_tv = sum(finite_float(row["filtered_total_variation"]) for row in rows)
    raw_peak = sum(finite_float(row["raw_dynamic_peak_sum"]) for row in rows)
    filtered_peak = sum(
        finite_float(row["filtered_dynamic_peak_sum"]) for row in rows
    )
    run_count = sum(histogram.values())
    changed_in_short_runs = sum(
        length * count for length, count in histogram.items() if length <= 3
    )
    changed_in_all_runs = sum(length * count for length, count in histogram.items())
    high_motion_values = sum(int(row["high_motion_values"]) for row in rows)
    high_motion_replacements = sum(
        int(row["high_motion_replacements"]) for row in rows
    )
    overall_replacement_rate = safe_ratio(replaced, values)
    high_motion_rate = safe_ratio(high_motion_replacements, high_motion_values)

    return {
        "half_window": half_window,
        "full_window_frames": full_window_frames(half_window),
        "n_sigma": N_SIGMA,
        "diagnostic_samples": len(rows),
        "frames": frames,
        "values": values,
        "replaced_values": replaced,
        "replacement_rate": overall_replacement_rate,
        "mad_zero_rate": safe_ratio(mad_zero, values),
        "zero_mad_share_of_replacements": safe_ratio(
            zero_mad_replacements,
            replaced,
        ),
        "frame_any_replacement_rate": safe_ratio(
            sum(int(row["frames_with_any_replacement"]) for row in rows),
            frames,
        ),
        "frame_1pct_channels_replaced_rate": safe_ratio(
            sum(
                int(row["frames_with_at_least_1pct_channels_replaced"])
                for row in rows
            ),
            frames,
        ),
        "total_variation_retention": safe_ratio(filtered_tv, raw_tv, default=1.0),
        "dynamic_peak_retention": safe_ratio(
            filtered_peak,
            raw_peak,
            default=1.0,
        ),
        "mean_waveform_correlation": float(
            np.mean([finite_float(row["waveform_correlation"]) for row in rows])
        ),
        "mean_normalized_mae": float(
            np.mean([finite_float(row["normalized_mae"]) for row in rows])
        ),
        "high_motion_replacement_rate": high_motion_rate,
        "high_motion_replacement_enrichment": safe_ratio(
            high_motion_rate,
            overall_replacement_rate,
        ),
        "median_sample_window_span_seconds": percentile(
            [finite_float(row["median_window_span_seconds"]) for row in rows],
            50,
        ),
        "p90_sample_window_span_seconds": percentile(
            [finite_float(row["p90_window_span_seconds"]) for row in rows],
            90,
        ),
        "median_window_fraction_of_sample": percentile(
            [finite_float(row["median_window_fraction_of_sample"]) for row in rows],
            50,
        ),
        "replacement_run_count": run_count,
        "single_frame_run_fraction": safe_ratio(histogram.get(1, 0), run_count),
        "runs_le3_fraction": safe_ratio(
            sum(count for length, count in histogram.items() if length <= 3),
            run_count,
        ),
        "changed_values_in_runs_le3_fraction": safe_ratio(
            changed_in_short_runs,
            changed_in_all_runs,
        ),
        "run_length_median_frames": histogram_quantile(histogram, 0.50),
        "run_length_p90_frames": histogram_quantile(histogram, 0.90),
        "run_length_max_frames": max(histogram, default=0),
        "run_duration_median_seconds": percentile(run_durations, 50),
        "run_duration_p90_seconds": percentile(run_durations, 90),
        "run_duration_max_seconds": percentile(run_durations, 100),
    }


def run_signal_diagnostics(
    records: list,
    windows: list[int],
    factor: float,
    diagnostic_samples: int,
) -> tuple[list[dict[str, Any]], dict[int, Counter]]:
    selected = select_stratified_records(records, diagnostic_samples)
    if not selected:
        raise RuntimeError("没有可用于信号诊断的样本")

    aggregate_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    histograms: dict[int, Counter] = {}
    source_equivalence_verified: dict[int, bool] = {}

    for half_window in windows:
        print(
            f"信号诊断: half_window={half_window}, "
            f"full_window={full_window_frames(half_window)}"
        )
        window_rows: list[dict[str, Any]] = []
        histogram: Counter = Counter()
        all_run_durations: list[float] = []
        for sample_index, record in enumerate(selected, start=1):
            raw, raw_timestamps = record_to_array_and_timestamps(record)
            timestamps = timestamps_in_seconds(raw_timestamps, factor)
            metrics, sample_histogram, run_durations, filtered = (
                analyse_signal_sample(raw, timestamps, half_window)
            )
            if sample_index == 1:
                source_filtered = hampel_filter(
                    raw,
                    window_size=half_window,
                    n_sigma=N_SIGMA,
                )
                if not np.array_equal(filtered, source_filtered, equal_nan=True):
                    raise AssertionError(
                        "诊断实现与 WSDP 源码 Hampel 不一致："
                        f"half_window={half_window}"
                    )
                source_equivalence_verified[half_window] = True

            metadata = record_metadata(record)
            sample_row = {
                "half_window": half_window,
                "full_window_frames": full_window_frames(half_window),
                "sample_index": sample_index,
                "file_name": str(record.file_name),
                **metadata,
                **metrics,
                "replacement_rate": safe_ratio(
                    metrics["replaced_values"],
                    metrics["values"],
                ),
                "mad_zero_rate": safe_ratio(
                    metrics["mad_zero_values"],
                    metrics["values"],
                ),
                "zero_mad_share_of_replacements": safe_ratio(
                    metrics["zero_mad_replacements"],
                    metrics["replaced_values"],
                ),
                "total_variation_retention": safe_ratio(
                    metrics["filtered_total_variation"],
                    metrics["raw_total_variation"],
                    default=1.0,
                ),
                "dynamic_peak_retention": safe_ratio(
                    metrics["filtered_dynamic_peak_sum"],
                    metrics["raw_dynamic_peak_sum"],
                    default=1.0,
                ),
            }
            window_rows.append(metrics)
            per_sample_rows.append(sample_row)
            histogram.update(sample_histogram)
            all_run_durations.extend(run_durations)

            if sample_index % 100 == 0 or sample_index == len(selected):
                print(
                    f"  half_window={half_window}: "
                    f"{sample_index}/{len(selected)}"
                )

        aggregate_rows.append(
            aggregate_signal_metrics(
                half_window,
                window_rows,
                histogram,
                all_run_durations,
            )
        )
        histograms[half_window] = histogram

    write_csv(SIGNAL_DIAGNOSTIC_PATH, aggregate_rows)
    write_csv(SIGNAL_SAMPLE_PATH, per_sample_rows)
    write_json(
        RUN_LENGTH_PATH,
        {
            "source_equivalence_verified_on_first_sample": {
                str(window): bool(source_equivalence_verified.get(window, False))
                for window in windows
            },
            "histograms": {
                str(window): {
                    str(length): int(count)
                    for length, count in sorted(histogram.items())
                }
                for window, histogram in histograms.items()
            },
        },
    )
    return aggregate_rows, histograms


def select_stratified_records(records: list, limit: int) -> list:
    """按用户×位置×动作轮询抽样，避免只取文件列表开头造成类别偏差。"""
    if limit == 0 or limit >= len(records):
        return list(records)
    buckets: dict[tuple[Any, Any, Any], list] = {}
    for record in records:
        metadata = record_metadata(record)
        key = (
            metadata["user_id"],
            metadata["position_id"],
            metadata["action_id"],
        )
        buckets.setdefault(key, []).append(record)

    selected: list = []
    depth = 0
    keys = sorted(buckets, key=lambda item: tuple(str(value) for value in item))
    while len(selected) < limit:
        added = False
        for key in keys:
            bucket = buckets[key]
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def select_representative_signal(
    records: list,
    factor: float,
    half_window: int,
    search_limit: int = 100,
) -> tuple[np.ndarray, np.ndarray, int, int, str]:
    """选择默认窗口改变量最大的真实通道。"""
    best_score = -1.0
    best = None
    for record in records[: min(search_limit, len(records))]:
        raw, raw_timestamps = record_to_array_and_timestamps(record)
        timestamps = timestamps_in_seconds(raw_timestamps, factor)
        filtered, _, _ = hampel_with_diagnostics(raw, half_window)
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
                timestamps,
                int(frequency_index),
                int(antenna_index),
                str(record.file_name),
            )
    if best is None:
        raise RuntimeError("无法选择代表性 ElderAL 信号")
    return best


# ---------------------------------------------------------------------------
# 科研绘图
# ---------------------------------------------------------------------------

def plot_timing_audit(
    timing_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    if not timing_rows:
        return
    frames = np.asarray([int(row["frames"]) for row in timing_rows], dtype=float)
    durations = np.asarray(
        [finite_float(row["duration_seconds"]) for row in timing_rows],
        dtype=float,
    )
    actions = np.asarray(
        [int(row["action_id"]) if str(row["action_id"]) else -1 for row in timing_rows]
    )
    intervals = np.asarray(
        [
            finite_float(row["interval_seconds"])
            for row in interval_rows
            if int(row["is_positive"]) == 1
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    axes[0, 0].hist(frames, bins=min(30, max(8, int(np.sqrt(len(frames))))),
                    color=OKABE_ITO["blue"], alpha=0.85)
    axes[0, 0].axvline(np.median(frames), color=OKABE_ITO["vermillion"],
                       linestyle="--", label=f"Median={np.median(frames):.1f}")
    axes[0, 0].set_xlabel("Retained frames per sample")
    axes[0, 0].set_ylabel("Samples")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(True)

    axes[0, 1].hist(durations[durations > 0], bins=25,
                    color=OKABE_ITO["green"], alpha=0.85)
    axes[0, 1].axvline(
        ACTION_DURATION_REFERENCE_SECONDS,
        color=OKABE_ITO["vermillion"],
        linestyle="--",
        label="5 s protocol reference",
    )
    axes[0, 1].set_xlabel("Timestamp-derived duration (s)")
    axes[0, 1].set_ylabel("Samples")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(True)

    scatter = axes[1, 0].scatter(
        durations,
        frames,
        c=actions,
        cmap="viridis",
        s=15,
        alpha=0.65,
        edgecolors="none",
    )
    axes[1, 0].set_xlabel("Timestamp-derived duration (s)")
    axes[1, 0].set_ylabel("Retained frames")
    axes[1, 0].grid(True)
    colorbar = fig.colorbar(scatter, ax=axes[1, 0], pad=0.02)
    colorbar.set_label("Action ID")

    positive = intervals[intervals > 0]
    if positive.size:
        axes[1, 1].hist(
            positive,
            bins=np.geomspace(max(positive.min(), 1e-9), positive.max(), 30)
            if positive.max() > positive.min()
            else 10,
            color=OKABE_ITO["orange"],
            alpha=0.85,
        )
        if positive.max() > positive.min():
            axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("Positive inter-frame interval (s)")
    axes[1, 1].set_ylabel("Intervals")
    axes[1, 1].grid(True, which="both")

    fig.suptitle("ElderAL timing characteristics", fontsize=10)
    save_publication_figure(fig, "figure_1_timing_audit")


def plot_window_spans(window_rows: list[dict[str, Any]]) -> None:
    if not window_rows:
        return
    windows = sorted({int(row["half_window"]) for row in window_rows})
    spans = [
        [
            finite_float(row["median_span_seconds"])
            for row in window_rows
            if int(row["half_window"]) == window
        ]
        for window in windows
    ]
    fractions = [
        [
            100.0 * finite_float(row["median_fraction_of_sample_duration"])
            for row in window_rows
            if int(row["half_window"]) == window
        ]
        for window in windows
    ]
    labels = [str(full_window_frames(window)) for window in windows]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)
    box_1 = axes[0].boxplot(spans, labels=labels, showfliers=False, patch_artist=True)
    box_2 = axes[1].boxplot(
        fractions,
        labels=labels,
        showfliers=False,
        patch_artist=True,
    )
    for box in box_1["boxes"]:
        box.set_facecolor(OKABE_ITO["sky"])
        box.set_alpha(0.75)
    for box in box_2["boxes"]:
        box.set_facecolor(OKABE_ITO["orange"])
        box.set_alpha(0.75)
    axes[0].set_xlabel("Full Hampel window (frames)")
    axes[0].set_ylabel("Median real window span per sample (s)")
    axes[1].set_xlabel("Full Hampel window (frames)")
    axes[1].set_ylabel("Window span / sample duration (%)")
    for ax in axes:
        ax.grid(True)
    save_publication_figure(fig, "figure_2_real_window_spans")


def plot_signal_metrics(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: int(row["half_window"]))
    x = np.asarray([int(row["full_window_frames"]) for row in rows])
    metrics = (
        ("replacement_rate", "Replaced CSI values (%)", 100.0),
        ("mad_zero_rate", "Local MAD = 0 (%)", 100.0),
        ("total_variation_retention", "Total variation retained (%)", 100.0),
        ("dynamic_peak_retention", "Dynamic peak retained (%)", 100.0),
        (
            "high_motion_replacement_enrichment",
            "High-motion replacement enrichment (×)",
            1.0,
        ),
        (
            "median_window_fraction_of_sample",
            "Window / sample duration (%)",
            100.0,
        ),
    )
    colors = (
        OKABE_ITO["blue"],
        OKABE_ITO["purple"],
        OKABE_ITO["green"],
        OKABE_ITO["orange"],
        OKABE_ITO["vermillion"],
        OKABE_ITO["sky"],
    )
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.0), constrained_layout=True)
    for ax, (key, ylabel, scale), color in zip(axes.flat, metrics, colors):
        y = np.asarray([finite_float(row[key]) * scale for row in rows])
        ax.plot(x, y, marker="o", color=color)
        if key == "high_motion_replacement_enrichment":
            ax.axhline(1.0, color=OKABE_ITO["grey"], linestyle="--", linewidth=1.0)
        ax.set_xlabel("Full Hampel window (frames)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.grid(True)
    save_publication_figure(fig, "figure_3_signal_metrics_vs_window")


def plot_run_length_histograms(
    histograms: dict[int, Counter],
    max_display_length: int = 15,
) -> None:
    if not histograms:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    colors = [
        OKABE_ITO["blue"],
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
        ax.plot(
            x,
            counts / counts.sum(),
            marker="o",
            color=color,
            label=f"{full_window_frames(window)}-frame window",
        )
    labels = [str(value) for value in x]
    labels[-1] = f"≥{max_display_length}"
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Consecutive replaced samples")
    ax.set_ylabel("Fraction of replacement runs")
    ax.set_yscale("log")
    ax.grid(True, which="both")
    ax.legend(frameon=False)
    save_publication_figure(fig, "figure_4_replacement_run_lengths")


def plot_representative_waveforms(
    records: list,
    windows: list[int],
    factor: float,
) -> None:
    default_window = 5 if 5 in windows else max(windows)
    raw, timestamps, frequency_index, antenna_index, file_name = (
        select_representative_signal(records, factor, default_window)
    )
    x = timestamps if timestamps.size and timestamps[-1] > 0 else np.arange(raw.shape[0])
    x_label = "Time from sample start (s)" if timestamps.size and timestamps[-1] > 0 else "Frame index"
    raw_signal = raw[:, frequency_index, antenna_index]

    fig, axes = plt.subplots(
        len(windows) + 1,
        1,
        figsize=(7.2, 1.55 * (len(windows) + 1)),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(x, raw_signal, color=OKABE_ITO["black"], label="Raw")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(True)

    colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["green"],
        OKABE_ITO["orange"],
        OKABE_ITO["vermillion"],
        OKABE_ITO["purple"],
    ]
    for ax, half_window, color in zip(axes[1:], windows, colors):
        filtered, changed, _ = hampel_with_diagnostics(raw, half_window)
        signal = filtered[:, frequency_index, antenna_index]
        mask = changed[:, frequency_index, antenna_index]
        ax.plot(x, raw_signal, color=OKABE_ITO["grey"], alpha=0.55, label="Raw")
        ax.plot(
            x,
            signal,
            color=color,
            label=f"Hampel {full_window_frames(half_window)} frames",
        )
        if np.any(mask):
            ax.scatter(
                x[mask],
                signal[mask],
                s=18,
                facecolors="none",
                edgecolors=OKABE_ITO["vermillion"],
                linewidths=0.9,
                label="Replaced",
                zorder=3,
            )
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend(frameon=False, loc="upper right", ncol=2)
    axes[-1].set_xlabel(x_label)
    fig.suptitle(
        f"Representative ElderAL channel: {Path(file_name).name}, "
        f"subcarrier={frequency_index}, link={antenna_index}",
        fontsize=9,
    )
    save_publication_figure(fig, "figure_5_representative_waveforms")


# ---------------------------------------------------------------------------
# 训练预处理与 CSI-Time
# ---------------------------------------------------------------------------

def build_conditions(windows: list[int]) -> list[dict[str, Any]]:
    conditions = [
        {
            "condition": "no_hampel",
            "half_window": None,
            "full_window_frames": 0,
        }
    ]
    conditions.extend(
        {
            "condition": f"hampel_w{window}",
            "half_window": window,
            "full_window_frames": full_window_frames(window),
        }
        for window in windows
    )
    return conditions


def build_pipeline_steps(condition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    steps: dict[str, dict[str, Any]] = {}
    if condition["half_window"] is not None:
        steps["denoise"] = {
            "method": "hampel",
            "window_size": int(condition["half_window"]),
            "n_sigma": N_SIGMA,
        }
    steps["outliers"] = {"method": "iqr", "factor": IQR_FACTOR}
    steps["normalize"] = {"method": "min-max"}
    steps["interpolate"] = {
        "method": "linear",
        "target_K": TARGET_SUBCARRIERS,
    }
    return steps


def process_record_worker(
    record,
    pipeline_steps: dict[str, dict[str, Any]],
) -> tuple[np.ndarray | None, int | None, int | None, str]:
    parsed = _parse_file_info_from_filename(record.file_name, DATASET_NAME)
    if parsed is None:
        return None, None, None, str(record.file_name)
    label, group = _selector(parsed, DATASET_NAME)
    frames = sorted(record.frames, key=lambda frame: float(frame.timestamp))
    if len(frames) < 2:
        return None, None, None, str(record.file_name)
    whole_csi = np.stack([frame.csi_array for frame in frames], axis=0)
    if whole_csi.ndim == 2:
        whole_csi = np.expand_dims(whole_csi, -1)
    cleaned = execute_pipeline(
        whole_csi,
        pipeline_steps,
        dataset=DATASET_NAME,
    )
    if np.iscomplexobj(cleaned):
        cleaned = np.real_if_close(cleaned, tol=1000)
    if np.iscomplexobj(cleaned):
        raise ValueError(f"ElderAL 处理后仍为复数: {record.file_name}")
    return np.asarray(cleaned), int(label), int(group), str(record.file_name)


def preprocess_condition(
    records: list,
    pipeline_steps: dict[str, dict[str, Any]],
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[str]]:
    worker = partial(process_record_worker, pipeline_steps=pipeline_steps)
    processed: list[np.ndarray] = []
    raw_labels: list[int] = []
    groups: list[int] = []
    file_names: list[str] = []

    if workers == 1:
        results = map(worker, records)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(worker, records)

    try:
        for index, (csi, label, group, file_name) in enumerate(results, start=1):
            if csi is not None:
                processed.append(csi)
                raw_labels.append(int(label))
                groups.append(int(group))
                file_names.append(file_name)
            if index % 200 == 0 or index == len(records):
                print(f"预处理: {index}/{len(records)}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if not processed:
        raise RuntimeError("预处理后没有有效样本")
    unique_labels = sorted(set(raw_labels))
    label_map = {label: index for index, label in enumerate(unique_labels)}
    labels = np.asarray([label_map[label] for label in raw_labels], dtype=int)
    resized = resize_csi_to_fixed_length(
        processed,
        target_length=PADDING_LENGTH,
    )
    return (
        np.asarray(resized),
        labels,
        np.asarray(groups, dtype=int),
        unique_labels,
        file_names,
    )


def split_elder(
    processed_data: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    pipeline_steps: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[np.ndarray, ...]:
    return _create_data_split(
        processed_data,
        labels,
        groups,
        test_split=TEST_SPLIT,
        val_split=VAL_SPLIT,
        seed=seed,
        use_simple_split=len(set(groups.tolist())) < 3,
        dataset=DATASET_NAME,
        pipeline_steps=pipeline_steps,
    )


def split_group_ids(groups: np.ndarray, seed: int) -> tuple[list[int], list[int], list[int]]:
    """在轻量占位数组上复现源码 group split，记录各集合的位置ID。"""
    dummy = np.arange(len(groups), dtype=float)[:, None, None]
    labels = np.zeros(len(groups), dtype=int)
    split = _create_data_split(
        dummy,
        labels,
        groups,
        test_split=TEST_SPLIT,
        val_split=VAL_SPLIT,
        seed=seed,
        use_simple_split=len(set(groups.tolist())) < 3,
        dataset=DATASET_NAME,
        pipeline_steps={},
    )
    train_ids = split[0].reshape(-1).astype(int)
    val_ids = split[1].reshape(-1).astype(int)
    test_ids = split[2].reshape(-1).astype(int)
    return (
        sorted(set(groups[train_ids].tolist())),
        sorted(set(groups[val_ids].tolist())),
        sorted(set(groups[test_ids].tolist())),
    )


def build_loaders(
    split: tuple[np.ndarray, ...],
    pipeline_steps: dict[str, dict[str, Any]],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_data, val_data, test_data, train_labels, val_labels, test_labels = split

    def make_loader(data, labels, shuffle: bool) -> DataLoader:
        dataset = CSIDataset(
            data,
            labels,
            dataset_name=DATASET_NAME,
            pipeline_steps=pipeline_steps,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    return (
        make_loader(train_data, train_labels, True),
        make_loader(val_data, val_labels, False),
        make_loader(test_data, test_labels, False),
    )


def save_training_history(history: dict[str, Any], path: Path) -> None:
    if not history:
        return
    keys = [key for key, value in history.items() if isinstance(value, list)]
    row_count = max((len(history[key]) for key in keys), default=0)
    rows = []
    for index in range(row_count):
        row = {"epoch": index + 1}
        for key in keys:
            row[key] = history[key][index] if index < len(history[key]) else ""
        rows.append(row)
    write_csv(path, rows)


def plot_training_loss(
    history: dict[str, Any],
    output_dir: Path,
) -> None:
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if not train_loss or not val_loss:
        return
    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    ax.plot(epochs, train_loss, color=OKABE_ITO["blue"], label="Train")
    ax.plot(epochs, val_loss, color=OKABE_ITO["vermillion"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(True)
    ax.legend(frameon=False)
    save_publication_figure(fig, "loss_curve", output_dir)


def plot_normalized_confusion_matrix(
    predictions: list[int],
    targets: list[int],
    num_classes: int,
    output_dir: Path,
) -> None:
    matrix = np.zeros((num_classes, num_classes), dtype=float)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix),
        where=row_sums > 0,
    )

    fig, ax = plt.subplots(figsize=(5.0, 4.3), constrained_layout=True)
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(num_classes):
        for column in range(num_classes):
            value = normalized[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.55 else "black",
            )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    fig.colorbar(image, ax=ax, label="Row-normalized proportion")
    save_publication_figure(fig, "confusion_matrix_normalized", output_dir)


def require_cuda0() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "本实验按要求固定使用物理 GPU 0，但当前环境没有可用 CUDA。"
        )
    device = torch.device("cuda:0")
    print(f"训练设备: {device}; CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    return device


def train_one_condition(
    condition: dict[str, Any],
    records: list,
    params: dict[str, Any],
    seed: int,
    epochs: int,
    workers: int,
) -> dict[str, Any]:
    pipeline_steps = build_pipeline_steps(condition)
    output_dir = RUNS_DIR / condition["condition"]
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    row: dict[str, Any] = {
        "condition": condition["condition"],
        "half_window": (
            "" if condition["half_window"] is None else condition["half_window"]
        ),
        "full_window_frames": condition["full_window_frames"],
        "n_sigma": "" if condition["half_window"] is None else N_SIGMA,
        "model": MODEL_NAME,
        "seed": seed,
        "epochs": epochs,
        "status": "failed",
        "best_val_acc": "",
        "test_acc": "",
        "train_size": "",
        "val_size": "",
        "test_size": "",
        "input_shape": "",
        "test_groups": "",
        "checkpoint": "",
        "output_dir": str(output_dir),
        "duration_sec": "",
        "error": "",
    }

    with (output_dir / "train_process.txt").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            try:
                print("=" * 80)
                print(f"条件: {condition['condition']}")
                print(json.dumps(pipeline_steps, ensure_ascii=False, indent=2))
                print("=" * 80)
                set_seed(seed)
                data, labels, groups, unique_labels, file_names = (
                    preprocess_condition(records, pipeline_steps, workers)
                )
                write_json(
                    output_dir / "preprocessing_metadata.json",
                    {
                        "condition": condition,
                        "pipeline_steps": pipeline_steps,
                        "samples": len(data),
                        "files_in_order": file_names,
                        "raw_label_values": unique_labels,
                        "groups": sorted(set(groups.tolist())),
                    },
                )
                split = split_elder(data, labels, groups, pipeline_steps, seed)
                train_groups, val_groups, test_groups = split_group_ids(groups, seed)
                print(
                    f"位置划分: train={train_groups}, val={val_groups}, "
                    f"test={test_groups}"
                )

                batch_size = int(params.get("batch", 32))
                loaders = build_loaders(split, pipeline_steps, batch_size)
                input_shape = tuple(loaders[0].dataset.data_list.shape[1:])
                print(f"CSI-Time 输入形状: {input_shape}")
                device = require_cuda0()
                model = create_model(
                    MODEL_NAME,
                    num_classes=len(unique_labels),
                    input_shape=input_shape,
                ).to(device)
                print(f"模型参数量: {sum(parameter.numel() for parameter in model.parameters())}")

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
                plot_training_loss(history, output_dir)
                if not checkpoint.exists():
                    raise RuntimeError(f"训练未生成 checkpoint: {checkpoint}")

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
                        "test_groups": ",".join(str(value) for value in test_groups),
                        "checkpoint": str(checkpoint),
                    }
                )
                print(
                    f"完成: best_val_acc={best_val_acc:.4f}, "
                    f"test_acc={test_acc:.4f}"
                )
                del model, loaders, data
            except Exception:
                row["error"] = traceback.format_exc()
                print(row["error"])

    row["duration_sec"] = f"{time.time() - start:.2f}"
    clear_cuda_cache()
    return row


def successful_training_rows() -> list[dict[str, str]]:
    successful: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(TRAINING_SUMMARY_PATH):
        if row.get("status") == "ok":
            successful[row["condition"]] = row
    return list(successful.values())


def plot_accuracy_results(
    training_rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
) -> None:
    successful = [row for row in training_rows if row.get("status") == "ok"]
    if not successful:
        return

    def order_key(row: dict[str, Any]) -> int:
        value = str(row.get("half_window", "")).strip()
        return -1 if not value else int(value)

    successful = sorted(successful, key=order_key)
    labels = [
        "No Hampel"
        if not str(row.get("half_window", "")).strip()
        else f"{int(row['full_window_frames'])} frames"
        for row in successful
    ]
    scores = np.asarray([100.0 * finite_float(row["test_acc"]) for row in successful])
    colors = [OKABE_ITO["grey"]] + [
        OKABE_ITO["blue"],
        OKABE_ITO["green"],
        OKABE_ITO["orange"],
        OKABE_ITO["vermillion"],
        OKABE_ITO["purple"],
    ][: max(0, len(successful) - 1)]

    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    bars = ax.bar(np.arange(len(scores)), scores, color=colors[: len(scores)], width=0.72)
    for bar, value in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlabel("Hampel condition")
    ax.grid(True, axis="y")
    lower = max(0.0, float(scores.min()) - 8.0)
    ax.set_ylim(lower, min(100.0, float(scores.max()) + 8.0))
    save_publication_figure(fig, "figure_6_test_accuracy_vs_window")

    signal_by_window = {
        int(row["half_window"]): row
        for row in signal_rows
        if str(row.get("half_window", "")).strip()
    }
    paired = [
        row
        for row in successful
        if str(row.get("half_window", "")).strip()
        and int(row["half_window"]) in signal_by_window
    ]
    if not paired:
        return
    retention = np.asarray(
        [
            100.0
            * finite_float(
                signal_by_window[int(row["half_window"])][
                    "total_variation_retention"
                ]
            )
            for row in paired
        ]
    )
    peak = np.asarray(
        [
            100.0
            * finite_float(
                signal_by_window[int(row["half_window"])]["dynamic_peak_retention"]
            )
            for row in paired
        ]
    )
    accuracy = np.asarray([100.0 * finite_float(row["test_acc"]) for row in paired])
    window_labels = [f"{int(row['full_window_frames'])}f" for row in paired]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    axes[0].scatter(retention, accuracy, color=OKABE_ITO["blue"], s=42)
    axes[1].scatter(peak, accuracy, color=OKABE_ITO["orange"], s=42)
    for index, label in enumerate(window_labels):
        axes[0].annotate(label, (retention[index], accuracy[index]), xytext=(4, 4),
                         textcoords="offset points", fontsize=8)
        axes[1].annotate(label, (peak[index], accuracy[index]), xytext=(4, 4),
                         textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("Total variation retained (%)")
    axes[1].set_xlabel("Dynamic peak retained (%)")
    for ax in axes:
        ax.set_ylabel("Test accuracy (%)")
        ax.grid(True)
    save_publication_figure(fig, "figure_7_accuracy_vs_signal_retention")


# ---------------------------------------------------------------------------
# 结果重画、CLI 与主流程
# ---------------------------------------------------------------------------

def load_run_histograms() -> dict[int, Counter]:
    if not RUN_LENGTH_PATH.exists():
        return {}
    payload = read_json(RUN_LENGTH_PATH)
    histograms = payload.get("histograms", payload)
    return {
        int(window): Counter(
            {int(length): int(count) for length, count in histogram.items()}
        )
        for window, histogram in histograms.items()
    }


def regenerate_summary_figures() -> None:
    timing_rows = read_csv_rows(TIMING_SAMPLE_PATH)
    interval_rows = read_csv_rows(FRAME_INTERVAL_PATH)
    window_rows = read_csv_rows(WINDOW_SPAN_SAMPLE_PATH)
    signal_rows = read_csv_rows(SIGNAL_DIAGNOSTIC_PATH)
    training_rows = successful_training_rows()
    if timing_rows and TIMING_REPORT_PATH.exists():
        plot_timing_audit(timing_rows, interval_rows, read_json(TIMING_REPORT_PATH))
    plot_window_spans(window_rows)
    plot_signal_metrics(signal_rows)
    plot_run_length_histograms(load_run_histograms())
    plot_accuracy_results(training_rows, signal_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        help="ElderAL 数据目录；默认自动查找 sdp_dataset/elderAL",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=list(DEFAULT_WINDOWS),
        help="Hampel 半窗口，默认 1 2 3 5（完整窗口3 5 7 11帧）",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--diagnostic-samples",
        type=int,
        default=DEFAULT_DIAGNOSTIC_SAMPLES,
        help=(
            "信号诊断分层抽样数；默认162以覆盖用户×位置×动作并节省时间，"
            "0表示全部样本。时间诊断始终使用全部读取样本"
        ),
    )
    parser.add_argument(
        "--timestamp-unit",
        choices=["auto", "s", "ms", "us", "ns"],
        default="auto",
        help="原始 timestamp 单位；默认根据5秒协议参考只推断量纲",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="调试用；0使用全部样本，正数只保留前N条记录",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="只执行时间/信号诊断和绘图，不训练",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="复用已有诊断结果，只训练尚未完成的条件",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="不读取数据或训练，只根据已有CSV/JSON重新生成汇总图",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_publication_style()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    windows = parse_positive_windows(args.windows)
    if args.seed < 0:
        raise ValueError("--seed 必须 >= 0")
    if args.epochs < 1:
        raise ValueError("--epochs 必须 >= 1")
    if args.workers < 1:
        raise ValueError("--workers 必须 >= 1")
    if args.diagnostic_samples < 0:
        raise ValueError("--diagnostic-samples 必须 >= 0")
    if args.max_samples < 0:
        raise ValueError("--max-samples 必须 >= 0")
    if args.diagnostics_only and args.skip_diagnostics:
        raise ValueError("--diagnostics-only 与 --skip-diagnostics 不能同时使用")

    if args.plot_only:
        regenerate_summary_figures()
        print(f"已根据现有结果重画科研图: {FIGURE_DIR}")
        return

    data_path = resolve_data_path(args.data_path)
    conditions = build_conditions(windows)
    write_json(
        SETTINGS_PATH,
        {
            "research_question": (
                "Does ElderAL Hampel fail because an 11-frame neighborhood "
                "covers a large part of a short/irregular action sample?"
            ),
            "sampling_rate_correction": (
                "Retained frame frequency is not treated as an adaptive device "
                "sampling rate. Physical spans are measured from timestamps."
            ),
            "windows": windows,
            "full_windows": [full_window_frames(window) for window in windows],
            "conditions": conditions,
            "fixed_pipeline": {
                "n_sigma": N_SIGMA,
                "outliers": {"method": "iqr", "factor": IQR_FACTOR},
                "normalize": {"method": "min-max"},
                "interpolate": {
                    "method": "linear",
                    "target_K": TARGET_SUBCARRIERS,
                },
                "padding_length": PADDING_LENGTH,
                "model": MODEL_NAME,
            },
            "training": {
                "physical_gpu": 0,
                "logical_device": "cuda:0",
                "seed": args.seed,
                "epochs": args.epochs,
                "test_split": TEST_SPLIT,
                "val_split_within_heldout": VAL_SPLIT,
            },
            "timestamp_unit_requested": args.timestamp_unit,
            "diagnostic_samples": args.diagnostic_samples,
            "max_samples_debug": args.max_samples,
            "data_path": str(data_path),
        },
    )

    records = load_raw_records(data_path)
    if args.max_samples > 0:
        records = records[: min(args.max_samples, len(records))]
        print(f"调试限制后样本数: {len(records)}")

    if not args.skip_diagnostics:
        timing_rows, interval_rows, window_rows, timing_report = run_timing_audit(
            records,
            windows,
            args.timestamp_unit,
        )
        timestamp_factor = finite_float(
            timing_report["timestamp_unit_report"]["seconds_per_timestamp_unit"],
            1.0,
        )
        signal_rows, histograms = run_signal_diagnostics(
            records,
            windows,
            timestamp_factor,
            args.diagnostic_samples,
        )
        plot_timing_audit(timing_rows, interval_rows, timing_report)
        plot_window_spans(window_rows)
        plot_signal_metrics(signal_rows)
        plot_run_length_histograms(histograms)
        plot_representative_waveforms(records, windows, timestamp_factor)
        print(f"时间与信号诊断完成: {RESULT_ROOT}")

    if args.diagnostics_only:
        print("按 --diagnostics-only 结束，未启动训练。")
        return

    completed = {
        row["condition"]
        for row in successful_training_rows()
        if int(row.get("seed", -1)) == args.seed
    }
    params = load_params(DATASET_NAME)
    for condition in conditions:
        if condition["condition"] in completed:
            print(f"跳过已有成功训练: {condition['condition']}")
            continue
        row = train_one_condition(
            condition,
            records,
            params,
            args.seed,
            args.epochs,
            args.workers,
        )
        append_csv(TRAINING_SUMMARY_PATH, row, TRAINING_FIELDS)

    regenerate_summary_figures()
    print(f"实验完成，结果目录: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
