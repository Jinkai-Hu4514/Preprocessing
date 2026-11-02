"""为 ElderAL 其余动作生成与原 Figure 5 完全同版式的 Hampel 波形图。

只负责读取原始 CSV 和画图，不训练模型，也不修改已有实验结果。

默认生成静止、跌倒、睡眠、喝水、坐着看电视五张图；走路图已经由
``ablation_elder_hampel.py`` 生成。如需重画走路，可在 ``--actions`` 中加入1。

服务器运行示例：

    python ablation/plot_elder_hampel_action_waveforms.py
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR / "ablation_elder_hampel_result" / "figures"
DEFAULT_DATA_PATH = SCRIPT_DIR.parent / "sdp_dataset" / "elderAL"
N_SIGMA = 3.0

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

ACTION_NAMES = {
    0: "static",
    1: "walk",
    2: "fall",
    3: "sleep",
    5: "drink",
    6: "sit_TV",
}


@dataclass
class CSVRecord:
    file_name: str
    action_id: int
    timestamps_raw: np.ndarray


def configure_publication_style() -> None:
    """与 ablation_elder_hampel.py 使用相同的论文绘图参数。"""
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
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def full_window_frames(half_window: int) -> int:
    return 2 * half_window + 1


def parse_positive_ints(values: list[int], option: str) -> list[int]:
    parsed = sorted(set(int(value) for value in values))
    if not parsed or parsed[0] < 0:
        raise ValueError(f"{option} 必须包含非负整数")
    return parsed


def parse_action_id(file_name: str) -> int | None:
    match = re.search(
        r"user\d+_position\d+_activity(\d+)",
        file_name,
    )
    return int(match.group(1)) if match else None


def read_timestamps_only(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            headers = [name.strip() for name in next(reader)]
        except StopIteration:
            return np.asarray([], dtype=float)
        if "timestamp" not in headers:
            return np.asarray([], dtype=float)
        if not any(name.startswith("amp_tx0_rx") for name in headers):
            return np.asarray([], dtype=float)
        timestamp_index = headers.index("timestamp")
        timestamps: list[float] = []
        for row in reader:
            if not row or timestamp_index >= len(row):
                continue
            try:
                timestamps.append(float(row[timestamp_index]))
            except ValueError:
                continue
    return np.asarray(timestamps, dtype=float)


def scan_records(data_path: Path) -> list[CSVRecord]:
    if not data_path.is_dir():
        raise FileNotFoundError(f"找不到 ElderAL 数据目录: {data_path}")
    paths = sorted(path for path in data_path.rglob("*.csv") if path.is_file())
    if not paths:
        raise RuntimeError(f"{data_path} 下没有找到CSV文件")

    print(f"扫描 ElderAL CSV: {data_path}；共发现{len(paths)}个CSV")
    records: list[CSVRecord] = []
    for index, path in enumerate(paths, start=1):
        action_id = parse_action_id(str(path))
        if action_id is not None:
            timestamps = read_timestamps_only(path)
            if timestamps.size:
                records.append(
                    CSVRecord(
                        file_name=str(path),
                        action_id=action_id,
                        timestamps_raw=timestamps,
                    )
                )
        if index % 250 == 0 or index == len(paths):
            print(f"  已读取时间戳 {index}/{len(paths)}")
    if not records:
        raise RuntimeError("没有找到有效 ElderAL CSV")
    return records


def read_raw_csi(record: CSVRecord) -> tuple[np.ndarray, np.ndarray]:
    """复现 ElderReader：只读取 tx0，输出形状为(T, subcarrier, rx)。"""
    path = Path(record.file_name)
    pattern = re.compile(r"amp_tx(\d+)_rx(\d+)_sub(\d+)")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            headers = [name.strip() for name in next(reader)]
        except StopIteration as error:
            raise ValueError(f"空CSV: {path}") from error
        if "timestamp" not in headers:
            raise ValueError(f"CSV缺少timestamp列: {path}")
        timestamp_index = headers.index("timestamp")

        mapping: list[tuple[int, int, int]] = []
        max_subcarrier = -1
        max_link = -1
        for column_index, name in enumerate(headers):
            match = pattern.fullmatch(name)
            if not match or int(match.group(1)) != 0:
                continue
            link = int(match.group(2))
            subcarrier = int(match.group(3))
            mapping.append((column_index, subcarrier, link))
            max_subcarrier = max(max_subcarrier, subcarrier)
            max_link = max(max_link, link)
        if not mapping:
            raise ValueError(f"CSV没有amp_tx0_rx*_sub*列: {path}")

        timestamps: list[float] = []
        frames: list[np.ndarray] = []
        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                timestamp = float(row[timestamp_index])
                frame = np.zeros(
                    (max_subcarrier + 1, max_link + 1),
                    dtype=float,
                )
                for column_index, subcarrier, link in mapping:
                    frame[subcarrier, link] = float(row[column_index])
            except (IndexError, ValueError):
                print(f"  跳过无法解析的行: {path.name}:{row_number}")
                continue
            timestamps.append(timestamp)
            frames.append(frame)

    if not frames:
        raise ValueError(f"CSV没有可解析的CSI帧: {path}")
    timestamp_array = np.asarray(timestamps, dtype=float)
    raw = np.stack(frames, axis=0)
    order = np.argsort(timestamp_array, kind="stable")
    return raw[order], timestamp_array[order]


def infer_timestamp_scale(
    records: list[CSVRecord],
    requested_unit: str,
) -> tuple[str, float]:
    factors = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    durations = [
        float(np.max(record.timestamps_raw) - np.min(record.timestamps_raw))
        for record in records
        if record.timestamps_raw.size >= 2
        and float(np.max(record.timestamps_raw) - np.min(record.timestamps_raw)) > 0
    ]
    if not durations:
        raise RuntimeError("ElderAL时间戳没有正的时间跨度")

    median_raw_duration = float(np.median(durations))
    if requested_unit != "auto":
        return requested_unit, factors[requested_unit]

    candidates: list[tuple[float, str, float]] = []
    for unit, factor in factors.items():
        duration_seconds = median_raw_duration * factor
        if 0.02 <= duration_seconds <= 600.0:
            score = abs(math.log(max(duration_seconds, 1e-12) / 5.0))
        else:
            score = 100.0 + abs(math.log10(max(duration_seconds, 1e-12)))
        candidates.append((score, unit, factor))
    _, unit, factor = min(candidates)
    return unit, factor


def timestamps_in_seconds(
    raw_timestamps: np.ndarray,
    factor: float,
) -> np.ndarray:
    if raw_timestamps.size == 0:
        return raw_timestamps.astype(float)
    return (raw_timestamps.astype(float) - float(raw_timestamps[0])) * factor


def hampel_with_diagnostics(
    raw: np.ndarray,
    half_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """与原实验相同，额外返回实际被替换位置的布尔掩码。"""
    source = np.asarray(raw)
    filtered = source.copy()
    changed = np.zeros(source.shape, dtype=bool)
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
    return filtered, changed


def select_representative_signal(
    records: list[CSVRecord],
    factor: float,
    half_window: int,
    search_limit: int,
) -> tuple[np.ndarray, np.ndarray, int, int, str]:
    """复现原Figure 5：在前N段中选择默认窗口改变量最大的真实通道。"""
    best_score = -1.0
    best: tuple[np.ndarray, np.ndarray, int, int, str] | None = None
    for index, record in enumerate(records[: min(search_limit, len(records))], start=1):
        raw, raw_timestamps = read_raw_csi(record)
        timestamps = timestamps_in_seconds(raw_timestamps, factor)
        filtered, _ = hampel_with_diagnostics(raw, half_window)
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
                record.file_name,
            )
        if index % 20 == 0 or index == min(search_limit, len(records)):
            print(f"    已检查候选片段 {index}/{min(search_limit, len(records))}")
    if best is None:
        raise RuntimeError("无法选择代表性 ElderAL 信号")
    return best


def plot_action_waveform(
    action_id: int,
    records: list[CSVRecord],
    windows: list[int],
    factor: float,
    search_limit: int,
    output_dir: Path,
) -> None:
    """保持原Figure 5的五层结构、颜色、图例和标记方式不变。"""
    default_window = 5 if 5 in windows else max(windows)
    raw, timestamps, frequency_index, antenna_index, file_name = (
        select_representative_signal(
            records,
            factor,
            default_window,
            search_limit,
        )
    )
    x = (
        timestamps
        if timestamps.size and timestamps[-1] > 0
        else np.arange(raw.shape[0])
    )
    x_label = (
        "Time from sample start (s)"
        if timestamps.size and timestamps[-1] > 0
        else "Frame index"
    )
    raw_signal = raw[:, frequency_index, antenna_index]

    fig, axes = plt.subplots(
        len(windows) + 1,
        1,
        figsize=(7.2, 1.55 * (len(windows) + 1)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
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
        filtered, changed = hampel_with_diagnostics(raw, half_window)
        signal = filtered[:, frequency_index, antenna_index]
        mask = changed[:, frequency_index, antenna_index]
        ax.plot(
            x,
            raw_signal,
            color=OKABE_ITO["grey"],
            alpha=0.55,
            label="Raw",
        )
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
    action_name = ACTION_NAMES.get(action_id, f"action_{action_id}")
    save_publication_figure(
        fig,
        f"figure_5_action_{action_id}_{action_name}_representative_waveforms",
        output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help=(
            "ElderAL原始数据目录；默认自动使用项目中的"
            "sdp_dataset/elderAL"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(FIGURE_DIR),
        help="输出目录；默认仍是原实验的figures文件夹",
    )
    parser.add_argument(
        "--timestamp-unit",
        choices=["auto", "s", "ms", "us", "ns"],
        default="auto",
        help="原始timestamp单位；建议使用与原实验一致的us",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5],
        help="Hampel半窗口；默认1 2 3 5，即完整3 5 7 11帧",
    )
    parser.add_argument(
        "--actions",
        type=int,
        nargs="+",
        default=[0, 2, 3, 5, 6],
        help="需要画的动作ID；默认画除走路以外的其余五个动作",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=100,
        help="每个动作按原Figure 5规则检查的候选片段数；默认100",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = parse_positive_ints(args.windows, "--windows")
    if windows[0] < 1:
        raise ValueError("--windows 必须全部 >= 1")
    actions = parse_positive_ints(args.actions, "--actions")
    if len(windows) > 5:
        raise ValueError("--windows 最多给5个值")
    if args.search_limit < 1:
        raise ValueError("--search-limit 必须 >= 1")

    configure_publication_style()
    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    records = scan_records(data_path)
    timestamp_unit, factor = infer_timestamp_scale(records, args.timestamp_unit)
    print(f"timestamp单位: {timestamp_unit}；1单位={factor:g}秒")

    grouped: dict[int, list[CSVRecord]] = {}
    for record in records:
        grouped.setdefault(record.action_id, []).append(record)

    generated = 0
    for action_id in actions:
        action_records = grouped.get(action_id, [])
        if not action_records:
            print(f"跳过action {action_id}：没有找到有效CSV")
            continue
        action_name = ACTION_NAMES.get(action_id, str(action_id))
        print(f"绘制 action {action_id} ({action_name})：{len(action_records)}段")
        plot_action_waveform(
            action_id,
            action_records,
            windows,
            factor,
            args.search_limit,
            output_dir,
        )
        generated += 1

    print(f"完成：生成{generated}个动作的同版式波形图，保存到 {output_dir}")


if __name__ == "__main__":
    main()
