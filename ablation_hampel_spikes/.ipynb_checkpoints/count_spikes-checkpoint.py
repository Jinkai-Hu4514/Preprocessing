"""统计 ElderAL 原始信号中被 11 帧 Hampel 改掉的点。"""

import argparse
import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
RESULTS = ROOT / "results"
WSDP_SRC = ROOT.parent / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main" / "src"
ACTIONS = {0: "Static", 1: "Walking", 2: "Fall", 3: "Sleeping", 5: "Drinking", 6: "Sitting / TV"}
HALF_WINDOW = 5
N_SIGMA = 3.0
TIMESTAMP_SECONDS = 1e-6  # ElderAL 原始 CSV 时间戳以微秒计。


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hampel_residual(raw):
    """对所有信号列一起计算，规则与 WSDP 的 Hampel 相同。"""
    residual = np.zeros_like(raw)
    if len(raw) < 3:
        return residual
    for index, frame in enumerate(raw):
        window = raw[max(0, index - HALF_WINDOW):index + HALF_WINDOW + 1]
        median = np.median(window, axis=0)
        mad = np.median(np.abs(window - median), axis=0)
        delta = frame - median
        residual[index] = np.where(np.abs(delta) > N_SIGMA * 1.4826 * mad, delta, 0)
    return residual


def describe_sample(raw, timestamps):
    residual = hampel_residual(raw)
    changed = residual != 0
    edges = np.diff(np.pad(changed, ((1, 1), (0, 0))).astype(np.int8), axis=0)
    channel, start = np.nonzero(edges.T == 1)
    _, end = np.nonzero(edges.T == -1)
    lengths = end - start

    # 只判断两侧都有有效帧的 1–2 帧替换段。进入和离开的方向相反，
    # 并且两侧原值之差不超过进出跳幅中较小值的一半，才计为回到原水平。
    bounded = (lengths <= 2) & (start > 0) & (end < len(raw))
    left, right, columns = start[bounded], end[bounded], channel[bounded]
    entry = raw[left, columns] - raw[left - 1, columns]
    exit_jump = raw[right, columns] - raw[right - 1, columns]
    neighbor_gap = np.abs(raw[right, columns] - raw[left - 1, columns])
    returned = (entry * exit_jump < 0) & (neighbor_gap <= 0.5 * np.minimum(np.abs(entry), np.abs(exit_jump)))
    left, right, columns = left[returned], right[returned], columns[returned]

    # 一个保存帧里的多条信号合在一起，相邻尖峰帧连成一次。
    # 每次用绝对改变量最大的点代表跳幅和方向。
    spike_points = np.zeros_like(changed)
    spike_points[left, columns] = True
    two_frames = right - left == 2
    spike_points[left[two_frames] + 1, columns[two_frames]] = True
    spike_frames = np.flatnonzero(spike_points.any(axis=1))
    events = np.split(spike_frames, np.flatnonzero(np.diff(spike_frames) > 1) + 1) if spike_frames.size else []
    signed_peaks = []
    for event in events:
        values = residual[event][spike_points[event]]
        signed_peaks.append(values[np.argmax(np.abs(values))])
    event_times = [timestamps[event[0]] for event in events]
    return_spans = timestamps[right] - timestamps[left - 1]
    return {
        "frames": len(raw),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "replaced_points": int(changed.sum()),
        "replacement_abs_sum": float(np.abs(residual).sum()),
        "replacement_max_amplitude": float(np.abs(residual).max()),
        "segments_1_frame": int(np.count_nonzero(lengths == 1)),
        "segments_2_frames": int(np.count_nonzero(lengths == 2)),
        "segments_3plus_frames": int(np.count_nonzero(lengths >= 3)),
        "short_segments_with_neighbors": int(bounded.sum()),
        "returned_segments": int(returned.sum()),
        "return_span_sum_s": float(return_spans.sum()),
        "spike_count": len(events),
        "spike_mean_amplitude": float(np.mean(np.abs(signed_peaks))) if events else 0.0,
        "spike_max_amplitude": float(np.max(np.abs(signed_peaks))) if events else 0.0,
        "negative_spike_fraction": float(np.mean(np.asarray(signed_peaks) < 0)) if events else 0.0,
        "spike_mean_interval_s": float(np.mean(np.diff(event_times))) if len(events) > 1 else 0.0,
    }


def read_sample(data_dir, sample, reader, parse_file_info, select_label_and_group):
    relative, sample_index = sample["id"].rsplit("#", 1)
    path = data_dir / relative
    if sample_index != "0" or path.stat().st_size != sample["size"]:
        raise ValueError(f"原文件与样本清单不一致: {sample['id']}")
    record = reader.read_file(str(path))
    parsed = parse_file_info(record.file_name, "elderAL")
    if parsed is None or select_label_and_group(parsed, "elderAL") != (sample["label"], sample["group"]):
        raise ValueError(f"动作或采集位置与样本清单不一致: {sample['id']}")
    frames = sorted(record.frames, key=lambda frame: frame.timestamp)
    if len(frames) < 2:
        raise ValueError(f"有效帧少于两帧: {sample['id']}")
    raw = np.stack([frame.csi_array.ravel() for frame in frames])
    timestamps = np.asarray([frame.timestamp for frame in frames], dtype=np.float64)
    if not np.isfinite(raw).all() or not np.isfinite(timestamps).all():
        raise ValueError(f"信号或时间戳包含非有限值: {sample['id']}")
    return raw, (timestamps - timestamps[0]) * TIMESTAMP_SECONDS


def summarize_actions(rows):
    summary = []
    for label, action in ACTIONS.items():
        selected = [row for row in rows if row["label"] == label]
        fields = (
            "replaced_points", "segments_1_frame", "segments_2_frames", "segments_3plus_frames",
            "short_segments_with_neighbors", "returned_segments", "spike_count",
        )
        totals = {field: sum(row[field] for row in selected) for field in fields}
        summary.append({
            "label": label,
            "action": action,
            "samples": len(selected),
            **totals,
            "mean_replacement_amplitude": sum(row["replacement_abs_sum"] for row in selected) / max(1, totals["replaced_points"]),
            "max_replacement_amplitude": max(row["replacement_max_amplitude"] for row in selected),
            "mean_return_span_s": sum(row["return_span_sum_s"] for row in selected) / max(1, totals["returned_segments"]),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="原始 Elder CSV 所在目录")
    args = parser.parse_args()
    sys.path.insert(0, str(WSDP_SRC))
    from wsdp.processors.base_processor import _parse_file_info_from_filename, _selector
    from wsdp.readers import get_reader_class

    samples = json.loads((INPUTS / "samples.json").read_text(encoding="utf-8"))["samples"]
    reader = get_reader_class("elderAL")()
    rows = []
    for index, sample in enumerate(samples, start=1):
        raw, timestamps = read_sample(args.data_dir, sample, reader, _parse_file_info_from_filename, _selector)
        rows.append({
            "sample_id": sample["id"],
            "label": sample["label"],
            "action": ACTIONS[sample["label"]],
            "position": sample["group"],
            **describe_sample(raw, timestamps),
        })
        if index % 200 == 0 or index == len(samples):
            print(f"已统计 {index}/{len(samples)} 段", flush=True)

    RESULTS.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=RESULTS) as directory:
        staging = Path(directory)
        write_rows(staging / "spikes_per_sample.csv", rows)
        write_rows(staging / "spikes_by_action.csv", summarize_actions(rows))
        for path in staging.iterdir():
            path.replace(RESULTS / path.name)


if __name__ == "__main__":
    main()
