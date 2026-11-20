"""从原始 ElderAL 记录重新统计 11 帧 Hampel 替换点。"""

import csv
from itertools import combinations
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np


FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parent
DATA_DIR = ROOT / "datasets" / "elder"
SPLIT_DIR = ROOT / "test_elder" / "splits"
RESULTS = FOLDER / "results"
FEATURES = RESULTS / "spikes_per_sample.csv"
ACTION_NAMES = {0: "Static", 1: "Walking", 2: "Fall", 3: "Sleeping", 5: "Drinking", 6: "Sitting / TV"}
TIME_BINS = 10
BIN_FIELDS = ("count", "mean_amplitude", "fraction")
FEATURE_COLUMNS = tuple(f"bin_{index:02d}_{field}" for index in range(1, TIME_BINS + 1) for field in BIN_FIELDS)
SYNC_THRESHOLDS = (25, 50, 75)
HALF_WINDOW = 5
N_SIGMA = 3.0
SECONDS_PER_TIMESTAMP_UNIT = 1e-6


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        pending = Path(temporary) / path.name
        with pending.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        pending.replace(path)


def hampel_residual(raw):
    """按现有 Hampel 的规则计算原值减替换值，同时处理全部子载波。"""
    residual = np.zeros_like(raw)
    if len(raw) < 3:
        return residual
    for frame, value in enumerate(raw):
        window = raw[max(0, frame - HALF_WINDOW):frame + HALF_WINDOW + 1]
        median = np.median(window, axis=0)
        mad = np.median(np.abs(window - median), axis=0)
        delta = value - median
        residual[frame] = np.where(np.abs(delta) > N_SIGMA * 1.4826 * mad, delta, 0)
    return residual


def describe(raw, seconds):
    residual = hampel_residual(raw)
    changed = residual != 0
    amplitudes = np.abs(residual)
    duration = float(seconds[-1])
    bins = np.minimum((seconds / duration * TIME_BINS).astype(int), TIME_BINS - 1) if duration else np.zeros(len(raw), dtype=int)

    features = {}
    for index in range(TIME_BINS):
        in_bin = bins == index
        count = int(changed[in_bin].sum())
        prefix = f"bin_{index + 1:02d}_"
        features[prefix + "count"] = count
        features[prefix + "mean_amplitude"] = float(amplitudes[in_bin].sum()) / max(count, 1)
        features[prefix + "fraction"] = count / max(changed[in_bin].size, 1)

    # 按接收链路统计。同一帧的原始值变化方向还需与被替换点的方向一致。
    step = np.diff(raw, axis=0)
    positive = (residual[1:] > 0) & (step > 0)
    negative = (residual[1:] < 0) & (step < 0)
    synchronized = np.maximum(positive.mean(axis=1), negative.mean(axis=1)).max(axis=1)
    affected_frames = int(changed[1:].any(axis=(1, 2)).sum())
    sync_counts = {f"sync_{pct}_frames": int((synchronized >= pct / 100).sum()) for pct in SYNC_THRESHOLDS}

    flat_changed = changed.reshape(len(raw), -1)
    flat_raw = raw.reshape(len(raw), -1)
    edges = np.diff(np.pad(flat_changed, ((1, 1), (0, 0))).astype(np.int8), axis=0)
    channel, start = np.nonzero(edges.T == 1)
    _, end = np.nonzero(edges.T == -1)
    lengths = end - start
    bounded = (start > 0) & (end < len(raw))
    left, right, columns = start[bounded], end[bounded], channel[bounded]
    entry = flat_raw[left, columns] - flat_raw[left - 1, columns]
    exit_jump = flat_raw[right, columns] - flat_raw[right - 1, columns]
    neighbor_gap = np.abs(flat_raw[right, columns] - flat_raw[left - 1, columns])
    # 进出方向相反，且前后水平差不超过较小跳幅的一半，计为恢复。
    returned = (entry * exit_jump < 0) & (neighbor_gap <= 0.5 * np.minimum(np.abs(entry), np.abs(exit_jump)))
    short = lengths[bounded] <= 2
    short_return = returned & short
    return_spans = seconds[right[short_return]] - seconds[left[short_return] - 1]
    rebound_points = np.zeros_like(flat_changed)
    rebound_points[left[short_return], columns[short_return]] = True
    second_frame = short_return & (right - left == 2)
    rebound_points[left[second_frame] + 1, columns[second_frame]] = True
    rebound = rebound_points.reshape(raw.shape)[1:]
    short_synchronized = np.maximum(
        (positive & rebound).mean(axis=1), (negative & rebound).mean(axis=1),
    ).max(axis=1)
    short_sync_counts = {
        f"sync_{pct}_short_return_frames": int((short_synchronized >= pct / 100).sum())
        for pct in SYNC_THRESHOLDS
    }
    frame_positions = np.flatnonzero(changed.any(axis=(1, 2)))
    return {
        "frames": len(raw),
        "duration_s": duration,
        "available_points": raw.size,
        "replaced_points": int(changed.sum()),
        "amplitude_sum": float(amplitudes.sum()),
        "max_amplitude": float(amplitudes.max()),
        "segments_1_frame": int((lengths == 1).sum()),
        "segments_2_frames": int((lengths == 2).sum()),
        "segments_3plus_frames": int((lengths >= 3).sum()),
        "segments_with_neighbors": int(bounded.sum()),
        "returned_segments": int(returned.sum()),
        "short_segments_with_neighbors": int(short.sum()),
        "returned_short_segments": int(short_return.sum()),
        "short_return_span_sum_s": float(return_spans.sum()),
        "affected_frames_with_previous": affected_frames,
        **sync_counts,
        **short_sync_counts,
        "spike_frame_indices_zero_based": json.dumps(frame_positions.tolist()),
        "spike_frame_times_s": json.dumps(seconds[frame_positions].tolist()),
        **features,
    }


def summarize(rows):
    count_fields = (
        "frames", "available_points", "replaced_points", "amplitude_sum", "segments_1_frame",
        "segments_2_frames", "segments_3plus_frames", "segments_with_neighbors", "returned_segments",
        "short_segments_with_neighbors", "returned_short_segments", "short_return_span_sum_s",
        "affected_frames_with_previous", *(f"sync_{pct}_frames" for pct in SYNC_THRESHOLDS),
        *(f"sync_{pct}_short_return_frames" for pct in SYNC_THRESHOLDS),
    )
    sums = {key: sum(row[key] for row in rows) for key in count_fields}
    affected = sums["affected_frames_with_previous"]
    return {
        "samples": len(rows),
        **sums,
        "spike_fraction": sums["replaced_points"] / sums["available_points"],
        "mean_amplitude": sums["amplitude_sum"] / max(sums["replaced_points"], 1),
        "max_amplitude": max(row["max_amplitude"] for row in rows),
        "short_return_fraction": sums["returned_short_segments"] / max(sums["short_segments_with_neighbors"], 1),
        "mean_short_return_span_s": sums["short_return_span_sum_s"] / max(sums["returned_short_segments"], 1),
        **{f"sync_{pct}_fraction": sums[f"sync_{pct}_frames"] / max(affected, 1) for pct in SYNC_THRESHOLDS},
        **{
            f"sync_{pct}_short_return_fraction": sums[f"sync_{pct}_short_return_frames"] / max(affected, 1)
            for pct in SYNC_THRESHOLDS
        },
    }


def compare_locations(cells):
    # 每组使用同一个动作或同一个位置内的成对差，避免大样本组压过小样本组。
    metrics = (
        "spike_fraction", "mean_amplitude", "short_return_fraction",
        *(f"sync_{pct}_fraction" for pct in SYNC_THRESHOLDS),
        *(f"sync_{pct}_short_return_fraction" for pct in SYNC_THRESHOLDS),
    )
    within_action = [(a, b) for a, b in combinations(cells, 2) if a["action_id"] == b["action_id"]]
    within_location = [(a, b) for a, b in combinations(cells, 2) if a["location"] == b["location"]]
    return [{
        "metric": metric,
        "same_action_different_location_pairs": len(within_action),
        "same_action_different_location_mean_abs_diff": np.mean([abs(a[metric] - b[metric]) for a, b in within_action]),
        "same_location_different_action_pairs": len(within_location),
        "same_location_different_action_mean_abs_diff": np.mean([abs(a[metric] - b[metric]) for a, b in within_location]),
    } for metric in metrics]


def main():
    sys.path.insert(0, str(ROOT / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main" / "src"))
    from wsdp.readers import get_reader_class

    reader = get_reader_class("elderAL")()
    samples = read_json(SPLIT_DIR / "samples.json")["samples"]
    rows = []
    for index, sample in enumerate(samples, start=1):
        relative, record_index = sample["id"].rsplit("#", 1)
        path = DATA_DIR / relative
        if path.stat().st_size != sample["size"]:
            raise ValueError(f"原始文件与样本清单不匹配: {path}")
        loaded = reader.read_file(str(path))
        records = loaded if isinstance(loaded, list) else [loaded]
        frames = sorted(records[int(record_index)].frames, key=lambda frame: frame.timestamp)
        if len(frames) < 2:
            raise ValueError(f"有效帧不足: {path}")
        timestamps = np.array([frame.timestamp for frame in frames], dtype=float)
        seconds = (timestamps - timestamps[0]) * SECONDS_PER_TIMESTAMP_UNIT
        raw = np.stack([frame.csi_array for frame in frames])
        if raw.ndim != 3 or not np.isfinite(raw).all() or not np.isfinite(seconds).all():
            raise ValueError(f"信号或时间戳有误: {path}")
        rows.append({
            "sample_id": sample["id"], "action_id": sample["label"],
            "action": ACTION_NAMES[sample["label"]], "location": sample["group"],
            **describe(raw, seconds),
        })
        if index % 250 == 0 or index == len(samples):
            print(f"已统计 {index}/{len(samples)} 段", flush=True)

    by_action = []
    by_location = []
    for action_id, action in ACTION_NAMES.items():
        action_rows = [row for row in rows if row["action_id"] == action_id]
        by_action.append({"action_id": action_id, "action": action, **summarize(action_rows)})
        for location in sorted({row["location"] for row in action_rows}):
            selected = [row for row in action_rows if row["location"] == location]
            by_location.append({
                "action_id": action_id, "action": action, "location": location, **summarize(selected),
            })
    write_csv(FEATURES, rows)
    write_csv(RESULTS / "spikes_by_action.csv", by_action)
    write_csv(RESULTS / "spikes_by_action_location.csv", by_location)
    write_csv(RESULTS / "location_comparison.csv", compare_locations(by_location))


if __name__ == "__main__":
    main()
