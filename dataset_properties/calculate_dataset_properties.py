"""按 benchmark 样本清单生成数据卡片，每个数据集一行。

运行 python dataset_properties/calculate_dataset_properties.py，可用 --workers 调整进程数。
数据读取自 datasets/<目录名>，清单来自 test_<目录名>/splits/。
帧间隔取同一序列内的正向时间差。有效帧比例以清单涵盖文件的存储记录为分母。
标称采样率采用已知配置，缺少采集信息时填 N/A。
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main" / "src"
sys.path.insert(0, str(SOURCE))
from wsdp.readers import get_reader_class


NA = "N/A"
BFEE_DATASETS = {"gait", "widar"}
HARDWARE = {
    "elderAL": "Commercial AP",
    "gait": "Intel 5300",
    "widar": "Intel 5300",
    "xrf55": "Intel 5300",
}
# XRF55 的 200 Hz 来自仓库中已有的标称采样率记录。
NOMINAL_SAMPLING_HZ = {"xrf55": 200}
SPLIT_METHOD = {
    "elderAL": "GroupShuffleSplit by position",
    "gait": "GroupShuffleSplit by track and receiver",
    "widar": "GroupShuffleSplit by position, orientation and receiver",
    "xrf55": "Fixed repetition split (train 1-12, validation 13-16, test 17-20)",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def count_raw_frames(path, dataset):
    if dataset in BFEE_DATASETS:
        count = 0
        with path.open("rb") as stream:
            while len(header := stream.read(3)) == 3:
                length = int.from_bytes(header[:2], "big")
                if length < 1:
                    return None
                count += header[2] == 0xBB
                stream.seek(length - 1, 1)
        return count
    if dataset == "elderAL" and path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            rows = csv.reader(stream)
            next(rows, None)
            return sum(bool(row) for row in rows)
    return None


def frame_intervals(frames, dataset, path):
    if dataset == "xrf55" or (dataset == "elderAL" and path.suffix.lower() != ".csv"):
        return "no", None
    if dataset not in BFEE_DATASETS and dataset != "elderAL":
        return NA, None

    # Elder CSV 沿用现有时序统计脚本的微秒口径。
    timestamps = np.asarray([frame.timestamp for frame in frames], dtype=np.float64)
    delta = np.diff(timestamps)
    if dataset in BFEE_DATASETS:
        # Intel 5300 使用 32 位微秒计数器。
        delta[delta > 2**31] -= 2**32
        delta[delta < -(2**31)] += 2**32
    return "yes", delta[np.isfinite(delta) & (delta > 0)] / 1_000


def measure_file(dataset, data_dir, entry):
    relative, indices = entry
    path = data_dir / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    records = get_reader_class(dataset)().read_file(str(path))
    if not isinstance(records, list):
        records = [records]

    measurements = []
    for index in indices:
        if not 0 <= index < len(records) or len(records[index].frames) < 2:
            raise ValueError(f"原始文件与 benchmark 清单不匹配: {path}#{index}")
        frames = records[index].frames
        data = np.stack([frame.csi_array for frame in frames])
        valid_frames = int(np.isfinite(data).reshape(len(frames), -1).all(axis=1).sum())
        subcarriers = data.shape[1]
        channels = int(np.prod(data.shape[2:]))
        representation = "complex" if np.iscomplexobj(data) else "real"
        antenna_layouts = set()
        if dataset in BFEE_DATASETS:
            antenna_counts = {(frame.n_tx, frame.n_rx) for frame in frames}
            antenna_layouts = {f"{tx} TX x {rx} RX" for tx, rx in antenna_counts}
        elif dataset == "elderAL" and path.suffix.lower() == ".csv":
            representation = "amplitude"
            antenna_layouts = {f"1 TX (TX0) x {channels} RX used by reader"}
        elif dataset == "xrf55":
            antenna_layouts = {"3 receivers x 3 antennas"}

        timestamp_status, gaps = frame_intervals(frames, dataset, path)
        measurements.append((len(frames), valid_frames, subcarriers, channels,
                             representation, antenna_layouts, timestamp_status, gaps))
    return count_raw_frames(path, dataset), measurements


def summarize(values, prefix, include_min=False):
    percentiles = {"min": 0, "median": 50, "p90": 90, "max": 100}
    if not include_min:
        del percentiles["min"]
    if len(values):
        numbers = np.percentile(values, list(percentiles.values()), overwrite_input=True)
    else:
        numbers = repeat(NA)
    return {f"{prefix}_{name}": value for name, value in zip(percentiles, numbers)}


def split_sizes(split_dir, manifest):
    data = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    sizes = {}
    for path in sorted(split_dir.glob("split*.json")):
        split = read_json(path)
        if split["sample_digest"] != digest:
            raise ValueError(f"划分与样本清单不匹配: {path}")
        sizes[f"test_samples_seed_{split['seed']}"] = len(split["test"])
    return sizes


def dataset_card(split_dir, executor):
    manifest = read_json(split_dir / "samples.json")
    dataset = manifest["dataset"]
    folder = split_dir.parent.name.removeprefix("test_")
    data_dir = ROOT / "datasets" / folder
    samples = manifest["samples"]
    if not samples:
        raise ValueError(f"样本清单为空: {split_dir}")
    sizes = split_sizes(split_dir, manifest)
    if not sizes:
        raise ValueError(f"缺少 benchmark 划分: {split_dir}")
    print(f"统计 {dataset}（{len(samples)} 条样本）", flush=True)

    files = defaultdict(list)
    for sample in samples:
        relative, index = sample["id"].rsplit("#", 1)
        files[relative].append(int(index))

    raw_counts, measurements = [], []
    for raw_count, records in executor.map(
        measure_file, repeat(dataset), repeat(data_dir), files.items(), chunksize=8
    ):
        raw_counts.append(raw_count)
        measurements.extend(records)

    lengths, valid, subcarriers, channels, representations, antennas, timestamps, gaps = zip(*measurements)
    intervals = [values for values in gaps if values is not None and len(values)]
    total_raw = sum(raw_counts) if None not in raw_counts else None
    valid_count = sum(valid)
    classes = len({sample["label"] for sample in samples})
    return {
        "dataset": dataset,
        "hardware": HARDWARE.get(dataset, NA),
        "nominal_sampling_hz": NOMINAL_SAMPLING_HZ.get(dataset, NA),
        "subcarriers": "/".join(map(str, sorted(set(subcarriers)))),
        "csi_channels": "/".join(map(str, sorted(set(channels)))),
        "antenna_layout": "/".join(sorted(set().union(*antennas))) or NA,
        "csi_representation": "/".join(sorted(set(representations))),
        "capture_timestamp": "/".join(sorted(set(timestamps))),
        **summarize(np.concatenate(intervals) if intervals else [], "frame_interval_ms"),
        "raw_frame_count": total_raw if total_raw is not None else NA,
        "valid_frame_count": valid_count,
        "valid_frame_ratio": valid_count / total_raw if total_raw else NA,
        **summarize(lengths, "sequence_frames", include_min=True),
        "classes": classes,
        "uniform_guess_accuracy": 1 / classes,
        "split_method": SPLIT_METHOD.get(dataset, NA),
        "benchmark_samples": len(samples),
        **sizes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers 必须至少为 1")

    manifests = sorted(ROOT.glob("test_*/splits/samples.json"))
    if not manifests:
        raise FileNotFoundError("找不到 test_*/splits/samples.json")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        cards = [dataset_card(path.parent, executor) for path in manifests]

    output = HERE / "dataset_properties.csv"
    columns = list(dict.fromkeys(key for card in cards for key in card))
    with TemporaryDirectory(dir=HERE) as temporary:
        pending = Path(temporary) / output.name
        with pending.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, columns, restval=NA)
            writer.writeheader()
            writer.writerows(cards)
        pending.replace(output)
    print(f"已写入 {output}")


if __name__ == "__main__":
    main()
