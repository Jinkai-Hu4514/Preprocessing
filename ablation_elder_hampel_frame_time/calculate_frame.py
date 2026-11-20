"""从原始 Elder CSV 读取有效帧和时间戳，重新统计帧数与帧间隔。"""

import math
import multiprocessing as mp
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

from common import DATA_DIR, FEATURES, ROOT, SPLIT_DIR, TIME_COLUMNS, read_json, write_csv

sys.path.insert(0, str(ROOT / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main" / "src"))
from wsdp.readers import get_reader_class

SECONDS_PER_TIMESTAMP_UNIT = 1e-6
TARGET_FRAMES = 80


def measure(sample):
    relative, index = sample["id"].rsplit("#", 1)
    source = DATA_DIR / relative
    reader = get_reader_class("elderAL")()
    loaded = reader.read_file(str(source))
    records = loaded if isinstance(loaded, list) else [loaded]
    timestamps = sorted(frame.timestamp for frame in records[int(index)].frames)
    if len(timestamps) < 2 or not all(math.isfinite(value) for value in timestamps):
        raise ValueError(f"缺少有效时间戳: {sample['id']}")
    gaps = [(right - left) * SECONDS_PER_TIMESTAMP_UNIT for left, right in zip(timestamps, timestamps[1:])]
    mean, std, maximum = statistics.fmean(gaps), statistics.pstdev(gaps), max(gaps)
    return {
        "sample_id": sample["id"],
        "action_id": sample["label"],
        "group": sample["group"],
        "raw_frame_count": len(timestamps),
        "frame_count": min(len(timestamps), TARGET_FRAMES),
        **dict(zip(TIME_COLUMNS, (mean, std, maximum))),
    }


if __name__ == "__main__":
    samples = read_json(SPLIT_DIR / "samples.json")["samples"]
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as executor:
        rows = list(executor.map(measure, samples, chunksize=8))
    write_csv(FEATURES, list(rows[0]), rows)
    print(f"已重新统计 {len(rows)} 段，结果在 {FEATURES}")
