"""Stretch short Elder CSI samples to 80 frames with linear interpolation."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from elder_common import INPUT_SHAPE, run_experiment


def interpolate_frames(data):
    amplitude = np.abs(data)
    target = INPUT_SHAPE[0]
    if len(amplitude) >= target:
        return amplitude[:target]

    positions = np.linspace(0, len(amplitude) - 1, target)
    left = positions.astype(np.intp)
    right = np.minimum(left + 1, len(amplitude) - 1)
    fraction = (positions - left)[:, None, None]
    return amplitude[left] * (1 - fraction) + amplitude[right] * fraction


def make_inputs(elder, samples, frame_counts):
    import torch

    data = np.empty((len(samples), *INPUT_SHAPE), dtype=np.float32)
    paths = [elder.DATA_PATH / sample["id"].rsplit("#", 1)[0] for sample in samples]
    with ProcessPoolExecutor(
        max_workers=elder.PARALLEL_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=elder.initialize_worker,
    ) as executor:
        parsed = elder.ordered_parallel_map(executor, elder.read_raw_file, paths)
        for index, (arrays, records, _) in enumerate(parsed):
            sample = samples[index]
            if len(records) != 1 or any(records[0][key] != sample[key] for key in ("id", "label", "group")):
                raise ValueError(f"原始 CSI 与样本清单不一致: {sample['id']}")
            expected_shape = (frame_counts[index], *INPUT_SHAPE[1:])
            if arrays[0].shape != expected_shape:
                raise ValueError(f"原始 CSI 的帧数或形状不一致: {sample['id']}")
            data[index] = interpolate_frames(arrays[0])
    return torch.from_numpy(data)


if __name__ == "__main__":
    run_experiment(make_inputs, "no_padding.csv", include_difference=True)
