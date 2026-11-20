"""用 Elder full-test 的两个模型比较帧数、尖峰和组合输入。"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from count_spikes import ACTIONS, INPUTS, RESULTS, ROOT, write_rows


CUDA_DEVICE_INDEX = 2  # 默认 cuda:0。改成 1 就用 cuda:1，也可以通过 --device 覆盖。
SEEDS = (42, 727, 914)
MODELS = ("CSI-Time", "ResNet2D")
MODEL_KEYS = {"CSI-Time": "csitime", "ResNet2D": "resnet2d"}
INPUT_SHAPE = (80, 64, 3)
TIMING_FIELDS = ("frames", "duration_s")
SPIKE_FIELDS = (
    "spike_count", "spike_mean_amplitude", "spike_max_amplitude",
    "negative_spike_fraction", "spike_mean_interval_s",
)
FEATURE_FIELDS = TIMING_FIELDS + SPIKE_FIELDS
CONDITIONS = (
    ("timing_only", TIMING_FIELDS),
    ("spikes_only", SPIKE_FIELDS),
    ("timing_and_spikes", FEATURE_FIELDS),
)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_features(samples):
    with (RESULTS / "spikes_per_sample.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    actual = [(row["sample_id"], int(row["label"]), int(row["position"])) for row in rows]
    expected = [(sample["id"], sample["label"], sample["group"]) for sample in samples]
    if actual != expected:
        raise ValueError("逐样本尖峰表与样本清单不一致，请重新运行 count_spikes.py")
    features = np.asarray([[float(row[field]) for field in FEATURE_FIELDS] for row in rows])
    if not np.isfinite(features).all():
        raise ValueError("逐样本尖峰表包含非有限值")
    return features


def load_seed(seed, manifest):
    split = json.loads((INPUTS / f"split{seed}.json").read_text(encoding="utf-8"))
    parts = [np.asarray(split[key], dtype=int) for key in ("train", "validation", "test")]
    samples = manifest["samples"]
    indices = np.sort(np.concatenate(parts))
    if (split["seed"] != seed or split["sample_digest"] != digest(manifest)
            or not all(part.size for part in parts) or not np.array_equal(indices, np.arange(len(samples)))):
        raise ValueError(f"seed {seed} 的划分与样本清单不一致")
    positions = [{samples[index]["group"] for index in part} for part in parts]
    if len(set.union(*positions)) != sum(map(len, positions)):
        raise ValueError(f"seed {seed} 的训练、验证和测试出现相同的采集位置")

    split_digest = digest(split)
    with (INPUTS / f"raw_seed{seed}.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["status"] == "ok" and int(row["seed"]) == seed
            and row["split_digest"] == split_digest and row["model"].lower() in MODEL_KEYS.values()
        ]
    baseline = {row["model"].lower(): float(row["test_acc"]) for row in rows}
    if len(rows) != len(MODELS) or baseline.keys() != set(MODEL_KEYS.values()):
        raise ValueError(f"seed {seed} 的原始成绩缺失、重复或使用了不同的划分")
    return parts, baseline


def feature_dataset(features, labels, train_indices, active_fields):
    # 每个数字固定占一组列。所有帧和天线填写同样的数字，整张表没有补零边界。
    # 每一列的换算只用训练部分计算，未选用的数字所在列填零。
    training = features[train_indices]
    scale = training.std(axis=0)
    scale[scale == 0] = 1
    values = (features - training.mean(axis=0)) / scale
    values[:, [field not in active_fields for field in FEATURE_FIELDS]] = 0
    frames, subcarriers, antennas = INPUT_SHAPE
    columns = np.arange(subcarriers) * len(FEATURE_FIELDS) // subcarriers
    grid = torch.from_numpy(values[:, columns].astype(np.float32))
    return TensorDataset(grid[:, None, :, None].expand(-1, frames, -1, antennas), labels)


def test_model(model, device, loader, checkpoint):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    matrix = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)
    with torch.inference_mode():
        for data, labels in loader:
            predicted = model(data.to(device)).argmax(1).cpu().numpy()
            np.add.at(matrix, (labels.numpy(), predicted), 1)
    return matrix


def result_row(seed, model_name, condition, matrix, raw_accuracy, timing_accuracy):
    totals = matrix.sum(axis=1)
    correct = matrix.diagonal()
    accuracy = correct.sum() / totals.sum()
    row = {
        "seed": seed,
        "model": model_name,
        "input": condition,
        "test_samples": int(totals.sum()),
        "accuracy_pct": 100 * accuracy,
        "raw_accuracy_pct": 100 * raw_accuracy,
        "vs_raw_pp": 100 * (accuracy - raw_accuracy),
        "vs_timing_only_pp": 100 * (accuracy - timing_accuracy) if timing_accuracy is not None else 0.0,
    }
    for index, label in enumerate(ACTIONS):
        row[f"action_{label}_correct"] = int(correct[index])
        row[f"action_{label}_total"] = int(totals[index])
        row[f"action_{label}_accuracy_pct"] = 100 * correct[index] / totals[index] if totals[index] else ""
    row.update({
        f"true_{actual}_pred_{predicted}": int(matrix[i, j])
        for i, actual in enumerate(ACTIONS) for j, predicted in enumerate(ACTIONS)
    })
    return accuracy, row


def select_device(requested):
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device_type, separator, device_index = requested.partition(":")
    if device_type != "cuda":
        return torch.device(requested)
    if separator and not device_index.isdecimal():
        raise ValueError(f"CUDA 设备格式无效: {requested}，例如 cuda:0")
    index = int(device_index) if separator else CUDA_DEVICE_INDEX
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise ValueError(f"无法使用 cuda:{index}，当前可见的 CUDA 设备有 {count} 张")
    return torch.device("cuda", index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        help=f"覆盖脚本中的设备设置，例如 cuda:1 或 cpu。默认 cuda:{CUDA_DEVICE_INDEX}，无 CUDA 时用 CPU。",
    )
    args = parser.parse_args()
    try:
        device = select_device(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    sys.path.insert(0, str(ROOT.parent))
    from test_elder.full_test.full_test_elder import (
        DATASET_NAME,
        create_model,
        load_params,
        set_seed,
        train_registered_model,
    )

    manifest = json.loads((INPUTS / "samples.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    labels = torch.tensor([list(ACTIONS).index(sample["label"]) for sample in samples])
    features = load_features(samples)
    splits = {seed: load_seed(seed, manifest) for seed in SEEDS}
    params = load_params(DATASET_NAME)
    if params["padding_length"] != INPUT_SHAPE[0]:
        raise ValueError("固定输入的帧数与 Elder full-test 配置不一致")
    print(f"训练设备 {device} | Elder 默认训练配置 {params}", flush=True)

    rows = []
    with TemporaryDirectory(dir=RESULTS) as directory:
        staging = Path(directory)
        for seed, (parts, baseline) in splits.items():
            timing_accuracy = {}
            for condition, fields in CONDITIONS:
                dataset = feature_dataset(features, labels, parts[0], fields)
                for name in MODELS:
                    print(f"{name} | seed={seed} | {condition} | {device}", flush=True)
                    set_seed(seed)
                    loaders = tuple(
                        DataLoader(
                            Subset(dataset, indices), batch_size=params["batch"], shuffle=part == 0,
                            generator=torch.Generator().manual_seed(seed),
                        )
                        for part, indices in enumerate(parts)
                    )
                    model = create_model(MODEL_KEYS[name], num_classes=len(ACTIONS), input_shape=INPUT_SHAPE).to(device)
                    checkpoint, _ = train_registered_model(model, device, loaders, params, staging)
                    matrix = test_model(model, device, loaders[2], checkpoint)
                    accuracy, row = result_row(seed, name, condition, matrix, baseline[MODEL_KEYS[name]], timing_accuracy.get(name))
                    if condition == "timing_only":
                        timing_accuracy[name] = accuracy
                    rows.append(row)
                    print(f"测试准确率 {accuracy:.2%}", flush=True)
                    del model
        output = staging / "classification_results.csv"
        write_rows(output, rows)
        output.replace(RESULTS / output.name)


if __name__ == "__main__":
    main()
