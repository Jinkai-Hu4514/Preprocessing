"""用十段尖峰统计训练 Elder full-test 的两个模型和随机森林。"""

import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset, TensorDataset

from count_spikes import BIN_FIELDS, FEATURES, FEATURE_COLUMNS, RESULTS, ROOT, SPLIT_DIR, TIME_BINS, read_json, write_csv


CUDA_DEVICE_INDEX = 0  # 改成 1 使用 cuda:1，设为 None 使用 CPU。
SEEDS = (42, 727, 914)
NEURAL_MODELS = ("CSITime", "ResNet2D")
RAW_DIR = ROOT / "test_elder" / "raw_test"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_features(samples):
    rows = read_csv(FEATURES)
    identities = [(row["sample_id"], int(row["action_id"]), int(row["location"])) for row in rows]
    expected = [(sample["id"], sample["label"], sample["group"]) for sample in samples]
    if identities != expected:
        raise ValueError("尖峰表与样本清单不匹配，请重跑 count_spikes.py")
    features = np.array([[float(row[column]) for column in FEATURE_COLUMNS] for row in rows])
    if not np.isfinite(features).all() or (features < 0).any():
        raise ValueError("尖峰表有无效数值")
    return features


def load_split(manifest, seed, digest):
    split = read_json(SPLIT_DIR / f"split{seed}.json")
    samples = manifest["samples"]
    parts = [split[name] for name in ("train", "validation", "test")]
    indices = sorted(index for part in parts for index in part)
    if split["seed"] != seed or split["sample_digest"] != digest(manifest) or indices != list(range(len(samples))):
        raise ValueError(f"seed {seed} 的划分与样本清单不匹配")
    locations = [{samples[index]["group"] for index in part} for part in parts]
    if len(set.union(*locations)) != sum(map(len, locations)):
        raise ValueError(f"seed {seed} 的训练、验证和测试存在位置重叠")
    return split, parts


def neural_inputs(features, labels, train, frame_count):
    if frame_count % TIME_BINS:
        raise ValueError("Elder full-test 的输入长度必须能被时间段数整除")
    values = np.log1p(features)
    scale = values[train].std(axis=0)
    normalized = (values - values[train].mean(axis=0)) / np.where(scale > 0, scale, 1)
    # 十段按时间排，每段的三个统计量占固定列。重复各段以沿用 Elder 的输入长度。
    inputs = torch.tensor(normalized, dtype=torch.float32).reshape(-1, TIME_BINS, 1, len(BIN_FIELDS))
    inputs = inputs.repeat_interleave(frame_count // TIME_BINS, dim=1)
    return TensorDataset(inputs, torch.from_numpy(labels))


def train_neural(elder, model_name, dataset, indices, class_count, seed, params, device):
    elder.set_seed(seed)
    loaders = [
        DataLoader(
            Subset(dataset, part), batch_size=params["batch"], shuffle=index == 0,
            generator=torch.Generator().manual_seed(seed),
        )
        for index, part in enumerate(indices)
    ]
    model = elder.create_model(model_name, class_count, tuple(dataset.tensors[0].shape[1:])).to(device)
    with TemporaryDirectory(dir=RESULTS) as directory:
        checkpoint, _ = elder.train_registered_model(model, device, loaders, params, Path(directory))
        _, accuracy, macro_f1 = elder.evaluate_checkpoint(model, device, loaders[2], checkpoint, class_count)
    return accuracy, macro_f1


def main():
    sys.path.insert(0, str(ROOT))
    from test_elder.full_test import full_test_elder as elder

    manifest = read_json(SPLIT_DIR / "samples.json")
    samples = manifest["samples"]
    features = load_features(samples)
    classes, labels = np.unique([sample["label"] for sample in samples], return_inverse=True)
    if CUDA_DEVICE_INDEX is not None and not 0 <= CUDA_DEVICE_INDEX < torch.cuda.device_count():
        raise ValueError(f"cuda:{CUDA_DEVICE_INDEX} 不可用，请修改脚本开头的 CUDA_DEVICE_INDEX")
    device = torch.device("cpu") if CUDA_DEVICE_INDEX is None else torch.device("cuda", CUDA_DEVICE_INDEX)
    params = elder.load_params(elder.DATASET_NAME)
    print(f"Elder full-test 配置 {params} | {device}", flush=True)

    rows = []
    for seed in SEEDS:
        split, indices = load_split(manifest, seed, elder.digest)
        train, _, test = indices
        targets = labels[test]
        class_indices = np.arange(len(classes))
        split_digest = elder.digest(split)
        raw_rows = read_csv(RAW_DIR / f"result{seed}" / "elder_raw_summary.csv")
        raw = {
            row["model"].lower(): row
            for row in raw_rows
            if row["status"] == "ok" and int(row["seed"]) == seed and row["split_digest"] == split_digest
        }
        if not all(name.lower() in raw for name in NEURAL_MODELS):
            raise ValueError(f"seed {seed} 缺少相同划分的 raw 成绩")

        majority = np.full(len(test), np.bincount(labels[train]).argmax())
        random_guess = np.random.default_rng(seed).integers(len(classes), size=len(test))
        forest = RandomForestClassifier(random_state=seed)
        forest.fit(features[train], labels[train])
        scores = [
            (name, inputs, accuracy_score(targets, predictions),
             f1_score(targets, predictions, labels=class_indices, average="macro", zero_division=0))
            for name, inputs, predictions in (
                ("Majority class", "baseline", majority),
                ("Uniform random", "baseline", random_guess),
                ("Random Forest", "spikes", forest.predict(features[test])),
            )
        ]
        dataset = neural_inputs(features, labels, train, params["padding_length"])
        for name in NEURAL_MODELS:
            print(f"{name} | seed {seed} | spikes", flush=True)
            accuracy, macro_f1 = train_neural(elder, name, dataset, indices, len(classes), seed, params, device)
            scores.append((name, "spikes", accuracy, macro_f1))
            reference = raw[name.lower()]
            scores.append((name, "raw", float(reference["test_acc"]), float(reference["macro_f1"])))

        for name, inputs, accuracy, macro_f1 in scores:
            rows.append({
                "seed": seed, "model": name, "input": inputs, "test_samples": len(test),
                "accuracy": accuracy, "macro_f1": macro_f1,
            })
            print(f"{name} | {inputs} | seed {seed} | Acc {accuracy:.2%} | Macro-F1 {macro_f1:.2%}")
    write_csv(RESULTS / "classification_results.csv", rows)


if __name__ == "__main__":
    main()
