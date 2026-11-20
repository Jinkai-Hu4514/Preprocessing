import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


CUDA_DEVICE_INDEX: int | None = 0  # 两个神经网络用的卡，设成 None 则用 CPU。

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parent
DATA_DIR = ROOT / "datasets" / "elder"
SPLIT_DIR = ROOT / "test_elder" / "splits"
RAW_DIR = ROOT / "test_elder" / "raw_test"
RESULTS = FOLDER / "results"
FEATURES = RESULTS / "frame_time_features.csv"
SEEDS = (42, 727, 914)
NEURAL_MODELS = ("CSITime", "ResNet2D")
TIME_COLUMNS = ("mean_gap_seconds", "std_gap_seconds", "max_gap_seconds")
CONDITIONS = ("frame", "time", "frame_time")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, fields, rows):
    path.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        pending = Path(temporary) / path.name
        with pending.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        pending.replace(path)


def load_split(elder, manifest, seed):
    split = read_json(SPLIT_DIR / f"split{seed}.json")
    if split["seed"] != seed or split["sample_digest"] != elder.digest(manifest):
        raise ValueError(f"seed {seed} 的划分与样本清单不匹配")
    indices = [split[key] for key in ("train", "validation", "test")]
    if sorted(index for part in indices for index in part) != list(range(len(manifest["samples"]))):
        raise ValueError(f"seed {seed} 的划分存在重复或缺失的样本")
    group_sets = [{manifest["samples"][index]["group"] for index in part} for part in indices]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise ValueError(f"seed {seed} 的划分存在相同位置跨集合")
    return split, indices


def update_comparison():
    paths = {name: RESULTS / f"{name}.csv" for name in CONDITIONS}
    if not all(path.is_file() for path in paths.values()):
        return
    runs = {
        name: {(row["model"], row["seed"]): row for row in read_csv(path)}
        for name, path in paths.items()
    }
    rows = []
    for (model, seed), first in runs["frame"].items():
        key = (model, seed)
        rows.append({
            "model": model,
            "seed": seed,
            **{name: runs[name][key]["test_accuracy"] for name in CONDITIONS},
            **{f"{name}_macro_f1": runs[name][key].get("test_macro_f1", "") for name in CONDITIONS},
            "majority_accuracy": first["majority_accuracy"],
            "majority_macro_f1": first.get("majority_macro_f1", ""),
            "raw_accuracy": first["raw_accuracy"],
        })
    write_csv(RESULTS / "comparison.csv", list(rows[0]), rows)


def run_training(condition, columns):
    sys.path.insert(0, str(ROOT))
    from test_elder.raw_test import raw_test_elder as elder
    import numpy as np
    import torch
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader, Subset, TensorDataset

    manifest = read_json(SPLIT_DIR / "samples.json")
    samples = manifest["samples"]
    feature_rows = read_csv(FEATURES)
    if [row["sample_id"] for row in feature_rows] != [sample["id"] for sample in samples]:
        raise ValueError("特征表和样本清单不匹配，请重跑 calculate_frame.py")
    features = np.array([[float(row[name]) for name in columns] for row in feature_rows])
    if not np.isfinite(features).all() or (features < 0).any():
        raise ValueError("特征表中存在无效数值")
    classes, labels = np.unique([sample["label"] for sample in samples], return_inverse=True)
    class_indices = list(range(len(classes)))
    params = elder.load_params(elder.DATASET_NAME)
    if CUDA_DEVICE_INDEX is None:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available() or not 0 <= CUDA_DEVICE_INDEX < torch.cuda.device_count():
            raise ValueError(f"CUDA {CUDA_DEVICE_INDEX} 不可用，请修改 common.py 中的 CUDA_DEVICE_INDEX")
        device = torch.device("cuda", CUDA_DEVICE_INDEX)

    rows = []
    for seed in SEEDS:
        split, indices = load_split(elder, manifest, seed)
        train, _, test = indices
        majority = np.bincount(labels[train]).argmax()
        majority_accuracy = float((labels[test] == majority).mean())
        majority_macro_f1 = f1_score(
            labels[test], np.full(len(test), majority),
            labels=class_indices, average="macro", zero_division=0,
        )
        raw_rows = read_csv(RAW_DIR / f"result{seed}" / "elder_raw_summary.csv")
        raw_accuracy = {
            row["model"].lower(): float(row["test_acc"])
            for row in raw_rows
            if row["status"] == "ok" and row["split_digest"] == elder.digest(split)
            and row["model"].lower() in {name.lower() for name in NEURAL_MODELS}
        }
        if len(raw_accuracy) != len(NEURAL_MODELS):
            raise ValueError(f"seed {seed} 缺少匹配的 raw 基线")

        # Neural models receive one value per feature. Compute scaling from training samples only.
        logged = np.log1p(features)
        train_mean = logged[train].mean(axis=0)
        train_std = logged[train].std(axis=0)
        normalized = (logged - train_mean) / np.where(train_std == 0, 1.0, train_std)
        inputs = torch.tensor(normalized, dtype=torch.float32)[:, None, None, :]
        dataset = TensorDataset(inputs, torch.from_numpy(labels))
        for model_name in NEURAL_MODELS:
            print(f"{condition} | {model_name} | seed {seed} | {device}", flush=True)
            elder.set_seed(seed)
            loaders = [
                DataLoader(
                    Subset(dataset, part), batch_size=params["batch"], shuffle=index == 0,
                    generator=torch.Generator().manual_seed(seed),
                )
                for index, part in enumerate(indices)
            ]
            model = elder.create_model(model_name, len(classes), tuple(inputs.shape[1:])).to(device)
            with TemporaryDirectory() as temporary:
                checkpoint, _ = elder.train_registered_model(model, device, loaders, params, Path(temporary))
                _, accuracy, macro_f1 = elder.evaluate_checkpoint(
                    model, device, loaders[2], checkpoint, len(classes)
                )
            rows.append({
                "model": model_name, "seed": seed, "test_accuracy": accuracy,
                "test_macro_f1": macro_f1, "majority_accuracy": majority_accuracy,
                "majority_macro_f1": majority_macro_f1, "raw_accuracy": raw_accuracy[model_name.lower()],
            })

        forest = RandomForestClassifier(random_state=seed)
        forest.fit(features[train], labels[train])
        predicted = forest.predict(features[test])
        rows.append({
            "model": "Random Forest", "seed": seed, "test_accuracy": float((labels[test] == predicted).mean()),
            "test_macro_f1": f1_score(labels[test], predicted, labels=class_indices, average="macro", zero_division=0),
            "majority_accuracy": majority_accuracy, "majority_macro_f1": majority_macro_f1, "raw_accuracy": "",
        })

    write_csv(RESULTS / f"{condition}.csv", list(rows[0]), rows)
    update_comparison()
