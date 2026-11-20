"""Train the ElderAL Hampel ablation and plot seed-averaged confusion matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
WSDP_SRC = ROOT / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main" / "src"
sys.path.insert(0, str(WSDP_SRC))

from wsdp.algorithms import execute_pipeline
from wsdp.datasets import CSIDataset
from wsdp.models import create_model
from wsdp.readers import get_reader_class
from wsdp.utils import load_params, resize_csi_to_fixed_length


DATASET = "elderAL"
DATA_DIR = ROOT / "datasets" / "elder"
SPLIT_DIR = ROOT / "test_elder" / "splits"
RESULTS_DIR = SCRIPT_DIR / "results" / "elder_hampel_ablation"
MODELS = (("csitime", "CSI-Time"), ("resnet2d", "ResNet2D"))
SEEDS = (42, 727, 914)
WINDOWS = (0, 3, 5, 7, 11)
ACTION_NAMES = {
    0: "Static",
    1: "Walking",
    2: "Fall",
    3: "Sleeping",
    5: "Drinking",
    6: "Sitting / TV",
}
CLASSES = tuple(ACTION_NAMES.values())
CLASS_COUNT = len(CLASSES)
PROTOCOL_VERSION = 1
GREY = "#d9d9d9"
BLUE = "#9edcf0"
EDGE = "#303030"
CMAP = LinearSegmentedColormap.from_list(
    "hampel_blue", ["#f7fbff", "#d7eaf4", BLUE, "#4a91c2", "#174a83"],
)


@dataclass(frozen=True)
class Prediction:
    truth: np.ndarray
    predicted: np.ndarray
    best_val_accuracy: float
    best_epoch: int


def digest(value) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_splits(directory: Path):
    manifest = json.loads((directory / "samples.json").read_text(encoding="utf-8-sig"))
    samples = manifest["samples"]
    sample_ids = [sample["id"] for sample in samples]
    if manifest["dataset"] != DATASET or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Expected a unique ElderAL sample list in samples.json")
    if {sample["label"] for sample in samples} != set(ACTION_NAMES):
        raise ValueError("The sample list must contain the six ElderAL actions")
    label_index = {label: index for index, label in enumerate(ACTION_NAMES)}
    labels = np.asarray([label_index[sample["label"]] for sample in samples], dtype=np.int64)
    groups = np.asarray([sample["group"] for sample in samples])
    manifest_digest = digest(manifest)
    splits = {}
    split_digests = {}
    for seed in SEEDS:
        path = directory / f"split{seed}.json"
        record = json.loads(path.read_text(encoding="utf-8-sig"))
        if record["seed"] != seed or record["sample_digest"] != manifest_digest:
            raise ValueError(f"{path} does not match the current samples.json")
        indices = tuple(
            np.asarray(record[name], dtype=np.int64) for name in ("train", "validation", "test")
        )
        if any(index.ndim != 1 or not index.size for index in indices):
            raise ValueError(f"Invalid split shape in {path}")
        if not np.array_equal(np.sort(np.concatenate(indices)), np.arange(len(samples))):
            raise ValueError(f"Duplicate, missing, or invalid sample indices in {path}")
        train, validation, test = (set(groups[index]) for index in indices)
        if train & validation or train & test or validation & test:
            raise ValueError(f"Acquisition locations overlap in {path}")
        if len(np.unique(labels[indices[2]])) != CLASS_COUNT:
            raise ValueError(f"The test set in {path} must contain all six actions")
        splits[seed] = indices
        split_digests[seed] = digest(record)
    return samples, labels, splits, split_digests


def read_sample(sample, data_dir: Path) -> np.ndarray:
    relative_path, record_index = sample["id"].rsplit("#", 1)
    path = (data_dir / relative_path).resolve()
    if not path.is_relative_to(data_dir) or path.stat().st_size != sample["size"]:
        raise ValueError(f"Raw sample does not match the manifest: {path}")
    loaded = get_reader_class(DATASET)().read_file(str(path))
    records = loaded if isinstance(loaded, list) else [loaded]
    frames = sorted(records[int(record_index)].frames, key=lambda frame: frame.timestamp)
    if len(frames) < 2:
        raise ValueError(f"Missing CSI frames for {sample['id']}")
    data = np.stack([frame.csi_array for frame in frames])
    if data.ndim == 2:
        data = data[..., None]
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError(f"Invalid CSI values for {sample['id']}")
    return data


def pipeline_for(window: int) -> dict:
    steps = {}
    if window:
        steps["denoise"] = {"method": "hampel", "window_size": window // 2, "n_sigma": 3.0}
    steps["outliers"] = {"method": "iqr", "factor": 1.5}
    steps["normalize"] = {"method": "min-max"}
    steps["interpolate"] = {"method": "linear", "target_K": 64}
    return steps


def preprocess(raw: np.ndarray, steps: dict, padding_length: int) -> np.ndarray:
    result = execute_pipeline(raw, steps, dataset=DATASET)
    if result.ndim != 3 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"Invalid preprocessed CSI shape or values: {result.shape}")
    return resize_csi_to_fixed_length([result], target_length=padding_length)[0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0
    with torch.set_grad_enabled(training):
        for values, labels in loader:
            values, labels = values.to(device), labels.to(device)
            logits = model(values)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite training or validation loss")
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.detach().argmax(1) == labels).sum().item()
            total += len(labels)
    return loss_sum / total, correct / total


def train_predict(model_name, seed, dataset, indices, params, device, checkpoint) -> Prediction:
    set_seed(seed)
    loaders = tuple(
        DataLoader(
            Subset(dataset, index.tolist()), batch_size=params["batch"],
            shuffle=part == 0, generator=torch.Generator().manual_seed(seed),
        )
        for part, index in enumerate(indices)
    )
    model = create_model(
        model_name, num_classes=CLASS_COUNT, input_shape=tuple(dataset.data_list.shape[1:]),
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5,
    )
    best = -1.0
    for epoch in range(1, params["num_epochs"] + 1):
        train_loss, train_acc = run_epoch(model, loaders[0], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders[1], criterion, device)
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch}/{params['num_epochs']}  train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2%}  val_loss={val_loss:.4f}  val_acc={val_acc:.2%}",
            flush=True,
        )
        if val_acc > best:
            best = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_accuracy": best,
                "best_epoch": epoch,
                "padding_length": params["padding_length"],
            }, checkpoint)

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    truth, predicted = [], []
    with torch.inference_mode():
        for values, labels in loaders[2]:
            truth.extend(labels.tolist())
            predicted.extend(model(values.to(device)).argmax(1).cpu().tolist())
    return Prediction(
        np.asarray(truth, dtype=np.int64), np.asarray(predicted, dtype=np.int64),
        float(saved["best_val_accuracy"]), int(saved["best_epoch"]),
    )


def completed_prediction(directory: Path, signature: str, indices, truth):
    path = directory / "test_predictions.npz"
    if not path.is_file() or not (directory / "best_checkpoint.pth").is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        if (
            saved["signature"].item() != signature
            or not np.array_equal(saved["test_indices"], indices)
            or not np.array_equal(saved["y_true"], truth)
        ):
            return None
        predicted = saved["y_pred"]
        if predicted.shape != truth.shape or not np.isin(predicted, np.arange(CLASS_COUNT)).all():
            return None
        return Prediction(
            truth, predicted, float(saved["best_val_accuracy"]), int(saved["best_epoch"]),
        )


def run_condition(model, seed, dataset, indices, params, device, directory, signature, samples, labels):
    directory.mkdir(parents=True, exist_ok=True)
    print(f"\n{model}, {directory.parent.name}, seed={seed}", flush=True)
    test_indices = indices[2]
    with TemporaryDirectory(dir=directory) as temp:
        pending = Path(temp)
        checkpoint = pending / "best_checkpoint.pth"
        result = train_predict(model, seed, dataset, indices, params, device, checkpoint)
        if not np.array_equal(result.truth, labels[test_indices]):
            raise RuntimeError("Test prediction order does not match the shared split")
        predictions = pending / "test_predictions.npz"
        np.savez_compressed(
            predictions, signature=signature, test_indices=test_indices,
            sample_ids=[samples[index]["id"] for index in test_indices],
            y_true=result.truth, y_pred=result.predicted, class_ids=list(ACTION_NAMES),
            class_names=CLASSES, best_val_accuracy=result.best_val_accuracy,
            best_epoch=result.best_epoch,
        )
        # Publish the prediction file last, since it marks a completed run.
        (directory / predictions.name).unlink(missing_ok=True)
        checkpoint.replace(directory / checkpoint.name)
        predictions.replace(directory / predictions.name)
    print(f"Test accuracy: {(result.truth == result.predicted).mean():.2%}", flush=True)
    return result


def confusion_counts(result: Prediction) -> np.ndarray:
    counts = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(counts, (result.truth, result.predicted), 1)
    return counts


def draw_model(model_name, model_label, matrices, means, deviations, directory):
    figure = plt.figure(figsize=(13.7, 9.25), layout="constrained")
    grid = figure.add_gridspec(2, 4, width_ratios=(1, 1, 1, 0.045))
    figure.suptitle(model_label)

    for index, (window, matrix) in enumerate(zip(WINDOWS, matrices, strict=True)):
        row, column = divmod(index, 3)
        axis = figure.add_subplot(grid[row, column])
        image = axis.imshow(matrix, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
        condition = "No Hampel" if window == 0 else f"{window}-frame Hampel"
        axis.set_title(f"({chr(97 + index)}) {condition}")
        axis.set_xticks(range(CLASS_COUNT), CLASSES, rotation=38, ha="right")
        axis.set_yticks(range(CLASS_COUNT), CLASSES)
        if row == 1:
            axis.set_xlabel("Predicted class")
        if column == 0:
            axis.set_ylabel("True class")
        for (true_class, predicted_class), value in np.ndenumerate(matrix):
            if value >= 0.005:
                axis.text(
                    predicted_class, true_class, f"{value:.2f}", ha="center", va="center",
                    color="white" if value >= 0.55 else "#202020", fontsize=8,
                )

    figure.colorbar(image, cax=figure.add_subplot(grid[:, 3])).set_label(
        "Mean row-normalized proportion",
    )
    accuracy = figure.add_subplot(grid[1, 2])
    accuracy.set_box_aspect(1)
    positions = np.arange(len(WINDOWS))
    means, deviations = means * 100, deviations * 100
    accuracy.bar(
        positions, means, yerr=deviations, color=[GREY] + [BLUE] * 4,
        edgecolor=EDGE, linewidth=0.8, capsize=3, width=0.64,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
    )
    accuracy.set(
        title="(f) Test accuracy (mean ± SD)", xlabel="Hampel window (frames)",
        ylabel="Accuracy (%)", ylim=(0, max(100, float((means + deviations).max()) + 8)),
    )
    accuracy.set_xticks(positions, ["Off", *map(str, WINDOWS[1:])])
    accuracy.set_axisbelow(True)
    accuracy.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    for position, mean, deviation in zip(positions, means, deviations):
        accuracy.text(position, mean + deviation + 2, f"{mean:.1f}%", ha="center", fontsize=8)

    for suffix in ("png", "pdf"):
        figure.savefig(
            directory / f"elder_hampel_confusion_{model_name}.{suffix}",
            dpi=300, facecolor="white",
        )
    plt.close(figure)


def save_results(results, directory: Path) -> None:
    counts = np.asarray([
        [[confusion_counts(results[model, window, seed]) for seed in SEEDS] for window in WINDOWS]
        for model, _ in MODELS
    ])
    normalized = counts / counts.sum(axis=-1, keepdims=True)
    mean_matrices = normalized.mean(axis=2)
    accuracy = np.trace(counts, axis1=-2, axis2=-1) / counts.sum(axis=(-2, -1))
    means, deviations = accuracy.mean(axis=2), accuracy.std(axis=2, ddof=1)
    np.savez_compressed(
        directory / "confusion_matrices.npz", models=[name for name, _ in MODELS],
        window_frames=WINDOWS, seeds=SEEDS, class_ids=list(ACTION_NAMES), class_names=CLASSES,
        counts=counts, normalized_by_seed=normalized, mean_normalized=mean_matrices,
    )
    with (directory / "accuracy_by_seed.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "full_window_frames", "seed", "test_accuracy", "best_val_accuracy", "best_epoch"])
        for model_index, (model, _) in enumerate(MODELS):
            for window_index, window in enumerate(WINDOWS):
                for seed_index, seed in enumerate(SEEDS):
                    result = results[model, window, seed]
                    writer.writerow([
                        model, window, seed, accuracy[model_index, window_index, seed_index],
                        result.best_val_accuracy, result.best_epoch,
                    ])
    with (directory / "accuracy_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "full_window_frames", "seed_count", "mean_test_accuracy", "sd_test_accuracy"])
        for model_index, (model, _) in enumerate(MODELS):
            for window_index, window in enumerate(WINDOWS):
                writer.writerow([
                    model, window, len(SEEDS), means[model_index, window_index],
                    deviations[model_index, window_index],
                ])

    with plt.rc_context({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.edgecolor": EDGE, "pdf.fonttype": 42,
    }):
        for index, (model, label) in enumerate(MODELS):
            draw_model(model, label, mean_matrices[index], means[index], deviations[index], directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="ElderAL raw CSV directory")
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR, help="Directory with samples.json and split files")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Experiment results directory")
    parser.add_argument("--device", default="cuda:0", help="Training device, for example cuda:2")
    parser.add_argument("--workers", type=int, default=4, help="Worker processes for reading and preprocessing")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        parser.error(f"CUDA device unavailable: {device}")

    data_dir = args.data_dir.expanduser().resolve()
    directory = args.output_dir.expanduser().resolve()
    samples, labels, splits, split_digests = load_splits(args.split_dir.expanduser())
    params = load_params(DATASET)
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    raw = None
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for window in WINDOWS:
            steps = pipeline_for(window)
            pending = []
            for model, _ in MODELS:
                for seed in SEEDS:
                    run_dir = directory / "runs" / model / f"window_{window}" / f"seed_{seed}"
                    signature = digest({
                        "protocol": PROTOCOL_VERSION, "model": model, "seed": seed,
                        "data_dir": str(data_dir), "steps": steps,
                        "params": params, "split": split_digests[seed],
                    })
                    test = splits[seed][2]
                    saved = completed_prediction(run_dir, signature, test, labels[test])
                    if saved is None:
                        pending.append((model, seed, run_dir, signature))
                    else:
                        results[model, window, seed] = saved
            if not pending:
                continue
            if raw is None:
                print(f"Reading {len(samples)} ElderAL samples", flush=True)
                raw = list(pool.map(partial(read_sample, data_dir=data_dir), samples, chunksize=8))
            print(f"Preprocessing window={window} with {len(pending)} runs pending", flush=True)
            process = partial(preprocess, steps=steps, padding_length=params["padding_length"])
            processed = np.stack(list(pool.map(process, raw, chunksize=4)))
            dataset = CSIDataset(processed, labels, dataset_name=DATASET, pipeline_steps=steps)
            del processed
            for model, seed, run_dir, signature in pending:
                results[model, window, seed] = run_condition(
                    model, seed, dataset, splits[seed], params, device, run_dir, signature, samples, labels,
                )
            del dataset
    save_results(results, directory)
    print(f"Results written to {directory}", flush=True)


if __name__ == "__main__":
    main()
