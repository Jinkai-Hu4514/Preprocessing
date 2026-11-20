"""Load the saved Elder splits and run the two models for each seed."""

import argparse
import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


# 两个实验共用这里的显卡编号。用 2 号卡就改成 2，使用 CPU 就改成 None。
CUDA_DEVICE_INDEX: int | None = 2

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
INPUT_SHAPE = (80, 512, 3)
SEEDS = (42, 727, 914)
MODELS = ("CSITime", "ResNet2D")


def load_samples():
    samples = json.loads((INPUTS / "samples.json").read_text(encoding="utf-8"))["samples"]
    timing = {}
    with (INPUTS / "timing_per_sample.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            _, separator, relative_path = row["file_name"].replace("\\", "/").partition("/elderAL/")
            if not separator:
                raise ValueError(f"无法识别时间表里的 Elder 路径: {row['file_name']}")
            # Each file in this archive contains one sample, with index 0.
            sample_id = f"{relative_path}#0"
            if sample_id in timing:
                raise ValueError(f"时间表里有重复样本: {sample_id}")
            timing[sample_id] = row

    frame_counts = []
    for sample in samples:
        sample_id = sample["id"]
        row = timing.get(sample_id)
        if row is None:
            raise ValueError(f"时间表缺少样本: {sample_id}")
        if int(row["action_id"]) != sample["label"] or int(row["position_id"]) != sample["group"]:
            raise ValueError(f"时间表的动作或位置与样本清单不一致: {sample_id}")
        frame_count = int(row["frames"])
        if frame_count < 2:
            raise ValueError(f"有效帧数少于 2: {sample_id}")
        frame_counts.append(frame_count)
    return samples, frame_counts


def load_seed(seed, sample_count):
    split = json.loads((INPUTS / f"split{seed}.json").read_text(encoding="utf-8"))
    indices = [split[name] for name in ("train", "validation", "test")]
    all_indices = [index for part in indices for index in part]
    if split["seed"] != seed or not all(indices) or sorted(all_indices) != list(range(sample_count)):
        raise ValueError(f"seed {seed} 的划分与样本清单不一致")

    model_keys = {name.lower() for name in MODELS}
    baseline = {}
    with (INPUTS / f"raw_seed{seed}.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            model = row["model"].lower()
            if model in model_keys and row["status"] == "ok" and int(row["seed"]) == seed:
                if model in baseline:
                    raise ValueError(f"seed {seed} 有重复的 {model} 原成绩")
                baseline[model] = float(row["test_acc"])
    if baseline.keys() != model_keys:
        raise ValueError(f"seed {seed} 缺少需要的原始补零成绩")
    return indices, baseline


def run_experiment(make_inputs, filename, *, include_difference=False):
    parser = argparse.ArgumentParser(description="在 elder_common.py 的 CUDA_DEVICE_INDEX 设置显卡编号。")
    parser.parse_args()

    # Direct script launches put ablation_time_frame, rather than the workspace, on sys.path.
    sys.path.insert(0, str(ROOT.parent))
    from test_elder.raw_test import raw_test_elder as elder
    import torch
    from torch.utils.data import DataLoader, Subset, TensorDataset

    if CUDA_DEVICE_INDEX is None:
        device = torch.device("cpu")
    else:
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if not 0 <= CUDA_DEVICE_INDEX < cuda_count:
            parser.error(
                f"cuda:{CUDA_DEVICE_INDEX} 不可用，当前进程可见的 CUDA 显卡有 {cuda_count} 张。"
                "请修改 elder_common.py 里的 CUDA_DEVICE_INDEX。"
            )
        device = torch.device("cuda", CUDA_DEVICE_INDEX)
        torch.cuda.set_device(device)

    params = elder.load_params(elder.DATASET_NAME)
    if params["padding_length"] != INPUT_SHAPE[0]:
        raise ValueError("Elder 的训练配置与本实验的 80 帧输入不一致")
    print(f"训练设备 {device}，Elder 默认配置 {json.dumps(params, sort_keys=True)}", flush=True)

    samples, frame_counts = load_samples()
    seed_inputs = {seed: load_seed(seed, len(samples)) for seed in SEEDS}
    classes = sorted({sample["label"] for sample in samples})
    labels = torch.tensor([classes.index(sample["label"]) for sample in samples])
    dataset = TensorDataset(make_inputs(elder, samples, frame_counts), labels)

    output = ROOT / "results" / filename
    output.parent.mkdir(exist_ok=True)
    fields = ["model", "seed", "raw_accuracy", "test_accuracy"]
    if include_difference:
        fields.append("accuracy_delta_pp")

    # Reuse the raw trainer's checkpoints and plots for this run only.
    with TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory)
        scores = temporary / filename
        with scores.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for seed, (indices, baseline) in seed_inputs.items():
                for model_name in MODELS:
                    print(f"{output.stem} | {model_name} | seed={seed} | {device}", flush=True)
                    elder.set_seed(seed)
                    loaders = tuple(
                        DataLoader(
                            Subset(dataset, part),
                            batch_size=params["batch"],
                            shuffle=split_index == 0,
                            generator=torch.Generator().manual_seed(seed),
                        )
                        for split_index, part in enumerate(indices)
                    )
                    model = elder.create_model(
                        model_name, num_classes=len(classes), input_shape=INPUT_SHAPE
                    ).to(device)
                    checkpoint, _ = elder.train_registered_model(model, device, loaders, params, temporary)
                    _, accuracy, _ = elder.evaluate_checkpoint(model, device, loaders[2], checkpoint, len(classes))
                    raw_accuracy = baseline[model_name.lower()]
                    row = {
                        "model": model_name,
                        "seed": seed,
                        "raw_accuracy": raw_accuracy,
                        "test_accuracy": accuracy,
                    }
                    if include_difference:
                        row["accuracy_delta_pp"] = round(100 * (accuracy - raw_accuracy), 6)
                    writer.writerow(row)
                    del model
        scores.replace(output)
    print(f"结果已保存到 {output}")
