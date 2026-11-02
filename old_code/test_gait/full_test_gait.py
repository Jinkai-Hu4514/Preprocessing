"""Gait 320 组自动 pipeline 全量实验脚本。

固定模型：resnet2d

组合空间：
denoise(5) -> outliers(2) -> calibrate(4) -> normalize(2) -> interpolate(4)
= 320 组 pipeline。

这个文件不依赖 pipline_gait_steps.py，而是直接调用 WSDP 源码模块。

Gait 当前实验约定：
- 标签是 user_id；
- 分组依据是 track_id * 100 + receiver_id；
- 模型输入由 dataset_name 自动选择为“幅度 + 相位”。
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import random
import sys
import time
import traceback
import types
from collections import Counter
from itertools import product
from pathlib import Path

# ==================== 环境设置 ====================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wsdp_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


# ==================== 路径设置 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WSDP_SRC = PROJECT_ROOT / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main" / "src"
DATA_ROOT = PROJECT_ROOT.parent / "sdp_dataset"

if not (WSDP_SRC / "wsdp").is_dir():
    raise FileNotFoundError(f"找不到本地 WSDP 源码目录: {WSDP_SRC}")

sys.path.insert(0, str(WSDP_SRC))

# 避免源码里 import kagglehub 时出错
sys.modules.setdefault("kagglehub", types.ModuleType("kagglehub"))


# ==================== WSDP 源码模块 ====================
from wsdp import readers
from wsdp.core import _create_data_split, _evaluate_model
from wsdp.datasets import CSIDataset
from wsdp.models import create_model
from wsdp.processors import ConfigurableProcessor
from wsdp.utils import load_params, resize_csi_to_fixed_length, train_model


# ==================== 数据集与实验设置 ====================
DATASET_NAME = "gait"
DATA_PATH = DATA_ROOT / "Gait_Dataset" / "CSI_Gait"

RUN_NAME = "gait_320_pipeline"
MODEL_NAME = "resnet2d"

BATCH_SIZE = None
LEARNING_RATE = None
WEIGHT_DECAY = None
NUM_EPOCHS = 60
PADDING_LENGTH = 1500
TEST_SPLIT = 0.3
VAL_SPLIT = 0.5
SEED = 42

RESULT_DIR = Path(__file__).resolve().parent / "result" / "full_tests"
SUMMARY_PATH = RESULT_DIR / f"{RUN_NAME}_{MODEL_NAME}_summary.csv"

SUMMARY_FIELDS = [
    "combo_index",
    "combo_id",
    "combo_name",
    "model",
    "status",
    "best_val_acc",
    "test_acc",
    "denoise",
    "outliers",
    "calibrate",
    "normalize",
    "interpolate",
    "pipeline_steps",
    "output_dir",
    "duration_sec",
    "error",
]


# ==================== 320 组算法组合配置 ====================
DENOISE_OPTIONS = [
    ("wavelet", {"method": "wavelet"}),
    ("butterworth_o5_c0.3", {"method": "butterworth", "order": 5, "cutoff": 0.3}),
    ("savgol_w7_p3", {"method": "savgol", "window_length": 7, "polyorder": 3}),
    (
        "bandpass_0.5-50",
        {
            "method": "bandpass",
            "order": 4,
            "low_freq": 0.5,
            "high_freq": 50.0,
            "fs": 1000.0,
        },
    ),
    ("hampel_w5_s3", {"method": "hampel", "window_size": 5, "n_sigma": 3.0}),
]

OUTLIER_OPTIONS = [
    ("iqr", {"method": "iqr", "factor": 1.5}),
    ("outlier_z-score", {"method": "z-score", "factor": 3.0}),
]

CALIBRATE_OPTIONS = [
    ("linear", {"method": "linear"}),
    ("polynomial_d3", {"method": "polynomial", "degree": 3}),
    ("stc", {"method": "stc"}),
    ("robust", {"method": "robust"}),
]

NORMALIZE_OPTIONS = [
    ("z-score", {"method": "z-score"}),
    ("min-max", {"method": "min-max"}),
]

# Gait 原始子载波维通常是 30。
# target_K=15 可以让 linear/cubic/nearest/decimate 四种方法都保持合法且输出维度一致。
INTERPOLATE_OPTIONS = [
    ("linear15", {"method": "linear", "target_K": 15}),
    ("cubic15", {"method": "cubic", "target_K": 15}),
    ("nearest15", {"method": "nearest", "target_K": 15}),
    ("decimate15", {"method": "decimate", "target_K": 15}),
]


class Tee:
    """同时把 stdout 写到终端和 train_process.txt。"""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)

    def flush(self):
        for file in self.files:
            file.flush()


def set_seed(seed: int) -> None:
    """固定随机性，方便复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pipeline_combinations() -> list[dict]:
    """自动生成 Gait 的 320 组 pipeline 组合。"""
    combinations = []

    for combo_index, (denoise, outliers, calibrate, normalize, interpolate) in enumerate(
        product(
            DENOISE_OPTIONS,
            OUTLIER_OPTIONS,
            CALIBRATE_OPTIONS,
            NORMALIZE_OPTIONS,
            INTERPOLATE_OPTIONS,
        ),
        start=1,
    ):
        denoise_name, denoise_cfg = denoise
        outliers_name, outliers_cfg = outliers
        calibrate_name, calibrate_cfg = calibrate
        normalize_name, normalize_cfg = normalize
        interpolate_name, interpolate_cfg = interpolate

        pipeline_steps = {
            "denoise": denoise_cfg.copy(),
            "outliers": outliers_cfg.copy(),
            "calibrate": calibrate_cfg.copy(),
            "normalize": normalize_cfg.copy(),
            "interpolate": interpolate_cfg.copy(),
        }

        combo_id = f"gait_{combo_index:03d}"
        combo_name = "+".join(
            [
                denoise_name,
                outliers_name,
                calibrate_name,
                normalize_name,
                interpolate_name,
            ]
        )

        combinations.append(
            {
                "combo_index": combo_index,
                "combo_id": combo_id,
                "combo_name": combo_name,
                "denoise": denoise_name,
                "outliers": outliers_name,
                "calibrate": calibrate_name,
                "normalize": normalize_name,
                "interpolate": interpolate_name,
                "pipeline_steps": pipeline_steps,
            }
        )

    return combinations


def save_loss_curve(
    history: dict,
    output_dir: Path,
    combo_id: str,
    model_name: str,
) -> None:
    """保存训练/验证 loss 曲线。"""
    if not history.get("train_loss") or not history.get("val_loss"):
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{combo_id}+{model_name} Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def load_raw_data():
    """第一步：读取 Gait 原始数据。"""
    print("step 1: load_raw_data")

    data = readers.load_data(str(DATA_PATH), DATASET_NAME)

    print(f"加载样本数: {len(data)}")

    return data


def process_data(csi_data_list, pipeline_steps, padding_length: int):
    """第二步：按源码 ConfigurableProcessor 处理数据，并映射标签和分组。"""
    print("step 2: process_data")

    processor = ConfigurableProcessor(pipeline_steps)

    all_data, all_labels, all_groups = processor.process(
        csi_data_list,
        dataset=DATASET_NAME,
    )

    processed_data = resize_csi_to_fixed_length(
        all_data,
        target_length=padding_length,
    )

    unique_labels = sorted(set(all_labels))
    unique_groups = sorted(set(all_groups))

    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    group_map = {group: idx for idx, group in enumerate(unique_groups)}

    processed_data = np.asarray(processed_data)
    labels = np.asarray([label_map[label] for label in all_labels])
    groups = np.asarray([group_map[group] for group in all_groups])

    summary_lines = [
        f"   处理完成: {len(processed_data)} 个样本",
        f"   标签分布: {dict(Counter(all_labels))}",
        f"   分组分布: {dict(Counter(all_groups))}",
        f"   样本形状: {processed_data[0].shape}",
        f"   类别数: {len(unique_labels)} (原始标签: {unique_labels})",
    ]

    for line in summary_lines:
        print(line)

    return processed_data, labels, groups, unique_labels


def split_data(
    processed_data,
    labels,
    groups,
    pipeline_steps,
    test_split: float,
    val_split: float,
    seed: int,
):
    """第三步：调用源码划分函数；Gait 使用 group split。"""
    print("step 3: split_data")

    split = _create_data_split(
        processed_data,
        labels,
        groups,
        test_split=test_split,
        val_split=val_split,
        seed=seed,
        use_simple_split=len(set(groups.tolist())) < 3,
        dataset=DATASET_NAME,
        pipeline_steps=pipeline_steps,
    )

    print(f"训练集 {len(split[0])} | 验证集 {len(split[1])} | 测试集 {len(split[2])}")

    return split


def build_loaders(split, pipeline_steps, batch_size: int):
    """第四步：构造训练、验证、测试 DataLoader。"""
    print("step 4: 构造 DataLoader")

    train_data, val_data, test_data, train_labels, val_labels, test_labels = split

    def make_loader(data, labels, shuffle: bool):
        return DataLoader(
            CSIDataset(
                data,
                labels,
                dataset_name=DATASET_NAME,
                pipeline_steps=pipeline_steps,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    return (
        make_loader(train_data, train_labels, shuffle=True),
        make_loader(val_data, val_labels, shuffle=False),
        make_loader(test_data, test_labels, shuffle=False),
    )


def create_registered_model(model_name: str, num_classes: int, input_shape):
    """第五步：创建源码注册模型。"""
    print("step 5: create_registered_model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(
        model_name,
        num_classes=num_classes,
        input_shape=input_shape,
    ).to(device)

    print(f"模型参数量 {sum(p.numel() for p in model.parameters())}")
    print(f"训练设备: {device}")

    return model, device


def train_registered_model(
    model,
    device,
    loaders,
    params: dict,
    output_dir: Path,
    learning_rate,
    weight_decay,
    num_epochs: int,
    padding_length: int,
    combo_id: str,
    model_name: str,
):
    """第六步：训练模型。"""
    print(f"step 6: 开始训练 {num_epochs} 轮")

    train_loader, val_loader, _ = loaders

    criterion = nn.CrossEntropyLoss()

    lr = learning_rate if learning_rate is not None else params.get("lr", 3e-4)
    wd = weight_decay if weight_decay is not None else params.get("wd", 1e-3)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=5,
    )

    checkpoint_path = output_dir / "best_checkpoint.pth"

    history = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        num_epochs,
        device,
        checkpoint_path,
        padding_length,
    )

    save_loss_curve(history, output_dir, combo_id, model_name)

    print(f"训练完成，最佳模型保存至: {checkpoint_path}")

    return checkpoint_path


def evaluate_checkpoint(model, device, test_loader, checkpoint_path: Path):
    """第七步：测试集评估。"""
    print("step 7: 测试集评估")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, test_acc = _evaluate_model(
        model,
        test_loader,
        device,
    )

    val_acc = checkpoint.get("best_val_acc", 0.0) / 100.0

    print(f"最佳验证准确率: {val_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    return val_acc, test_acc


def load_done_records() -> set[str]:
    """读取 summary 中已有 combo_id，用于断点续跑。"""
    if not SUMMARY_PATH.exists():
        return set()

    with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        return {
            row["combo_id"]
            for row in csv.DictReader(f)
            if row.get("combo_id") and row.get("model") == MODEL_NAME
        }


def append_summary(row: dict) -> None:
    """追加一条组合结果到 summary。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_PATH.exists()

    with SUMMARY_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def output_dir_for(combo: dict) -> Path:
    """单个组合的输出目录。"""
    return RESULT_DIR / f"{combo['combo_id']}+{combo['combo_name']}+{MODEL_NAME}"


def pipeline_steps_to_json(pipeline_steps: dict) -> str:
    return json.dumps(pipeline_steps, ensure_ascii=False, sort_keys=True)


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one_combo(
    combo: dict,
    combo_total: int,
    csi_data_list,
    params: dict,
) -> dict:
    """按 step 流程处理、训练并测试一个 pipeline 组合。"""
    output_dir = output_dir_for(combo)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_steps = combo["pipeline_steps"]
    start_time = time.time()

    with (output_dir / "train_process.txt").open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
            try:
                print("\n" + "=" * 80)
                print(
                    f"当前算法组合: {combo['combo_index']}/{combo_total}"
                    f" | combo_id={combo['combo_id']}"
                    f" | model={MODEL_NAME}"
                )
                print(f"combo_name: {combo['combo_name']}")
                print(f"保存目录: {output_dir}")
                print("当前组合的具体算法:")
                print(json.dumps(pipeline_steps, ensure_ascii=False, indent=2))
                print("=" * 80)

                set_seed(SEED)

                processed_data, labels, groups, unique_labels = process_data(
                    csi_data_list,
                    pipeline_steps,
                    PADDING_LENGTH,
                )

                split = split_data(
                    processed_data,
                    labels,
                    groups,
                    pipeline_steps,
                    TEST_SPLIT,
                    VAL_SPLIT,
                    SEED,
                )

                batch_size = BATCH_SIZE if BATCH_SIZE is not None else params.get("batch", 32)

                loaders = build_loaders(
                    split,
                    pipeline_steps,
                    batch_size,
                )

                input_shape = tuple(loaders[0].dataset.data_list.shape[1:])
                print(f"模型实际输入形状: {input_shape}")

                model, device = create_registered_model(
                    MODEL_NAME,
                    len(unique_labels),
                    input_shape,
                )

                checkpoint_path = train_registered_model(
                    model,
                    device,
                    loaders,
                    params,
                    output_dir,
                    LEARNING_RATE,
                    WEIGHT_DECAY,
                    NUM_EPOCHS,
                    PADDING_LENGTH,
                    combo["combo_id"],
                    MODEL_NAME,
                )

                val_acc, test_acc = evaluate_checkpoint(
                    model,
                    device,
                    loaders[2],
                    checkpoint_path,
                )

                duration_sec = time.time() - start_time

                print("\n" + "=" * 80)
                print(
                    f"组合完成: {combo['combo_index']}/{combo_total}"
                    f" | combo_id={combo['combo_id']}"
                    f" | best_val_acc={val_acc:.4f}"
                    f" | test_acc={test_acc:.4f}"
                    f" | duration={duration_sec:.2f}s"
                )
                print("=" * 80)

                return {
                    "combo_index": combo["combo_index"],
                    "combo_id": combo["combo_id"],
                    "combo_name": combo["combo_name"],
                    "model": MODEL_NAME,
                    "status": "ok",
                    "best_val_acc": val_acc,
                    "test_acc": test_acc,
                    "denoise": combo["denoise"],
                    "outliers": combo["outliers"],
                    "calibrate": combo["calibrate"],
                    "normalize": combo["normalize"],
                    "interpolate": combo["interpolate"],
                    "pipeline_steps": pipeline_steps_to_json(pipeline_steps),
                    "output_dir": str(output_dir),
                    "duration_sec": f"{duration_sec:.2f}",
                    "error": "",
                }
            except Exception:
                duration_sec = time.time() - start_time
                print("\n组合失败，错误如下：")
                traceback.print_exc()

                return {
                    "combo_index": combo["combo_index"],
                    "combo_id": combo["combo_id"],
                    "combo_name": combo["combo_name"],
                    "model": MODEL_NAME,
                    "status": "failed",
                    "best_val_acc": "",
                    "test_acc": "",
                    "denoise": combo["denoise"],
                    "outliers": combo["outliers"],
                    "calibrate": combo["calibrate"],
                    "normalize": combo["normalize"],
                    "interpolate": combo["interpolate"],
                    "pipeline_steps": pipeline_steps_to_json(pipeline_steps),
                    "output_dir": str(output_dir),
                    "duration_sec": f"{duration_sec:.2f}",
                    "error": traceback.format_exc().splitlines()[-1],
                }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 Gait 数据目录: {DATA_PATH}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    combinations = build_pipeline_combinations()
    combo_total = len(combinations)
    done_records = load_done_records()
    params = load_params(DATASET_NAME)

    print(f"Gait 全量 pipeline 组合数: {combo_total}")
    print(f"固定模型: {MODEL_NAME}")
    print(f"结果目录: {RESULT_DIR}")
    print(f"summary: {SUMMARY_PATH}")
    print(f"已完成组合数: {len(done_records)}")

    # step 1: 原始数据只读取一次，后续 320 个 pipeline 复用这份 csi_data_list。
    set_seed(SEED)
    csi_data_list = load_raw_data()

    for combo in combinations:
        if combo["combo_id"] in done_records:
            print(
                f"\n跳过已有记录: {combo['combo_index']}/{combo_total}"
                f" | combo_id={combo['combo_id']}"
                f" | combo_name={combo['combo_name']}"
            )
            continue

        print(
            f"\n开始组合: {combo['combo_index']}/{combo_total}"
            f" | combo_id={combo['combo_id']}"
            f" | combo_name={combo['combo_name']}"
        )

        try:
            row = run_one_combo(combo, combo_total, csi_data_list, params)
        finally:
            clear_cuda_cache()

        append_summary(row)
        done_records.add(combo["combo_id"])

    print(f"\n全部完成，汇总已保存到: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
