"""Widar step 版单组实验脚本。

这个版本把关键阶段展开，方便检查每一步。核心能力仍然来自最终版源码。

步骤：
load_raw_data -> process_data -> split_data -> 构造 DataLoader ->
create_registered_model -> 训练 -> 测试

Widar 当前源码逻辑：
- 任务标签是 gesture_type。
- 分组依据是 position_id * 1000 + orientation_id * 100 + receiver_id。
- 模型输入由 dataset_name 自动选择为“幅度 + 相位”。
"""

from __future__ import annotations
import contextlib
import os
import random
import sys
import time
import types
from collections import Counter
from pathlib import Path

# ==================== 环境设置 ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from wsdp.algorithms import apply_preset
from wsdp.core import _create_data_split, _evaluate_model
from wsdp.datasets import CSIDataset
from wsdp.models import create_model
from wsdp.processors import ConfigurableProcessor
from wsdp.processors.base_processor import BaseProcessor
from wsdp.utils import load_params, resize_csi_to_fixed_length, train_model


# ==================== 数据集设置 ====================
DATASET_NAME = "widar"
DATA_PATH = DATA_ROOT / "widar_common3"


# ==================== 配置区 ====================
# 更换算法：
# 1. 使用 BaseProcessor：
#    PRESET_NAME = "baseprocessor"
#    PIPELINE_STEPS = None
#
# 2. 使用预设算法：
#    PRESET_NAME = "fast" / "robust" / ...
#    PIPELINE_STEPS = None
#
# 3. 使用自定义算法组合：
#    PIPELINE_STEPS = {"denoise": {"method": "savgol", "window_length": 7, "polyorder": 3}}

PRESET_NAME = "gesture_recognition"
PIPELINE_STEPS = None

# 更换模型
MODEL_NAME = "mlpmodel"

# 训练超参数；None 表示使用源码或模型参数文件里的默认值
BATCH_SIZE = None
LEARNING_RATE = None
WEIGHT_DECAY = None
NUM_EPOCHS = 80
PADDING_LENGTH = 1500
TEST_SPLIT = 0.3
VAL_SPLIT = 0.5
SEED = 42

# 输出位置
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "result"
    / "self_design_test"
    / f"auto_amp_phase+{PRESET_NAME}+{MODEL_NAME}"
)
# ================== 配置区结束 ==================

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


def resolve_pipeline_steps(preset_name: str, pipeline_steps_override):
    """优先使用自定义算法组合，其次使用预设，最后使用 BaseProcessor。"""
    if pipeline_steps_override is not None:
        return pipeline_steps_override
    if preset_name == "baseprocessor":
        return None
    return apply_preset(preset_name)


def save_loss_curve(
    history: dict,
    output_dir: Path,
    preset_name: str,
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
    plt.title(f"{preset_name}+{model_name} Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def load_raw_data():
    """第一步：读取 Widar 原始数据。"""
    print("step 1: load_raw_data")
    data = readers.load_data(str(DATA_PATH), DATASET_NAME)
    print(f"加载样本数: {len(data)}")
    # 加载完成后暂停 2 秒
    time.sleep(2)
    return data


def process_data(csi_data_list, pipeline_steps, padding_length: int):
    """第二步：按源码 Processor 处理数据，并映射标签和分组。"""
    print("step 2: process_data")
    processor = (
        BaseProcessor()
        if pipeline_steps is None
        else ConfigurableProcessor(pipeline_steps)
    )
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
    """第三步：调用源码划分函数；Widar 使用 group split。"""
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
        make_loader(val_data, val_labels, shuffle=True),
        make_loader(test_data, test_labels, shuffle=True),
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
    preset_name: str,
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

    # 只保存 loss 曲线，不保存 training_history.csv
    save_loss_curve(history, output_dir, preset_name, model_name)

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


def run_experiment(
    preset_name: str,
    model_name: str,
    output_dir: Path,
    pipeline_steps_override=None,
):
    """给 full_test 复用的单组合入口。"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 Widar 数据目录: {DATA_PATH}")

    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)

    pipeline_steps = resolve_pipeline_steps(preset_name, pipeline_steps_override)
    params = load_params(DATASET_NAME)

    # step 1 不写入 train_process.txt，只在终端显示
    csi_data_list = load_raw_data()

    # 从 step 2 开始，既输出到终端，也写入 train_process.txt
    with (output_dir / "train_process.txt").open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
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

            model, device = create_registered_model(
                model_name,
                len(unique_labels),
                tuple(loaders[0].dataset.data_list.shape[1:]),
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
                preset_name,
                model_name,
            )

            val_acc, test_acc = evaluate_checkpoint(
                model,
                device,
                loaders[2],
                checkpoint_path,
            )

            print("\n" + "=" * 60)
            print("pipeline 演示完成")
            print("=" * 60)

    return {
        "pipeline": preset_name,
        "model": model_name,
        "best_val_acc": val_acc,
        "test_acc": test_acc,
        "output_dir": str(output_dir),
    }


def main() -> None:
    """运行单组 step 实验。"""
    run_experiment(
        PRESET_NAME,
        MODEL_NAME,
        OUTPUT_DIR,
        PIPELINE_STEPS,
    )


if __name__ == "__main__":
    main()
