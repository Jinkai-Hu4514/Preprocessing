"""XRF55 raw-minimal 基线测试，仅使用前三个用户。

本实验不调用 BaseProcessor，也不执行 80 组流程中的去噪、异常值处理、
归一化或子载波插值。唯一保留的处理是模型输入不可缺少的帧排序、张量堆叠、
无效样本过滤、固定长度截断/零填充和幅度输入转换。

数据划分与 xrf55 80 组实验一致：repetition 1-12 训练、13-16 验证、
17-20 测试；只读取 user id 为 1、2、3 的文件。
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import random
import re
import sys
import time
import traceback
import types
from collections import Counter
from pathlib import Path

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WSDP_SRC = PROJECT_ROOT / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main" / "src"
DATA_ROOT = PROJECT_ROOT.parent / "sdp_dataset"

if not (WSDP_SRC / "wsdp").is_dir():
    raise FileNotFoundError(f"找不到本地 WSDP 源码目录: {WSDP_SRC}")

sys.path.insert(0, str(WSDP_SRC))
sys.modules.setdefault("kagglehub", types.ModuleType("kagglehub"))

from wsdp import readers
from wsdp.core import _create_data_split, _evaluate_model
from wsdp.datasets import CSIDataset
from wsdp.models import create_model
from wsdp.processors import ConfigurableProcessor
from wsdp.utils import load_params, resize_csi_to_fixed_length, train_model


DATASET_NAME = "xrf55"
DATA_PATH = DATA_ROOT / "xrf55"
XRF55_USER_LIMIT = 3
RUN_NAME = "xrf55_raw_minimal_first3"
MODEL_NAME = "resnet1d"

# 与 full_test_xrf55.py 保持一致，保证和既有 80 组结果同口径。
BATCH_SIZE = None
LEARNING_RATE = None
WEIGHT_DECAY = None
NUM_EPOCHS = 50
PADDING_LENGTH = 1000
TEST_SPLIT = 0.3
VAL_SPLIT = 0.5
SEED = 654

# 空流程不会触发 execute_pipeline 中的任何算法步骤。
RAW_MINIMAL_STEPS: dict = {}

RESULT_DIR = Path(__file__).resolve().parent / "result" / "raw_minimal_tests"
SUMMARY_PATH = RESULT_DIR / f"{RUN_NAME}_{MODEL_NAME}_summary.csv"
OUTPUT_DIR = RESULT_DIR / f"{RUN_NAME}+{MODEL_NAME}+seed{SEED}"
SUMMARY_FIELDS = [
    "run_name",
    "dataset",
    "user_scope",
    "model",
    "seed",
    "status",
    "best_val_acc",
    "test_acc",
    "pipeline_steps",
    "output_dir",
    "duration_sec",
    "error",
]


class Tee:
    """同时把 stdout 写到终端和训练日志。"""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)

    def flush(self):
        for file in self.files:
            file.flush()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_first_three_user_file(file_path: Path) -> bool:
    match = re.search(r"(\d+)_(\d+)_(\d+)", file_path.stem)
    return bool(match and int(match.group(1)) <= XRF55_USER_LIMIT)


def load_raw_data():
    """只读取前三个用户的 XRF55 原始 CSI 文件。"""
    print("step 1: load_raw_data (XRF55 前三个用户)")
    reader = readers.get_reader_class(DATASET_NAME)()
    files = [
        path
        for path in sorted(DATA_PATH.rglob("*"))
        if path.is_file()
        and "truth" not in path.name
        and is_first_three_user_file(path)
    ]
    if not files:
        raise IOError(f"没有找到前三个用户的 XRF55 文件: {DATA_PATH}")

    csi_data_list = []
    skipped = 0
    for file_path in files:
        if not reader.sniff(str(file_path)):
            skipped += 1
            continue
        data = reader.read_file(str(file_path))
        csi_data_list.extend(data if isinstance(data, list) else [data])

    print(f"匹配文件数: {len(files)}")
    print(f"跳过格式不匹配文件数: {skipped}")
    print(f"加载样本数: {len(csi_data_list)}")
    return csi_data_list


def process_data(csi_data_list, padding_length: int):
    """执行最小输入适配，不调用 BaseProcessor 或任何信号处理步骤。"""
    print("step 2: raw-minimal process_data")
    processor = ConfigurableProcessor(RAW_MINIMAL_STEPS)
    all_data, all_labels, all_groups = processor.process(
        csi_data_list,
        dataset=DATASET_NAME,
    )

    processed_data = np.asarray(
        resize_csi_to_fixed_length(all_data, target_length=padding_length)
    )
    unique_labels = sorted(set(all_labels))
    unique_groups = sorted(set(all_groups))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    group_map = {group: idx for idx, group in enumerate(unique_groups)}
    labels = np.asarray([label_map[label] for label in all_labels])
    groups = np.asarray([group_map[group] for group in all_groups])

    print(f"处理完成: {len(processed_data)} 个样本")
    print(f"标签分布: {dict(Counter(all_labels))}")
    print(f"分组分布: {dict(Counter(all_groups))}")
    print(f"样本形状: {processed_data[0].shape}")
    print(f"类别数: {len(unique_labels)} (原始标签: {unique_labels})")
    return processed_data, labels, groups, unique_labels


def split_data(processed_data, labels, groups):
    print("step 3: fixed repetition split")
    split = _create_data_split(
        processed_data,
        labels,
        groups,
        test_split=TEST_SPLIT,
        val_split=VAL_SPLIT,
        seed=SEED,
        use_simple_split=len(set(groups.tolist())) < 3,
        dataset=DATASET_NAME,
        pipeline_steps=RAW_MINIMAL_STEPS,
    )
    print(f"训练集 {len(split[0])} | 验证集 {len(split[1])} | 测试集 {len(split[2])}")
    return split


def build_loaders(split, batch_size: int):
    print("step 4: build DataLoader")
    train_data, val_data, test_data, train_labels, val_labels, test_labels = split

    def make_loader(data, labels, shuffle: bool):
        return DataLoader(
            CSIDataset(
                data,
                labels,
                dataset_name=DATASET_NAME,
                pipeline_steps=RAW_MINIMAL_STEPS,
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


def save_loss_curve(history: dict, output_dir: Path) -> None:
    if not history.get("train_loss") or not history.get("val_loss"):
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{RUN_NAME}+{MODEL_NAME} Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def train_and_evaluate(loaders, num_classes: int, input_shape, params: dict, output_dir: Path):
    print("step 5: create_registered_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(MODEL_NAME, num_classes=num_classes, input_shape=input_shape).to(device)
    print(f"模型参数量 {sum(p.numel() for p in model.parameters())}")
    print(f"训练设备: {device}")

    print(f"step 6: 开始训练 {NUM_EPOCHS} 轮")
    lr = LEARNING_RATE if LEARNING_RATE is not None else params.get("lr", 3e-4)
    wd = WEIGHT_DECAY if WEIGHT_DECAY is not None else params.get("wd", 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    checkpoint_path = output_dir / "best_checkpoint.pth"
    history = train_model(
        model,
        nn.CrossEntropyLoss(),
        optimizer,
        scheduler,
        loaders[0],
        loaders[1],
        NUM_EPOCHS,
        device,
        checkpoint_path,
        PADDING_LENGTH,
    )
    save_loss_curve(history, output_dir)

    print("step 7: 测试集评估")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, test_acc = _evaluate_model(model, loaders[2], device)
    val_acc = checkpoint.get("best_val_acc", 0.0) / 100.0
    print(f"最佳验证准确率: {val_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    return val_acc, test_acc


def write_summary(row: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 XRF55 数据目录: {DATA_PATH}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    set_seed(SEED)
    csi_data_list = load_raw_data()

    with (OUTPUT_DIR / "train_process.txt").open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
            try:
                print(json.dumps({"pipeline_steps": RAW_MINIMAL_STEPS}, ensure_ascii=False))
                processed_data, labels, groups, unique_labels = process_data(
                    csi_data_list,
                    PADDING_LENGTH,
                )
                split = split_data(processed_data, labels, groups)
                params = load_params(DATASET_NAME)
                batch_size = BATCH_SIZE if BATCH_SIZE is not None else params.get("batch", 32)
                loaders = build_loaders(split, batch_size)
                input_shape = tuple(loaders[0].dataset.data_list.shape[1:])
                print(f"模型实际输入形状: {input_shape}")
                val_acc, test_acc = train_and_evaluate(
                    loaders,
                    len(unique_labels),
                    input_shape,
                    params,
                    OUTPUT_DIR,
                )
                row = {
                    "run_name": RUN_NAME,
                    "dataset": DATASET_NAME,
                    "user_scope": "users_1_to_3",
                    "model": MODEL_NAME,
                    "seed": SEED,
                    "status": "ok",
                    "best_val_acc": val_acc,
                    "test_acc": test_acc,
                    "pipeline_steps": json.dumps(RAW_MINIMAL_STEPS),
                    "output_dir": str(OUTPUT_DIR),
                    "duration_sec": f"{time.time() - start_time:.2f}",
                    "error": "",
                }
            except Exception:
                traceback.print_exc()
                row = {
                    "run_name": RUN_NAME,
                    "dataset": DATASET_NAME,
                    "user_scope": "users_1_to_3",
                    "model": MODEL_NAME,
                    "seed": SEED,
                    "status": "failed",
                    "best_val_acc": "",
                    "test_acc": "",
                    "pipeline_steps": json.dumps(RAW_MINIMAL_STEPS),
                    "output_dir": str(OUTPUT_DIR),
                    "duration_sec": f"{time.time() - start_time:.2f}",
                    "error": traceback.format_exc().splitlines()[-1],
                }

    write_summary(row)
    print(f"结果已保存到: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
