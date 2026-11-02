"""在当前 Gait 测试集上对比旧、新 MLP checkpoint。

本脚本只读取已有数据和 checkpoint，不训练模型，也不修改现有实验文件。
"""

from __future__ import annotations

import contextlib
import gc
import os
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import pipline_gait_steps as gait
from wsdp.datasets import CSIDataset
from wsdp.models import create_model


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

OLD_CHECKPOINT = (
    PROJECT_ROOT
    / "test1_gait"
    / "result"
    / "user_id_v2+fast+mlpmodel"
    / "best_checkpoint.pth"
)
CURRENT_CHECKPOINT = (
    HERE
    / "result"
    / "self_design_test"
    / "fast+mlpmodel"
    / "best_checkpoint.pth"
)

PRESET_NAME = "fast"
MODEL_NAME = "mlpmodel"
BATCH_SIZE = 32


def evaluate_checkpoint(model, loader, device, checkpoint_path: Path) -> float:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, accuracy = gait._evaluate_model(model, loader, device)
    return float(accuracy)


def main() -> None:
    for checkpoint_path in (OLD_CHECKPOINT, CURRENT_CHECKPOINT):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    gait.set_seed(gait.SEED)
    pipeline_steps = gait.resolve_pipeline_steps(PRESET_NAME, None)

    print("1/4 使用当前源码读取 Gait 数据……", flush=True)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            csi_data_list = gait.load_raw_data()
    print(f"    有效读取对象数: {len(csi_data_list)}", flush=True)

    print("2/4 使用当前 fast pipeline 处理数据……", flush=True)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            processed_data, labels, groups, unique_labels = gait.process_data(
                csi_data_list,
                pipeline_steps,
                gait.PADDING_LENGTH,
            )
    del csi_data_list
    gc.collect()
    print(f"    处理后样本数: {len(processed_data)}", flush=True)

    print("3/4 使用当前 group split 生成测试集……", flush=True)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            split = gait.split_data(
                processed_data,
                labels,
                groups,
                pipeline_steps,
                gait.TEST_SPLIT,
                gait.VAL_SPLIT,
                gait.SEED,
            )

    (
        train_data,
        val_data,
        test_data,
        train_labels,
        val_labels,
        test_labels,
    ) = split
    print(
        f"    train/val/test: "
        f"{len(train_data)}/{len(val_data)}/{len(test_data)}",
        flush=True,
    )
    print(f"    测试集标签分布: {dict(Counter(test_labels.tolist()))}", flush=True)

    del processed_data, labels, groups
    del train_data, val_data, train_labels, val_labels, split
    gc.collect()

    # 旧、新 checkpoint 都来自历史 amplitude-only 输入；此诊断脚本显式
    # 使用空数据集策略以保持旧 checkpoint 的输入形状。
    test_dataset = CSIDataset(
        test_data,
        test_labels,
        dataset_name="",
        pipeline_steps=pipeline_steps,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    input_shape = tuple(test_dataset.data_list.shape[1:])
    del test_data, test_labels
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        MODEL_NAME,
        num_classes=len(unique_labels),
        input_shape=input_shape,
    ).to(device)

    print(f"4/4 在同一当前测试集上评估，设备: {device}", flush=True)
    old_accuracy = evaluate_checkpoint(
        model,
        test_loader,
        device,
        OLD_CHECKPOINT,
    )
    current_accuracy = evaluate_checkpoint(
        model,
        test_loader,
        device,
        CURRENT_CHECKPOINT,
    )

    print("\n对比结果")
    print(f"旧 checkpoint: {old_accuracy:.4f}  ({OLD_CHECKPOINT})")
    print(f"新 checkpoint: {current_accuracy:.4f}  ({CURRENT_CHECKPOINT})")

    if abs(current_accuracy - 0.8302) > 0.005:
        print(
            "警告: 新 checkpoint 未复现日志中的 0.8302，"
            "说明当前生成的测试输入与训练时已经不完全一致。"
        )
    elif old_accuracy >= 0.90:
        print(
            "结论: 旧 checkpoint 在当前测试输入上仍保持高准确率；"
            "主要差距来自训练过程，而不是当前测试集预处理。"
        )
    else:
        print(
            "结论: 旧 checkpoint 在当前测试输入上也明显下降；"
            "旧、新实验的输入语义并不一致。旧版 registry 没有把 "
            "fast 的 method='min-max' 传给 normalize_amplitude，"
            "实际会落到默认 z-score；当前源码已修复为真正的 min-max。"
        )


if __name__ == "__main__":
    main()
