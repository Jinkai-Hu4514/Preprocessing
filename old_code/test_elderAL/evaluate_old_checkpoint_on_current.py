"""在当前 ElderAL 测试集上对比旧、新 high_quality+csitime checkpoint。

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

import pipline_elderAL_steps as elder
from wsdp.datasets import CSIDataset
from wsdp.models import create_model


HERE = Path(__file__).resolve().parent

OLD_CHECKPOINT = (
    HERE
    / "result"
    / "preset_tests"
    / "source_action_position+high_quality+csitime"
    / "best_checkpoint.pth"
)
CURRENT_CHECKPOINT = (
    HERE
    / "result"
    / "self_design_test"
    / "high_quality+csitime"
    / "best_checkpoint.pth"
)

PRESET_NAME = "high_quality"
MODEL_NAME = "csitime"
BATCH_SIZE = 32


@contextlib.contextmanager
def suppress_process_stdout():
    """Silence stdout/stderr in the parent and worker processes."""
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def evaluate_checkpoint(model, loader, device, checkpoint_path: Path) -> float:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, accuracy = elder._evaluate_model(model, loader, device)
    return float(accuracy)


def main() -> None:
    for checkpoint_path in (OLD_CHECKPOINT, CURRENT_CHECKPOINT):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    elder.set_seed(elder.SEED)
    pipeline_steps = elder.resolve_pipeline_steps(PRESET_NAME, None)

    print("1/4 使用当前源码读取 ElderAL 数据……", flush=True)
    with suppress_process_stdout():
        csi_data_list = elder.load_raw_data()
    print(f"    有效读取对象数: {len(csi_data_list)}", flush=True)

    print("2/4 使用当前 high_quality pipeline 处理数据……", flush=True)
    with suppress_process_stdout():
        processed_data, labels, groups, unique_labels = elder.process_data(
            csi_data_list,
            pipeline_steps,
            elder.PADDING_LENGTH,
        )
    del csi_data_list
    gc.collect()
    print(f"    处理后样本数: {len(processed_data)}", flush=True)

    print("3/4 使用当前 group split 生成测试集……", flush=True)
    with suppress_process_stdout():
        split = elder.split_data(
            processed_data,
            labels,
            groups,
            pipeline_steps,
            elder.TEST_SPLIT,
            elder.VAL_SPLIT,
            elder.SEED,
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

    test_dataset = CSIDataset(
        test_data,
        test_labels,
        dataset_name=elder.DATASET_NAME,
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

    if abs(current_accuracy - 0.8852) > 0.005:
        print(
            "警告: 新 checkpoint 未复现日志中的 0.8852，"
            "说明当前生成的测试输入与新训练时已经不完全一致。"
        )
    elif abs(old_accuracy - 0.9148) <= 0.005:
        print(
            "结论: 旧、新 checkpoint 在当前测试集上都复现原日志；"
            "预处理和测试集一致，差距来自训练过程。"
        )
    elif old_accuracy >= current_accuracy:
        print(
            "结论: 旧 checkpoint 在当前输入上仍优于新 checkpoint；"
            "主要差距来自训练过程，而不是当前测试集预处理。"
        )
    else:
        print(
            "结论: 旧 checkpoint 在当前输入上无法复现旧准确率；"
            "旧、新实验的预处理、测试集内容或模型实现存在版本差异。"
        )


if __name__ == "__main__":
    main()
