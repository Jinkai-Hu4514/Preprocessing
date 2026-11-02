"""在当前 ElderAL 测试输入上复评 self_design_test 的全部匹配实验。

对比范围：
- activity_detection + csitime
- high_quality + csitime
- localization + csitime

本脚本只读取数据和 checkpoint，不训练模型，也不修改现有实验结果。
"""

from __future__ import annotations

import contextlib
import gc
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import pipline_elderAL_steps as elder
from wsdp.datasets import CSIDataset
from wsdp.models import create_model


HERE = Path(__file__).resolve().parent
PRESETS = ("activity_detection", "high_quality", "localization")
MODEL_NAME = "csitime"
BATCH_SIZE = 32

LOGGED_ACCURACIES = {
    "activity_detection": {"old": 0.8741, "new": 0.7778},
    "high_quality": {"old": 0.9148, "new": 0.8852},
    "localization": {"old": 0.8815, "new": 0.8333},
}


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


def checkpoint_paths(preset_name: str) -> tuple[Path, Path]:
    old_checkpoint = (
        HERE
        / "result"
        / "preset_tests"
        / f"source_action_position+{preset_name}+{MODEL_NAME}"
        / "best_checkpoint.pth"
    )
    new_checkpoint = (
        HERE
        / "result"
        / "self_design_test"
        / f"{preset_name}+{MODEL_NAME}"
        / "best_checkpoint.pth"
    )
    return old_checkpoint, new_checkpoint


def evaluate(model, loader, device, checkpoint_path: Path) -> float:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, accuracy = elder._evaluate_model(model, loader, device)
    return float(accuracy)


def build_current_test_loader(csi_data_list, preset_name: str):
    pipeline_steps = elder.resolve_pipeline_steps(preset_name, None)

    with suppress_process_stdout():
        processed_data, labels, groups, unique_labels = elder.process_data(
            csi_data_list,
            pipeline_steps,
            elder.PADDING_LENGTH,
        )
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
    split_sizes = (len(train_data), len(val_data), len(test_data))

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
    num_classes = len(unique_labels)

    del test_data, test_labels, test_dataset
    gc.collect()
    return test_loader, input_shape, num_classes, split_sizes


def main() -> None:
    for preset_name in PRESETS:
        for path in checkpoint_paths(preset_name):
            if not path.is_file():
                raise FileNotFoundError(f"找不到 checkpoint: {path}")

    elder.set_seed(elder.SEED)
    print("读取一次当前 ElderAL 原始数据……", flush=True)
    with suppress_process_stdout():
        csi_data_list = elder.load_raw_data()
    print(f"有效读取对象数: {len(csi_data_list)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for index, preset_name in enumerate(PRESETS, start=1):
        print(
            f"\n[{index}/{len(PRESETS)}] 重建当前 {preset_name} 测试输入……",
            flush=True,
        )
        loader, input_shape, num_classes, split_sizes = build_current_test_loader(
            csi_data_list,
            preset_name,
        )
        print(
            f"split={split_sizes}, input_shape={input_shape}, device={device}",
            flush=True,
        )

        model = create_model(
            MODEL_NAME,
            num_classes=num_classes,
            input_shape=input_shape,
        ).to(device)
        old_checkpoint, new_checkpoint = checkpoint_paths(preset_name)
        old_accuracy = evaluate(model, loader, device, old_checkpoint)
        new_accuracy = evaluate(model, loader, device, new_checkpoint)

        expected = LOGGED_ACCURACIES[preset_name]
        rows.append(
            {
                "preset": preset_name,
                "old": old_accuracy,
                "old_logged": expected["old"],
                "new": new_accuracy,
                "new_logged": expected["new"],
            }
        )

        del model, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n同一当前测试集复评汇总")
    print("preset                  old(eval/log)      new(eval/log)      delta")
    for row in rows:
        print(
            f"{row['preset']:<23}"
            f"{row['old']:.4f}/{row['old_logged']:.4f}     "
            f"{row['new']:.4f}/{row['new_logged']:.4f}     "
            f"{row['new'] - row['old']:+.4f}"
        )

    all_reproduced = all(
        abs(row["old"] - row["old_logged"]) <= 0.005
        and abs(row["new"] - row["new_logged"]) <= 0.005
        for row in rows
    )
    if all_reproduced:
        print(
            "\n结论: 所有旧、新 checkpoint 都在当前测试输入上复现原日志。"
            "预处理、划分和模型 forward 一致，准确率差距来自训练得到的权重不同。"
        )
    else:
        print(
            "\n结论: 至少一组无法复现原日志，需要继续检查该 preset 的"
            "预处理或模型版本。"
        )


if __name__ == "__main__":
    main()
