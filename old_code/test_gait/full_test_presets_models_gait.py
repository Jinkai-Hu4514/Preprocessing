"""Gait 全预设、全模型实验；输入自动使用幅度和相位。"""

from __future__ import annotations

import contextlib
import csv
import sys
import traceback
from pathlib import Path

import pipline_gait_steps as step_pipeline
from wsdp.algorithms import apply_preset, list_presets
from wsdp.models import list_models


RUN_NAME = "gait"
DATASET_NAME = step_pipeline.DATASET_NAME
DATA_PATH = step_pipeline.DATA_PATH

RESULT_DIR = Path(__file__).resolve().parent / "result" / "new_preset_tests"
SUMMARY_PATH = RESULT_DIR / f"{RUN_NAME}_all_presets_models_summary.csv"
SUMMARY_FIELDS = [
    "preset",
    "model",
    "status",
    "best_val_acc",
    "test_acc",
    "output_dir",
    "error",
]

SKIP_MODELS = {
    "efficientnetcsi",
    "visiontransformercsi",
    "graphneuralcsi",
    "mambacsi",
}

BATCH_SIZE = step_pipeline.BATCH_SIZE
LEARNING_RATE = step_pipeline.LEARNING_RATE
WEIGHT_DECAY = step_pipeline.WEIGHT_DECAY
NUM_EPOCHS = step_pipeline.NUM_EPOCHS
PADDING_LENGTH = step_pipeline.PADDING_LENGTH
TEST_SPLIT = step_pipeline.TEST_SPLIT
VAL_SPLIT = step_pipeline.VAL_SPLIT
SEED = step_pipeline.SEED


def load_done_records() -> set[tuple[str, str]]:
    """读取 summary 中已有的组合，用于断点续跑。"""
    if not SUMMARY_PATH.exists():
        return set()

    with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        return {
            (row["preset"], row["model"])
            for row in csv.DictReader(f)
            if row.get("preset") and row.get("model")
        }


def append_summary(row: dict) -> None:
    """把一个模型结果追加到 summary。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_PATH.exists()

    with SUMMARY_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def output_dir_for(preset_name: str, model_name: str) -> Path:
    return RESULT_DIR / f"{RUN_NAME}+{preset_name}+{model_name}"


def available_models() -> list[str]:
    """源码注册模型，排除本批次不跑的模型。"""
    return [
        model_name
        for model_name in list_models().keys()
        if model_name not in SKIP_MODELS
    ]


def train_and_evaluate_model(
    preset_name: str,
    model_name: str,
    model_index: int,
    model_total: int,
    params: dict,
    loaders,
    unique_labels,
    input_shape,
) -> dict:
    """训练并测试一个模型。"""
    output_dir = output_dir_for(preset_name, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "train_process.txt").open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(step_pipeline.Tee(sys.stdout, log_file)):
            print("\n" + "=" * 60)
            print(
                f"开始模型训练: preset={preset_name}, "
                f"model={model_name} ({model_index}/{model_total})"
            )
            print("=" * 60)

            step_pipeline.set_seed(SEED)

            model, device = step_pipeline.create_registered_model(
                model_name,
                len(unique_labels),
                input_shape,
            )

            checkpoint_path = step_pipeline.train_registered_model(
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

            val_acc, test_acc = step_pipeline.evaluate_checkpoint(
                model,
                device,
                loaders[2],
                checkpoint_path,
            )

    return {
        "preset": preset_name,
        "model": model_name,
        "status": "ok",
        "best_val_acc": val_acc,
        "test_acc": test_acc,
        "output_dir": str(output_dir),
        "error": "",
    }


def failure_row(preset_name: str, model_name: str, error: BaseException) -> dict:
    output_dir = output_dir_for(preset_name, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")

    return {
        "preset": preset_name,
        "model": model_name,
        "status": "failed",
        "best_val_acc": "",
        "test_acc": "",
        "output_dir": str(output_dir),
        "error": f"{type(error).__name__}: {error}",
    }


def clear_cuda_cache() -> None:
    if step_pipeline.torch.cuda.is_available():
        step_pipeline.torch.cuda.empty_cache()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 Gait 数据目录: {DATA_PATH}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    done_records = load_done_records()
    presets = list(list_presets().keys())
    models = available_models()
    model_total = len(models)
    params = step_pipeline.load_params(DATASET_NAME)

    print(f"预设数量: {len(presets)}")
    print(f"模型数量: {model_total} (已跳过: {sorted(SKIP_MODELS)})")
    print(f"summary: {SUMMARY_PATH}")

    # step 1: 读取数据。全量实验只需要读一次原始数据。
    step_pipeline.set_seed(SEED)
    csi_data_list = step_pipeline.load_raw_data()

    for preset_name in presets:
        pending_models = [
            (model_index, model_name)
            for model_index, model_name in enumerate(models, start=1)
            if (preset_name, model_name) not in done_records
        ]

        if not pending_models:
            print(f"\n跳过预设 {preset_name}: CSV 中所有模型已有记录")
            continue

        preset_dir = RESULT_DIR / f"{RUN_NAME}+{preset_name}"
        preset_dir.mkdir(parents=True, exist_ok=True)

        with (preset_dir / "preset_process.txt").open("w", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(step_pipeline.Tee(sys.stdout, log_file)):
                print("\n" + "=" * 60)
                print(f"开始预设处理: {preset_name}")
                print("=" * 60)

                # step 2: 使用源码预设算法处理数据。这里全量脚本只跑预设，不需要自定义判断。
                pipeline_steps = apply_preset(preset_name)
                processed_data, labels, groups, unique_labels = step_pipeline.process_data(
                    csi_data_list,
                    pipeline_steps,
                    PADDING_LENGTH,
                )

                # step 3: 数据划分。一个预设只划分一次，后续模型共用同一份 split。
                split = step_pipeline.split_data(
                    processed_data,
                    labels,
                    groups,
                    pipeline_steps,
                    TEST_SPLIT,
                    VAL_SPLIT,
                    SEED,
                )

                # step 4: 构造 DataLoader。一个预设只构造一次，后续模型共用。
                batch_size = BATCH_SIZE if BATCH_SIZE is not None else params.get("batch", 32)
                loaders = step_pipeline.build_loaders(split, pipeline_steps, batch_size)
                input_shape = tuple(loaders[0].dataset.data_list.shape[1:])
                print(f"模型实际输入形状: {input_shape}")

        # step 5-7: 在同一个预设处理结果上，依次训练和测试所有模型。
        for model_index, model_name in pending_models:
            print(
                f"\n运行组合: preset={preset_name}, "
                f"model={model_name} ({model_index}/{model_total})"
            )

            try:
                row = train_and_evaluate_model(
                    preset_name,
                    model_name,
                    model_index,
                    model_total,
                    params,
                    loaders,
                    unique_labels,
                    input_shape,
                )
            except Exception as exc:
                print(f"组合失败: preset={preset_name}, model={model_name}: {exc}")
                row = failure_row(preset_name, model_name, exc)
            finally:
                clear_cuda_cache()

            append_summary(row)
            done_records.add((preset_name, model_name))

    print(f"\n全部完成，汇总已保存到: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
