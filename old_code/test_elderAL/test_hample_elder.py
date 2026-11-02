"""Run one ElderAL experiment with a shorter Hampel time window.

Pipeline:
    Hampel(window_size=2, full window=5 frames, n_sigma=3)
    -> IQR(factor=1.5)
    -> min-max
    -> linear interpolation(target_K=64)

The model is CSI-Time. This script intentionally has no phase-calibration
step and reuses the ElderAL loading, split, training, and evaluation helpers
from ``full_test_elder.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import full_test_elder as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "elderAL_hampel_w2_iqr_minmax_linear64"
MODEL_NAME = "csitime"
DATA_PATH = PROJECT_ROOT.parent / "sdp_dataset" / "elderAL"
RESULT_DIR = Path(__file__).resolve().parent / "result" / "hampel_elder_w3"
SUMMARY_PATH = RESULT_DIR / f"{RUN_NAME}_{MODEL_NAME}_summary.csv"

COMBO = {
    "combo_index": 1,
    "combo_id": "elderAL_hampel_w2",
    "combo_name": "hampel_w2_s3+zscore+min-max+linear64",
    "denoise": "hampel_w2_s3",
    "outliers": "iqr",
    "normalize": "min-max",
    "interpolate": "linear64",
    "pipeline_steps": {
        "denoise": {
            "method": "hampel",
            "window_size": 3,
            "n_sigma": 3.0,
        },
        # "outlier":{"method": "z-score", "factor": 3.0},
        "outliers": {"method": "iqr", "factor": 1.5},
        "normalize": {"method": "min-max"},
        "interpolate": {"method": "linear", "target_K": 64},
    },
}


def configure_base_experiment() -> None:
    """Point the ElderAL helpers at this experiment and its output paths."""
    base.DATA_PATH = DATA_PATH
    base.RUN_NAME = RUN_NAME
    base.MODEL_NAME = MODEL_NAME
    base.RESULT_DIR = RESULT_DIR
    base.SUMMARY_PATH = SUMMARY_PATH


def validate_configuration() -> None:
    """Fail early if the intended single-combination experiment drifts."""
    pipeline_steps = COMBO["pipeline_steps"]
    denoise = pipeline_steps["denoise"]

    if denoise.get("method") != "hampel":
        raise ValueError("denoise method must be Hampel")
    if denoise.get("window_size") != 3:
        raise ValueError("Hampel window_size must be 2 (full window: 5 frames)")
    if "calibrate" in pipeline_steps:
        raise ValueError("this ElderAL experiment must not use phase calibration")
    if MODEL_NAME != "csitime":
        raise ValueError("this ElderAL experiment must use the csitime model")


def main() -> None:
    configure_base_experiment()
    validate_configuration()

    if not base.DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 ElderAL 数据目录: {base.DATA_PATH}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("ElderAL Hampel 短窗口单组合实验")
    print(f"固定模型: {MODEL_NAME}")
    print(f"数据目录: {base.DATA_PATH}")
    print(f"训练轮数: {base.NUM_EPOCHS}")
    print(f"时间补齐/截断长度: {base.PADDING_LENGTH}")
    print(f"结果目录: {RESULT_DIR}")
    print(f"summary: {SUMMARY_PATH}")
    print("pipeline:")
    print(json.dumps(COMBO["pipeline_steps"], ensure_ascii=False, indent=2))

    done_records = base.load_done_records()
    if COMBO["combo_id"] in done_records:
        print(f"已有完成记录，跳过: {COMBO['combo_id']}")
        return

    params = base.load_params(base.DATASET_NAME)
    base.set_seed(base.SEED)
    csi_data_list = base.load_raw_data()

    try:
        row = base.run_one_combo(COMBO, 1, csi_data_list, params)
    finally:
        base.clear_cuda_cache()

    base.append_summary(row)

    if row["status"] != "ok":
        raise RuntimeError(f"实验失败: {row['error']}")

    print(f"best val acc: {float(row['best_val_acc']):.4f}")
    print(f"test acc: {float(row['test_acc']):.4f}")
    print(f"结果已保存到: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
