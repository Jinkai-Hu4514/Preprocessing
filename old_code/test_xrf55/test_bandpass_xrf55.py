"""Run one XRF55 experiment with the corrected 200 Hz sampling rate.

Pipeline:
    bandpass(fs=200, 0.5-50 Hz, order=4)
    -> IQR(factor=1.5)
    -> z-score
    -> cubic interpolation(target_K=15)

The experiment uses the same first-three-user scope, repetition split, training
settings, and ResNet1D model as ``full_test_xrf55.py``.  No phase calibration
step is included because the current XRF55 ``.npy`` input is real-valued.
"""

from __future__ import annotations

import json
from pathlib import Path

import full_test_xrf55 as base


RUN_NAME = "xrf55_bandpass_fs200_iqr_zscore_cubic15"
MODEL_NAME = "resnet1d"
DATA_PATH = base.PROJECT_ROOT.parent / "sdp_dataset" / "xrf55" / "wifi"
RESULT_DIR = Path(__file__).resolve().parent / "result" / "bandpass_fs200"
SUMMARY_PATH = RESULT_DIR / f"{RUN_NAME}_{MODEL_NAME}_summary.csv"

COMBO = {
    "combo_index": 1,
    "combo_id": "xrf55_bandpass_fs200",
    "combo_name": "bandpass_0.5-50_fs200+iqr+z-score+cubic15",
    "denoise": "bandpass_0.5-50_fs200",
    "outliers": "iqr",
    "normalize": "z-score",
    "interpolate": "cubic15",
    "pipeline_steps": {
        "denoise": {
            "method": "bandpass",
            "order": 4,
            "low_freq": 0.5,
            "high_freq": 50.0,
            "fs": 200.0,
        },
        "outliers": {"method": "iqr", "factor": 1.5},
        "normalize": {"method": "z-score"},
        "interpolate": {"method": "cubic", "target_K": 15},
    },
}


def configure_base_experiment() -> None:
    """Point the shared training helpers at this experiment's own outputs."""
    base.RUN_NAME = RUN_NAME
    base.MODEL_NAME = MODEL_NAME
    base.DATA_PATH = DATA_PATH
    base.RESULT_DIR = RESULT_DIR
    base.SUMMARY_PATH = SUMMARY_PATH


def validate_configuration() -> None:
    """Fail early if this single experiment is accidentally changed."""
    pipeline_steps = COMBO["pipeline_steps"]
    denoise = pipeline_steps["denoise"]

    if denoise.get("method") != "bandpass" or denoise.get("fs") != 200.0:
        raise ValueError("bandpass must explicitly use fs=200 Hz")
    if "calibrate" in pipeline_steps:
        raise ValueError("this XRF55 experiment must not use phase calibration")


def main() -> None:
    configure_base_experiment()
    validate_configuration()

    if not base.DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 XRF55 数据目录: {base.DATA_PATH}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("XRF55 bandpass fs=200 单组合实验")
    print(f"固定模型: {MODEL_NAME}")
    print(f"数据目录: {base.DATA_PATH}")
    print(f"数据范围: 前 {base.XRF55_USER_LIMIT} 个用户")
    print(f"训练轮数: {base.NUM_EPOCHS}")
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
