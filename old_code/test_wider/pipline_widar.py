"""Widar 单组实验脚本。

这个脚本只跑一个“算法组合 + 模型”的实验。核心流程全部交给最终版源码
``wsdp.pipeline``，本文件只负责配置DATA_PATH、算法组合、模型和OUTPUT_DIR。

使用方式：
1. 使用预设算法：设置 ``PRESET_NAME``，保持 ``PIPELINE_STEPS = None``。
2. 使用自定义算法组合：填写 ``PIPELINE_STEPS``，它会优先于 ``PRESET_NAME``。
3. 使用源码默认 BaseProcessor：设置 ``PRESET_NAME = "baseprocessor"``。

Widar 当前源码逻辑：
- 任务标签是 gesture_type。
- 分组依据是 position_id * 1000 + orientation_id * 100 + receiver_id。
- `dataset="widar"` 会让公共源码自动构造“幅度 + 相位”模型输入。
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WSDP_SRC = PROJECT_ROOT / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main" / "src"
DATA_ROOT = PROJECT_ROOT.parent / "sdp_dataset"

if not (WSDP_SRC / "wsdp").is_dir():
    raise FileNotFoundError(f"找不到本地 WSDP 源码目录: {WSDP_SRC}")
sys.path.insert(0, str(WSDP_SRC))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wsdp_mplconfig")

# 最终版源码在导入 wsdp 时会加载下载模块；本测试不使用下载功能。
sys.modules.setdefault("kagglehub", types.ModuleType("kagglehub"))

from wsdp.core import pipeline as wsdp_pipeline
from wsdp.dataset_policy import uses_phase_amplitude


DATASET_NAME = "widar"
DATA_PATH = DATA_ROOT / "widar_common3"

# ==================== 配置区 ====================
# 更换算法：修改 PRESET_NAME，或填写 PIPELINE_STEPS。
PRESET_NAME = "high_quality"
PIPELINE_STEPS = None
# 自定义算法组合示例：
# PIPELINE_STEPS = {
#     "denoise": {"method": "savgol", "window_length": 7, "polyorder": 3},
#     "normalize": {"method": "min-max"},
# }

# 更换模型：修改 MODEL_NAME。
# 可用模型：运行 list_algorithms_models.py 查看。
MODEL_NAME = "csitime"

# 训练超参数；None 表示继续使用源码或模型参数文件里的默认值。
BATCH_SIZE = None
LEARNING_RATE = None
WEIGHT_DECAY = None
NUM_EPOCHS = 80
NUM_SEEDS = 1
PADDING_LENGTH = 1500
TEST_SPLIT = 0.3
VAL_SPLIT = 0.5

# 输出位置。
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "result"
    / "self_design_test"
    / f"auto_amp_phase+{PRESET_NAME}+{MODEL_NAME}"
)
# ================== 配置区结束 ==================


def main() -> None:
    """运行一个随机种子的 Widar 实验。"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 Widar 数据集目录: {DATA_PATH}")

    algorithm_preset = None
    if PIPELINE_STEPS is None and PRESET_NAME != "baseprocessor":
        algorithm_preset = PRESET_NAME

    input_representation = (
        "幅度 + 相位"
        if uses_phase_amplitude(DATASET_NAME)
        else "幅度"
    )
    print(f"模型输入策略: {input_representation}")

    wsdp_pipeline(
        input_path=str(DATA_PATH),
        output_folder=str(OUTPUT_DIR),
        dataset=DATASET_NAME,
        model_name=MODEL_NAME,
        pipeline_steps=PIPELINE_STEPS,
        algorithm_preset=algorithm_preset,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_epochs=NUM_EPOCHS,
        padding_length=PADDING_LENGTH,
        test_split=TEST_SPLIT,
        val_split=VAL_SPLIT,
        num_seeds=NUM_SEEDS,
    )
    print(f"结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
