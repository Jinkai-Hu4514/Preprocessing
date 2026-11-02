"""XRF55 单组实验脚本。

这个脚本只跑一个“算法组合 + 模型”的实验。核心流程全部交给最终版源码
``wsdp.pipeline``，本文件只负责配置DATA_PATH、算法组合、模型和OUTPUT_DIR。

使用方式：
1. 使用预设算法：设置 ``PRESET_NAME``，保持 ``PIPELINE_STEPS = None``。
2. 使用自定义算法组合：填写 ``PIPELINE_STEPS``，它会优先于 ``PRESET_NAME``。
3. 使用源码默认 BaseProcessor：设置 ``PRESET_NAME = "baseprocessor"``。

XRF55 当前源码逻辑：
- 任务标签是 action_id。
- 分组依据是 repetition_id。
- 数据划分固定为 repetition 1-12 train，13-16 valid，17-20 test。
- `dataset="xrf55"` 使用幅度模型输入，不增加相位通道。
"""

from __future__ import annotations

import os
import re
import shutil
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


DATASET_NAME = "xrf55"
DATA_PATH = DATA_ROOT / "xrf55"
XRF55_USER_LIMIT = 3
FILTERED_DATA_PATH = Path(__file__).resolve().parent / ".xrf55_first3_input"

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
NUM_EPOCHS = 20
NUM_SEEDS = 1
PADDING_LENGTH = 1000
TEST_SPLIT = 0.3
VAL_SPLIT = 0.5

# 输出位置。
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "result"
    / "self_design_test"
    / f"{PRESET_NAME}+{MODEL_NAME}"
)
# ================== 配置区结束 ==================


def is_first_three_user_file(file_path: Path) -> bool:
    """判断 XRF55 文件名是否属于前三个用户。"""
    match = re.search(r"(\d+)_(\d+)_(\d+)", file_path.stem)
    if not match:
        return False

    return int(match.group(1)) <= XRF55_USER_LIMIT


def prepare_filtered_data_path() -> Path:
    """创建只包含前三个用户文件链接的输入目录。"""
    script_dir = Path(__file__).resolve().parent
    filtered_root = FILTERED_DATA_PATH.resolve()

    if script_dir.resolve() not in filtered_root.parents:
        raise RuntimeError(f"过滤目录不在当前测试目录下: {filtered_root}")

    if FILTERED_DATA_PATH.exists():
        shutil.rmtree(FILTERED_DATA_PATH)
    FILTERED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    linked_count = 0
    for source_path in sorted(DATA_PATH.rglob("*")):
        if not source_path.is_file():
            continue
        if "truth" in source_path.name:
            continue
        if not is_first_three_user_file(source_path):
            continue

        target_path = FILTERED_DATA_PATH / source_path.relative_to(DATA_PATH)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            os.link(source_path, target_path)
        except OSError:
            try:
                target_path.symlink_to(source_path.resolve())
            except OSError as exc:
                raise RuntimeError(
                    "无法为前三个用户数据创建硬链接或符号链接，请确认数据和工程在同一磁盘，"
                    "或系统允许创建符号链接。"
                ) from exc

        linked_count += 1

    if linked_count == 0:
        raise IOError(f"没有找到前三个用户的 XRF55 文件: {DATA_PATH}")

    print(f"前三个用户过滤目录: {FILTERED_DATA_PATH} ({linked_count} files)")

    return FILTERED_DATA_PATH


def main() -> None:
    """运行一个随机种子的 XRF55 实验。"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 XRF55 数据集目录: {DATA_PATH}")

    input_path = prepare_filtered_data_path()

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
        input_path=str(input_path),
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
