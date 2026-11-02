"""列出最终版 WSDP 源码中可用的数据集、算法、模型和预设。

这个脚本只做信息展示，不运行训练。它直接读取当前工作区里的最终版源码，
方便确认测试脚本中可以填写哪些 ``PRESET_NAME``、``MODEL_NAME`` 和自定义算法组合。
"""

from __future__ import annotations

import sys
import types
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WSDP_SRC = PROJECT_ROOT / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main" / "src"

if not (WSDP_SRC / "wsdp").is_dir():
    raise FileNotFoundError(f"找不到本地 WSDP 源码目录: {WSDP_SRC}")
sys.path.insert(0, str(WSDP_SRC))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wsdp_mplconfig")

# 最终版源码在导入 wsdp 时会加载下载模块；本脚本不使用下载功能。
sys.modules.setdefault("kagglehub", types.ModuleType("kagglehub"))

from wsdp.algorithms import apply_preset, list_algorithms, list_presets
from wsdp.models import list_models
from wsdp.readers import list_datasets


def main() -> None:
    """打印当前源码注册表中的可用选项。"""
    print("可用数据集")
    print("-" * 40)
    for dataset in list_datasets():
        print(dataset)

    print("\n算法列表")
    print("-" * 40)
    for category, methods in list_algorithms().items():
        print(category)
        for method in methods:
            print(f"-- {method}")

    print("\n模型列表")
    print("-" * 40)
    for model_name, category in list_models().items():
        print(f"{model_name:<24} {category}")

    print("\n可用预设")
    print("-" * 40)
    for preset_name in list_presets().keys():
        print(f"-- {preset_name}")
        for step_name, step_config in apply_preset(preset_name).items():
            print(f"  --{step_name} : {step_config}")


if __name__ == "__main__":
    main()
