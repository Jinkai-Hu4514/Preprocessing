"""Launch gait combinations 290-320 on physical GPU 1.

This launcher intentionally shares the implementation in full_test_gait_new.py
while assigning an independent range, result directory, and summary CSV.
"""

from __future__ import annotations

import os


os.environ["GAIT_CUDA_VISIBLE_DEVICES"] = "1"
os.environ["GAIT_COMBO_START"] = "290"
os.environ["GAIT_COMBO_END"] = "320"
os.environ["GAIT_RESULT_DIR_NAME"] = "full_tests_new_gpu1_290_320"
os.environ["GAIT_RUN_NAME"] = "gait_320_pipeline_gpu1_290_320"

import full_test_gait_new as experiment


if __name__ == "__main__":
    experiment.main()
