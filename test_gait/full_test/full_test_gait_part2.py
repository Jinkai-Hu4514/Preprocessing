"""第二阶段第 161–320 种组合；复用原脚本，结果写入同一目录下独立的汇总 CSV。"""
import full_test_gait

CUDA_DEVICE_INDEX = 2  # 独立于第 1–160 种组合的 GPU 编号。


if __name__ == "__main__":
    full_test_gait.CUDA_DEVICE_INDEX = CUDA_DEVICE_INDEX
    full_test_gait.main(start=160, stop=320)
