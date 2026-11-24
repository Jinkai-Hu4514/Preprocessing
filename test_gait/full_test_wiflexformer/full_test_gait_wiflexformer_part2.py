"""运行第 161–320 种组合，并写入第二阶段的汇总 CSV。"""
import full_test_gait_wiflexformer

CUDA_DEVICE_INDEX = 2  # 第 161–320 种组合使用的 GPU 编号，可单独修改。


if __name__ == "__main__":
    full_test_gait_wiflexformer.CUDA_DEVICE_INDEX = CUDA_DEVICE_INDEX
    full_test_gait_wiflexformer.main(start=160, stop=320)
