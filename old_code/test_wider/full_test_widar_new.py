"""Widar 320 组公共前缀复用版全量实验脚本。

固定模型：mlpmodel

组合空间：
denoise(5) -> outliers(2) -> calibrate(4) -> normalize(2) -> interpolate(4)
= 320 组 pipeline。

与 full_test_widar.py 的逐组合完整预处理不同，本脚本按前三步划分为
40 个公共前缀，每个公共前缀只执行一次，再派生 8 个最终组合：

- Widar + z-score：公共前缀 -> interpolate -> z-score 幅相实数通道；
- Widar + min-max：公共前缀 -> min-max（复用一次）-> interpolate。

这个文件不依赖任何测试脚本，直接调用 WSDP 源码模块。
"""

from __future__ import annotations

import contextlib
import csv
import gc
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
import types
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product
from pathlib import Path

# ==================== 环境设置 ====================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wsdp_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


# ==================== 路径设置 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WSDP_SRC = (
    PROJECT_ROOT
    / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
    / "src"
)
DATA_ROOT = PROJECT_ROOT.parent / "sdp_dataset"

if not (WSDP_SRC / "wsdp").is_dir():
    raise FileNotFoundError(f"找不到本地 WSDP 源码目录: {WSDP_SRC}")

sys.path.insert(0, str(WSDP_SRC))

# 避免源码里 import kagglehub 时出错。
sys.modules.setdefault("kagglehub", types.ModuleType("kagglehub"))


# ==================== WSDP 源码模块 ====================
from wsdp import readers
from wsdp.algorithms import execute_pipeline
from wsdp.algorithms.amplitude import normalize_amplitude
from wsdp.core import _create_data_split, _evaluate_model
from wsdp.datasets import CSIDataset
from wsdp.models import create_model
from wsdp.processors.configurable_processor import _process_single_csi_configurable
from wsdp.utils import load_params, resize_csi_to_fixed_length, train_model


# ==================== 数据集与实验设置 ====================
DATASET_NAME = "widar"
DATA_PATH = DATA_ROOT / "widar_common3"

RUN_NAME = "widar_320_pipeline_optimized"
MODEL_NAME = "mlpmodel"

BATCH_SIZE = None
LEARNING_RATE = None
WEIGHT_DECAY = None
NUM_EPOCHS = 80
PADDING_LENGTH = 1500
TEST_SPLIT = 0.3
VAL_SPLIT = 0.5
SEED = 42
PREPROCESS_WORKERS = 4

RESULT_DIR = Path(__file__).resolve().parent / "result" / "full_tests_new"
SUMMARY_PATH = RESULT_DIR / f"{RUN_NAME}_{MODEL_NAME}_summary.csv"

SUMMARY_FIELDS = [
    "combo_index",
    "combo_id",
    "combo_name",
    "model",
    "status",
    "best_val_acc",
    "test_acc",
    "denoise",
    "outliers",
    "calibrate",
    "normalize",
    "interpolate",
    "pipeline_steps",
    "output_dir",
    "duration_sec",
    "error",
]


# ==================== 320 组算法组合配置 ====================
DENOISE_OPTIONS = [
    ("wavelet", {"method": "wavelet"}),
    ("butterworth_o5_c0.3", {"method": "butterworth", "order": 5, "cutoff": 0.3}),
    ("savgol_w7_p3", {"method": "savgol", "window_length": 7, "polyorder": 3}),
    (
        "bandpass_0.5-50",
        {
            "method": "bandpass",
            "order": 4,
            "low_freq": 0.5,
            "high_freq": 50.0,
            "fs": 1000.0,
        },
    ),
    ("hampel_w5_s3", {"method": "hampel", "window_size": 5, "n_sigma": 3.0}),
]

OUTLIER_OPTIONS = [
    ("iqr", {"method": "iqr", "factor": 1.5}),
    ("outlier_z-score", {"method": "z-score", "factor": 3.0}),
]

CALIBRATE_OPTIONS = [
    ("linear", {"method": "linear"}),
    ("polynomial_d3", {"method": "polynomial", "degree": 3}),
    ("stc", {"method": "stc"}),
    ("robust", {"method": "robust"}),
]

NORMALIZE_OPTIONS = [
    ("z-score", {"method": "z-score"}),
    ("min-max", {"method": "min-max"}),
]

# Widar 原始子载波维通常是 30；统一输出 15 个子载波。
INTERPOLATE_OPTIONS = [
    ("linear15", {"method": "linear", "target_K": 15}),
    ("cubic15", {"method": "cubic", "target_K": 15}),
    ("nearest15", {"method": "nearest", "target_K": 15}),
    ("decimate15", {"method": "decimate", "target_K": 15}),
]


class Tee:
    """同时把 stdout 写到终端和 train_process.txt。"""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)

    def flush(self):
        for file in self.files:
            file.flush()


def set_seed(seed: int) -> None:
    """固定随机性，使各组合的模型初始化和 DataLoader 洗牌可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pipeline_combinations() -> list[dict]:
    """按旧脚本完全相同的顺序生成 320 组组合。"""
    combinations = []

    for combo_index, (denoise, outliers, calibrate, normalize, interpolate) in enumerate(
        product(
            DENOISE_OPTIONS,
            OUTLIER_OPTIONS,
            CALIBRATE_OPTIONS,
            NORMALIZE_OPTIONS,
            INTERPOLATE_OPTIONS,
        ),
        start=1,
    ):
        denoise_name, denoise_cfg = denoise
        outliers_name, outliers_cfg = outliers
        calibrate_name, calibrate_cfg = calibrate
        normalize_name, normalize_cfg = normalize
        interpolate_name, interpolate_cfg = interpolate

        pipeline_steps = {
            "denoise": denoise_cfg.copy(),
            "outliers": outliers_cfg.copy(),
            "calibrate": calibrate_cfg.copy(),
            "normalize": normalize_cfg.copy(),
            "interpolate": interpolate_cfg.copy(),
        }

        combinations.append(
            {
                "combo_index": combo_index,
                "combo_id": f"widar_{combo_index:03d}",
                "combo_name": "+".join(
                    [
                        denoise_name,
                        outliers_name,
                        calibrate_name,
                        normalize_name,
                        interpolate_name,
                    ]
                ),
                "denoise": denoise_name,
                "outliers": outliers_name,
                "calibrate": calibrate_name,
                "normalize": normalize_name,
                "interpolate": interpolate_name,
                "pipeline_steps": pipeline_steps,
            }
        )

    return combinations


def build_prefix_groups(combinations: list[dict]) -> list[dict]:
    """将320组按相邻的8组归为40个公共前缀，并检查顺序不变量。"""
    if len(combinations) != 320:
        raise RuntimeError(f"预期生成320组组合，实际为{len(combinations)}组")

    groups = []
    for offset in range(0, len(combinations), 8):
        group_combos = combinations[offset : offset + 8]
        first = group_combos[0]
        prefix_key = (first["denoise"], first["outliers"], first["calibrate"])

        if any(
            (combo["denoise"], combo["outliers"], combo["calibrate"]) != prefix_key
            for combo in group_combos
        ):
            raise RuntimeError(f"公共前缀分组顺序异常: {prefix_key}")

        normalize_order = [combo["normalize"] for combo in group_combos]
        if normalize_order != ["z-score"] * 4 + ["min-max"] * 4:
            raise RuntimeError(f"归一化组合顺序异常: {normalize_order}")

        prefix_steps = {
            "denoise": first["pipeline_steps"]["denoise"].copy(),
            "outliers": first["pipeline_steps"]["outliers"].copy(),
            "calibrate": first["pipeline_steps"]["calibrate"].copy(),
        }
        groups.append(
            {
                "prefix_index": len(groups) + 1,
                "prefix_name": "+".join(prefix_key),
                "prefix_steps": prefix_steps,
                "combinations": group_combos,
            }
        )

    if len(groups) != 40:
        raise RuntimeError(f"预期生成40个公共前缀，实际为{len(groups)}个")

    return groups


def save_loss_curve(
    history: dict,
    output_dir: Path,
    combo_id: str,
    model_name: str,
) -> None:
    """保存训练/验证 loss 曲线。"""
    if not history.get("train_loss") or not history.get("val_loss"):
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{combo_id}+{model_name} Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def load_and_parse_raw_data_once(executor: ProcessPoolExecutor):
    """只读取、堆叠原始帧并解析标签/分组一次，不执行候选算法。"""
    print("step 1: load_raw_data（全程只执行一次）")
    csi_data_list = readers.load_data(str(DATA_PATH), DATASET_NAME)
    print(f"读取原始文件数: {len(csi_data_list)}")

    # 直接复用 ConfigurableProcessor 的源码单样本函数，确保帧排序、堆叠、
    # 文件名解析、label/group selector 和无效样本过滤完全等价；同时使用
    # 本脚本的常驻 spawn 进程池，避免读取大数据后由默认 fork 继承整份内存。
    worker = partial(
        _process_single_csi_configurable,
        dataset=DATASET_NAME,
        pipeline_steps={},
    )
    raw_data, all_labels, all_groups = [], [], []
    for csi, label, group in executor.map(worker, csi_data_list, chunksize=1):
        if csi is not None:
            raw_data.append(csi)
            all_labels.append(label)
            all_groups.append(group)
    del csi_data_list
    gc.collect()

    if not raw_data:
        raise RuntimeError("Widar 原始数据解析后没有有效样本")

    unique_labels = sorted(set(all_labels))
    unique_groups = sorted(set(all_groups))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    group_map = {group: idx for idx, group in enumerate(unique_groups)}

    labels = np.asarray([label_map[label] for label in all_labels])
    groups = np.asarray([group_map[group] for group in all_groups])

    metadata = {
        "sample_count": len(raw_data),
        "label_distribution": dict(Counter(all_labels)),
        "group_distribution": dict(Counter(all_groups)),
        "unique_labels": unique_labels,
        "raw_sample_shape": tuple(raw_data[0].shape),
    }

    print(f"有效样本数: {metadata['sample_count']}")
    print(f"原始样本形状: {metadata['raw_sample_shape']}")
    print(f"类别数: {len(unique_labels)} (原始标签: {unique_labels})")

    return raw_data, labels, groups, unique_labels, metadata


def _execute_steps_worker(csi: np.ndarray, steps: dict) -> np.ndarray:
    """子进程执行一段普通 WSDP pipeline。"""
    return execute_pipeline(csi, steps, dataset=DATASET_NAME)


def _execute_zscore_tail_worker(csi: np.ndarray, interpolate_cfg: dict) -> np.ndarray:
    """子进程执行 Widar z-score 的真实尾部：插值后输出幅相实数通道。"""
    interpolated = execute_pipeline(
        csi,
        {"interpolate": interpolate_cfg},
        dataset=DATASET_NAME,
    )
    return normalize_amplitude(
        interpolated,
        method="z-score",
        return_phase_channels=True,
    )


def parallel_execute_steps(
    executor: ProcessPoolExecutor,
    data_list: list[np.ndarray],
    steps: dict,
) -> list[np.ndarray]:
    """并行执行一段 pipeline，executor.map 保持样本顺序。"""
    worker = partial(_execute_steps_worker, steps=steps)
    return list(executor.map(worker, data_list, chunksize=1))


def parallel_execute_zscore_tail(
    executor: ProcessPoolExecutor,
    data_list: list[np.ndarray],
    interpolate_cfg: dict,
) -> list[np.ndarray]:
    """并行执行 interpolate -> signed z-score amplitude + phase。"""
    worker = partial(
        _execute_zscore_tail_worker,
        interpolate_cfg=interpolate_cfg,
    )
    return list(executor.map(worker, data_list, chunksize=1))


def prepare_final_data(
    executor: ProcessPoolExecutor,
    source_data: list[np.ndarray],
    combo: dict,
) -> list[np.ndarray]:
    """从公共缓存生成当前组合的最终变长样本。"""
    interpolate_cfg = combo["pipeline_steps"]["interpolate"].copy()

    if combo["normalize"] == "z-score":
        return parallel_execute_zscore_tail(
            executor,
            source_data,
            interpolate_cfg,
        )

    # min-max 已在当前公共前缀的共享缓存中执行，这里只做插值。
    return parallel_execute_steps(
        executor,
        source_data,
        {"interpolate": interpolate_cfg},
    )


def resize_and_summarize(
    final_data: list[np.ndarray],
    metadata: dict,
) -> np.ndarray:
    """最后才补齐时间维，并记录与旧 step 脚本一致的处理信息。"""
    resized = resize_csi_to_fixed_length(
        final_data,
        target_length=PADDING_LENGTH,
    )
    processed_data = np.asarray(resized)

    if len(processed_data) != metadata["sample_count"]:
        raise RuntimeError(
            "最终样本数与一次性解析结果不一致: "
            f"{len(processed_data)} != {metadata['sample_count']}"
        )

    summary_lines = [
        f"   处理完成: {len(processed_data)} 个样本",
        f"   标签分布: {metadata['label_distribution']}",
        f"   分组分布: {metadata['group_distribution']}",
        f"   样本形状: {processed_data[0].shape}",
        (
            f"   类别数: {len(metadata['unique_labels'])} "
            f"(原始标签: {metadata['unique_labels']})"
        ),
    ]
    for line in summary_lines:
        print(line)

    return processed_data


def split_data(processed_data, labels, groups, pipeline_steps):
    """调用源码 group split；相同 seed 和 labels/groups 保证各组合划分一致。"""
    print("step 3: split_data")
    split = _create_data_split(
        processed_data,
        labels,
        groups,
        test_split=TEST_SPLIT,
        val_split=VAL_SPLIT,
        seed=SEED,
        use_simple_split=len(set(groups.tolist())) < 3,
        dataset=DATASET_NAME,
        pipeline_steps=pipeline_steps,
    )
    print(f"训练集 {len(split[0])} | 验证集 {len(split[1])} | 测试集 {len(split[2])}")
    return split


def build_loaders(split, pipeline_steps, batch_size: int):
    """构造训练、验证、测试 DataLoader。"""
    print("step 4: 构造 DataLoader")
    train_data, val_data, test_data, train_labels, val_labels, test_labels = split

    def make_loader(data, labels, shuffle: bool):
        return DataLoader(
            CSIDataset(
                data,
                labels,
                dataset_name=DATASET_NAME,
                pipeline_steps=pipeline_steps,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    return (
        make_loader(train_data, train_labels, shuffle=True),
        make_loader(val_data, val_labels, shuffle=False),
        make_loader(test_data, test_labels, shuffle=False),
    )


def create_registered_model(model_name: str, num_classes: int, input_shape):
    """创建源码注册模型。"""
    print("step 5: create_registered_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name,
        num_classes=num_classes,
        input_shape=input_shape,
    ).to(device)
    print(f"模型参数量 {sum(p.numel() for p in model.parameters())}")
    print(f"训练设备: {device}")
    return model, device


def train_registered_model(
    model,
    device,
    loaders,
    params: dict,
    output_dir: Path,
    combo_id: str,
):
    """训练模型并保存 checkpoint 和 loss 曲线。"""
    print(f"step 6: 开始训练 {NUM_EPOCHS} 轮")
    train_loader, val_loader, _ = loaders

    criterion = nn.CrossEntropyLoss()
    lr = LEARNING_RATE if LEARNING_RATE is not None else params.get("lr", 3e-4)
    wd = WEIGHT_DECAY if WEIGHT_DECAY is not None else params.get("wd", 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=5,
    )

    checkpoint_path = output_dir / "best_checkpoint.pth"
    history = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        NUM_EPOCHS,
        device,
        checkpoint_path,
        PADDING_LENGTH,
    )
    save_loss_curve(history, output_dir, combo_id, MODEL_NAME)
    print(f"训练完成，最佳模型保存至: {checkpoint_path}")
    return checkpoint_path


def evaluate_checkpoint(model, device, test_loader, checkpoint_path: Path):
    """加载最佳 checkpoint 并评估测试集。"""
    print("step 7: 测试集评估")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, test_acc = _evaluate_model(model, test_loader, device)
    val_acc = checkpoint.get("best_val_acc", 0.0) / 100.0
    print(f"最佳验证准确率: {val_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    return val_acc, test_acc


def load_done_records() -> set[str]:
    """断点续跑只跳过 status=ok；失败记录会在下次运行时重试。"""
    if not SUMMARY_PATH.exists():
        return set()

    with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as file:
        return {
            row["combo_id"]
            for row in csv.DictReader(file)
            if (
                row.get("combo_id")
                and row.get("model") == MODEL_NAME
                and row.get("status") == "ok"
            )
        }


def append_summary(row: dict) -> None:
    """立即追加一条结果，进程中断时已完成记录仍然保留。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_PATH.exists() or SUMMARY_PATH.stat().st_size == 0
    with SUMMARY_PATH.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        file.flush()


def output_dir_for(combo: dict) -> Path:
    return RESULT_DIR / f"{combo['combo_id']}+{combo['combo_name']}+{MODEL_NAME}"


def pipeline_steps_to_json(pipeline_steps: dict) -> str:
    return json.dumps(pipeline_steps, ensure_ascii=False, sort_keys=True)


def actual_execution_order(combo: dict) -> str:
    prefix = " -> ".join(
        [combo["denoise"], combo["outliers"], combo["calibrate"]]
    )
    if combo["normalize"] == "z-score":
        return f"{prefix} -> {combo['interpolate']} -> z-score -> 训练"
    return f"{prefix} -> min-max -> {combo['interpolate']} -> 训练"


def result_row(
    combo: dict,
    status: str,
    duration_sec: float,
    best_val_acc="",
    test_acc="",
    error: str = "",
) -> dict:
    return {
        "combo_index": combo["combo_index"],
        "combo_id": combo["combo_id"],
        "combo_name": combo["combo_name"],
        "model": MODEL_NAME,
        "status": status,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "denoise": combo["denoise"],
        "outliers": combo["outliers"],
        "calibrate": combo["calibrate"],
        "normalize": combo["normalize"],
        "interpolate": combo["interpolate"],
        "pipeline_steps": pipeline_steps_to_json(combo["pipeline_steps"]),
        "output_dir": str(output_dir_for(combo)),
        "duration_sec": f"{duration_sec:.2f}",
        "error": error,
    }


def clear_runtime_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one_combo(
    combo: dict,
    combo_total: int,
    source_data: list[np.ndarray],
    source_description: str,
    executor: ProcessPoolExecutor,
    labels: np.ndarray,
    groups: np.ndarray,
    unique_labels: list,
    metadata: dict,
    params: dict,
    prefix_index: int,
    prefix_name: str,
    prefix_duration_sec: float,
) -> dict:
    """从复用缓存生成最终输入，随后完成一次训练和测试。"""
    output_dir = output_dir_for(combo)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # 显式声明，finally 中逐项释放，避免长时间全量实验积累内存。
    final_data = None
    processed_data = None
    split = None
    loaders = None
    model = None
    device = None

    with (output_dir / "train_process.txt").open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
            try:
                print("\n" + "=" * 80)
                print(
                    f"当前算法组合: {combo['combo_index']}/{combo_total}"
                    f" | combo_id={combo['combo_id']}"
                    f" | model={MODEL_NAME}"
                )
                print(f"combo_name: {combo['combo_name']}")
                print(f"保存目录: {output_dir}")
                print(f"公共前缀: {prefix_index}/40 | {prefix_name}")
                print("公共前缀 pipeline_steps（本组只处理一次）:")
                print(
                    json.dumps(
                        {
                            key: combo["pipeline_steps"][key]
                            for key in ("denoise", "outliers", "calibrate")
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                print(f"公共前缀处理耗时: {prefix_duration_sec:.2f}s")
                print(f"当前输入缓存: {source_description}")
                print(f"实际执行顺序: {actual_execution_order(combo)}")
                print("当前组合的完整 pipeline_steps:")
                print(json.dumps(combo["pipeline_steps"], ensure_ascii=False, indent=2))
                print("=" * 80)

                set_seed(SEED)

                print("step 2: 从公共缓存生成当前组合的最终处理数据")
                final_data = prepare_final_data(
                    executor,
                    source_data,
                    combo,
                )
                processed_data = resize_and_summarize(final_data, metadata)
                del final_data
                final_data = None
                gc.collect()

                split = split_data(
                    processed_data,
                    labels,
                    groups,
                    combo["pipeline_steps"],
                )
                del processed_data
                processed_data = None
                gc.collect()

                batch_size = (
                    BATCH_SIZE if BATCH_SIZE is not None else params.get("batch", 32)
                )
                loaders = build_loaders(split, combo["pipeline_steps"], batch_size)
                del split
                split = None
                gc.collect()

                input_shape = tuple(loaders[0].dataset.data_list.shape[1:])
                print(f"模型实际输入形状: {input_shape}")
                model, device = create_registered_model(
                    MODEL_NAME,
                    len(unique_labels),
                    input_shape,
                )
                checkpoint_path = train_registered_model(
                    model,
                    device,
                    loaders,
                    params,
                    output_dir,
                    combo["combo_id"],
                )
                val_acc, test_acc = evaluate_checkpoint(
                    model,
                    device,
                    loaders[2],
                    checkpoint_path,
                )

                duration_sec = time.time() - start_time
                print("\n" + "=" * 80)
                print(
                    f"组合完成: {combo['combo_index']}/{combo_total}"
                    f" | combo_id={combo['combo_id']}"
                    f" | best_val_acc={val_acc:.4f}"
                    f" | test_acc={test_acc:.4f}"
                    f" | 当前组合耗时={duration_sec:.2f}s"
                )
                print("=" * 80)
                return result_row(
                    combo,
                    "ok",
                    duration_sec,
                    best_val_acc=val_acc,
                    test_acc=test_acc,
                )
            except Exception:
                duration_sec = time.time() - start_time
                traceback_text = traceback.format_exc()
                print("\n组合失败，完整 traceback 如下：")
                print(traceback_text, end="")
                return result_row(
                    combo,
                    "failed",
                    duration_sec,
                    error=traceback_text.strip().splitlines()[-1],
                )
            finally:
                # 这些变量可能包含整份数据集的多个副本，必须逐组合释放。
                if final_data is not None:
                    del final_data
                if processed_data is not None:
                    del processed_data
                if split is not None:
                    del split
                if loaders is not None:
                    del loaders
                if model is not None:
                    del model
                if device is not None:
                    del device
                clear_runtime_memory()


def record_shared_preprocess_failure(
    combinations: list[dict],
    combo_total: int,
    stage_name: str,
    traceback_text: str,
    duration_sec: float,
    prefix_index: int,
    prefix_name: str,
) -> None:
    """共享预处理失败时，为每个待跑组合分别写日志和summary。"""
    for combo in combinations:
        output_dir = output_dir_for(combo)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "train_process.txt").open("w", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
                print("\n" + "=" * 80)
                print(
                    f"当前算法组合: {combo['combo_index']}/{combo_total}"
                    f" | combo_id={combo['combo_id']}"
                    f" | model={MODEL_NAME}"
                )
                print(f"combo_name: {combo['combo_name']}")
                print(f"公共前缀: {prefix_index}/40 | {prefix_name}")
                print("公共前缀 pipeline_steps（本组只处理一次）:")
                print(
                    json.dumps(
                        {
                            key: combo["pipeline_steps"][key]
                            for key in ("denoise", "outliers", "calibrate")
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                print(f"失败阶段: {stage_name}")
                print(f"实际执行顺序: {actual_execution_order(combo)}")
                print("当前组合的完整 pipeline_steps:")
                print(json.dumps(combo["pipeline_steps"], ensure_ascii=False, indent=2))
                print("=" * 80)
                print("共享预处理失败，当前组合未进入训练。完整 traceback 如下：")
                print(traceback_text, end="" if traceback_text.endswith("\n") else "\n")

        append_summary(
            result_row(
                combo,
                "failed",
                duration_sec,
                error=traceback_text.strip().splitlines()[-1],
            )
        )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 Widar 数据目录: {DATA_PATH}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    combinations = build_pipeline_combinations()
    prefix_groups = build_prefix_groups(combinations)
    combo_total = len(combinations)
    done_records = load_done_records()
    params = load_params(DATASET_NAME)

    print(f"Widar 全量 pipeline 组合数: {combo_total}")
    print(f"公共前缀数: {len(prefix_groups)}（每个前缀派生8组）")
    print(f"固定模型: {MODEL_NAME}")
    print(f"结果目录: {RESULT_DIR}")
    print(f"summary: {SUMMARY_PATH}")
    print(f"已成功完成组合数: {len(done_records)}")

    set_seed(SEED)

    # 同一个常驻 spawn 池负责一次性原始解析和后续全部公共前缀/尾部处理。
    # 它不会在读取大数据后 fork，也不会在首次训练初始化 CUDA 后再 fork。
    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=PREPROCESS_WORKERS,
        mp_context=mp_context,
    ) as executor:
        raw_data, labels, groups, unique_labels, metadata = (
            load_and_parse_raw_data_once(executor)
        )

        for prefix_group in prefix_groups:
            prefix_index = prefix_group["prefix_index"]
            prefix_name = prefix_group["prefix_name"]
            group_combos = prefix_group["combinations"]
            pending = [
                combo for combo in group_combos if combo["combo_id"] not in done_records
            ]

            if not pending:
                print(
                    f"\n跳过公共前缀 {prefix_index}/40: {prefix_name}"
                    " | 8个组合均已成功完成"
                )
                continue

            for combo in group_combos:
                if combo["combo_id"] in done_records:
                    print(
                        f"跳过成功记录: {combo['combo_index']}/{combo_total}"
                        f" | {combo['combo_id']} | {combo['combo_name']}"
                    )

            print("\n" + "#" * 80)
            print(
                f"处理公共前缀 {prefix_index}/40: {prefix_name}"
                f" | 本组待运行 {len(pending)}/8"
            )
            print(json.dumps(prefix_group["prefix_steps"], ensure_ascii=False, indent=2))
            print("#" * 80)

            prefix_data = None
            minmax_data = None
            prefix_start = time.time()
            try:
                # denoise + outliers + calibrate 在本组8个组合之间只执行一次。
                prefix_data = parallel_execute_steps(
                    executor,
                    raw_data,
                    prefix_group["prefix_steps"],
                )
                prefix_duration_sec = time.time() - prefix_start
                print(
                    f"公共前缀 {prefix_index}/40 处理完成，"
                    f"耗时 {prefix_duration_sec:.2f}s"
                )
            except Exception:
                prefix_duration_sec = time.time() - prefix_start
                traceback_text = traceback.format_exc()
                print(traceback_text, end="")
                record_shared_preprocess_failure(
                    pending,
                    combo_total,
                    "denoise -> outliers -> calibrate 公共前缀",
                    traceback_text,
                    prefix_duration_sec,
                    prefix_index,
                    prefix_name,
                )
                if prefix_data is not None:
                    del prefix_data
                clear_runtime_memory()
                continue

            try:
                # z-score 四组：必须分别 interpolate 后再做带符号幅度+相位输出。
                zscore_pending = [
                    combo for combo in pending if combo["normalize"] == "z-score"
                ]
                for combo in zscore_pending:
                    print(
                        f"\n开始组合: {combo['combo_index']}/{combo_total}"
                        f" | {combo['combo_id']} | {combo['combo_name']}"
                    )
                    row = run_one_combo(
                        combo,
                        combo_total,
                        prefix_data,
                        "公共前缀缓存（当前组合再执行 interpolate -> z-score）",
                        executor,
                        labels,
                        groups,
                        unique_labels,
                        metadata,
                        params,
                        prefix_index,
                        prefix_name,
                        prefix_duration_sec,
                    )
                    append_summary(row)
                    if row["status"] == "ok":
                        done_records.add(combo["combo_id"])

                # min-max 四组：公共前缀上只归一化一次，然后分别插值。
                minmax_pending = [
                    combo for combo in pending if combo["normalize"] == "min-max"
                ]
                if minmax_pending:
                    minmax_start = time.time()
                    try:
                        minmax_data = parallel_execute_steps(
                            executor,
                            prefix_data,
                            {"normalize": {"method": "min-max"}},
                        )
                        print(
                            "min-max共享缓存处理完成，耗时 "
                            f"{time.time() - minmax_start:.2f}s"
                        )
                    except Exception:
                        minmax_duration_sec = time.time() - minmax_start
                        traceback_text = traceback.format_exc()
                        print(traceback_text, end="")
                        record_shared_preprocess_failure(
                            minmax_pending,
                            combo_total,
                            "公共前缀 -> min-max 共享缓存",
                            traceback_text,
                            minmax_duration_sec,
                            prefix_index,
                            prefix_name,
                        )
                        minmax_pending = []

                    # min-max共享缓存已经独立，后续四组不再需要复数公共前缀。
                    # 立即释放它，避免训练期间同时常驻两份全数据缓存。
                    if prefix_data is not None:
                        del prefix_data
                        prefix_data = None
                        clear_runtime_memory()

                    for combo in minmax_pending:
                        print(
                            f"\n开始组合: {combo['combo_index']}/{combo_total}"
                            f" | {combo['combo_id']} | {combo['combo_name']}"
                        )
                        row = run_one_combo(
                            combo,
                            combo_total,
                            minmax_data,
                            "公共前缀+min-max共享缓存（当前组合只执行 interpolate）",
                            executor,
                            labels,
                            groups,
                            unique_labels,
                            metadata,
                            params,
                            prefix_index,
                            prefix_name,
                            prefix_duration_sec,
                        )
                        append_summary(row)
                        if row["status"] == "ok":
                            done_records.add(combo["combo_id"])
            finally:
                if minmax_data is not None:
                    del minmax_data
                if prefix_data is not None:
                    del prefix_data
                clear_runtime_memory()

    del raw_data
    clear_runtime_memory()
    print(f"\n全部可运行组合处理结束，汇总已保存到: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
