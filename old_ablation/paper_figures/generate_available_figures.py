"""Generate publication-ready figures that do not require new experiments.

Outputs
-------
paper_figures/output/
    fig01_benchmark_framework.{png,pdf}
    fig02_stage1_model_preset_heatmaps.{png,pdf}
    fig03_stage2_complete_distributions.{png,pdf}
    fig04_matched_component_effects.{png,pdf}
paper_figures/derived_data/
    stage1_model_preset_long.csv
    stage2_complete_configurations.csv
    matched_component_effects.csv
paper_figures/figure_manifest.json

The script deliberately excludes Gait from inferential-looking Stage-2 plots:
only 253 of 320 configurations are currently present and the missing tail is
not random.  It also labels configuration bootstrap intervals as descriptive,
not as random-seed uncertainty.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PACKAGE_DIR / "output"
DERIVED_DIR = PACKAGE_DIR / "derived_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_DIR.mkdir(parents=True, exist_ok=True)


STAGE1_FILES = {
    "Widar": ROOT / "SDP/test_wider/result/preset_tests/condition_v2_all_presets_models_summary.csv",
    "Gait": ROOT / "SDP/test_gait/result/preset_tests/user_id_v2_all_presets_models_summary.csv",
    "XRF55": ROOT / "SDP/test_xrf55/result/preset_tests/train_norm_nosplit_repetition_all_presets_models_summary.csv",
    "ElderAL": ROOT / "SDP/test_elderAL/result/preset_tests/source_action_position_all_presets_models_summary.csv",
}

STAGE2_COMPLETE_FILES = {
    "Widar": ROOT / "SDP/test_wider/result/full_tests_new/widar_320_pipeline_optimized_mlpmodel_summary.csv",
    "XRF55": ROOT / "SDP/test_xrf55/result/full_tests/xrf55_80_pipeline_resnet1d_summary.csv",
    "ElderAL": ROOT / "SDP/test_elderAL/result/full_tests/elderAL_80_pipeline_csitime_summary.csv",
}

GAIT_INCOMPLETE_FILE = ROOT / "SDP/test_gait/result/full_tests_new/gait_320_pipeline_optimized_mlpmodel_summary.csv"

PRESET_ORDER = [
    "high_quality",
    "fast",
    "robust",
    "gesture_recognition",
    "activity_detection",
    "localization",
]

PRESET_LABELS = {
    "high_quality": "HQ",
    "fast": "Fast",
    "robust": "Robust",
    "gesture_recognition": "Gesture",
    "activity_detection": "Activity",
    "localization": "Local.",
}

MODEL_ORDER = [
    "csimodel",
    "mlpmodel",
    "cnn1dmodel",
    "cnn2dmodel",
    "resnet1d",
    "resnet2d",
    "lstmmodel",
    "bilstmattention",
    "attentiongru",
    "csitime",
    "ei",
    "fewsense",
    "pa_csi",
    "that",
    "wiflexformer",
    "efficientnetcsi",
    "mambacsi",
    "graphneuralcsi",
    "visiontransformercsi",
]

MODEL_LABELS = {
    "csimodel": "CSIModel",
    "mlpmodel": "MLP",
    "cnn1dmodel": "CNN1D",
    "cnn2dmodel": "CNN2D",
    "resnet1d": "ResNet1D",
    "resnet2d": "ResNet2D",
    "lstmmodel": "LSTM",
    "bilstmattention": "BiLSTM-Attn",
    "attentiongru": "Attn-GRU",
    "csitime": "CSITime",
    "ei": "EI",
    "fewsense": "FewSense",
    "pa_csi": "PA-CSI",
    "that": "THAT",
    "wiflexformer": "WiFlexFormer",
    "efficientnetcsi": "EfficientNet-CSI",
    "mambacsi": "Mamba-CSI",
    "graphneuralcsi": "GNN-CSI",
    "visiontransformercsi": "ViT-CSI",
}

DATASET_COLORS = {
    "Widar": "#0077BB",
    "XRF55": "#EE7733",
    "ElderAL": "#009988",
}

DATASET_MARKERS = {"Widar": "o", "XRF55": "s", "ElderAL": "^"}

FACTOR_COLORS = {
    "denoise": "#0077BB",
    "outliers": "#009988",
    "calibrate": "#EE7733",
    "normalize": "#CC3311",
    "interpolate_method": "#AA3377",
}


def configure_matplotlib() -> FontProperties:
    """Use a CJK-capable font and vector-friendly PDF settings."""

    candidates = [
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is None:
        raise FileNotFoundError("No supported CJK font found in C:/Windows/Fonts")
    fontManager.addfont(str(font_path))
    font_prop = FontProperties(fname=str(font_path))
    font_name = font_prop.get_name()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )
    return font_prop


FONT_PROP = configure_matplotlib()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: mpl.figure.Figure, stem: str) -> dict[str, str]:
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return {
        "png": str(png_path.relative_to(ROOT)).replace("\\", "/"),
        "pdf": str(pdf_path.relative_to(ROOT)).replace("\\", "/"),
        "png_sha256": hash_file(png_path),
        "pdf_sha256": hash_file(pdf_path),
    }


def load_stage1() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset, path in STAGE1_FILES.items():
        frame = pd.read_csv(path)
        required = {"preset", "model", "status", "test_acc"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if len(frame) != 114:
            raise ValueError(f"{path} expected 114 rows, found {len(frame)}")
        if frame.duplicated(["preset", "model"]).any():
            raise ValueError(f"{path} has duplicate preset/model rows")
        if set(frame["preset"]) != set(PRESET_ORDER):
            raise ValueError(f"{path} preset set differs from expected set")
        if set(frame["model"]) != set(MODEL_ORDER):
            raise ValueError(f"{path} model set differs from expected set")
        frame = frame.copy()
        frame.insert(0, "dataset", dataset)
        frame["test_acc_pct"] = pd.to_numeric(frame["test_acc"], errors="coerce") * 100.0
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(DERIVED_DIR / "stage1_model_preset_long.csv", index=False, encoding="utf-8-sig")
    return result


def _strip_interpolation_target(value: str) -> str:
    return re.sub(r"\d+$", "", str(value))


def load_stage2_complete() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    expected_rows = {"Widar": 320, "XRF55": 80, "ElderAL": 80}
    for dataset, path in STAGE2_COMPLETE_FILES.items():
        frame = pd.read_csv(path)
        required = {
            "combo_index",
            "combo_id",
            "status",
            "test_acc",
            "denoise",
            "outliers",
            "normalize",
            "interpolate",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if len(frame) != expected_rows[dataset]:
            raise ValueError(
                f"{path} expected {expected_rows[dataset]} rows, found {len(frame)}"
            )
        if set(frame["status"]) != {"ok"}:
            raise ValueError(f"{path} contains non-ok Stage-2 rows")
        if frame["combo_index"].nunique() != expected_rows[dataset]:
            raise ValueError(f"{path} combo_index is not unique")
        frame = frame.copy()
        frame.insert(0, "dataset", dataset)
        frame["test_acc"] = pd.to_numeric(frame["test_acc"], errors="raise")
        frame["test_acc_pct"] = frame["test_acc"] * 100.0
        frame["interpolate_method"] = frame["interpolate"].map(
            _strip_interpolation_target
        )
        if "calibrate" not in frame.columns:
            frame["calibrate"] = pd.NA
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(
        DERIVED_DIR / "stage2_complete_configurations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gait = pd.read_csv(GAIT_INCOMPLETE_FILE)
    if len(gait) != 253 or gait["combo_index"].nunique() != 253:
        raise ValueError(
            "Gait completion guard changed. Re-audit before including or excluding it."
        )
    present = set(pd.to_numeric(gait["combo_index"], errors="raise").astype(int))
    missing = sorted(set(range(1, 321)) - present)
    if missing != list(range(254, 321)):
        raise ValueError(
            "Gait missing-index pattern changed. Re-audit before generating figures."
        )
    return result


def add_box(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 8.0,
    linewidth: float = 1.0,
    radius: float = 0.015,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        transform=ax.transAxes,
        linespacing=1.22,
    )
    return patch


def add_arrow(
    ax: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#555555",
    connectionstyle: str = "arc3,rad=0",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.1,
        color=color,
        connectionstyle=connectionstyle,
        transform=ax.transAxes,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def figure_01_framework() -> tuple[mpl.figure.Figure, dict]:
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    ax.set_axis_off()

    blue_fill = "#E5F2FA"
    teal_fill = "#E3F3EF"
    orange_fill = "#FCEBDD"
    purple_fill = "#F2E8F3"
    gray_fill = "#F1F1F1"

    ax.text(0.105, 0.96, "异构 CSI 数据", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(0.43, 0.96, "统一且语义感知的 SDP", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(0.80, 0.96, "两阶段系统评测", ha="center", va="center", fontsize=10, weight="bold")

    dataset_specs = [
        (0.79, "Widar\n手势 · 复数 CSI\n1500 × 15 × 6"),
        (0.60, "Gait\n用户识别 · 复数 CSI\n1500 × 15 × 6"),
        (0.41, "XRF55\n动作 · 带符号实数\n1000 × 15 × 9"),
        (0.22, "ElderAL\n动作 · 实数幅度\n80 × 64 × 3"),
    ]
    for y, label in dataset_specs:
        add_box(ax, 0.02, y, 0.17, 0.135, label, blue_fill, "#0077BB", fontsize=7.6)

    add_box(
        ax,
        0.27,
        0.77,
        0.32,
        0.13,
        "数据集适配与 group-aware 划分\n固定样本、标签、split 与训练统计量",
        teal_fill,
        "#009988",
        fontsize=8.0,
    )
    ax.text(0.43, 0.695, "可插拔预处理管线", ha="center", va="center", fontsize=9, weight="bold")

    modules = [
        ("去噪", "#E5F2FA", "#0077BB"),
        ("异常值", "#E3F3EF", "#009988"),
        ("相位*", "#FCEBDD", "#EE7733"),
        ("归一化", "#FBE5E1", "#CC3311"),
        ("插值", "#F2E8F3", "#AA3377"),
    ]
    x_positions = np.linspace(0.265, 0.535, len(modules))
    for idx, (x, (label, face, edge)) in enumerate(zip(x_positions, modules)):
        add_box(ax, float(x), 0.56, 0.055, 0.085, label, face, edge, fontsize=7.2, radius=0.009)
        if idx < len(modules) - 1:
            add_arrow(
                ax,
                (float(x) + 0.061, 0.602),
                (float(x_positions[idx + 1]) - 0.010, 0.602),
                color="#777777",
            )

    add_box(
        ax,
        0.27,
        0.38,
        0.32,
        0.115,
        "输入语义保持\n复数：幅度 + 相位；实数：保留归一化符号",
        gray_fill,
        "#666666",
        fontsize=7.8,
    )
    add_box(
        ax,
        0.27,
        0.20,
        0.32,
        0.105,
        "统一训练与评测协议\n固定模型超参数 · Accuracy · 配置追溯",
        gray_fill,
        "#666666",
        fontsize=7.8,
    )

    add_box(
        ax,
        0.66,
        0.71,
        0.30,
        0.18,
        "阶段 1：广度评测\n6 presets × 19 models\n每数据集 114 个任务\n→ 模型–预处理依赖与覆盖率",
        orange_fill,
        "#EE7733",
        fontsize=8.0,
    )
    add_box(
        ax,
        0.66,
        0.44,
        0.30,
        0.18,
        "阶段 2：细粒度评测\n固定代表模型\n复数 320 / 实数 80 组\n→ 组件效应与交互",
        purple_fill,
        "#AA3377",
        fontsize=8.0,
    )
    add_box(
        ax,
        0.66,
        0.16,
        0.30,
        0.18,
        "机制诊断与选择指南\n波形、统计和针对性消融\n→ 解释何时有效、何时失效",
        teal_fill,
        "#009988",
        fontsize=8.0,
    )

    for y in [0.857, 0.667, 0.477, 0.287]:
        add_arrow(ax, (0.195, y), (0.265, 0.835), color="#777777", connectionstyle="arc3,rad=0.12")
    add_arrow(ax, (0.43, 0.765), (0.43, 0.65), color="#555555")
    add_arrow(ax, (0.43, 0.555), (0.43, 0.50), color="#555555")
    add_arrow(ax, (0.43, 0.375), (0.43, 0.31), color="#555555")
    add_arrow(ax, (0.595, 0.43), (0.655, 0.53), color="#777777")
    add_arrow(ax, (0.595, 0.82), (0.655, 0.80), color="#777777")
    add_arrow(ax, (0.81, 0.705), (0.81, 0.625), color="#777777")
    add_arrow(ax, (0.81, 0.435), (0.81, 0.345), color="#777777")

    ax.text(
        0.43,
        0.11,
        "* 相位校准仅用于保留复数 CSI 的 Widar 与 Gait",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#555555",
    )
    ax.text(0.015, 0.915, "(a)", fontsize=9, weight="bold")
    ax.text(0.255, 0.915, "(b)", fontsize=9, weight="bold")
    ax.text(0.645, 0.915, "(c)", fontsize=9, weight="bold")

    trace = {
        "artifact_id": "fig-01",
        "caption": "异构 CSI 预处理基准的两阶段评测框架。",
        "claim": "本文将异构数据适配、语义感知的可插拔预处理、跨模型广度评测和代表模型下的细粒度评测组织为统一框架。",
        "limitations": ["该图是方法结构图，不编码性能结果。"],
    }
    return fig, trace


def figure_02_stage1_heatmaps(stage1: pd.DataFrame) -> tuple[mpl.figure.Figure, dict]:
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 9.2))
    axes_flat = axes.ravel()
    cmap = mpl.colormaps["cividis"].copy()
    cmap.set_bad("#F2F2F2")
    norm = mpl.colors.Normalize(vmin=15, vmax=95)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for panel_idx, (ax, dataset) in enumerate(zip(axes_flat, STAGE1_FILES)):
        subset = stage1[stage1["dataset"] == dataset].copy()
        status_matrix = (
            subset.pivot(index="model", columns="preset", values="status")
            .reindex(index=MODEL_ORDER, columns=PRESET_ORDER)
        )
        acc_matrix = (
            subset.pivot(index="model", columns="preset", values="test_acc_pct")
            .reindex(index=MODEL_ORDER, columns=PRESET_ORDER)
            .to_numpy(dtype=float)
        )

        image = ax.imshow(acc_matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        for row_idx, model in enumerate(MODEL_ORDER):
            for col_idx, preset in enumerate(PRESET_ORDER):
                status = status_matrix.loc[model, preset]
                if status == "failed":
                    patch = Rectangle(
                        (col_idx - 0.5, row_idx - 0.5),
                        1,
                        1,
                        facecolor="#F4D9D4",
                        edgecolor="#8C2D1F",
                        hatch="////",
                        linewidth=0.65,
                    )
                    ax.add_patch(patch)
                elif status == "skipped":
                    patch = Rectangle(
                        (col_idx - 0.5, row_idx - 0.5),
                        1,
                        1,
                        facecolor="#E5E5E5",
                        edgecolor="#666666",
                        hatch="....",
                        linewidth=0.55,
                    )
                    ax.add_patch(patch)

        ax.set_xticks(range(len(PRESET_ORDER)))
        ax.set_xticklabels([PRESET_LABELS[item] for item in PRESET_ORDER], rotation=38, ha="right")
        ax.set_yticks(range(len(MODEL_ORDER)))
        if panel_idx % 2 == 0:
            ax.set_yticklabels([MODEL_LABELS[item] for item in MODEL_ORDER])
            ax.set_ylabel("模型")
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ok_count = int((subset["status"] == "ok").sum())
        ax.set_title(f"{panel_labels[panel_idx]} {dataset}（有效任务 {ok_count}/114）", loc="left", pad=5)
        ax.set_xticks(np.arange(-0.5, len(PRESET_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.45)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("#777777")

    fig.subplots_adjust(left=0.17, right=0.985, top=0.94, bottom=0.115, hspace=0.22, wspace=0.08)
    legend_handles = [
        Patch(facecolor="#F4D9D4", edgecolor="#8C2D1F", hatch="////", label="运行失败"),
        Patch(facecolor="#E5E5E5", edgecolor="#666666", hatch="....", label="配置跳过"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.60, 0.985), ncol=2, frameon=False)
    cbar_ax = fig.add_axes([0.27, 0.055, 0.46, 0.018])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("测试准确率（%）")
    cbar.set_ticks([15, 35, 55, 75, 95])
    fig.text(0.82, 0.064, "预设管线", ha="center", va="center", fontsize=9)

    trace = {
        "artifact_id": "fig-02",
        "caption": "四个数据集上 19 种模型与 6 种预设管线的测试准确率及任务状态。",
        "claim": "模型性能同时依赖数据集、下游模型和预处理 preset，且名义设计空间中存在失败与跳过任务。",
        "limitations": [
            "颜色仅表示 status=ok 的单随机种子 Accuracy。",
            "failed/skipped 是覆盖率结果，不应解释为 0% Accuracy。",
            "不同模型的参数规模与资源需求不同。",
        ],
    }
    return fig, trace


def figure_03_stage2_distributions(stage2: pd.DataFrame) -> tuple[mpl.figure.Figure, dict]:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    datasets = ["Widar", "XRF55", "ElderAL"]
    positions = np.arange(1, len(datasets) + 1)
    values = [
        stage2.loc[stage2["dataset"] == dataset, "test_acc_pct"].to_numpy()
        for dataset in datasets
    ]

    violins = ax.violinplot(
        values,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.25,
    )
    for body, dataset in zip(violins["bodies"], datasets):
        body.set_facecolor(DATASET_COLORS[dataset])
        body.set_edgecolor(DATASET_COLORS[dataset])
        body.set_alpha(0.20)
        body.set_linewidth(1.0)

    rng = np.random.default_rng(20260812)
    for position, dataset, array in zip(positions, datasets, values):
        jitter = rng.normal(0.0, 0.055, size=array.size)
        jitter = np.clip(jitter, -0.16, 0.16)
        ax.scatter(
            np.full_like(array, position, dtype=float) + jitter,
            array,
            s=9,
            color=DATASET_COLORS[dataset],
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        q1, median, q3 = np.percentile(array, [25, 50, 75])
        minimum, maximum = float(array.min()), float(array.max())
        ax.plot([position, position], [minimum, maximum], color="#333333", linewidth=0.8, zorder=3)
        ax.add_patch(
            Rectangle(
                (position - 0.08, q1),
                0.16,
                q3 - q1,
                facecolor="white",
                edgecolor="#333333",
                linewidth=0.9,
                zorder=4,
            )
        )
        ax.plot([position - 0.08, position + 0.08], [median, median], color="#111111", linewidth=1.5, zorder=5)
        ax.plot([position - 0.05, position + 0.05], [minimum, minimum], color="#333333", linewidth=0.8, zorder=3)
        ax.plot([position - 0.05, position + 0.05], [maximum, maximum], color="#333333", linewidth=0.8, zorder=3)
        ax.text(
            position,
            98.2,
            f"n={array.size}\n{minimum:.1f} / {median:.1f} / {maximum:.1f}",
            ha="center",
            va="top",
            fontsize=8,
            linespacing=1.25,
        )

    ax.set_xlim(0.45, 3.55)
    ax.set_ylim(30, 100)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Widar\nMLP", "XRF55\nResNet1D", "ElderAL\nCSITime"])
    ax.set_ylabel("测试准确率（%）")
    ax.set_xlabel("数据集与固定代表模型")
    ax.set_yticks(np.arange(30, 101, 10))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.text(
        0.01,
        0.02,
        "标注顺序：最小值 / 中位数 / 最大值；每个点为一条预处理配置",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        color="#555555",
    )
    fig.tight_layout()

    trace = {
        "artifact_id": "fig-03",
        "caption": "三个已完成数据集上全部细粒度预处理配置的测试准确率分布。",
        "claim": "在固定模型、split 和训练设置时，仅改变预处理组合仍会导致宽幅的性能分布，而且分布形态具有明显数据集依赖性。",
        "limitations": [
            "每个点是 seed=42 下的配置条件，不是独立随机重复。",
            "Gait 仅完成 253/320 且缺失非随机，因此未纳入。",
            "不同数据集使用不同代表模型，图用于比较敏感性而非模型优劣。",
        ],
    }
    return fig, trace


CONTRASTS = [
    {
        "factor": "denoise",
        "reference": "savgol_w7_p3",
        "comparison": "wavelet",
        "label": "去噪 | Wavelet − Savitzky–Golay",
    },
    {
        "factor": "denoise",
        "reference": "savgol_w7_p3",
        "comparison": "butterworth_o5_c0.3",
        "label": "去噪 | Butterworth − Savitzky–Golay",
    },
    {
        "factor": "denoise",
        "reference": "savgol_w7_p3",
        "comparison": "bandpass_0.5-50",
        "label": "去噪 | Bandpass − Savitzky–Golay",
    },
    {
        "factor": "denoise",
        "reference": "savgol_w7_p3",
        "comparison": "hampel_w5_s3",
        "label": "去噪 | Hampel − Savitzky–Golay",
    },
    {
        "factor": "outliers",
        "reference": "iqr",
        "comparison": "outlier_z-score",
        "label": "异常值 | Z-score − IQR",
    },
    {
        "factor": "calibrate",
        "reference": "linear",
        "comparison": "polynomial_d3",
        "label": "相位 | Polynomial − Linear",
    },
    {
        "factor": "calibrate",
        "reference": "linear",
        "comparison": "stc",
        "label": "相位 | STC − Linear",
    },
    {
        "factor": "calibrate",
        "reference": "linear",
        "comparison": "robust",
        "label": "相位 | Robust − Linear",
    },
    {
        "factor": "normalize",
        "reference": "z-score",
        "comparison": "min-max",
        "label": "归一化 | Min–max − Z-score",
    },
    {
        "factor": "interpolate_method",
        "reference": "linear",
        "comparison": "cubic",
        "label": "插值 | Cubic − Linear",
    },
    {
        "factor": "interpolate_method",
        "reference": "linear",
        "comparison": "nearest",
        "label": "插值 | Nearest − Linear",
    },
    {
        "factor": "interpolate_method",
        "reference": "linear",
        "comparison": "decimate",
        "label": "插值 | Decimate − Linear",
    },
]


def bootstrap_mean_ci(values: np.ndarray, seed_text: str, n_boot: int = 10_000) -> tuple[float, float]:
    if values.size < 2:
        return float(values.mean()), float(values.mean())
    seed_bytes = hashlib.sha256(seed_text.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "little", signed=False)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[indices].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return float(low), float(high)


def calculate_matched_effects(stage2: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    factor_columns = ["denoise", "outliers", "calibrate", "normalize", "interpolate_method"]

    for dataset in ["Widar", "XRF55", "ElderAL"]:
        subset = stage2[stage2["dataset"] == dataset].copy()
        available_factors = [
            factor
            for factor in factor_columns
            if factor in subset.columns and subset[factor].notna().any()
        ]
        for contrast_index, contrast in enumerate(CONTRASTS):
            factor = contrast["factor"]
            if factor not in available_factors:
                continue
            reference = contrast["reference"]
            comparison = contrast["comparison"]
            if reference not in set(subset[factor].dropna()):
                continue
            if comparison not in set(subset[factor].dropna()):
                continue

            keys = [column for column in available_factors if column != factor]
            reference_rows = subset[subset[factor] == reference][keys + ["test_acc_pct"]].copy()
            comparison_rows = subset[subset[factor] == comparison][keys + ["test_acc_pct"]].copy()
            if reference_rows.duplicated(keys).any() or comparison_rows.duplicated(keys).any():
                raise ValueError(f"Non-unique matched keys for {dataset} / {contrast['label']}")
            paired = comparison_rows.merge(
                reference_rows,
                on=keys,
                how="inner",
                validate="one_to_one",
                suffixes=("_comparison", "_reference"),
            )
            if paired.empty:
                raise ValueError(f"No matched pairs for {dataset} / {contrast['label']}")
            differences = (
                paired["test_acc_pct_comparison"] - paired["test_acc_pct_reference"]
            ).to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_mean_ci(
                differences, f"{dataset}|{contrast['label']}|20260812"
            )
            rows.append(
                {
                    "contrast_order": contrast_index,
                    "dataset": dataset,
                    "factor": factor,
                    "label": contrast["label"],
                    "comparison": comparison,
                    "reference": reference,
                    "n_pairs": int(differences.size),
                    "mean_difference_pp": float(differences.mean()),
                    "median_difference_pp": float(np.median(differences)),
                    "bootstrap_ci_low_pp": ci_low,
                    "bootstrap_ci_high_pp": ci_high,
                    "min_difference_pp": float(differences.min()),
                    "max_difference_pp": float(differences.max()),
                }
            )
    effects = pd.DataFrame(rows).sort_values(["contrast_order", "dataset"]).reset_index(drop=True)
    effects.to_csv(
        DERIVED_DIR / "matched_component_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return effects


def figure_04_matched_effects(effects: pd.DataFrame) -> tuple[mpl.figure.Figure, dict]:
    fig, ax = plt.subplots(figsize=(7.5, 6.9))
    labels = [contrast["label"] for contrast in CONTRASTS]
    y_base = np.arange(len(labels), dtype=float)
    offsets = {"Widar": -0.21, "XRF55": 0.0, "ElderAL": 0.21}

    for dataset in ["Widar", "XRF55", "ElderAL"]:
        subset = effects[effects["dataset"] == dataset]
        for _, row in subset.iterrows():
            y = float(row["contrast_order"]) + offsets[dataset]
            mean = float(row["mean_difference_pp"])
            low = float(row["bootstrap_ci_low_pp"])
            high = float(row["bootstrap_ci_high_pp"])
            ax.errorbar(
                mean,
                y,
                xerr=np.array([[mean - low], [high - mean]]),
                fmt=DATASET_MARKERS[dataset],
                markersize=5.1,
                color=DATASET_COLORS[dataset],
                ecolor=DATASET_COLORS[dataset],
                elinewidth=1.0,
                capsize=2.2,
                capthick=0.9,
                markerfacecolor="white" if dataset == "XRF55" else DATASET_COLORS[dataset],
                markeredgewidth=1.0,
                zorder=3,
            )

    ax.axvline(0, color="#444444", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_yticks(y_base)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("相对参考水平的测试准确率差（百分点）")
    ax.set_ylabel("匹配组件对比")
    all_bounds = pd.concat(
        [effects["bootstrap_ci_low_pp"], effects["bootstrap_ci_high_pp"]]
    ).to_numpy(dtype=float)
    span = float(all_bounds.max() - all_bounds.min())
    padding = max(2.0, span * 0.06)
    ax.set_xlim(float(all_bounds.min() - padding), float(all_bounds.max() + padding))
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    for separator in [3.5, 4.5, 7.5, 8.5]:
        ax.axhline(separator, color="#B8B8B8", linewidth=0.55)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=DATASET_MARKERS[dataset],
            color=DATASET_COLORS[dataset],
            markerfacecolor="white" if dataset == "XRF55" else DATASET_COLORS[dataset],
            markeredgecolor=DATASET_COLORS[dataset],
            linestyle="none",
            markersize=5.5,
            label=dataset,
        )
        for dataset in ["Widar", "XRF55", "ElderAL"]
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, ncol=3)
    fig.text(
        0.5,
        0.012,
        "点为匹配配置差值的均值；误差条为配置层 bootstrap 95% CI（描述性，不是随机种子 CI）",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.39, right=0.98, top=0.98, bottom=0.12)

    trace = {
        "artifact_id": "fig-04",
        "caption": "三个完整数据集上预处理组件水平相对参考水平的匹配测试准确率差。",
        "claim": "组件影响具有显著的数据集依赖性；某些失配组件会造成大幅退化，而异常值与插值的平均影响通常更小。",
        "limitations": [
            "bootstrap 单位是匹配的配置对，不代表训练随机性。",
            "图中不提供显著性检验，也不把同一配置网格视为独立随机样本。",
            "Gait 因 67 个配置非随机缺失而未纳入。",
            "相位校准只适用于 Widar，因此相位行仅显示 Widar。",
        ],
    }
    return fig, trace


def build_manifest(
    traces: list[dict],
    output_records: dict[str, dict[str, str]],
    source_paths: list[Path],
) -> dict:
    source_records = []
    for path in source_paths:
        source_records.append(
            {
                "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": hash_file(path),
            }
        )
    trace_by_id = {trace["artifact_id"]: trace for trace in traces}
    artifacts = []
    for artifact_id, files in output_records.items():
        artifacts.append(
            {
                **trace_by_id[artifact_id],
                "files": files,
                "transformation": {
                    "script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
                    "script_sha256": hash_file(Path(__file__).resolve()),
                },
            }
        )
    return {
        "generated_on": "2026-08-12",
        "git_commit": "3063477",
        "source_files": source_records,
        "derived_files": [
            "paper_figures/derived_data/stage1_model_preset_long.csv",
            "paper_figures/derived_data/stage2_complete_configurations.csv",
            "paper_figures/derived_data/matched_component_effects.csv",
        ],
        "artifacts": artifacts,
        "global_limitations": [
            "主结果来自单随机种子 42。",
            "Gait 细粒度实验当前为 253/320，未进入 Fig. 3–4。",
            "所有置信区间若无特别说明均为配置层描述性 bootstrap，而非 seed CI。",
        ],
    }


def main() -> None:
    stage1 = load_stage1()
    stage2 = load_stage2_complete()
    effects = calculate_matched_effects(stage2)

    figure_builders = [
        ("fig-01", "fig01_benchmark_framework", lambda: figure_01_framework()),
        ("fig-02", "fig02_stage1_model_preset_heatmaps", lambda: figure_02_stage1_heatmaps(stage1)),
        ("fig-03", "fig03_stage2_complete_distributions", lambda: figure_03_stage2_distributions(stage2)),
        ("fig-04", "fig04_matched_component_effects", lambda: figure_04_matched_effects(effects)),
    ]
    traces: list[dict] = []
    output_records: dict[str, dict[str, str]] = {}
    for artifact_id, stem, builder in figure_builders:
        figure, trace = builder()
        traces.append(trace)
        output_records[artifact_id] = save_figure(figure, stem)

    manifest = build_manifest(
        traces,
        output_records,
        source_paths=[
            *STAGE1_FILES.values(),
            *STAGE2_COMPLETE_FILES.values(),
            GAIT_INCOMPLETE_FILE,
        ],
    )
    manifest_path = PACKAGE_DIR / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Generated {len(output_records)} figures in {OUTPUT_DIR}")
    print(f"Derived data written to {DERIVED_DIR}")
    print(f"Manifest written to {manifest_path}")
    print("\nMatched-effect summary (mean percentage-point difference):")
    summary = effects.pivot(index="label", columns="dataset", values="mean_difference_pp")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
