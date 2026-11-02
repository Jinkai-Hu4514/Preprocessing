"""Plot Stage-I test accuracy for 19 models under six preset pipelines.

The figure is intentionally a connected-dot plot rather than a conventional
continuous line chart: model names are discrete categories, and the light
lines only help the reader follow one preset across models.  Only successful
(`status == "ok"`) runs are plotted.  Failed or skipped runs remain missing
and are never encoded as zero accuracy.

Outputs
-------
paper_figures/output/四个数据集19种模型六种预设测试准确率_四宫格.svg
paper_figures/output/四个数据集19种模型六种预设测试准确率_四宫格.png
paper_figures/output/四个数据集19种模型六种预设测试准确率_纵向.svg
paper_figures/output/四个数据集19种模型六种预设测试准确率_纵向.png
paper_figures/derived_data/stage1_19models_6presets_accuracy.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PACKAGE_DIR / "output"
DERIVED_DIR = PACKAGE_DIR / "derived_data"

STAGE1_FILES = {
    "Widar": ROOT
    / "SDP/test_wider/result/preset_tests/condition_v2_all_presets_models_summary.csv",
    "Gait": ROOT
    / "SDP/test_gait/result/preset_tests/user_id_v2_all_presets_models_summary.csv",
    "XRF55": ROOT
    / "SDP/test_xrf55/result/preset_tests/"
    "train_norm_nosplit_repetition_all_presets_models_summary.csv",
    "ElderAL": ROOT
    / "SDP/test_elderAL/result/preset_tests/"
    "source_action_position_all_presets_models_summary.csv",
}

DATASET_ORDER = ["Widar", "Gait", "XRF55", "ElderAL"]

PRESET_ORDER = [
    "high_quality",
    "fast",
    "robust",
    "gesture_recognition",
    "activity_detection",
    "localization",
]

PRESET_LABELS = {
    "high_quality": "High Quality",
    "fast": "Fast",
    "robust": "Robust",
    "gesture_recognition": "Gesture Recognition",
    "activity_detection": "Activity Detection",
    "localization": "Localization",
}

# Okabe-Ito-derived palette: distinguishable under common colour-vision
# deficiencies.  Marker shape and line pattern provide redundant encoding.
PRESET_STYLES = {
    "high_quality": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "fast": {"color": "#E69F00", "marker": "s", "linestyle": "--"},
    "robust": {"color": "#009E73", "marker": "^", "linestyle": "-."},
    "gesture_recognition": {
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": ":",
    },
    "activity_detection": {
        "color": "#D55E00",
        "marker": "P",
        "linestyle": (0, (5, 2)),
    },
    "localization": {
        "color": "#56B4E9",
        "marker": "X",
        "linestyle": (0, (2, 1, 1, 1)),
    },
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


def configure_matplotlib() -> None:
    """Configure a publication-oriented Matplotlib style."""

    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is None:
        raise FileNotFoundError("No supported publication font found")

    fontManager.addfont(str(font_path))
    font_name = FontProperties(fname=str(font_path)).get_name()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, "Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "text.color": "#202020",
            "axes.labelcolor": "#202020",
            "axes.titlecolor": "#202020",
            "xtick.color": "#3A3A3A",
            "ytick.color": "#3A3A3A",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",  # Keep SVG labels editable.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
        }
    )


def load_stage1() -> pd.DataFrame:
    """Load, audit, and combine the four complete 19-by-6 registries."""

    frames: list[pd.DataFrame] = []
    required = {"preset", "model", "status", "test_acc"}

    for dataset in DATASET_ORDER:
        path = STAGE1_FILES[dataset]
        frame = pd.read_csv(path)
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")
        if len(frame) != len(PRESET_ORDER) * len(MODEL_ORDER):
            raise ValueError(f"{path} expected 114 rows, found {len(frame)}")
        if frame.duplicated(["preset", "model"]).any():
            raise ValueError(f"{path} contains duplicate preset/model rows")
        if set(frame["preset"]) != set(PRESET_ORDER):
            raise ValueError(f"{path} preset set differs from expected set")
        if set(frame["model"]) != set(MODEL_ORDER):
            raise ValueError(f"{path} model set differs from expected set")

        frame = frame.copy()
        frame["status"] = frame["status"].astype(str).str.strip().str.lower()
        if not set(frame["status"]).issubset({"ok", "failed", "skipped"}):
            raise ValueError(f"{path} contains an unknown task status")

        raw_accuracy = pd.to_numeric(frame["test_acc"], errors="coerce")
        valid = frame["status"].eq("ok")
        if raw_accuracy[valid].isna().any():
            raise ValueError(f"{path} has successful runs without test accuracy")

        max_valid = float(raw_accuracy[valid].max())
        scale = 100.0 if max_valid <= 1.000001 else 1.0
        frame["test_acc_pct"] = raw_accuracy * scale
        frame.loc[~valid, "test_acc_pct"] = np.nan
        if not frame.loc[valid, "test_acc_pct"].between(0, 100).all():
            raise ValueError(f"{path} has accuracy outside [0, 100]")

        frame.insert(0, "dataset", dataset)
        frame["source_file"] = path.relative_to(ROOT).as_posix()
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    output_columns = [
        "dataset",
        "preset",
        "model",
        "status",
        "test_acc",
        "test_acc_pct",
        "source_file",
    ]
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    combined[output_columns].to_csv(
        DERIVED_DIR / "stage1_19models_6presets_accuracy.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return combined


def build_figure(data: pd.DataFrame, layout: str = "grid") -> mpl.figure.Figure:
    """Build a four-panel connected-dot figure in grid or stacked layout."""

    if layout == "grid":
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(15.8, 11.2),
            sharey=True,
            constrained_layout=False,
        )
    elif layout == "stacked":
        figure, axes = plt.subplots(
            4,
            1,
            figsize=(7.2, 10.0),
            sharex=True,
            sharey=True,
            constrained_layout=False,
        )
    else:
        raise ValueError("layout must be 'grid' or 'stacked'")

    axes_flat = np.asarray(axes).ravel()
    x_positions = np.arange(len(MODEL_ORDER), dtype=float)
    # A small horizontal dodge exposes coincident markers.  It does not encode
    # another variable; category centres remain the integer tick positions.
    offsets = np.linspace(-0.14, 0.14, len(PRESET_ORDER))
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    legend_handles = []

    for panel_index, (axis, dataset) in enumerate(zip(axes_flat, DATASET_ORDER)):
        subset = data[data["dataset"].eq(dataset)].copy()
        status_matrix = (
            subset.pivot(index="model", columns="preset", values="status")
            .reindex(index=MODEL_ORDER, columns=PRESET_ORDER)
        )

        for preset_index, preset in enumerate(PRESET_ORDER):
            style = PRESET_STYLES[preset]
            values = (
                subset[subset["preset"].eq(preset)]
                .set_index("model")["test_acc_pct"]
                .reindex(MODEL_ORDER)
                .to_numpy(dtype=float)
            )
            (line,) = axis.plot(
                x_positions + offsets[preset_index],
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.0,
                marker=style["marker"],
                markersize=4.7,
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.55,
                alpha=0.84,
                zorder=3,
                label=PRESET_LABELS[preset],
            )
            if panel_index == 0:
                legend_handles.append(line)

        axis.set_title(
            f"{panel_labels[panel_index]} {dataset}",
            loc="left",
            pad=5,
            fontweight="semibold",
        )
        axis.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
        axis.set_ylim(0, 100)
        axis.set_xticks(x_positions)
        show_x_labels = layout == "grid" or panel_index == len(DATASET_ORDER) - 1
        if show_x_labels:
            axis.set_xticklabels(
                [MODEL_LABELS[model] for model in MODEL_ORDER],
                rotation=48,
                ha="right",
                rotation_mode="anchor",
            )
        else:
            axis.tick_params(axis="x", which="both", labelbottom=False)
        axis.yaxis.set_major_locator(MultipleLocator(20))
        axis.yaxis.set_minor_locator(MultipleLocator(10))
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        axis.grid(axis="y", which="major", color="#D8D8D8", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", length=3.0, pad=2.5)
        axis.tick_params(axis="y", which="major", length=3.2)
        axis.tick_params(axis="y", which="minor", length=2.0)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")

    figure.legend(
        handles=legend_handles,
        labels=[PRESET_LABELS[preset] for preset in PRESET_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6 if layout == "grid" else 3,
        frameon=False,
        handlelength=3.2,
        columnspacing=1.7,
        handletextpad=0.6,
    )
    figure.supylabel(
        "Test Accuracy (%)",
        x=0.016 if layout == "grid" else 0.012,
        fontsize=11,
    )
    figure.supxlabel("Model", y=0.025 if layout == "grid" else 0.025, fontsize=11)
    if layout == "grid":
        figure.subplots_adjust(
            left=0.062,
            right=0.992,
            top=0.92,
            bottom=0.125,
            hspace=0.58,
            wspace=0.075,
        )
    else:
        figure.subplots_adjust(
            left=0.105,
            right=0.985,
            top=0.885,
            bottom=0.12,
            hspace=0.28,
        )
    return figure


def save_figure(figure: mpl.figure.Figure, stem_name: str) -> tuple[Path, Path]:
    """Save an editable SVG and a high-resolution PNG preview."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / stem_name
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    metadata = {
        "Title": "Test accuracy of 19 models under six presets across four datasets",
        "Description": (
            "Stage-I test accuracy. Only successful runs are plotted; "
            "failed and skipped tasks remain missing."
        ),
        "Creator": "Matplotlib; generated by generate_stage1_accuracy_lines.py",
    }
    figure.savefig(svg_path, format="svg", metadata=metadata, facecolor="white")
    figure.savefig(png_path, format="png", dpi=400, facecolor="white")
    plt.close(figure)
    return svg_path, png_path


def main() -> None:
    configure_matplotlib()
    data = load_stage1()
    grid_figure = build_figure(data, layout="grid")
    grid_svg, grid_png = save_figure(
        grid_figure,
        "四个数据集19种模型六种预设测试准确率_四宫格",
    )
    stacked_figure = build_figure(data, layout="stacked")
    stacked_svg, stacked_png = save_figure(
        stacked_figure,
        "四个数据集19种模型六种预设测试准确率_纵向",
    )

    summary = (
        data.assign(valid=data["status"].eq("ok"))
        .groupby("dataset", sort=False)
        .agg(
            total_tasks=("status", "size"),
            valid_tasks=("valid", "sum"),
            minimum_accuracy=("test_acc_pct", "min"),
            maximum_accuracy=("test_acc_pct", "max"),
        )
        .reindex(DATASET_ORDER)
    )
    print(summary.round(2).to_string())
    print(f"\nGrid SVG: {grid_svg}")
    print(f"Grid PNG: {grid_png}")
    print(f"Stacked SVG: {stacked_svg}")
    print(f"Stacked PNG: {stacked_png}")
    print(
        "Data: "
        f"{DERIVED_DIR / 'stage1_19models_6presets_accuracy.csv'}"
    )


if __name__ == "__main__":
    main()
