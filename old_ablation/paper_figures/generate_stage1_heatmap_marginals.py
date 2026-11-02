"""Generate the Stage-I model-by-preset heatmap with marginal means.

Each dataset is shown as a 6 x 19 matrix: preprocessing presets are rows,
models are columns, and color encodes test accuracy on a shared 0--1 scale.
The narrow chart to the right of each matrix summarizes the preset mean over
successful runs. Failed and skipped runs are explicitly hatched and are never
treated as zero accuracy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from generate_stage1_accuracy_lines import (
    DATASET_ORDER,
    MODEL_LABELS,
    MODEL_ORDER,
    load_stage1,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# This row order follows the supplied reference figure.
PRESET_ORDER = [
    "fast",
    "high_quality",
    "robust",
    "activity_detection",
    "gesture_recognition",
    "localization",
]

PRESET_LABELS = {
    "fast": "fast",
    "high_quality": "high_quality",
    "robust": "robust",
    "activity_detection": "activity_detection",
    "gesture_recognition": "gesture_recognition",
    "localization": "localization",
}

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

# [left, bottom, width, height] in figure coordinates. Explicit placement
# keeps all four matrices geometrically identical and leaves room for the
# long preset and model labels at final double-column size.
PANEL_LAYOUT = [
    ([0.120, 0.650, 0.280, 0.245], [0.412, 0.650, 0.047, 0.245]),
    ([0.575, 0.650, 0.280, 0.245], [0.867, 0.650, 0.047, 0.245]),
    ([0.120, 0.205, 0.280, 0.245], [0.412, 0.205, 0.047, 0.245]),
    ([0.575, 0.205, 0.280, 0.245], [0.867, 0.205, 0.047, 0.245]),
]


def configure_matplotlib() -> None:
    """Apply compact journal-figure typography."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 6.4,
            "axes.titlesize": 8.8,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 6.3,
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "savefig.bbox": None,
            "axes.linewidth": 0.55,
            "hatch.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _matrices(
    data: pd.DataFrame, dataset: str
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return accuracy and status matrices in display order."""

    subset = data[data["dataset"].eq(dataset)].copy()
    accuracy = (
        subset.pivot(index="preset", columns="model", values="test_acc_pct")
        .reindex(index=PRESET_ORDER, columns=MODEL_ORDER)
        .to_numpy(dtype=float)
        / 100.0
    )
    status = (
        subset.pivot(index="preset", columns="model", values="status")
        .reindex(index=PRESET_ORDER, columns=MODEL_ORDER)
    )
    expected_shape = (len(PRESET_ORDER), len(MODEL_ORDER))
    if accuracy.shape != expected_shape or status.shape != expected_shape:
        raise ValueError(
            f"Unexpected Stage-I matrix shape for {dataset}: "
            f"accuracy={accuracy.shape}, status={status.shape}"
        )
    return accuracy, status


def _annotation_cells(accuracy: np.ndarray, count: int = 3) -> list[tuple[int, int]]:
    """Select high-value cells while spreading labels across rows and columns."""

    candidates = [
        (float(accuracy[row, column]), row, column)
        for row in range(accuracy.shape[0])
        for column in range(accuracy.shape[1])
        if np.isfinite(accuracy[row, column])
    ]
    candidates.sort(reverse=True)

    selected: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_columns: set[int] = set()
    for _, row, column in candidates:
        if row in used_rows or column in used_columns:
            continue
        selected.append((row, column))
        used_rows.add(row)
        used_columns.add(column)
        if len(selected) == count:
            return selected

    # The fallback is only relevant if a dataset has too few distinct rows or
    # models with successful results.
    for _, row, column in candidates:
        if (row, column) in selected or row in used_rows:
            continue
        selected.append((row, column))
        used_rows.add(row)
        if len(selected) == count:
            break
    return selected


def _style_heatmap_axis(axis: mpl.axes.Axes) -> None:
    axis.set_xticks(np.arange(len(MODEL_ORDER)))
    axis.set_xticklabels(
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        rotation=58,
        ha="right",
        va="top",
        rotation_mode="anchor",
    )
    axis.set_yticks(np.arange(len(PRESET_ORDER)))
    axis.set_yticklabels([PRESET_LABELS[preset] for preset in PRESET_ORDER])
    axis.tick_params(axis="both", which="major", length=0, pad=1.6)

    axis.set_xticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(PRESET_ORDER), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.33)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#777777")
        spine.set_linewidth(0.55)


def _draw_status_hatching(
    axis: mpl.axes.Axes, status: pd.DataFrame
) -> None:
    """Mark all non-numeric cells without assigning them a numeric color."""

    for row_index, preset in enumerate(PRESET_ORDER):
        for column_index, model in enumerate(MODEL_ORDER):
            if status.loc[preset, model] == "ok":
                continue
            axis.add_patch(
                Rectangle(
                    (column_index - 0.5, row_index - 0.5),
                    1,
                    1,
                    facecolor="#F0F0F0",
                    edgecolor="#A0A0A0",
                    hatch="////",
                    linewidth=0.25,
                    zorder=3,
                )
            )


def _draw_annotations(axis: mpl.axes.Axes, accuracy: np.ndarray) -> None:
    for row, column in _annotation_cells(accuracy):
        value = float(accuracy[row, column])
        label = axis.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=5.7,
            weight="bold",
            zorder=5,
        )
        label.set_path_effects(
            [
                path_effects.Stroke(linewidth=0.85, foreground="#16324A"),
                path_effects.Normal(),
            ]
        )


def _draw_mean_axis(
    axis: mpl.axes.Axes, accuracy: np.ndarray, heatmap_axis: mpl.axes.Axes
) -> None:
    means = np.nanmean(accuracy, axis=1)
    rows = np.arange(len(PRESET_ORDER))
    axis.barh(
        rows,
        means,
        height=0.58,
        color="#4C9ED9",
        edgecolor="none",
        zorder=2,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(heatmap_axis.get_ylim())
    axis.set_yticks([])
    axis.set_xticks([0.0, 0.5, 1.0])
    axis.set_xticklabels(["0", ".5", "1"])
    axis.tick_params(axis="x", length=2.2, width=0.45, pad=1.2)
    axis.axvline(0.5, color="#DDDDDD", linewidth=0.45, zorder=0)
    axis.set_title("Preset\nmean", fontsize=6.4, pad=2.2, linespacing=0.9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#777777")
    axis.spines["bottom"].set_linewidth(0.5)


def build_figure(data: pd.DataFrame) -> mpl.figure.Figure:
    """Build the four-panel heatmap and marginal-mean figure."""

    figure = plt.figure(figsize=(7.2, 4.4), facecolor="white")
    colormap = mpl.colormaps["YlGnBu"].copy()
    colormap.set_bad("#F0F0F0")
    normalization = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    image = None

    for panel_index, dataset in enumerate(DATASET_ORDER):
        heat_position, mean_position = PANEL_LAYOUT[panel_index]
        heat_axis = figure.add_axes(heat_position)
        mean_axis = figure.add_axes(mean_position)
        accuracy, status = _matrices(data, dataset)
        image = heat_axis.imshow(
            np.ma.masked_invalid(accuracy),
            cmap=colormap,
            norm=normalization,
            aspect="auto",
            interpolation="nearest",
            zorder=1,
        )

        _style_heatmap_axis(heat_axis)
        _draw_status_hatching(heat_axis, status)
        _draw_annotations(heat_axis, accuracy)
        _draw_mean_axis(mean_axis, accuracy, heat_axis)
        heat_axis.set_title(
            f"{PANEL_LABELS[panel_index]}  {dataset}",
            loc="left",
            pad=4.5,
            weight="bold",
        )

    if image is None:
        raise RuntimeError("No Stage-I panels were generated")

    colorbar_axis = figure.add_axes([0.952, 0.205, 0.016, 0.690])
    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="vertical")
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    colorbar.set_ticklabels(["0", ".2", ".4", ".6", ".8", "1"])
    colorbar.ax.tick_params(length=2.2, width=0.45, pad=1.8)
    colorbar.ax.set_title("Test\naccuracy", fontsize=6.6, pad=3.0, linespacing=0.9)
    colorbar.outline.set_linewidth(0.5)
    colorbar.outline.set_edgecolor("#777777")

    figure.legend(
        handles=[
            Patch(
                facecolor="#F0F0F0",
                edgecolor="#A0A0A0",
                hatch="////",
                linewidth=0.35,
                label="Failed / skipped",
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.505, 0.992),
        frameon=False,
        handlelength=1.45,
        handleheight=0.95,
    )
    figure.text(0.493, 0.018, "Model", ha="center", va="bottom", fontsize=6.8)
    figure.text(
        0.985,
        0.018,
        "Preset means use successful runs only.",
        ha="right",
        va="bottom",
        fontsize=5.6,
        color="#555555",
    )
    return figure


def save_figure(figure: mpl.figure.Figure) -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "fig02_stage1_model_preset_heatmap_marginals"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    description = (
        "Four Stage-I heatmaps compare six preprocessing presets across 19 "
        "models, with per-preset marginal means and explicit unavailable cells."
    )
    image_metadata = {
        "Title": "Stage-I model and preprocessing-preset performance heatmaps",
        "Description": description,
        "Creator": "Matplotlib; generated by generate_stage1_heatmap_marginals.py",
    }
    pdf_metadata = {
        "Title": image_metadata["Title"],
        "Subject": description,
        "Creator": image_metadata["Creator"],
    }
    figure.savefig(png_path, facecolor="white", metadata=image_metadata)
    figure.savefig(pdf_path, facecolor="white", metadata=pdf_metadata)
    figure.savefig(svg_path, facecolor="white", metadata=image_metadata)
    plt.close(figure)
    return png_path, pdf_path, svg_path


def main() -> None:
    configure_matplotlib()
    stage1 = load_stage1()
    figure = build_figure(stage1)
    for output in save_figure(figure):
        print(output)


if __name__ == "__main__":
    main()
