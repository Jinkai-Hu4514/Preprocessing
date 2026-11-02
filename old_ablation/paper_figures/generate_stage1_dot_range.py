"""Generate a publication-oriented Stage-I model/preset dot-range figure.

The figure keeps every successful test accuracy on a common 0--100% scale.
Failed and skipped runs are shown in a separate non-numeric status gutter so
that missing tasks cannot be mistaken for zero accuracy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from generate_stage1_accuracy_lines import (
    DATASET_ORDER,
    MODEL_LABELS,
    MODEL_ORDER,
    PRESET_LABELS,
    PRESET_ORDER,
    PRESET_STYLES,
    configure_matplotlib,
    load_stage1,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"

REPRESENTATIVE_MODELS = {
    "Widar": "mlpmodel",
    "Gait": "mlpmodel",
    "XRF55": "resnet1d",
    "ElderAL": "csitime",
}

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]
STATUS_X = {"failed": -7.0, "skipped": -2.8}


def _panel_title(frame: pd.DataFrame, label: str, dataset: str) -> str:
    counts = frame["status"].value_counts()
    return (
        f"{label} {dataset}   "
        f"{int(counts.get('ok', 0))} OK / "
        f"{int(counts.get('failed', 0))} failed / "
        f"{int(counts.get('skipped', 0))} skipped"
    )


def build_figure(data: pd.DataFrame) -> mpl.figure.Figure:
    """Build four shared-scale horizontal dot-range panels."""

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15.8, 13.0),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes_flat = axes.ravel()
    y_positions = np.arange(len(MODEL_ORDER), dtype=float)
    preset_offsets = dict(
        zip(PRESET_ORDER, np.linspace(-0.28, 0.28, len(PRESET_ORDER)))
    )
    status_offsets = dict(
        zip(PRESET_ORDER, np.linspace(-0.62, 0.62, len(PRESET_ORDER)))
    )

    for panel_index, (axis, dataset) in enumerate(
        zip(axes_flat, DATASET_ORDER)
    ):
        subset = data[data["dataset"].eq(dataset)].copy()
        representative_index = MODEL_ORDER.index(REPRESENTATIVE_MODELS[dataset])

        # Alternating row guides make it possible to trace a model across the
        # six slightly dodged preset markers without adding heavy grid lines.
        for row_index in range(len(MODEL_ORDER)):
            if row_index % 2 == 0:
                axis.axhspan(
                    row_index - 0.5,
                    row_index + 0.5,
                    color="#F5F5F5",
                    linewidth=0,
                    zorder=0,
                )
        axis.axhspan(
            representative_index - 0.5,
            representative_index + 0.5,
            color="#ECE8DC",
            linewidth=0,
            zorder=0.2,
        )

        # The gutter is deliberately separated from the quantitative scale.
        axis.axvspan(-10.0, -0.7, color="#FAFAFA", linewidth=0, zorder=0.1)
        axis.axvline(-0.7, color="#A7A7A7", linewidth=0.8, zorder=1)
        for tick in range(0, 101, 20):
            axis.axvline(tick, color="#DEDEDE", linewidth=0.65, zorder=0.5)

        # A thin interval summarizes the successful preset range for each
        # model. It connects values within a model, never different models.
        for row_index, model in enumerate(MODEL_ORDER):
            values = subset.loc[
                subset["model"].eq(model) & subset["status"].eq("ok"),
                "test_acc_pct",
            ].dropna()
            if len(values) >= 2:
                axis.plot(
                    [float(values.min()), float(values.max())],
                    [row_index, row_index],
                    color="#A8A8A8",
                    linewidth=1.15,
                    solid_capstyle="round",
                    zorder=1.5,
                )

        # Preset identity uses both colour and marker shape. Failed and skipped
        # tasks retain that identity while moving into status-specific columns.
        for preset in PRESET_ORDER:
            style = PRESET_STYLES[preset]
            preset_rows = subset[subset["preset"].eq(preset)].set_index("model")
            for row_index, model in enumerate(MODEL_ORDER):
                row = preset_rows.loc[model]
                y_value = row_index + preset_offsets[preset]
                status = row["status"]
                if status == "ok":
                    axis.scatter(
                        float(row["test_acc_pct"]),
                        y_value,
                        s=30,
                        marker=style["marker"],
                        facecolor=style["color"],
                        edgecolor="white",
                        linewidth=0.55,
                        alpha=0.94,
                        zorder=3,
                    )
                else:
                    axis.scatter(
                        STATUS_X[status] + status_offsets[preset],
                        y_value,
                        s=30,
                        marker=style["marker"],
                        facecolor="none",
                        edgecolor=style["color"],
                        linewidth=1.05,
                        zorder=3,
                    )

        # Ring every row-wise best successful preset. Multiple exact ties are
        # retained rather than broken arbitrarily.
        for row_index, model in enumerate(MODEL_ORDER):
            model_rows = subset.loc[
                subset["model"].eq(model) & subset["status"].eq("ok")
            ]
            if model_rows.empty:
                continue
            best_accuracy = float(model_rows["test_acc_pct"].max())
            best_rows = model_rows[np.isclose(
                model_rows["test_acc_pct"].astype(float), best_accuracy
            )]
            for _, row in best_rows.iterrows():
                axis.scatter(
                    best_accuracy,
                    row_index + preset_offsets[row["preset"]],
                    s=63,
                    marker="o",
                    facecolor="none",
                    edgecolor="#202020",
                    linewidth=0.8,
                    zorder=4,
                )

        axis.set_title(
            _panel_title(subset, PANEL_LABELS[panel_index], dataset),
            loc="left",
            pad=8,
            fontsize=11,
            fontweight="semibold",
        )
        axis.set_xlim(-10.0, 100.0)
        axis.set_ylim(len(MODEL_ORDER) - 0.5, -0.5)
        axis.set_xticks([-7.0, -2.8, 0, 20, 40, 60, 80, 100])
        axis.set_xticklabels(["F", "S", "0", "20", "40", "60", "80", "100"])
        axis.set_yticks(y_positions)
        if panel_index % 2 == 0:
            axis.set_yticklabels([MODEL_LABELS[item] for item in MODEL_ORDER])
            for tick_label, model in zip(axis.get_yticklabels(), MODEL_ORDER):
                if model == REPRESENTATIVE_MODELS[dataset]:
                    tick_label.set_fontweight("semibold")
        else:
            axis.tick_params(axis="y", labelleft=False)

        axis.tick_params(axis="x", length=3.0, pad=3)
        axis.tick_params(axis="y", length=0, pad=5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#707070")
        axis.spines["bottom"].set_color("#707070")
        axis.set_axisbelow(True)

        for tick_index, tick_label in enumerate(axis.get_xticklabels()):
            if tick_index == 0:
                tick_label.set_color("#A33A2A")
                tick_label.set_fontweight("semibold")
            elif tick_index == 1:
                tick_label.set_color("#666666")
                tick_label.set_fontweight("semibold")

    preset_handles = [
        Line2D(
            [0],
            [0],
            marker=PRESET_STYLES[preset]["marker"],
            linestyle="none",
            markersize=6.3,
            markerfacecolor=PRESET_STYLES[preset]["color"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=PRESET_LABELS[preset],
        )
        for preset in PRESET_ORDER
    ]
    figure.legend(
        handles=preset_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.45,
    )
    figure.supxlabel("Test accuracy (%)", x=0.55, y=0.038, fontsize=11)
    figure.text(
        0.5,
        0.014,
        (
            "F = failed; S = skipped (status gutter, not 0% accuracy).  "
            "Grey segment = successful preset range; black ring = row best; "
            "shaded row = Stage-II representative model."
        ),
        ha="center",
        va="bottom",
        fontsize=8.6,
        color="#4A4A4A",
    )
    figure.subplots_adjust(
        left=0.12,
        right=0.992,
        top=0.925,
        bottom=0.075,
        hspace=0.18,
        wspace=0.07,
    )
    return figure


def save_figure(figure: mpl.figure.Figure) -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "fig02_stage1_model_preset_dot_range"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    description = (
        "Four shared-scale dot-range panels preserve all successful test "
        "accuracies and show failed/skipped runs in a separate gutter."
    )
    image_metadata = {
        "Title": "Stage-I model sensitivity to six preprocessing presets",
        "Description": description,
        "Creator": "Matplotlib; generated by generate_stage1_dot_range.py",
    }
    pdf_metadata = {
        "Title": image_metadata["Title"],
        "Subject": description,
        "Creator": image_metadata["Creator"],
    }
    figure.savefig(png_path, dpi=400, facecolor="white", metadata=image_metadata)
    figure.savefig(pdf_path, facecolor="white", metadata=pdf_metadata)
    figure.savefig(svg_path, facecolor="white", metadata=image_metadata)
    plt.close(figure)
    return png_path, pdf_path, svg_path


def main() -> None:
    configure_matplotlib()
    stage1 = load_stage1()
    figure = build_figure(stage1)
    outputs = save_figure(figure)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
