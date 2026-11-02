"""Create a unified three-part Hampel figure for the manuscript.

Panel (a) shows aggregate ElderAL Hampel effects and action-dependent window
coverage. Panel (b) contains the six ElderAL action waveforms, and panel (c)
contains the six XRF55 Hampel-window waveforms.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

import plot_combined_hampel_waveforms as waveforms
from plot_elder_hampel_effect_and_action_coverage import (
    read_action_coverage,
    read_signal_metrics,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = (
    SCRIPT_DIR.parent / "output" / "pdf" / "hampel_effect_and_waveforms_abc_revised.pdf"
)

AGGREGATE_GREY = "#D9D9D9"
COVERAGE_BLUE = "#9EDCF0"
EDGE = "#303030"


def draw_aggregate_panel(ax: plt.Axes) -> None:
    metrics = read_signal_metrics()
    action_coverage, _, _ = read_action_coverage()

    effect_labels = [
        "CSI values\nreplaced",
        "Frames with ≥1\nreplacement",
        "High-change\nCSI replaced",
        "Temporal\nvariation removed",
        "Peak intensity\nremoved",
    ]
    effect_values = np.asarray(
        [
            100.0 * metrics["replacement_rate"],
            100.0 * metrics["frame_any_replacement_rate"],
            100.0 * metrics["high_motion_replacement_rate"],
            100.0 * (1.0 - metrics["total_variation_retention"]),
            100.0 * (1.0 - metrics["dynamic_peak_retention"]),
        ]
    )

    coverage_by_name = dict(action_coverage)
    actions = ["Static", "Walking", "Fall", "Sleeping", "Drinking", "Sitting / TV"]
    action_values = np.asarray([coverage_by_name[name] for name in actions])

    labels = effect_labels + actions
    values = np.concatenate([effect_values, action_values])
    positions = np.asarray(
        [0, 1.05, 2.10, 3.15, 4.20, 5.80, 6.85, 7.90, 8.95, 10.00, 11.05]
    )
    colors = [AGGREGATE_GREY] * len(effect_labels) + [COVERAGE_BLUE] * len(actions)

    bars = ax.bar(
        positions,
        values,
        width=0.64,
        color=colors,
        edgecolor=EDGE,
        linewidth=0.75,
        zorder=2,
    )

    ax.set_ylim(0, 113)
    ax.set_yticks(np.arange(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)])
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", pad=4, length=2.5, width=0.65)
    ax.tick_params(axis="y", length=2.5, width=0.65)
    for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
        decimals = 2 if index < len(effect_labels) else 1
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.{decimals}f}%",
            ha="center",
            va="bottom",
            fontsize=7.4,
        )

    ax.text(
        2.10,
        106.0,
        f"Waveform correlation, $r$ = {metrics['mean_waveform_correlation']:.3f}",
        ha="center",
        va="center",
        fontsize=8.0,
    )
    ax.legend(
        handles=[
            Patch(facecolor=AGGREGATE_GREY, edgecolor=EDGE, label=r"Hampel effect ($N=162$)"),
            Patch(facecolor=COVERAGE_BLUE, edgecolor=EDGE, label="Window coverage (11 frames)"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.6,
    )


def draw_waveform_block(
    axes: np.ndarray,
    elder_panels: list[tuple[str, waveforms.PanelData]],
    xrf55_panels: list[tuple[str, int, waveforms.PanelData]],
) -> None:
    for index, (label, panel) in enumerate(elder_panels):
        row, column = divmod(index, 2)
        waveforms.draw_panel(
            axes[row, column],
            panel,
            label,
            show_xlabels=row == 2,
            show_ylabels=column == 0,
        )

    for index, (label, half_window, panel) in enumerate(xrf55_panels):
        row, local_column = divmod(index, 2)
        column = local_column + 2
        short_label = label.replace(", ", " (", 1) + ")"
        short_label = short_label.replace(" (default)", "")
        waveforms.draw_panel(
            axes[row, column],
            panel,
            short_label,
            show_xlabels=row == 2,
            show_ylabels=local_column == 0,
            highlight=half_window == 5,
        )


def main() -> None:
    waveforms.configure_style()
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 8.0,
        }
    )

    elder_panels = waveforms.extract_elder_panels()
    xrf55_panels = waveforms.extract_xrf55_panels()

    fig = plt.figure(figsize=(14.4, 10.3))
    outer = fig.add_gridspec(
        5,
        1,
        height_ratios=[2.75, 0.34, 0.30, 5.25, 0.62],
        left=0.055,
        right=0.992,
        bottom=0.025,
        top=0.972,
        hspace=0.10,
    )

    aggregate_ax = fig.add_subplot(outer[0])
    draw_aggregate_panel(aggregate_ax)

    aggregate_caption_ax = fig.add_subplot(outer[1])
    aggregate_caption_ax.axis("off")
    aggregate_caption_ax.text(
        0.5,
        0.45,
        "(a) Aggregate effect and action coverage",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    waveform_legend_ax = fig.add_subplot(outer[2])
    waveform_legend_ax.axis("off")
    waveform_legend_ax.legend(
        handles=waveforms.legend_handles(),
        loc="center",
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )

    bottom = outer[3].subgridspec(
        3,
        4,
        wspace=0.12,
        hspace=0.27,
    )
    axes = np.empty((3, 4), dtype=object)
    for row in range(3):
        for column in range(4):
            axes[row, column] = fig.add_subplot(bottom[row, column])
    draw_waveform_block(axes, elder_panels, xrf55_panels)

    footer_ax = fig.add_subplot(outer[4])
    footer_ax.axis("off")
    footer_ax.text(0.5, 0.78, "Time (s)", ha="center", va="center", fontsize=9)
    footer_ax.text(
        0.255,
        0.17,
        "(b) ElderAL actions (11-frame Hampel)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    footer_ax.text(
        0.755,
        0.17,
        "(c) XRF55 Hampel windows",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    bottom_box = outer[3].get_position(fig)
    fig.text(
        0.016,
        bottom_box.y0 + 0.50 * bottom_box.height,
        "Normalized amplitude",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=9,
    )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
