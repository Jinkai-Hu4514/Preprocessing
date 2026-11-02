"""Combine ElderAL Hampel signal effects and action coverage in one bar chart."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "ablation_elder_hampel_result"
FIGURE_DIR = RESULT_DIR / "figures"
SIGNAL_CSV = RESULT_DIR / "signal_diagnostics.csv"
TIMING_CSV = RESULT_DIR / "timing_per_sample.csv"
OUTPUT_STEM = FIGURE_DIR / "figure_hampel_effect_and_action_coverage"

ACTION_NAMES = {
    0: "Static",
    1: "Walking",
    2: "Fall",
    3: "Sleeping",
    5: "Drinking",
    6: "Sitting / TV",
}

# Quiet two-color palette adapted from the supplied reference figure.
AGGREGATE_GREY = "#D9D9D9"
COVERAGE_BLUE = "#9EDCF0"
EDGE = "#303030"
TEXT = "#202020"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.8,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": EDGE,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "text.color": TEXT,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_signal_metrics() -> dict[str, float]:
    with SIGNAL_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(
            item
            for item in csv.DictReader(handle)
            if int(item["full_window_frames"]) == 11
        )
    return {key: float(value) for key, value in row.items() if value not in (None, "")}


def read_action_coverage() -> tuple[list[tuple[str, float]], float, int]:
    grouped: dict[int, list[float]] = {action_id: [] for action_id in ACTION_NAMES}
    positive_duration_values: list[float] = []

    with TIMING_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        action_id = int(row["action_id"])
        if action_id not in grouped:
            continue
        value = float(row["default_window_median_fraction_of_duration"])
        grouped[action_id].append(value)
        if float(row["duration_seconds"]) > 0:
            positive_duration_values.append(value)

    action_values = [
        (ACTION_NAMES[action_id], 100.0 * float(np.median(values)))
        for action_id, values in grouped.items()
    ]
    action_values.sort(key=lambda item: item[1], reverse=True)
    overall_median = 100.0 * float(np.median(positive_duration_values))
    return action_values, overall_median, len(rows)


def draw() -> None:
    configure_style()
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
    # Only the four activities discussed in the manuscript are shown, keeping
    # the single-axis figure focused and consistent with the supplied example.
    coverage_by_name = dict(action_coverage)
    selected_actions = ["Walking", "Fall", "Static", "Sleeping"]
    action_values = np.asarray([coverage_by_name[name] for name in selected_actions])

    labels = effect_labels + selected_actions
    values = np.concatenate([effect_values, action_values])
    positions = np.asarray([0, 1.15, 2.30, 3.45, 4.60, 6.30, 7.45, 8.60, 9.75], dtype=float)
    colors = [AGGREGATE_GREY] * len(effect_labels) + [COVERAGE_BLUE] * len(selected_actions)

    fig, ax = plt.subplots(figsize=(9.2, 3.8), constrained_layout=True)
    bars = ax.bar(
        positions,
        values,
        width=0.64,
        color=colors,
        edgecolor=EDGE,
        linewidth=0.8,
        zorder=2,
    )

    ax.set_ylim(0, 114)
    ax.set_yticks(np.arange(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)])
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", pad=5, length=3.5, width=0.8)
    ax.tick_params(axis="y", length=3.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color(EDGE)

    for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
        decimals = 2 if index < len(effect_labels) else 1
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.4,
            f"{value:.{decimals}f}%",
            ha="center",
            va="bottom",
            fontsize=8.2,
        )

    ax.text(
        2.30,
        108.0,
        f"Waveform correlation, $r$ = {metrics['mean_waveform_correlation']:.3f}",
        ha="center",
        va="center",
        fontsize=8.6,
    )
    ax.legend(
        handles=[
            Patch(facecolor=AGGREGATE_GREY, edgecolor=EDGE, label=r"Hampel effect ($N=162$)"),
            Patch(facecolor=COVERAGE_BLUE, edgecolor=EDGE, label="Window coverage (11 frames)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8.8,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.8,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    draw()
