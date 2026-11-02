"""Render a compact two-panel preview for the ElderAL Hampel analysis.

Panel (a) uses the existing high-distortion fall example selected by the
component-analysis scripts. Panel (b) summarizes the exact aggregate metrics
for the default 11-frame Hampel neighborhood over 162 stratified samples.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_elder_hampel_action_waveforms import (
    CSVRecord,
    hampel_with_diagnostics,
    read_raw_csi,
    timestamps_in_seconds,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULT_DIR = SCRIPT_DIR / "ablation_elder_hampel_result"
FIGURE_DIR = RESULT_DIR / "figures"
EXAMPLE_PATH = (
    PROJECT_ROOT
    / "sdp_dataset"
    / "elderAL"
    / "action2_fall_new"
    / "user0_position1_activity2"
    / "20250813=230850_mimo5s_part0.csv"
)

COLORS = {
    "raw": "#4D4D4D",
    "filtered": "#0F4D92",
    "replaced": "#B64342",
    "scope": "#3775BA",
    "high_change": "#D98C32",
    "loss": "#B64342",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_default_metrics() -> dict[str, float]:
    path = RESULT_DIR / "signal_diagnostics.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(item for item in rows if int(item["full_window_frames"]) == 11)
    return {key: float(value) for key, value in row.items() if value not in (None, "")}


def plot_preview() -> None:
    configure_style()
    if not EXAMPLE_PATH.is_file():
        raise FileNotFoundError(EXAMPLE_PATH)

    record = CSVRecord(
        file_name=str(EXAMPLE_PATH),
        action_id=2,
        timestamps_raw=np.asarray([], dtype=float),
    )
    raw, timestamps_raw = read_raw_csi(record)
    timestamps = timestamps_in_seconds(timestamps_raw, 1e-6)
    filtered, changed = hampel_with_diagnostics(raw, half_window=5)

    subcarrier = 193
    link = 2
    raw_signal = raw[:, subcarrier, link]
    filtered_signal = filtered[:, subcarrier, link]
    changed_signal = changed[:, subcarrier, link]

    metrics = read_default_metrics()
    labels = [
        "CSI values\nreplaced",
        "Frames with >=1\nreplacement",
        "Values replaced in\nhigh-change frames",
        "Temporal variation\nreduced",
        "Dynamic-peak intensity\nreduced",
    ]
    values = np.asarray(
        [
            100.0 * metrics["replacement_rate"],
            100.0 * metrics["frame_any_replacement_rate"],
            100.0 * metrics["high_motion_replacement_rate"],
            100.0 * (1.0 - metrics["total_variation_retention"]),
            100.0 * (1.0 - metrics["dynamic_peak_retention"]),
        ]
    )
    bar_colors = [
        COLORS["scope"],
        COLORS["scope"],
        COLORS["high_change"],
        COLORS["loss"],
        COLORS["loss"],
    ]

    fig, (ax_signal, ax_summary) = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.05),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
        constrained_layout=True,
    )

    ax_signal.plot(
        timestamps,
        raw_signal,
        color=COLORS["raw"],
        alpha=0.60,
        marker="o",
        markersize=2.6,
        linewidth=1.3,
        label="Raw CSI",
        zorder=1,
    )
    ax_signal.plot(
        timestamps,
        filtered_signal,
        color=COLORS["filtered"],
        marker="o",
        markersize=2.6,
        linewidth=1.4,
        label="Hampel (11 frames)",
        zorder=2,
    )
    ax_signal.vlines(
        timestamps[changed_signal],
        raw_signal[changed_signal],
        filtered_signal[changed_signal],
        color=COLORS["replaced"],
        alpha=0.45,
        linewidth=0.8,
        zorder=2,
    )
    ax_signal.scatter(
        timestamps[changed_signal],
        filtered_signal[changed_signal],
        s=34,
        facecolors="none",
        edgecolors=COLORS["replaced"],
        linewidths=1.25,
        label="Replacement (output)",
        zorder=3,
    )
    ax_signal.set_xlabel("Time from sample start (s)")
    ax_signal.set_ylabel("CSI amplitude")
    ax_signal.set_title(
        "(a) High-distortion example",
        loc="left",
        fontweight="bold",
    )
    ax_signal.grid(axis="both", color="#D9D9D9", linewidth=0.6, alpha=0.55)
    ax_signal.legend(frameon=False, loc="lower right")
    ax_signal.spines[["top", "right"]].set_visible(False)

    y = np.arange(len(labels))
    ax_summary.barh(
        y,
        values,
        color=bar_colors,
        edgecolor="#333333",
        linewidth=0.6,
        height=0.66,
    )
    ax_summary.set_yticks(y, labels)
    ax_summary.invert_yaxis()
    ax_summary.set_xlim(0, 105)
    ax_summary.set_xlabel("Percentage (%)")
    ax_summary.set_title(
        "(b) Aggregate effect ($N=162$)",
        loc="left",
        fontweight="bold",
    )
    ax_summary.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.55)
    ax_summary.set_axisbelow(True)
    ax_summary.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(values):
        ax_summary.text(
            value + 1.5,
            index,
            f"{value:.2f}%",
            va="center",
            ha="left",
            fontsize=8,
        )
    ax_summary.text(
        0.98,
        0.98,
        f"Correlation = {metrics['mean_waveform_correlation']:.3f}\n"
        f"Enrichment = {metrics['high_motion_replacement_enrichment']:.2f}x",
        transform=ax_summary.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        color="#333333",
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / "figure_x_hampel_signal_effect_preview"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    plot_preview()
