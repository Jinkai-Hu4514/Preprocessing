#!/usr/bin/env python3
"""Render the Robust mechanism and Gait component ablation."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
SUMMARY = ROOT / "ablation/robust_component_results/gait/summary.csv"

ORDER = [
    "linear_reference",
    "no_calibration",
    "common_only",
    "detrend_first50_only",
    "robust_first50",
    "robust_shared_first50",
    "robust_fullspan50",
    "robust_window_limited",
]
LABELS = {
    "linear_reference": "Linear reference",
    "no_calibration": "No calibration",
    "common_only": "Common phase only",
    "detrend_first50_only": "Independent detrend only",
    "robust_first50": "Original robust",
    "robust_shared_first50": "Shared slope",
    "robust_fullspan50": "50 points over model horizon",
    "robust_window_limited": "Correction frozen after frame 50",
}


def load_values() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY)
    if set(df["condition"]) != set(ORDER) or len(df) != len(ORDER):
        raise RuntimeError("Expected the eight completed Gait component conditions")
    df = df.set_index("condition").loc[ORDER].reset_index()
    df["test_pct"] = 100.0 * df["test_acc"]
    expected = {
        "linear_reference": 95.8667,
        "no_calibration": 95.8347,
        "common_only": 90.2916,
        "detrend_first50_only": 70.6825,
        "robust_first50": 67.3182,
        "robust_shared_first50": 93.5918,
        "robust_fullspan50": 67.8308,
        "robust_window_limited": 61.2624,
    }
    for row in df.itertuples(index=False):
        assert np.isclose(row.test_pct, expected[row.condition], atol=0.005)
    return df


def add_box(ax, xy, width, height, text, face, edge, fontsize=9.0, weight="normal"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.4,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#222222",
    )
    return box


def add_arrow(ax, start, end, color="#555555"):
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.4,
        color=color,
        clip_on=False,
    )
    ax.add_patch(arrow)


def draw_mechanism(ax):
    ax.set_axis_off()
    neutral_fill = "#F3F5F7"
    neutral_edge = "#66717E"
    adverse_fill = "#F6CFCB"
    adverse_edge = "#B64342"
    good_fill = "#DCEEDC"
    good_edge = "#2E7D4F"

    xs = [0.02, 0.265, 0.51, 0.755]
    texts = [
        "Temporal unwrap\nfor each $(f,a)$",
        "Subtract per-frame\nmedian across $f$",
        "Per-subcarrier slope\n$\\hat{\\beta}_{f,a}$ from first 50",
        "Apply $-t\\hat{\\beta}_{f,a}$\nto all frames",
    ]
    for i, (x, text) in enumerate(zip(xs, texts)):
        adverse = i >= 2
        add_box(
            ax,
            (x, 0.70),
            0.205,
            0.16,
            text,
            adverse_fill if adverse else neutral_fill,
            adverse_edge if adverse else neutral_edge,
            fontsize=8.6,
            weight="bold" if adverse else "normal",
        )
        if i < 3:
            add_arrow(ax, (x + 0.208, 0.78), (xs[i + 1] - 0.008, 0.78))

    ax.text(
        0.50,
        0.56,
        r"$\Delta\phi'_{f,g}(t)=\Delta\phi_{f,g}(t)-t\left(\hat{\beta}_{f,a}-\hat{\beta}_{g,a}\right)$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12.3,
        color="#222222",
    )
    ax.text(
        0.50,
        0.47,
        "Independent slopes add an unwrapped relative correction proportional to time (wrapped modulo $2\\pi$).",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.2,
        color=adverse_edge,
        fontweight="bold",
    )

    add_box(
        ax,
        (0.17, 0.15),
        0.66,
        0.20,
        r"Shared alternative:  $\hat{\beta}_{a}=\mathrm{median}_{f}\,\hat{\beta}_{f,a}$"
        "\n"
        r"All subcarriers use the same slope, so $\hat{\beta}_{f,a}-\hat{\beta}_{g,a}=0$.",
        good_fill,
        good_edge,
        fontsize=9.5,
        weight="bold",
    )
    ax.text(
        0.50,
        0.06,
        "A shared slope preserves cross-subcarrier phase differences.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.2,
        color=good_edge,
        fontweight="bold",
    )


def draw_ablation(ax, values):
    colors = {
        "linear_reference": "#3775BA",
        "no_calibration": "#A7ADB4",
        "common_only": "#E5A94D",
        "detrend_first50_only": "#D97941",
        "robust_first50": "#B64342",
        "robust_shared_first50": "#2F9E69",
        "robust_fullspan50": "#8E9AA7",
        "robust_window_limited": "#B8BEC5",
    }
    y = np.arange(len(values))[::-1]
    vals = values["test_pct"].to_numpy()
    bars = ax.barh(
        y,
        vals,
        height=0.68,
        color=[colors[c] for c in values["condition"]],
        edgecolor="#303030",
        linewidth=0.9,
        zorder=3,
    )
    ax.set_yticks(y, [LABELS[c] for c in values["condition"]])
    ax.set_xlim(0, 103)
    ax.set_xlabel("Test accuracy (%)")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=6)
    for bar, val in zip(bars, vals):
        ax.text(
            val + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=8.4,
            fontweight="bold",
        )

    row = {c: float(v) for c, v in zip(values["condition"], vals)}
    notes = [
        ("common_only", f"−5.54 pp vs no calibration", "#8A5C12"),
        ("detrend_first50_only", f"−25.15 pp vs no calibration", "#A94420"),
        ("robust_shared_first50", f"+26.27 pp vs original robust", "#237A50"),
        ("robust_fullspan50", f"+0.51 pp vs original robust", "#5F6872"),
    ]
    ypos = {c: yi for c, yi in zip(values["condition"], y)}
    for condition, note, color in notes:
        x = min(row[condition] - 2.0, 72.0)
        if condition == "robust_shared_first50":
            x = 52.5
        elif condition == "common_only":
            x = 55.0
        elif condition == "detrend_first50_only":
            x = 24.0
        elif condition == "robust_fullspan50":
            x = 24.0
        ax.text(
            x,
            ypos[condition],
            note,
            va="center",
            ha="left",
            fontsize=7.8,
            color="white" if condition in {"common_only", "detrend_first50_only", "robust_shared_first50", "robust_fullspan50"} else color,
            fontweight="bold",
        )

    ax.text(
        0.0,
        1.04,
        "Same 18,657 samples, group split, downstream pipeline, and MLP",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#555555",
    )
    ax.set_title("Gait component ablation (seed 42)", fontsize=10.5, pad=28,
                 fontweight="bold")


def render(values):
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(12.0, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.29)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    draw_mechanism(ax0)
    draw_ablation(ax1, values)
    ax0.text(-0.04, 1.02, "a", transform=ax0.transAxes, fontsize=18, fontweight="bold")
    ax1.text(-0.09, 1.02, "b", transform=ax1.transAxes, fontsize=18, fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.985, top=0.91, bottom=0.13)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT_DIR / f"fig_x_robust_root_cause.{ext}",
            dpi=500 if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    render(load_values())
