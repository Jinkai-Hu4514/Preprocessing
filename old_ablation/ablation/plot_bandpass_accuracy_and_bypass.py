#!/usr/bin/env python3
"""Draw the combined XRF55 correction and ElderAL bypass figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ELDER_SUMMARY = (
    REPO_ROOT
    / "ablation"
    / "bandpass_server_results"
    / "signal_analysis"
    / "elder_bandpass_summary.json"
)
OUTPUT = REPO_ROOT / "output" / "pdf" / "figure_bandpass_accuracy_and_bypass.pdf"

# Palette matched to the ablation figures used elsewhere in the manuscript.
GRAY = "#D9D9D9"
BLUE = "#9EDCF0"
PEACH = "#F2C6A5"
EDGE = "#2F2F2F"
GRID = "#E5E5E5"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.linewidth": 0.85,
            "axes.edgecolor": EDGE,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_elder_summary() -> dict:
    with ELDER_SUMMARY.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    total = int(summary["valid_samples"])
    bypassed = int(summary["bypass_samples"])
    filtered = int(summary["eligible_samples"])
    assert total == 2404
    assert bypassed == 2010
    assert filtered == 394
    assert bypassed + filtered == total
    return summary


def add_value_label(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    label: str,
    *,
    offset: float = 1.5,
    fontsize: float = 8.8,
) -> None:
    ax.text(
        x,
        y + offset,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=EDGE,
    )


def draw_gain_bracket(
    ax: mpl.axes.Axes, x0: float, x1: float, y: float, label: str
) -> None:
    stem = 1.25
    ax.plot(
        [x0, x0, x1, x1],
        [y - stem, y, y, y - stem],
        color=EDGE,
        linewidth=0.75,
        clip_on=False,
    )
    ax.text(
        (x0 + x1) / 2,
        y + 0.75,
        label,
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=EDGE,
    )


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{value}%" for value in range(0, 101, 20)])
    ax.grid(axis="y", linestyle=(0, (3, 2)), color=GRID, linewidth=0.7, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)
        spine.set_color(EDGE)


def render(summary: dict) -> Path:
    configure_style()

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.35),
        gridspec_kw={"width_ratios": [1.38, 1.0]},
    )

    # (a) XRF55: two controlled accuracy comparisons plus the signed-value audit.
    x = np.array([0.00, 0.72, 1.86, 2.58, 3.72])
    values = np.array([38.18, 61.97, 62.58, 64.70, 49.759])
    colors = [GRAY, BLUE, GRAY, BLUE, PEACH]
    bars = ax_left.bar(
        x,
        values,
        width=0.54,
        color=colors,
        edgecolor=EDGE,
        linewidth=0.85,
        zorder=3,
    )

    labels = ["38.18%", "61.97%", "62.58%", "64.70%", "49.759%"]
    for bar, label in zip(bars, labels):
        add_value_label(
            ax_left,
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
        )

    draw_gain_bracket(ax_left, x[0], x[1], 70.5, "+23.79 pp")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(
        ["1000 Hz", "200 Hz", "Absolute", "Signed", "Altered\ndirections"]
    )
    ax_left.text(
        np.mean(x[:2]),
        -0.125,
        "Sampling rate",
        transform=ax_left.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8.8,
    )
    ax_left.text(
        np.mean(x[2:4]),
        -0.125,
        "Sign handling",
        transform=ax_left.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8.8,
    )
    ax_left.text(
        x[4],
        -0.125,
        "Signal audit",
        transform=ax_left.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8.8,
    )
    ax_left.set_ylabel("Percentage (%)")
    ax_left.set_xlim(-0.48, 4.20)
    style_axis(ax_left)

    # (b) ElderAL: sample routing is separated from the reported mean accuracy.
    total = int(summary["valid_samples"])
    bypassed = int(summary["bypass_samples"])
    filtered = int(summary["eligible_samples"])
    bypass_rate = 100.0 * bypassed / total
    filtered_rate = 100.0 * filtered / total
    reported_accuracy = 82.20

    route_x = 0.0
    accuracy_x = 1.15
    width = 0.62
    ax_right.bar(
        route_x,
        bypass_rate,
        width=width,
        color=GRAY,
        edgecolor=EDGE,
        linewidth=0.85,
        zorder=3,
    )
    ax_right.bar(
        route_x,
        filtered_rate,
        width=width,
        bottom=bypass_rate,
        color=BLUE,
        edgecolor=EDGE,
        linewidth=0.85,
        zorder=3,
    )
    ax_right.bar(
        accuracy_x,
        reported_accuracy,
        width=width,
        color=PEACH,
        edgecolor=EDGE,
        linewidth=0.85,
        zorder=3,
    )

    ax_right.text(
        route_x,
        bypass_rate / 2,
        f"Bypassed\n{bypass_rate:.2f}%\n({bypassed})",
        ha="center",
        va="center",
        fontsize=8.8,
        color=EDGE,
    )
    ax_right.text(
        route_x,
        bypass_rate + filtered_rate / 2,
        f"Filtered  {filtered_rate:.2f}%  ({filtered})",
        ha="center",
        va="center",
        fontsize=8.2,
        color=EDGE,
    )
    add_value_label(
        ax_right,
        accuracy_x,
        reported_accuracy,
        f"{reported_accuracy:.2f}%",
    )

    ax_right.set_xticks([route_x, accuracy_x])
    ax_right.set_xticklabels(["Valid segments\n($N=2{,}404$)", "Reported mean\naccuracy"])
    ax_right.set_xlim(-0.60, 1.75)
    style_axis(ax_right)
    ax_right.tick_params(axis="y", labelleft=False)

    # Subfigure captions are deliberately placed below both panels.
    ax_left.text(
        0.5,
        -0.285,
        "(a) XRF55 Bandpass corrections",
        transform=ax_left.transAxes,
        ha="center",
        va="top",
        fontsize=10.0,
        fontweight="bold",
    )
    ax_right.text(
        0.5,
        -0.285,
        "(b) ElderAL short-sequence bypass",
        transform=ax_right.transAxes,
        ha="center",
        va="top",
        fontsize=10.0,
        fontweight="bold",
    )

    fig.subplots_adjust(
        left=0.072,
        right=0.985,
        top=0.965,
        bottom=0.285,
        wspace=0.23,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, facecolor="white")
    plt.close(fig)
    return OUTPUT


def main() -> None:
    summary = load_elder_summary()
    output = render(summary)
    print(output)


if __name__ == "__main__":
    main()
