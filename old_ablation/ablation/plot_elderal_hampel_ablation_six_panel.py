"""Combine ElderAL Hampel-window confusion matrices and test accuracy."""

from __future__ import annotations

import csv
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
from pypdf import PdfReader


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "ablation_elder_hampel_result"
RUN_DIR = RESULT_DIR / "runs"
SUMMARY_CSV = RESULT_DIR / "training_summary.csv"
OUTPUT_PDF = (
    SCRIPT_DIR.parent
    / "output"
    / "pdf"
    / "elderal_hampel_ablation_six_panel_aligned.pdf"
)

CONDITIONS = [
    ("no_hampel", "No Hampel", "No Hampel"),
    ("hampel_w1", "3-frame Hampel", "3"),
    ("hampel_w2", "5-frame Hampel", "5"),
    ("hampel_w3", "7-frame Hampel", "7"),
    ("hampel_w5", "11-frame Hampel", "11"),
]

GREY = "#D9D9D9"
BLUE = "#9EDCF0"
EDGE = "#303030"
GRID = "#D9D9D9"
CMAP = LinearSegmentedColormap.from_list(
    "hampel_blue",
    ["#F7FBFF", "#D7EAF4", BLUE, "#4A91C2", "#174A83"],
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.labelsize": 8.6,
            "axes.titlesize": 9.0,
            "axes.linewidth": 0.8,
            "axes.edgecolor": EDGE,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.6,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


ROW_PATTERN = re.compile(r"^(?:\d\.\d{2}\s+){5}\d\.\d{2}$")


def read_confusion_matrix(condition: str) -> np.ndarray:
    pdf_path = RUN_DIR / condition / "confusion_matrix_normalized.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    text = PdfReader(str(pdf_path)).pages[0].extract_text()
    rows = [
        [float(value) for value in line.strip().split()]
        for line in text.splitlines()
        if ROW_PATTERN.fullmatch(line.strip())
    ]
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (6, 6):
        raise ValueError(f"Expected a 6 x 6 matrix in {pdf_path}, found {matrix.shape}")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=0.02):
        raise ValueError(f"Unexpected row sums in {pdf_path}: {matrix.sum(axis=1)}")
    return matrix


def read_accuracies() -> np.ndarray:
    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {row["condition"]: row for row in csv.DictReader(handle)}
    values = []
    for condition, _, _ in CONDITIONS:
        row = rows.get(condition)
        if row is None or row.get("status") != "ok":
            raise ValueError(f"Missing successful training summary for {condition}")
        values.append(100.0 * float(row["test_acc"]))
    return np.asarray(values)


def add_caption(ax: plt.Axes, label: str, offset: float) -> None:
    ax.text(
        0.5,
        offset,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.8,
        fontweight="semibold",
        clip_on=False,
    )


def draw_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    caption: str,
    *,
    show_y_label: bool,
    show_x_label: bool,
    caption_offset: float,
) -> matplotlib.image.AxesImage:
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap=CMAP, interpolation="nearest")
    classes = np.arange(matrix.shape[0])
    ax.set_xticks(classes)
    ax.set_yticks(classes)
    ax.set_xlabel("Predicted class" if show_x_label else "", labelpad=2)
    ax.set_ylabel("True class" if show_y_label else "", labelpad=2)
    ax.tick_params(direction="out", length=2.5, width=0.7)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if value < 0.005:
                continue
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if value >= 0.55 else "#202020",
            )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(EDGE)
    add_caption(ax, caption, caption_offset)
    return image


def draw_accuracy(ax: plt.Axes, accuracies: np.ndarray) -> None:
    positions = np.arange(len(CONDITIONS))
    colors = [GREY] + [BLUE] * (len(CONDITIONS) - 1)
    bars = ax.bar(
        positions,
        accuracies,
        width=0.62,
        color=colors,
        edgecolor=EDGE,
        linewidth=0.8,
        zorder=2,
    )
    ax.set_xticks(positions, [entry[2] for entry in CONDITIONS])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 100)
    ax.set_yticks(np.arange(60, 101, 10), [f"{value}%" for value in range(60, 101, 10)])
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=2.5, width=0.7)

    for bar, value in zip(bars, accuracies, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    ax.legend(
        handles=[
            Patch(facecolor=GREY, edgecolor=EDGE, label="No Hampel"),
            Patch(facecolor=BLUE, edgecolor=EDGE, label="Hampel"),
        ],
        loc="upper right",
        ncol=1,
        frameon=False,
        handlelength=1.7,
        handletextpad=0.5,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(EDGE)
    add_caption(ax, "(f) Test accuracy", -0.25)


def main() -> None:
    configure_style()
    matrices = [read_confusion_matrix(condition) for condition, _, _ in CONDITIONS]
    accuracies = read_accuracies()

    fig = plt.figure(figsize=(10.8, 7.35))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.045],
        left=0.06,
        right=0.955,
        bottom=0.105,
        top=0.975,
        wspace=0.24,
        hspace=0.38,
    )

    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]

    captions = [
        "(a) No Hampel",
        "(b) 3-frame Hampel",
        "(c) 5-frame Hampel",
        "(d) 7-frame Hampel",
        "(e) 11-frame Hampel",
    ]
    image = None
    for index, (ax, matrix, caption) in enumerate(
        zip(axes[:5], matrices, captions, strict=True)
    ):
        bottom_row = index >= 3
        image = draw_matrix(
            ax,
            matrix,
            caption,
            show_y_label=index in {0, 3},
            show_x_label=bottom_row,
            caption_offset=-0.25 if bottom_row else -0.15,
        )

    draw_accuracy(axes[5], accuracies)

    colorbar_ax = fig.add_subplot(grid[:, 3])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Row-normalized proportion", rotation=90, labelpad=7)
    colorbar.set_ticks(np.linspace(0, 1, 6))
    colorbar.ax.tick_params(length=2.5, width=0.7)

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
