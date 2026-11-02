"""Plot ElderAL sampling density and physical Hampel-window span."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "ablation_elder_hampel_result"
TIMING_CSV = RESULT_DIR / "timing_per_sample.csv"
WINDOW_CSV = RESULT_DIR / "window_span_per_sample.csv"
OUTPUT_PDF = (
    SCRIPT_DIR.parent
    / "output"
    / "pdf"
    / "elderal_sampling_and_hampel_window_span_revised.pdf"
)

WINDOWS = [3, 5, 7, 11]
GREY = "#D9D9D9"
POINT_GREY = "#9B9B9B"
BLUE = "#9EDCF0"
BLUE_DARK = "#297D99"
EDGE = "#303030"
GRID = "#D9D9D9"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.8,
            "axes.edgecolor": EDGE,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.0,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_sampling_data() -> pd.DataFrame:
    frame = pd.read_csv(TIMING_CSV, encoding="utf-8-sig")
    required = {"duration_seconds", "median_interval_seconds", "frames"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing timing columns: {sorted(missing)}")

    frame = frame.loc[
        frame["duration_seconds"].between(3.0, 7.0, inclusive="both")
        & np.isfinite(frame["median_interval_seconds"])
        & (frame["median_interval_seconds"] > 0)
        & np.isfinite(frame["frames"])
    ].copy()
    if len(frame) != 2015:
        raise ValueError(f"Expected 2,015 duration-filtered samples, found {len(frame):,}")
    return frame


def load_window_data() -> list[np.ndarray]:
    frame = pd.read_csv(WINDOW_CSV, encoding="utf-8-sig")
    required = {"sample_index", "full_window_frames", "median_span_seconds"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing window-span columns: {sorted(missing)}")

    groups: list[np.ndarray] = []
    for window in WINDOWS:
        subset = frame.loc[
            frame["full_window_frames"] == window,
            ["sample_index", "median_span_seconds"],
        ].dropna()
        if subset["sample_index"].nunique() != 2405:
            raise ValueError(
                f"Expected 2,405 segments for {window} frames, "
                f"found {subset['sample_index'].nunique():,}"
            )
        groups.append(subset["median_span_seconds"].to_numpy(dtype=float))
    return groups


def binned_medians(x: np.ndarray, y: np.ndarray, bin_count: int = 10) -> tuple[np.ndarray, np.ndarray]:
    log_x = np.log10(x)
    edges = np.linspace(float(log_x.min()), float(log_x.max()), bin_count + 1)
    indices = np.digitize(log_x, edges[1:-1], right=False)

    median_x: list[float] = []
    median_y: list[float] = []
    for bin_index in range(bin_count):
        mask = indices == bin_index
        if np.count_nonzero(mask) < 8:
            continue
        median_x.append(float(np.median(x[mask])))
        median_y.append(float(np.median(y[mask])))
    return np.asarray(median_x), np.asarray(median_y)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.2, width=0.75)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color(EDGE)


def draw_sampling_panel(ax: plt.Axes, frame: pd.DataFrame) -> None:
    x = frame["median_interval_seconds"].to_numpy(dtype=float)
    y = frame["frames"].to_numpy(dtype=float)
    median_x, median_y = binned_medians(x, y)

    spearman = float(frame["median_interval_seconds"].corr(frame["frames"], method="spearman"))

    ax.scatter(
        x,
        y,
        s=8,
        color=POINT_GREY,
        alpha=0.38,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    ax.plot(
        median_x,
        median_y,
        color=BLUE_DARK,
        linewidth=1.5,
        marker="D",
        markersize=4.0,
        markerfacecolor=BLUE,
        markeredgecolor=EDGE,
        markeredgewidth=0.6,
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_ylabel("Retained frames per sample")
    ax.set_ylim(0, 125)
    ax.text(
        0.97,
        0.94,
        rf"$N$ = {len(frame):,},  Spearman $\rho$ = {spearman:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="o",
                markersize=4.5,
                markerfacecolor=POINT_GREY,
                markeredgecolor="none",
                alpha=0.65,
                label="Samples",
            ),
            Line2D(
                [0],
                [0],
                color=BLUE_DARK,
                linewidth=1.5,
                marker="D",
                markersize=4.0,
                markerfacecolor=BLUE,
                markeredgecolor=EDGE,
                markeredgewidth=0.6,
                label="Binned median",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 0.87),
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
    )
    style_axes(ax)


def draw_window_panel(ax: plt.Axes, groups: list[np.ndarray]) -> None:
    positions = np.arange(1, len(WINDOWS) + 1)
    medians = np.asarray([np.median(values) for values in groups])
    boxplot = ax.boxplot(
        groups,
        positions=positions,
        widths=0.52,
        whis=(10, 90),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": EDGE, "linewidth": 1.5},
        boxprops={"edgecolor": EDGE, "linewidth": 0.8},
        whiskerprops={"color": EDGE, "linewidth": 0.8},
        capprops={"color": EDGE, "linewidth": 0.8},
    )

    for index, patch in enumerate(boxplot["boxes"]):
        patch.set_facecolor(BLUE if WINDOWS[index] == 11 else GREY)
        patch.set_alpha(0.92)

    for position, median in zip(positions, medians, strict=True):
        ax.text(
            position,
            median + 0.10,
            f"{median:.3f} s",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )

    ax.set_xticks(positions, [str(window) for window in WINDOWS])
    ax.set_ylabel("Median physical span per segment (s)")
    ax.set_ylim(0, 5.25)
    ax.set_yticks(np.arange(0, 5.1, 1.0))
    ax.legend(
        handles=[
            Patch(facecolor=GREY, edgecolor=EDGE, label="3/5/7 frames"),
            Patch(facecolor=BLUE, edgecolor=EDGE, label="11 frames (default)"),
        ],
        loc="upper left",
        frameon=False,
        handlelength=1.7,
        handletextpad=0.5,
    )
    style_axes(ax)


def main() -> None:
    configure_style()
    sampling = load_sampling_data()
    window_groups = load_window_data()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.15), constrained_layout=False)
    draw_sampling_panel(axes[0], sampling)
    draw_window_panel(axes[1], window_groups)

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.965, wspace=0.09)
    captions = [
        "(a) Sampling interval and retained frames",
        "(b) Physical span of frame-count neighborhoods",
    ]
    for ax, caption in zip(axes, captions, strict=True):
        box = ax.get_position()
        fig.text(
            box.x0 + box.width / 2,
            0.045,
            caption,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="semibold",
        )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
