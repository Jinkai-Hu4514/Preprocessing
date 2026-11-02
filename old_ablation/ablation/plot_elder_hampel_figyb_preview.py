"""Plot Fig. Y(b): physical time spanned by frame-count Hampel windows.

The input table already contains one record per ElderAL segment and window
setting.  Each record reports the median physical span of all local windows
within that segment.  This script visualizes the distribution of those
sample-level medians without using timestamp-derived means or maxima.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "ablation_elder_hampel_result"
INPUT_CSV = RESULT_DIR / "window_span_per_sample.csv"
FIGURE_DIR = RESULT_DIR / "figures"
OUTPUT_STEM = "figure_yb_physical_window_span_preview"
WINDOWS = [3, 5, 7, 11]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    required = {
        "sample_index",
        "full_window_frames",
        "frames",
        "duration_seconds",
        "median_span_seconds",
        "median_fraction_of_sample_duration",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame[frame["full_window_frames"].isin(WINDOWS)].copy()
    counts = frame.groupby("full_window_frames")["sample_index"].nunique()
    if list(counts.reindex(WINDOWS)) != [2405, 2405, 2405, 2405]:
        raise ValueError(f"Unexpected per-window sample counts: {counts.to_dict()}")
    return frame


def draw_figure(frame: pd.DataFrame) -> tuple[plt.Figure, dict[str, float]]:
    groups = [
        frame.loc[
            frame["full_window_frames"] == window,
            "median_span_seconds",
        ].to_numpy(dtype=float)
        for window in WINDOWS
    ]
    medians = np.asarray([np.median(values) for values in groups])
    p90_values = np.asarray([np.percentile(values, 90) for values in groups])

    default = frame[frame["full_window_frames"] == 11]
    n_segments = int(default["sample_index"].nunique())
    # A segment with a single retained frame has zero duration, so its duration
    # coverage ratio is undefined and is excluded from this aggregate statistic.
    positive_duration = default[default["duration_seconds"] > 0]
    coverage_median = float(
        100.0
        * np.median(positive_duration["median_fraction_of_sample_duration"])
    )
    short_count = int((default["frames"] <= 11).sum())
    short_fraction = 100.0 * short_count / n_segments

    configure_style()
    fig, ax = plt.subplots(figsize=(6.6, 3.65), constrained_layout=True)
    positions = np.arange(1, len(WINDOWS) + 1)

    boxplot = ax.boxplot(
        groups,
        positions=positions,
        widths=0.56,
        whis=(10, 90),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#202020", "linewidth": 1.8},
        boxprops={"edgecolor": "#303030", "linewidth": 0.9},
        whiskerprops={"color": "#4A4A4A", "linewidth": 0.9},
        capprops={"color": "#4A4A4A", "linewidth": 0.9},
    )

    fill_colors = ["#9ECAE1", "#9ECAE1", "#9ECAE1", "#F4A582"]
    for patch, color in zip(boxplot["boxes"], fill_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    # Print the central value for every setting; the default setting receives
    # an explicit P90 marker because this is the quantity discussed in text.
    for x_position, median in zip(positions, medians, strict=True):
        ax.annotate(
            f"{median:.3f} s",
            xy=(x_position, median),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="semibold",
            color="#202020",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )

    default_p90 = float(p90_values[-1])
    ax.scatter(
        [positions[-1]],
        [default_p90],
        marker="D",
        s=30,
        color="#A63603",
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    ax.annotate(
        f"P90 = {default_p90:.3f} s",
        xy=(positions[-1], default_p90),
        xytext=(-10, 13),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="#8C2D04",
        fontsize=8.2,
        fontweight="semibold",
        arrowprops={
            "arrowstyle": "-",
            "color": "#8C2D04",
            "linewidth": 0.8,
        },
    )

    note = (
        "Default 11-frame neighborhood\n"
        f"Median segment coverage: {coverage_median:.2f}%\n"
        f"Segments with $T\\leq11$: {short_fraction:.2f}% "
        f"({short_count:,}/{n_segments:,})"
    )
    ax.text(
        0.035,
        0.965,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        linespacing=1.28,
        bbox={
            "boxstyle": "round,pad=0.36",
            "facecolor": "#FFF5EB",
            "edgecolor": "#D55E00",
            "linewidth": 0.8,
            "alpha": 0.96,
        },
    )

    ax.set_xticks(positions, [str(window) for window in WINDOWS])
    ax.set_xlabel("Full Hampel neighborhood (frames)")
    ax.set_ylabel("Per-segment median physical span (s)")
    ax.set_ylim(0, 5.7)
    ax.set_yticks(np.arange(0, 5.1, 1.0))
    ax.set_title(
        "(b) Physical time represented by frame-count neighborhoods",
        loc="left",
        pad=8,
        fontweight="semibold",
    )
    ax.text(
        0.99,
        0.02,
        f"$N$ = {n_segments:,} segments per setting",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="#555555",
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    metrics = {
        "n_segments": float(n_segments),
        "median_11": float(medians[-1]),
        "p90_11": default_p90,
        "coverage_median_percent": coverage_median,
        "short_count": float(short_count),
        "short_fraction_percent": short_fraction,
    }
    return fig, metrics


def main() -> None:
    frame = load_data()
    fig, metrics = draw_figure(frame)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            FIGURE_DIR / f"{OUTPUT_STEM}.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)

    print(f"Saved figure to {FIGURE_DIR / OUTPUT_STEM}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
