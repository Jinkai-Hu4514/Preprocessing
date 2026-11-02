#!/usr/bin/env python3
"""Render the XRF55 sampling-rate mismatch figure.

Panel (a) uses the exact Butterworth design in ``butterworth_bandpass``:
``butter(order, [low/(fs/2), high/(fs/2)], btype="band")``.  Because the
project applies the filter with ``filtfilt``, the plotted effective magnitude
is |H|^2.  Both responses are expressed on XRF55's physical 200-Hz axis.

Panel (b) reads the three reported test accuracies from the experiment CSVs.
All bars use the matched IQR + z-score + cubic15 downstream configuration and
seed 42; no uncertainty bars are drawn because these are single-seed results.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bandpass_paper_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.signal import butter, freqz


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAIN_SUMMARY = (
    PROJECT_ROOT
    / "SDP/test_xrf55/result/full_tests/"
    / "xrf55_80_pipeline_resnet1d_summary.csv"
)
FS200_SUMMARY = (
    PROJECT_ROOT
    / "SDP/test_xrf55/result/bandpass_fs200/"
    / "xrf55_bandpass_fs200_iqr_zscore_cubic15_resnet1d_summary.csv"
)

FILTER_ORDER = 4
LOW_HZ = 0.5
HIGH_HZ = 50.0
XRF55_FS_HZ = 200.0
SEED = 42

# Colorblind-safe palette (Okabe-Ito inspired).
ORANGE = "#D55E00"
BLUE = "#0072B2"
GREEN = "#009E73"
CHARCOAL = "#333333"
LIGHT_GRID = "#D9D9D9"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment summary: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_unique(rows: list[dict[str, str]], **conditions: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(row.get(column) == value for column, value in conditions.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one row for {conditions}, found {len(matches)}"
        )
    row = matches[0]
    if row.get("status") != "ok":
        raise RuntimeError(f"Selected experiment did not finish successfully: {row}")
    return row


def load_test_accuracies() -> np.ndarray:
    """Load the exact matched-configuration test accuracies, in percent."""
    main_rows = read_rows(MAIN_SUMMARY)
    bandpass_1000 = select_unique(
        main_rows,
        denoise="bandpass_0.5-50",
        outliers="iqr",
        normalize="z-score",
        interpolate="cubic15",
    )
    savgol = select_unique(
        main_rows,
        denoise="savgol_w7_p3",
        outliers="iqr",
        normalize="z-score",
        interpolate="cubic15",
    )
    bandpass_200 = select_unique(
        read_rows(FS200_SUMMARY),
        denoise="bandpass_0.5-50_fs200",
        outliers="iqr",
        normalize="z-score",
        interpolate="cubic15",
    )

    accuracies = 100.0 * np.array(
        [
            float(bandpass_1000["test_acc"]),
            float(bandpass_200["test_acc"]),
            float(savgol["test_acc"]),
        ]
    )
    expected = np.array([38.1818181818, 61.9696969697, 85.1515151515])
    if not np.allclose(accuracies, expected, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            f"Unexpected source accuracies: {accuracies}; expected {expected}"
        )
    return accuracies


def effective_filtfilt_response(
    configured_fs_hz: float, physical_fs_hz: float, frequencies_hz: np.ndarray
) -> np.ndarray:
    """Return |H|^2 for the project's zero-phase bandpass design."""
    nyquist = configured_fs_hz / 2.0
    b, a = butter(
        FILTER_ORDER,
        [LOW_HZ / nyquist, HIGH_HZ / nyquist],
        btype="band",
    )
    angular_frequency = 2.0 * np.pi * frequencies_hz / physical_fs_hz
    _, single_pass = freqz(b, a, worN=angular_frequency)
    return np.abs(single_pass) ** 2


def add_double_arrow(
    ax: plt.Axes,
    start: float,
    stop: float,
    y: float,
    text: str,
    color: str,
) -> None:
    ax.annotate(
        "",
        xy=(stop, y),
        xytext=(start, y),
        arrowprops={
            "arrowstyle": "<->",
            "color": color,
            "linewidth": 1.1,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )
    ax.text(
        np.sqrt(start * stop),
        y - 0.035,
        text,
        ha="center",
        va="top",
        color=color,
        fontsize=7.1,
        fontweight="semibold",
    )


def add_bracket(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    label: str,
    color: str = CHARCOAL,
) -> None:
    height = 1.6
    ax.plot(
        [x0, x0, x1, x1],
        [y - height, y, y, y - height],
        color=color,
        linewidth=0.9,
        clip_on=False,
    )
    ax.text(
        (x0 + x1) / 2.0,
        y + 1.0,
        label,
        ha="center",
        va="bottom",
        color=color,
        fontsize=7.2,
        fontweight="semibold",
    )


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 8.7,
            "axes.labelsize": 8.2,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def render() -> None:
    configure_style()
    accuracies = load_test_accuracies()

    frequencies = np.geomspace(0.05, 99.5, 8000)
    response_1000 = effective_filtfilt_response(1000.0, XRF55_FS_HZ, frequencies)
    response_200 = effective_filtfilt_response(200.0, XRF55_FS_HZ, frequencies)

    actual_1000_band = (
        LOW_HZ * XRF55_FS_HZ / 1000.0,
        HIGH_HZ * XRF55_FS_HZ / 1000.0,
    )
    target_band = (LOW_HZ, HIGH_HZ)
    if not np.allclose(actual_1000_band, (0.1, 10.0)):
        raise RuntimeError(f"Unexpected mapped passband: {actual_1000_band}")

    figure, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.16, 3.25),
        gridspec_kw={"width_ratios": [1.42, 1.0]},
    )
    figure.subplots_adjust(left=0.078, right=0.992, bottom=0.245, top=0.88, wspace=0.34)

    # Panel (a): the effective forward-backward response on the true time axis.
    ax_a.axvspan(
        actual_1000_band[0],
        actual_1000_band[1],
        color=ORANGE,
        alpha=0.075,
        linewidth=0,
        zorder=0,
    )
    ax_a.axvspan(
        target_band[0],
        target_band[1],
        color=BLUE,
        alpha=0.065,
        linewidth=0,
        zorder=0,
    )
    ax_a.plot(
        frequencies,
        response_1000,
        color=ORANGE,
        linewidth=1.75,
        linestyle=(0, (4, 2)),
        label=r"Configured $f_s=1000$ Hz",
        zorder=3,
    )
    ax_a.plot(
        frequencies,
        response_200,
        color=BLUE,
        linewidth=1.75,
        label=r"Aligned $f_s=200$ Hz",
        zorder=4,
    )
    for boundary in actual_1000_band:
        ax_a.axvline(boundary, color=ORANGE, linewidth=0.7, linestyle=":", alpha=0.9)
    for boundary in target_band:
        ax_a.axvline(boundary, color=BLUE, linewidth=0.7, linestyle=":", alpha=0.9)

    add_double_arrow(
        ax_a,
        actual_1000_band[0],
        actual_1000_band[1],
        0.12,
        "Actual 0.1–10 Hz",
        ORANGE,
    )
    add_double_arrow(
        ax_a,
        target_band[0],
        target_band[1],
        0.27,
        "Target 0.5–50 Hz",
        BLUE,
    )
    ax_a.set_xscale("log")
    ax_a.set_xlim(0.05, 100.0)
    ax_a.set_ylim(-0.01, 1.08)
    tick_positions = [0.1, 0.5, 1.0, 10.0, 50.0, 100.0]
    ax_a.xaxis.set_major_locator(FixedLocator(tick_positions))
    ax_a.xaxis.set_major_formatter(
        FixedFormatter(["0.1", "0.5", "1", "10", "50", "100"])
    )
    ax_a.minorticks_off()
    ax_a.set_yticks([0.0, 0.5, 1.0])
    ax_a.set_xlabel("Physical frequency on XRF55 time axis (Hz)")
    ax_a.set_ylabel(r"Effective magnitude, $|H(f)|^2$")
    ax_a.set_title("Zero-phase bandpass response (XRF55: 200 Hz)", pad=7)
    ax_a.grid(axis="y", color=LIGHT_GRID, linewidth=0.55, alpha=0.75)
    ax_a.legend(loc="upper left", ncol=1, handlelength=2.5)

    # Panel (b): exact single-seed, matched-pipeline classification results.
    x = np.arange(3)
    colors = [ORANGE, BLUE, GREEN]
    bars = ax_b.bar(
        x,
        accuracies,
        width=0.62,
        color=colors,
        edgecolor=CHARCOAL,
        linewidth=0.75,
        zorder=3,
    )
    for bar, value in zip(bars, accuracies):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 1.6,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="semibold",
        )

    improvement = accuracies[1] - accuracies[0]
    remaining_gap = accuracies[2] - accuracies[1]
    add_bracket(ax_b, 0.0, 1.0, 70.0, f"+{improvement:.2f} pp", BLUE)
    add_bracket(ax_b, 1.0, 2.0, 94.0, f"{remaining_gap:.2f} pp gap", CHARCOAL)

    ax_b.set_ylim(0.0, 105.0)
    ax_b.set_yticks(np.arange(0, 101, 20))
    ax_b.set_ylabel("Test accuracy (%)")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(
        [
            "Bandpass\n" + r"$f_s=1000$ Hz",
            "Bandpass\n" + r"$f_s=200$ Hz",
            "Savgol\n(matched)",
        ]
    )
    ax_b.set_title("Matched downstream configuration", pad=7)
    ax_b.text(
        0.5,
        0.985,
        f"IQR + z-score + cubic15; seed = {SEED}",
        transform=ax_b.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
        color="#555555",
    )
    ax_b.grid(axis="y", color=LIGHT_GRID, linewidth=0.55, alpha=0.75, zorder=0)

    for axis, label in [(ax_a, "a"), (ax_b, "b")]:
        axis.text(
            -0.13,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
            ha="left",
            va="top",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    output_stem = HERE / "fig_x_sampling_rate"
    for extension in ("png", "pdf", "svg"):
        options = {"bbox_inches": "tight", "pad_inches": 0.025}
        if extension == "png":
            options["dpi"] = 600
        figure.savefig(output_stem.with_suffix(f".{extension}"), **options)
    plt.close(figure)

    print(f"Source accuracies (%): {accuracies.tolist()}")
    print(f"Sampling-rate correction: +{improvement:.6f} percentage points")
    print(f"Remaining gap to matched Savgol: {remaining_gap:.6f} percentage points")
    print(f"Misconfigured physical passband: {actual_1000_band[0]:.1f}–{actual_1000_band[1]:.1f} Hz")
    print(f"Target/aligned passband: {target_band[0]:.1f}–{target_band[1]:.1f} Hz")
    print(f"Saved: {output_stem}.{{png,pdf,svg}}")


if __name__ == "__main__":
    render()
