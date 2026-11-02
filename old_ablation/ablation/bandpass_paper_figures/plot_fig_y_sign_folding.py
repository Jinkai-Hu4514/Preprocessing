#!/usr/bin/env python3
"""Generate Fig. Y: signed bandpass output and downstream absolute-value folding.

Panel (a) is an explicitly illustrative, fixed XRF55 example. Panel (b) uses
the complete 3,300-sample signal audit stored by bandpass_server_signal_analysis.py.
The script validates the audit scope before plotting and reproduces the repository's
Butterworth bandpass implementation directly from its source file.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bandpass_paper_mplconfig")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_STEM = Path(__file__).resolve().parent / "fig_y_sign_folding"
SUMMARY_PATH = (
    PROJECT_ROOT
    / "ablation/bandpass_server_results/signal_analysis/xrf55_negative_summary.json"
)
EXAMPLE_PATH = PROJECT_ROOT / "sdp_dataset/xrf55/wifi/01_01_01.npy"
FILTER_SOURCE = (
    PROJECT_ROOT
    / "SDP/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
    / "src/wsdp/algorithms/denoising_butterworth.py"
)

# Fixed example coordinates, matching the existing diagnostic for 01_01_01.npy.
# Labels below are one-based for readability; array indices remain zero-based.
EXAMPLE_SUBCARRIER_INDEX = 26
EXAMPLE_LINK_INDEX = 0
EXAMPLE_START_FRAME = 430
EXAMPLE_STOP_FRAME = 731
XRF55_FS_HZ = 200.0

# Okabe-Ito-derived, color-vision-safe colors.
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
CHARCOAL = "#2B2B2B"
LIGHT_GRAY = "#D9D9D9"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_bandpass_function():
    spec = importlib.util.spec_from_file_location("wsdp_bandpass_source", FILTER_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import filter source: {FILTER_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.butterworth_bandpass


def load_example() -> tuple[np.ndarray, np.ndarray]:
    raw_file = np.load(EXAMPLE_PATH)
    if raw_file.shape != (270, 1000):
        raise ValueError(f"Unexpected XRF55 array shape: {raw_file.shape}")

    # Reproduce XrfReader's layout: (Rx, Sub, Ant, T) -> (T, Sub, Rx*Ant).
    raw = (
        raw_file.reshape(3, 30, 3, 1000)
        .transpose(3, 1, 0, 2)
        .reshape(1000, 30, 9)
    )
    raw_channel = raw[:, EXAMPLE_SUBCARRIER_INDEX, EXAMPLE_LINK_INDEX]
    bandpass = load_bandpass_function()
    signed = bandpass(
        raw_channel[:, None],
        order=4,
        low_freq=0.5,
        high_freq=50.0,
        fs=XRF55_FS_HZ,
    )[:, 0]
    return raw_channel, np.asarray(signed, dtype=np.float64)


def load_complete_audit() -> dict[str, dict[str, float]]:
    with SUMMARY_PATH.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)

    discovery = summary["discovery"]
    assert discovery["valid_records"] == 3300
    assert discovery["fully_analyzed_records"] == 3300
    assert discovery["failed_files"] == 0
    assert discovery["selected_user_ids"] == [1, 2, 3]

    rows = {row["method"]: row for row in summary["methods"]}
    for method in ("bandpass_fs1000", "bandpass_fs200"):
        assert rows[method]["samples"] == 3300
    return rows


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def clean_axes(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)


def make_figure() -> mpl.figure.Figure:
    configure_style()
    raw_full, signed_full = load_example()
    audit = load_complete_audit()

    selection = slice(EXAMPLE_START_FRAME, EXAMPLE_STOP_FRAME)
    raw = raw_full[selection]
    signed = signed_full[selection]
    folded = np.abs(signed)
    time_s = np.arange(EXAMPLE_START_FRAME, EXAMPLE_STOP_FRAME) / XRF55_FS_HZ

    fig = plt.figure(figsize=(7.2, 3.3))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.32)
    example_grid = outer[0].subgridspec(
        2,
        1,
        height_ratios=[0.62, 1.38],
        hspace=0.12,
    )
    raw_ax = fig.add_subplot(example_grid[0])
    signal_ax = fig.add_subplot(example_grid[1], sharex=raw_ax)
    summary_ax = fig.add_subplot(outer[1])

    # (a) Fixed illustrative sample; no claim of representativeness.
    ax = raw_ax
    ax.plot(time_s, raw, color=BLUE, linewidth=1.1)
    ax.set_ylabel("Raw input\n(a.u.)")
    ax.set_title(
        "Illustrative example: 01_01_01, subcarrier 27, link 1",
        loc="left",
        pad=5,
    )
    ax.text(
        0.015,
        0.88,
        "Original input",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=BLUE,
    )
    ax.tick_params(labelbottom=False)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5, alpha=0.65)
    clean_axes(ax)
    panel_label(ax, "a")

    ax = signal_ax
    ax.axhline(0, color=CHARCOAL, linewidth=0.85, alpha=0.8, zorder=1)
    negative = signed < 0
    ax.fill_between(
        time_s,
        signed,
        0,
        where=negative,
        color=ORANGE,
        alpha=0.13,
        interpolate=True,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        time_s,
        signed,
        color=GREEN,
        label="Signed bandpass output",
        zorder=3,
    )
    ax.plot(
        time_s,
        folded,
        color=ORANGE,
        linestyle=(0, (3.2, 2.0)),
        linewidth=1.15,
        label=r"After folding, $|x|$",
        zorder=2,
    )
    # Use the first pronounced negative lobe so the annotation stays clear of
    # the legend and does not imply that this fixed example is representative.
    fold_index = int(np.argmin(signed[:120]))
    ax.annotate(
        "",
        xy=(time_s[fold_index], folded[fold_index]),
        xytext=(time_s[fold_index], signed[fold_index]),
        arrowprops={
            "arrowstyle": "->",
            "color": CHARCOAL,
            "linewidth": 0.75,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )
    ax.text(
        time_s[fold_index] + 0.035,
        0.35,
        "folded\nupward",
        fontsize=6.8,
        ha="left",
        va="center",
        color=CHARCOAL,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Filtered\nsignal (a.u.)")
    ax.text(
        0.015,
        0.96,
        r"Bandpass: 0.5--50 Hz, $f_s=200$ Hz",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=CHARCOAL,
    )
    ax.legend(frameon=False, loc="upper right", handlelength=2.7)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5, alpha=0.65)
    clean_axes(ax)

    # (b) Exact complete-population metrics from the 3,300-sample audit.
    metric_keys = [
        "element_weighted_meaningful_rate",
        "mean_negative_energy_fraction",
        "element_weighted_abs_slope_direction_changed_or_lost_rate",
    ]
    metric_labels = [
        "Meaningful\nnegative values",
        "Negative-energy\nshare",
        "Slope direction\nchanged/lost",
    ]
    fs1000 = 100 * np.asarray(
        [float(audit["bandpass_fs1000"][key]) for key in metric_keys]
    )
    fs200 = 100 * np.asarray(
        [float(audit["bandpass_fs200"][key]) for key in metric_keys]
    )

    # Fail closed if the trusted audit changes unexpectedly.
    np.testing.assert_allclose(
        fs1000,
        [49.759202356902354, 50.45685738496705, 49.75924747416297],
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        fs200,
        [49.99104747474747, 50.32305715584755, 49.960397225782854],
        rtol=0,
        atol=1e-10,
    )

    ax = summary_ax
    x = np.arange(len(metric_keys))
    width = 0.34
    bars_1000 = ax.bar(
        x - width / 2,
        fs1000,
        width,
        color=BLUE,
        edgecolor=CHARCOAL,
        linewidth=0.55,
        label=r"$f_s=1000$ Hz",
    )
    bars_200 = ax.bar(
        x + width / 2,
        fs200,
        width,
        color=GREEN,
        edgecolor=CHARCOAL,
        linewidth=0.55,
        label=r"$f_s=200$ Hz",
    )
    ax.set_xticks(x, metric_labels)
    ax.set_ylabel("Rate or share (%)")
    ax.set_ylim(0, 57)
    ax.set_title("Complete audit: 3,300 samples", loc="left", pad=5)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        columnspacing=0.9,
    )
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5, alpha=0.65)
    for bars in (bars_1000, bars_200):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.65,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=0,
            )
    clean_axes(ax)
    panel_label(ax, "b")

    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.17, top=0.88)
    return fig


def main() -> None:
    fig = make_figure()
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_STEM}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
