#!/usr/bin/env python3
"""Render Fig. Z: ElderAL short-sequence bypass behavior.

The figure is derived from the per-sample Bandpass signal audit. Panel (a)
shows the raw sequence-length distribution and the source-code threshold.
Panel (b) compares the input/output NRMSE for bypassed and filtered samples.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    REPO_ROOT / "ablation" / "bandpass_server_results" / "signal_analysis"
)
DEFAULT_OUTPUT_PREFIX = Path(__file__).resolve().parent / "fig_z_elder_bypass"

# Okabe--Ito colors: color-blind safe and stable in grayscale.
BLUE = "#0072B2"
ORANGE = "#E69F00"
RED = "#D55E00"
CHARCOAL = "#333333"
LIGHT_GRAY = "#D9D9D9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-sample",
        type=Path,
        default=DEFAULT_INPUT_DIR / "elder_bandpass_per_sample.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_INPUT_DIR / "elder_bandpass_summary.json",
    )
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX
    )
    return parser.parse_args()


def load_and_validate(
    per_sample_path: Path, summary_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with per_sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    frames = np.asarray([int(row["frames"]) for row in rows], dtype=int)
    nrmse = np.asarray([float(row["nrmse"]) for row in rows], dtype=float)
    exact = np.asarray(
        [row["output_exactly_equal_to_input"].lower() == "true" for row in rows],
        dtype=bool,
    )
    threshold = int(summary["source_min_length"])
    bypass = frames < threshold

    # Fail closed if the source audit and its summary no longer agree.
    assert len(rows) == int(summary["valid_samples"]), "valid-sample mismatch"
    assert int(bypass.sum()) == int(summary["bypass_samples"]), "bypass mismatch"
    assert int((~bypass).sum()) == int(summary["eligible_samples"]), "filter mismatch"
    assert bool(np.all(exact[bypass])), "not every bypassed sample is an exact match"
    assert bool(np.all(~exact[~bypass])), "an eligible sample was not changed"
    assert np.allclose(nrmse[bypass], 0.0), "bypass NRMSE is not exactly zero"
    assert np.isclose(
        np.median(nrmse[~bypass]),
        float(summary["eligible_nrmse_median"]),
        rtol=0,
        atol=1e-12,
    ), "filtered-sample median mismatch"
    return frames, nrmse, bypass, summary


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": CHARCOAL,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.045,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=CHARCOAL,
    )


def render(
    frames: np.ndarray,
    nrmse: np.ndarray,
    bypass: np.ndarray,
    summary: dict,
    output_prefix: Path,
) -> list[Path]:
    set_publication_style()
    threshold = int(summary["source_min_length"])
    bypass_count = int(summary["bypass_samples"])
    filtered_count = int(summary["eligible_samples"])
    total_count = int(summary["valid_samples"])
    bypass_rate = 100.0 * float(summary["bypass_sample_rate"])
    filtered_median = float(summary["eligible_nrmse_median"])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.15, 3.05))

    # Panel (a): one-frame bins retain the discrete raw-length distribution.
    bins = np.arange(frames.min() - 0.5, frames.max() + 1.5, 1.0)
    ax_a.axvspan(
        frames.min() - 0.5,
        threshold,
        color=ORANGE,
        alpha=0.12,
        linewidth=0,
        zorder=0,
    )
    ax_a.hist(
        frames,
        bins=bins,
        color=BLUE,
        edgecolor="white",
        linewidth=0.28,
        zorder=2,
    )
    ax_a.axvline(
        threshold,
        color=RED,
        linestyle=(0, (4, 2)),
        linewidth=1.25,
        zorder=3,
    )
    ax_a.text(
        threshold + 1.6,
        0.91,
        "Filter threshold\n$T=28$ frames",
        transform=ax_a.get_xaxis_transform(),
        color=RED,
        ha="left",
        va="top",
        fontsize=7.3,
    )
    ax_a.text(
        0.97,
        0.68,
        (
            "Short-sequence bypass\n"
            f"{bypass_count}/{total_count} samples\n"
            f"({bypass_rate:.2f}%)"
        ),
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": LIGHT_GRAY,
            "linewidth": 0.65,
            "alpha": 0.96,
        },
    )
    ax_a.set_xlabel("Raw sequence length, $T$ (frames)")
    ax_a.set_ylabel("Number of samples")
    ax_a.set_xlim(frames.min() - 2, frames.max() + 3)
    ax_a.set_ylim(bottom=0)
    panel_label(ax_a, "a")

    # Panel (b): compact boxplots plus every audited sample (deterministic jitter).
    groups = [nrmse[bypass], nrmse[~bypass]]
    positions = [1, 2]
    bp = ax_b.boxplot(
        groups,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.5},
        whiskerprops={"color": CHARCOAL, "linewidth": 0.9},
        capprops={"color": CHARCOAL, "linewidth": 0.9},
        boxprops={"edgecolor": CHARCOAL, "linewidth": 0.9},
    )
    for patch, color in zip(bp["boxes"], [ORANGE, BLUE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.86)

    rng = np.random.default_rng(20250815)
    for position, values, color in zip(positions, groups, [ORANGE, BLUE]):
        jitter = rng.uniform(-0.17, 0.17, size=len(values))
        ax_b.scatter(
            position + jitter,
            values,
            s=5.5,
            facecolor=color,
            edgecolor="none",
            alpha=0.16 if position == 1 else 0.34,
            rasterized=False,
            zorder=1,
        )

    ax_b.axhline(0, color=CHARCOAL, linewidth=0.65, zorder=0)
    ax_b.annotate(
        "All exact matches\nmedian = 0",
        xy=(1, 0),
        xytext=(1.22, 0.24),
        textcoords="data",
        ha="center",
        va="center",
        color=CHARCOAL,
        fontsize=7.3,
        arrowprops={
            "arrowstyle": "-",
            "color": CHARCOAL,
            "linewidth": 0.7,
            "shrinkA": 2,
            "shrinkB": 3,
        },
    )
    ax_b.annotate(
        f"median = {filtered_median:.3f}",
        xy=(2.0, filtered_median),
        xytext=(1.65, 1.23),
        textcoords="data",
        ha="left",
        va="center",
        color=BLUE,
        fontsize=7.5,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-",
            "color": BLUE,
            "linewidth": 0.8,
            "shrinkA": 2,
            "shrinkB": 3,
        },
    )
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(
        [
            f"Bypassed\n$T<28$ ($n={bypass_count}$)",
            f"Filtered\n$T\\geq28$ ($n={filtered_count}$)",
        ]
    )
    ax_b.set_ylabel("Normalized RMSE")
    ax_b.set_xlim(0.55, 2.45)
    ax_b.set_ylim(-0.075, 1.43)
    panel_label(ax_b, "b")

    for ax in (ax_a, ax_b):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#ECECEC", linewidth=0.55, zorder=0)

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.205, top=0.94, wspace=0.34)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("png", "pdf", "svg"):
        path = output_prefix.with_suffix(f".{suffix}")
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    frames, nrmse, bypass, summary = load_and_validate(
        args.per_sample, args.summary
    )
    outputs = render(frames, nrmse, bypass, summary, args.out_prefix)
    print(
        json.dumps(
            {
                "valid_samples": int(len(frames)),
                "bypass_samples": int(bypass.sum()),
                "bypass_rate_percent": 100.0 * float(bypass.mean()),
                "filtered_samples": int((~bypass).sum()),
                "bypass_nrmse_median": float(np.median(nrmse[bypass])),
                "filtered_nrmse_median": float(np.median(nrmse[~bypass])),
                "outputs": [str(path) for path in outputs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
