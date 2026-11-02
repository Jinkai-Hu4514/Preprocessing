#!/usr/bin/env python3
"""Plot cross-dataset signal evidence for the Robust root cause.

The plotted confidence intervals describe variation across the 64 diagnostic
samples in each dataset.  They are not model-training uncertainty estimates.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
DATASETS = {
    "Widar": ROOT / "ablation/robust_rootcause_results/widar/variant_diagnostics_per_sample.csv",
    "Gait": ROOT / "ablation/robust_rootcause_results/gait/variant_diagnostics_per_sample.csv",
}
CONDITIONS = [
    "common_only",
    "detrend_first50_only",
    "robust_first50",
    "robust_shared_first50",
]
LABELS = [
    "Common phase\nonly",
    "Independent\ndetrend only",
    "Original\nrobust",
    "Shared\nslope",
]
COLORS = {"Widar": "#3775BA", "Gait": "#E1812C"}
MARKERS = {"Widar": "o", "Gait": "s"}


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Return mean and deterministic percentile-bootstrap 95% interval."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10_000, len(values)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def load_statistics() -> pd.DataFrame:
    rows = []
    metrics = [
        "adjacent_phase_jump_late_mean_rad",
        "late_relative_rotation_gt_pi_over_2",
    ]
    seed = 42
    for dataset, path in DATASETS.items():
        frame = pd.read_csv(path)
        subset = frame.loc[frame["variant"].isin(CONDITIONS)].copy()
        counts = subset.groupby("variant").size().reindex(CONDITIONS)
        if not counts.eq(64).all():
            raise RuntimeError(f"{dataset}: expected 64 diagnostic samples per condition")
        for condition in CONDITIONS:
            condition_rows = subset.loc[subset["variant"].eq(condition)]
            for metric in metrics:
                mean, low, high = bootstrap_mean_ci(
                    condition_rows[metric].to_numpy(), seed
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "n": len(condition_rows),
                    }
                )
                seed += 1

    stats = pd.DataFrame(rows)
    expected = {
        ("Widar", "common_only"): (0.811821, 0.0),
        ("Widar", "detrend_first50_only"): (1.438973, 0.424687),
        ("Widar", "robust_first50"): (1.443611, 0.418520),
        ("Widar", "robust_shared_first50"): (0.811821, 0.0),
        ("Gait", "common_only"): (0.819748, 0.0),
        ("Gait", "detrend_first50_only"): (1.548432, 0.493583),
        ("Gait", "robust_first50"): (1.551651, 0.486217),
        ("Gait", "robust_shared_first50"): (0.819700, 0.001254),
    }
    for (dataset, condition), (jump, rotation) in expected.items():
        got_jump = stats.loc[
            (stats["dataset"].eq(dataset))
            & (stats["condition"].eq(condition))
            & stats["metric"].eq("adjacent_phase_jump_late_mean_rad"),
            "mean",
        ].item()
        got_rotation = stats.loc[
            (stats["dataset"].eq(dataset))
            & (stats["condition"].eq(condition))
            & stats["metric"].eq("late_relative_rotation_gt_pi_over_2"),
            "mean",
        ].item()
        assert np.isclose(got_jump, jump, atol=5e-6)
        assert np.isclose(got_rotation, rotation, atol=5e-6)
    return stats


def plot_metric(ax, stats: pd.DataFrame, metric: str, percent: bool = False) -> None:
    x = np.arange(len(CONDITIONS), dtype=float)
    offsets = {"Widar": -0.105, "Gait": 0.105}

    ax.axvspan(0.55, 2.45, color="#F6CFCB", alpha=0.42, zorder=0)
    ax.axvspan(2.55, 3.45, color="#DCEEDC", alpha=0.55, zorder=0)

    scale = 100.0 if percent else 1.0
    for dataset in DATASETS:
        selected = (
            stats.loc[
                stats["dataset"].eq(dataset) & stats["metric"].eq(metric)
            ]
            .set_index("condition")
            .loc[CONDITIONS]
        )
        means = scale * selected["mean"].to_numpy()
        lows = scale * selected["ci_low"].to_numpy()
        highs = scale * selected["ci_high"].to_numpy()
        xpos = x + offsets[dataset]
        ax.errorbar(
            xpos,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            fmt=MARKERS[dataset],
            color=COLORS[dataset],
            markerfacecolor=COLORS[dataset],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=7.2,
            elinewidth=1.6,
            capsize=3.5,
            capthick=1.4,
            label=dataset,
            zorder=4,
        )
        for xx, yy in zip(xpos, means):
            label = f"{yy:.1f}" if percent else f"{yy:.3f}"
            text_x = -13 if dataset == "Widar" else 13
            ax.annotate(
                label,
                (xx, yy),
                xytext=(text_x, 8),
                textcoords="offset points",
                ha="right" if dataset == "Widar" else "left",
                va="bottom",
                fontsize=7.4,
                color=COLORS[dataset],
                fontweight="bold",
            )

    ax.set_xticks(x, LABELS)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.set_xlim(-0.45, 3.45)


def render(stats: pd.DataFrame) -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.7))

    plot_metric(
        axes[0], stats, "adjacent_phase_jump_late_mean_rad", percent=False
    )
    axes[0].set_ylabel(
        r"Mean $|\mathrm{wrap}(\Delta\phi_{\rm adj})|$, final quarter (rad)"
    )
    axes[0].set_ylim(0.65, 1.72)

    plot_metric(
        axes[1], stats, "late_relative_rotation_gt_pi_over_2", percent=True
    )
    axes[1].set_ylabel(
        r"Pairs with additional rotation $>90^{\circ}$" "\nvs common-only, final quarter (%)"
    )
    axes[1].set_ylim(-2.5, 58.5)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 1.005),
        frameon=False,
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.4,
    )

    axes[0].text(-0.13, 1.04, "a", transform=axes[0].transAxes,
                 fontsize=16, fontweight="bold")
    axes[1].text(-0.13, 1.04, "b", transform=axes[1].transAxes,
                 fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        -0.01,
        "Mean and 95% bootstrap CI across 64 diagnostic samples per dataset",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.25, wspace=0.31)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT_DIR / f"fig_y_robust_signal_structure.{ext}",
            dpi=500 if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    stats.to_csv(OUT_DIR / "fig_y_robust_signal_structure_statistics.csv", index=False)


if __name__ == "__main__":
    render(load_statistics())
