"""Compare model rankings across the six presets using the saved seed summaries."""

import argparse
from itertools import combinations
from math import ceil
from pathlib import Path

import matplotlib
import pandas as pd
from scipy.stats import kendalltau

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT = ANALYSIS_DIR.parent
DATASETS = {
    "ElderAL": ("test_elder", "elder"),
    "XRF55": ("test_xrf55", "xrf55"),
    "Widar3.0": ("test_widar3.0", "widar3.0"),
    "Gait": ("test_gait", "gait"),
}
PRESET_LABELS = {
    "high_quality": "High\nquality",
    "fast": "Fast",
    "robust": "Robust",
    "gesture_recognition": "Gesture\nrecognition",
    "activity_detection": "Activity\ndetection",
    "localization": "Localization",
}
PRESETS = tuple(PRESET_LABELS)
SEEDS = (42, 727, 914)
REFERENCE = "high_quality"
HEATMAP_CMAP = "Blues"


def load_seed_scores(dataset, metric):
    folder, prefix = DATASETS[dataset]
    runs = []
    for seed in SEEDS:
        path = ROOT / folder / "preset_test" / f"result{seed}" / f"{prefix}_preset_summary.csv"
        rows = pd.read_csv(path, float_precision="round_trip")
        if not rows["seed"].eq(seed).all() or rows.duplicated(["model", "preset"]).any():
            raise ValueError(f"Mismatched seed or duplicate model/preset in {path}")
        if set(rows["preset"]) != set(PRESETS):
            raise ValueError(f"Expected all six presets in {path}")
        splits = rows["split_digest"]
        if splits.isna().any() or splits.nunique() != 1:
            raise ValueError(f"Expected the same split for all runs in {path}")
        scores = pd.to_numeric(rows[metric].where(rows["status"].eq("ok")))
        if not scores.dropna().between(0, 1).all():
            raise ValueError(f"Expected {metric} between 0 and 1 in {path}")
        runs.append(rows.assign(score=scores))

    columns = pd.MultiIndex.from_product([PRESETS, SEEDS], names=["preset", "seed"])
    return pd.concat(runs).pivot(index="model", columns=["preset", "seed"], values="score").reindex(
        columns=columns
    )


def top_models(scores):
    # Equality to the maximum also captures ties for first place.
    return " | ".join(scores.index[scores.eq(scores.max())])


def analyze_dataset(dataset, metric):
    seed_scores = load_seed_scores(dataset, metric)
    seed_counts = pd.DataFrame({p: seed_scores[p].count(axis=1) for p in PRESETS})
    means = pd.DataFrame({p: seed_scores[p].mean(axis=1, skipna=False) for p in PRESETS})
    # Remove floating-point artifacts before comparing scores for ties.
    scores = means.round(12)
    common = scores.dropna()
    if len(common) < 2:
        raise ValueError(f"{dataset} has fewer than two models with complete results")
    within_ranks = scores.rank(ascending=False, method="average")
    common_ranks = common.rank(ascending=False, method="average")

    winners = []
    ranks = []
    for preset in PRESETS:
        winners.append(
            {
                "dataset": dataset,
                "preset": preset,
                "n_models_in_preset": scores[preset].count(),
                "top_models_in_preset": top_models(scores[preset]),
                "n_common_models": len(common),
                "top_models_common": top_models(common[preset]),
            }
        )
        preset_ranks = pd.DataFrame(
            {
                "dataset": dataset,
                "preset": preset,
                "n_valid_seeds": seed_counts[preset],
                f"mean_{metric}": means[preset],
                "rank_in_preset": within_ranks[preset],
                "rank_common": common_ranks[preset],
                "rank_gain_vs_high_quality": common_ranks[REFERENCE] - common_ranks[preset],
            }
        )
        ranks.append(preset_ranks.sort_values("rank_in_preset", kind="stable").reset_index())

    rank_changes = common_ranks.assign(
        rank_span=common_ranks.max(axis=1) - common_ranks.min(axis=1)
    ).sort_values("rank_span", ascending=False, kind="stable")
    rank_changes.insert(0, "dataset", dataset)

    matrix = pd.DataFrame(float("nan"), index=PRESETS, columns=PRESETS)
    for preset in PRESETS:
        if common[preset].nunique() > 1:
            matrix.loc[preset, preset] = 1.0

    pairs = []
    common_winners = {row["preset"]: row["top_models_common"] for row in winners}
    for a, b in combinations(PRESETS, 2):
        tau = kendalltau(common[a], common[b], variant="b").statistic
        matrix.loc[a, b] = matrix.loc[b, a] = tau
        reversed_pairs = sum(
            (common.at[m, a] - common.at[n, a]) * (common.at[m, b] - common.at[n, b]) < 0
            for m, n in combinations(common.index, 2)
        )
        pairs.append(
            {
                "dataset": dataset,
                "preset_a": a,
                "preset_b": b,
                "n_common_models": len(common),
                "kendall_tau_b": tau,
                "n_reversed_model_pairs": reversed_pairs,
                "top_models_a": common_winners[a],
                "top_models_b": common_winners[b],
                "top_model_set_changed": common_winners[a] != common_winners[b],
            }
        )

    return pd.concat(ranks), rank_changes.reset_index(), winners, pairs, matrix


def heatmap_text_color(image, value):
    rgb = image.cmap(image.norm(value))[:3]
    linear_rgb = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in rgb
    ]
    luminance = sum(weight * channel for weight, channel in zip((0.2126, 0.7152, 0.0722), linear_rgb))
    return "white" if luminance < 0.18 else "black"


def add_heatmap_grid(ax, shape):
    rows, columns = shape
    ax.set_xticks([col + 0.5 for col in range(columns - 1)], minor=True)
    ax.set_yticks([row + 0.5 for row in range(rows - 1)], minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", linewidth=0.65)
    ax.spines[:].set_visible(False)


def save_heatmap(fig, path):
    fig.savefig(path, dpi=200)
    with plt.rc_context({"pdf.fonttype": 42}):
        fig.savefig(path.with_suffix(".pdf"), dpi=200)
    plt.close(fig)


def plot_kendall(matrices, counts, metric, path):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), layout="constrained")
    labels = list(PRESET_LABELS.values())
    for ax, (dataset, matrix) in zip(axes.flat, matrices.items()):
        values = matrix.to_numpy()
        image = ax.imshow(values, cmap=HEATMAP_CMAP, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"{dataset} ({counts[dataset]} common models)")
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.tick_params(length=0)
        add_heatmap_grid(ax, values.shape)
        for (row, column), value in matrix.stack().items():
            ax.text(
                PRESETS.index(column), PRESETS.index(row), f"{value:.2f}",
                ha="center", va="center", color=heatmap_text_color(image, value),
            )
    metric_label = "test accuracy" if metric == "test_acc" else "macro-F1"
    fig.suptitle(f"Model ranking agreement across preprocessing presets ({metric_label})")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Kendall's τ-b")
    save_heatmap(fig, path)


def plot_model_rank_heatmap(rank_changes, path):
    ranks_by_dataset = {
        dataset: group.set_index("model")[list(PRESETS)].sort_values(REFERENCE, kind="stable")
        for dataset, group in rank_changes.groupby("dataset", sort=False)
    }
    gains_by_dataset = {
        dataset: ranks.rsub(ranks[REFERENCE], axis="index")
        for dataset, ranks in ranks_by_dataset.items()
    }
    max_gain = max(gains.abs().to_numpy().max() for gains in gains_by_dataset.values())
    limit = max(1, ceil(max_gain))

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.8))
    fig.subplots_adjust(left=0.145, right=0.97, top=0.94, bottom=0.14, wspace=0.34, hspace=0.28)
    for ax, (dataset, ranks) in zip(axes.flat, ranks_by_dataset.items()):
        gains = gains_by_dataset[dataset].to_numpy()
        image = ax.imshow(
            gains, cmap=HEATMAP_CMAP, vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest"
        )
        ax.set_title(f"{dataset} (n = {len(ranks)})", loc="left", fontweight="bold", pad=9)
        ax.set_xticks(range(len(PRESETS)), PRESET_LABELS.values(), fontsize=9)
        ax.set_yticks(range(len(ranks)), ranks.index, fontsize=9)
        ax.tick_params(which="both", length=0, pad=5)
        add_heatmap_grid(ax, gains.shape)

        for row, rank_values in enumerate(ranks.to_numpy()):
            for col, rank in enumerate(rank_values):
                color = heatmap_text_color(image, gains[row, col])
                ax.text(col, row, f"{rank:g}", ha="center", va="center", color=color, fontsize=9.5)

    colorbar_ax = fig.add_axes([0.36, 0.057, 0.32, 0.015])
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal", ticks=[-limit, 0, limit])
    colorbar.ax.set_xticklabels([f"−{limit}", "0", f"+{limit}"])
    colorbar.ax.tick_params(length=0, pad=4)
    colorbar.set_label("Places gained vs. High quality")
    colorbar.outline.set_visible(False)
    save_heatmap(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=("test_acc", "macro_f1"), default="test_acc")
    parser.add_argument(
        "--output-dir", type=Path, help="Defaults to preset_rankings/results with a _macro_f1 suffix for Macro-F1 files."
    )
    args = parser.parse_args()

    ranks, changes, winners, pairs, matrices, counts = [], [], [], [], {}, {}
    for dataset in DATASETS:
        dataset_ranks, dataset_changes, dataset_winners, dataset_pairs, matrix = analyze_dataset(
            dataset, args.metric
        )
        ranks.append(dataset_ranks)
        changes.append(dataset_changes)
        winners.extend(dataset_winners)
        pairs.extend(dataset_pairs)
        matrices[dataset] = matrix
        counts[dataset] = len(dataset_changes)

    results_dir = ANALYSIS_DIR / "results"
    output_dir = args.output_dir or results_dir
    suffix = "_macro_f1" if args.metric == "macro_f1" and output_dir.resolve() == results_dir else ""
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_changes = pd.concat(changes, ignore_index=True)
    pd.concat(ranks).to_csv(output_dir / f"model_ranks{suffix}.csv", index=False)
    rank_changes.to_csv(output_dir / f"model_rank_changes{suffix}.csv", index=False)
    pd.DataFrame(winners).to_csv(output_dir / f"preset_winners{suffix}.csv", index=False)
    pd.DataFrame(pairs).to_csv(output_dir / f"kendall_pairs{suffix}.csv", index=False)
    plot_kendall(matrices, counts, args.metric, output_dir / f"kendall_heatmaps{suffix}.png")
    plot_model_rank_heatmap(rank_changes, output_dir / f"model_rank_heatmap{suffix}.png")
    print(f"Wrote preset ranking results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
