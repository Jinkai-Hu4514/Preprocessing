"""Render the ElderAL preset test-accuracy heatmap from the three seed summaries."""

import csv
from pathlib import Path
from statistics import mean

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR.parents[1] / "preset_test"
SEEDS = (42, 727, 914)
# The ElderAL split manifest and training logs contain six activity labels.
RANDOM_BASELINE = 100 / 6

PRESETS = {
    "high_quality": "High\nQuality",
    "fast": "Fast",
    "robust": "Robust",
    "gesture_recognition": "Gesture\nRecognition",
    "activity_detection": "Activity\nDetection",
    "localization": "Localization",
}
MODELS = (
    "csimodel",
    "mlpmodel",
    "cnn1dmodel",
    "cnn2dmodel",
    "resnet1d",
    "resnet2d",
    "lstmmodel",
    "bilstmattention",
    "attentiongru",
    "csitime",
    "ei",
    "fewsense",
    "pa_csi",
    "that",
    "wiflexformer",
    "efficientnetcsi",
    "mambacsi",
    "graphneuralcsi",
    "visiontransformercsi",
)
MISSING_STYLE = {"facecolor": "#eeeeee", "edgecolor": "#9a9a9a", "hatch": "////"}


def load_mean_accuracy(results_dir):
    """Return mean test accuracy (%) for cells with three successful scores."""
    scores = {}
    for seed in SEEDS:
        path = results_dir / f"result{seed}" / "elder_preset_summary.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (seed, row["model"], row["preset"])
                if int(row["seed"]) != seed or key in scores:
                    raise ValueError(f"Mismatched or duplicate run {key} in {path}")
                scores[key] = float("nan")
                if row["status"] == "ok" and row["test_acc"].strip():
                    scores[key] = float(row["test_acc"])

    accuracy = np.full((len(MODELS), len(PRESETS)), np.nan)
    for y, model in enumerate(MODELS):
        for x, preset in enumerate(PRESETS):
            runs = [scores.get((seed, model, preset), np.nan) for seed in SEEDS]
            if all(0 <= value <= 1 for value in runs):
                accuracy[y, x] = 100 * mean(runs)
    return accuracy


def build_heatmap(accuracy):
    fig, (ax, cax) = plt.subplots(
        ncols=2, figsize=(8.1, 7.5), width_ratios=(1, 0.035)
    )
    fig.subplots_adjust(left=0.15, right=0.885, bottom=0.075, top=0.94, wspace=0.055)

    # ElderAL has its own scale, with room for scores below random guessing.
    norm = Normalize(vmin=10, vmax=100)
    mesh = ax.pcolormesh(
        np.ma.masked_invalid(accuracy), cmap="cividis", norm=norm,
        edgecolors="white", linewidth=0.5,
    )
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(PRESETS)) + 0.5, list(PRESETS.values()))
    ax.tick_params(axis="both", length=0, pad=6, labelsize=9)
    ax.set_yticks(np.arange(len(MODELS)) + 0.5, MODELS, fontsize=7.5)
    ax.set_title(
        f"(a)ElderAL   Random baseline = {RANDOM_BASELINE:.2f}%",
        loc="left", fontsize=13, fontweight="bold", pad=11,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    for (y, x), value in np.ndenumerate(accuracy):
        if np.isnan(value):
            ax.add_patch(Rectangle((x, y), 1, 1, linewidth=0, zorder=0.9, **MISSING_STYLE))
            ax.text(
                x + 0.5, y + 0.5, "N/A", ha="center", va="center",
                fontsize=8, color="#404040",
                bbox={"facecolor": MISSING_STYLE["facecolor"], "edgecolor": "none", "pad": 1},
            )
        else:
            rgb = mesh.cmap(norm(value))[:3]
            brightness = np.dot(rgb, (0.2126, 0.7152, 0.0722))
            ax.text(
                x + 0.5, y + 0.5, f"{value:.2f}", ha="center", va="center",
                fontsize=9, color="white" if brightness < 0.46 else "#111111",
            )

    colorbar = fig.colorbar(mesh, cax=cax)
    colorbar.set_ticks(
        [10, RANDOM_BASELINE, 40, 60, 80, 100],
        labels=["10", f"{RANDOM_BASELINE:.2f}", "40", "60", "80", "100"],
    )
    colorbar.set_label("Mean test accuracy (%)", labelpad=8)
    colorbar.ax.tick_params(labelsize=9, length=3)
    colorbar.ax.axhline(RANDOM_BASELINE, color="white", linewidth=1.3)
    colorbar.outline.set_linewidth(0.5)
    return fig


def main():
    with plt.rc_context({"pdf.fonttype": 42, "hatch.linewidth": 0.65}):
        fig = build_heatmap(load_mean_accuracy(RESULTS_DIR))
        for suffix in ("png", "pdf"):
            fig.savefig(OUTPUT_DIR / f"elder_preset_heatmap.{suffix}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
