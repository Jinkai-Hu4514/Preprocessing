"""Plot gait preset test accuracy averaged over three random seeds."""

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
RANDOM_BASELINE = 100 / 9  # Uniform guessing across the nine gait classes.

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
        path = results_dir / f"result{seed}" / "gait_preset_summary.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (seed, row["model"], row["preset"])
                if int(row["seed"]) != seed or key in scores:
                    raise ValueError(f"Mismatched or duplicate run {key} in {path}")
                value = (row["test_acc"] or "").strip()
                scores[key] = float(value) if row["status"] == "ok" and value else np.nan

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

    norm = Normalize(vmin=0, vmax=100)
    mesh = ax.pcolormesh(
        np.ma.masked_invalid(accuracy), cmap="cividis", norm=norm,
        edgecolors="white", linewidth=0.5,
    )
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(PRESETS)) + 0.5, list(PRESETS.values()))
    ax.tick_params(axis="both", length=0, pad=6, labelsize=9)
    ax.set_yticks(np.arange(len(MODELS)) + 0.5, MODELS, fontsize=7.5)
    ax.set_title(
        f"(d)gait   Random baseline = {RANDOM_BASELINE:.2f}%",
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
        [0, RANDOM_BASELINE, 25, 50, 75, 100],
        labels=["0", f"{RANDOM_BASELINE:.2f}", "25", "50", "75", "100"],
    )
    colorbar.set_label("Mean test accuracy (%)", labelpad=8)
    colorbar.ax.tick_params(labelsize=9, length=3)
    colorbar.ax.axhline(RANDOM_BASELINE, color="white", linewidth=1.3)
    colorbar.outline.set_linewidth(0.5)
    return fig


def main():
    style = {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "hatch.linewidth": 0.65}
    with plt.rc_context(style):
        fig = build_heatmap(load_mean_accuracy(RESULTS_DIR))
        for suffix in ("png", "pdf"):
            fig.savefig(OUTPUT_DIR / f"gait_preset_heatmap.{suffix}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
