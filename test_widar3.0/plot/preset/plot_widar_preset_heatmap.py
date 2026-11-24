"""Plot Widar preset test accuracy averaged over three random seeds."""

import csv
import math
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


PLOT_DIR = Path(__file__).resolve().parent
RESULT_DIR = PLOT_DIR.parents[1] / "preset_test"
OUTPUT_PATH = PLOT_DIR / "widar_preset_heatmap.png"
SEEDS = (42, 727, 914)
RANDOM_BASELINE = 100 / 6  # The Widar experiment uses six gesture classes.

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


def read_seed_results(result_dir: Path, seed: int) -> dict[tuple[str, str], float]:
    path = result_dir / f"result{seed}" / "widar3.0_preset_summary.csv"
    scores = {}
    if not path.is_file():
        return scores

    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["preset"], row["model"])
            if int(row["seed"]) != seed or key in scores:
                raise ValueError(f"Unexpected seed or duplicate run for {key} in {path}")

            value = row["test_acc"].strip()
            scores[key] = float(value) if row["status"] == "ok" and value else math.nan
    return scores


def mean_test_accuracy(result_dir: Path) -> np.ndarray:
    runs = [read_seed_results(result_dir, seed) for seed in SEEDS]
    accuracy = np.full((len(MODELS), len(PRESETS)), np.nan)
    for row, model in enumerate(MODELS):
        for col, preset in enumerate(PRESETS):
            scores = [run.get((preset, model), math.nan) for run in runs]
            if all(0 <= score <= 1 for score in scores):
                accuracy[row, col] = 100 * fmean(scores)
    return accuracy


def build_heatmap(accuracy: np.ndarray) -> tuple[Figure, Colorbar]:
    valid = accuracy[np.isfinite(accuracy)]
    # Keep all observed means and the random baseline within Widar's color scale.
    color_min = 10 * math.floor(valid.min(initial=10) / 10)
    color_max = 10 * math.ceil(valid.max(initial=30) / 10)
    norm = Normalize(vmin=color_min, vmax=color_max)

    fig, (ax, cax) = plt.subplots(ncols=2, figsize=(8.1, 7.5), width_ratios=(1, 0.035))
    fig.subplots_adjust(left=0.15, right=0.885, bottom=0.075, top=0.94, wspace=0.055)
    mesh = ax.pcolormesh(
        np.ma.masked_invalid(accuracy), cmap="cividis", norm=norm,
        edgecolors="white", linewidth=0.5,
    )
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(PRESETS)) + 0.5, list(PRESETS.values()))
    ax.set_yticks(np.arange(len(MODELS)) + 0.5, MODELS)
    ax.tick_params(axis="both", length=0, pad=6, labelsize=9)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_title(
        f"(c)Widar   Random baseline = {RANDOM_BASELINE:.2f}%",
        loc="left", fontsize=13, fontweight="bold", pad=11,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    for (row, col), value in np.ndenumerate(accuracy):
        if np.isnan(value):
            ax.add_patch(Rectangle((col, row), 1, 1, linewidth=0, zorder=0.9, **MISSING_STYLE))
            ax.text(
                col + 0.5, row + 0.5, "N/A", ha="center", va="center", fontsize=8, color="#404040",
                bbox={"facecolor": MISSING_STYLE["facecolor"], "edgecolor": "none", "pad": 1},
            )
        else:
            brightness = np.dot(mesh.cmap(norm(value))[:3], (0.2126, 0.7152, 0.0722))
            ax.text(
                col + 0.5, row + 0.5, f"{value:.2f}", ha="center", va="center", fontsize=9,
                color="white" if brightness < 0.46 else "#111111",
            )

    ticks = sorted([RANDOM_BASELINE, *(
        tick for tick in range(color_min, color_max + 1, 10)
        if abs(tick - RANDOM_BASELINE) >= 5
    )])
    colorbar = fig.colorbar(mesh, cax=cax)
    colorbar.set_ticks(
        ticks, labels=[f"{tick:.2f}" if tick == RANDOM_BASELINE else str(tick) for tick in ticks],
    )
    colorbar.set_label("Mean test accuracy (%)", labelpad=8)
    colorbar.ax.tick_params(labelsize=9, length=3)
    colorbar.ax.axhline(RANDOM_BASELINE, color="white", linewidth=1.3)
    colorbar.outline.set_linewidth(0.5)
    return fig, colorbar


def plot_heatmap(accuracy: np.ndarray, output_path: Path) -> None:
    with plt.rc_context({"hatch.linewidth": 0.65, "pdf.fonttype": 42}):
        fig, colorbar = build_heatmap(accuracy)
        fig.savefig(output_path, dpi=300, facecolor="white")
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor("face")
        fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    plot_heatmap(mean_test_accuracy(RESULT_DIR), OUTPUT_PATH)
    print(OUTPUT_PATH)
