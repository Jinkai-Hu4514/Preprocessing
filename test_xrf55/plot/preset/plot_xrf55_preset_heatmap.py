"""Plot XRF55 preset test accuracy averaged over three random seeds."""

import csv
import math
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


PLOT_DIR = Path(__file__).resolve().parent
RESULT_DIR = PLOT_DIR.parents[1] / "preset_test"
SEEDS = (42, 727, 914)
RANDOM_BASELINE = 100 / 55  # Uniform guessing across 55 activity classes.

PRESET_LABELS = {
    "high_quality": "High\nQuality",
    "fast": "Fast",
    "robust": "Robust",
    "gesture_recognition": "Gesture\nRecognition",
    "activity_detection": "Activity\nDetection",
    "localization": "Localization",
}

MODEL_LABELS = {
    "csimodel": "csimodel",
    "mlpmodel": "mlpmodel",
    "cnn1dmodel": "cnn1dmodel",
    "cnn2dmodel": "cnn2dmodel",
    "resnet1d": "resnet1d",
    "resnet2d": "resnet2d",
    "lstmmodel": "lstmmodel",
    "bilstmattention": "bilstmattention",
    "attentiongru": "attentiongru",
    "csitime": "csitime",
    "ei": "ei",
    "fewsense": "fewsense",
    "pa_csi": "pa_csi",
    "that": "that",
    "wiflexformer": "wiflexformer",
    "efficientnetcsi": "efficientnetcsi",
    "mambacsi": "mambacsi",
    "graphneuralcsi": "graphneuralcsi",
    "visiontransformercsi": "visiontransformercsi",
}

MISSING_STYLE = {"facecolor": "#eeeeee", "edgecolor": "#9a9a9a", "hatch": "////"}


def read_seed_results(result_dir: Path, seed: int) -> dict[tuple[str, str], float]:
    path = result_dir / f"result{seed}" / "xrf55_preset_summary.csv"
    results = {}
    if not path.is_file():
        return results

    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["preset"], row["model"])
            if int(row["seed"]) != seed or key in results:
                raise ValueError(f"Unexpected seed or duplicate result for {key} in {path}")

            accuracy = math.nan
            if row["status"] == "ok" and row["test_acc"].strip():
                accuracy = float(row["test_acc"])
                if not math.isnan(accuracy) and not 0 <= accuracy <= 1:
                    raise ValueError(f"Test accuracy must be between 0 and 1 for {key} in {path}")
            results[key] = accuracy

    return results


def mean_test_accuracy(result_dir: Path) -> np.ndarray:
    runs = [read_seed_results(result_dir, seed) for seed in SEEDS]
    accuracy = np.full((len(MODEL_LABELS), len(PRESET_LABELS)), np.nan)
    for row, model in enumerate(MODEL_LABELS):
        for col, preset in enumerate(PRESET_LABELS):
            scores = [run.get((preset, model), math.nan) for run in runs]
            if all(math.isfinite(score) for score in scores):
                accuracy[row, col] = 100 * fmean(scores)
    return accuracy


def build_heatmap(accuracy: np.ndarray):
    fig, (ax, cax) = plt.subplots(
        ncols=2, figsize=(8.1, 7.5), width_ratios=(1, 0.035),
    )
    fig.subplots_adjust(left=0.15, right=0.885, bottom=0.075, top=0.94, wspace=0.055)

    # Keep the XRF55 scale independent and include scores below its random baseline.
    norm = Normalize(vmin=0, vmax=100)
    mesh = ax.pcolormesh(
        np.ma.masked_invalid(accuracy), cmap="cividis", norm=norm,
        edgecolors="white", linewidth=0.5,
    )
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(PRESET_LABELS)) + 0.5, list(PRESET_LABELS.values()))
    ax.set_yticks(np.arange(len(MODEL_LABELS)) + 0.5, list(MODEL_LABELS.values()))
    ax.tick_params(axis="both", length=0, pad=6, labelsize=9)
    for label in ax.get_yticklabels():
        if label.get_text() == "visiontransformercsi":
            label.set_fontsize(7.5)  # Keep the full name within the existing margin.
    ax.set_title(
        f"(b)xrf55   Random baseline = {RANDOM_BASELINE:.2f}%",
        loc="left", fontsize=13, fontweight="bold", pad=11,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    for (row, col), value in np.ndenumerate(accuracy):
        if np.isnan(value):
            ax.add_patch(Rectangle((col, row), 1, 1, linewidth=0, zorder=0.9, **MISSING_STYLE))
            ax.text(
                col + 0.5, row + 0.5, "N/A", ha="center", va="center",
                fontsize=8, color="#404040",
                bbox={"facecolor": MISSING_STYLE["facecolor"], "edgecolor": "none", "pad": 1},
            )
        else:
            rgb = mesh.cmap(norm(value))[:3]
            brightness = np.dot(rgb, (0.2126, 0.7152, 0.0722))
            ax.text(
                col + 0.5, row + 0.5, f"{value:.2f}", ha="center", va="center",
                fontsize=9, color="white" if brightness < 0.46 else "#111111",
            )

    colorbar = fig.colorbar(mesh, cax=cax)
    colorbar.set_ticks(
        [RANDOM_BASELINE, 20, 40, 60, 80, 100],
        labels=[f"{RANDOM_BASELINE:.2f}", "20", "40", "60", "80", "100"],
    )
    colorbar.set_label("Mean test accuracy (%)", labelpad=8)
    colorbar.ax.tick_params(labelsize=9, length=3)
    colorbar.ax.axhline(RANDOM_BASELINE, color="white", linewidth=1.3)
    colorbar.outline.set_linewidth(0.5)
    return fig


def main() -> None:
    with plt.rc_context({"pdf.fonttype": 42, "hatch.linewidth": 0.65}):
        fig = build_heatmap(mean_test_accuracy(RESULT_DIR))
        for suffix in ("png", "pdf"):
            fig.savefig(PLOT_DIR / f"xrf55_preset_heatmap.{suffix}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
