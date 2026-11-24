"""Plot paired accuracy changes relative to Raw and within each preprocessing step."""

import csv
from itertools import product
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


PLOT_DIR = Path(__file__).resolve().parent
RESULT_DIR = PLOT_DIR / "results"
ROOT = PLOT_DIR.parent
SEEDS = (42, 727, 914)
DATASETS = (
    ("elder", "Elder", 64, (("full_test", "CSITime"), ("full_test_resnet2d", "ResNet2D"))),
    ("xrf55", "XRF55", 15, (("full_test", "ResNet1D"), ("full_test_that", "THAT"))),
)
CURVE_STYLES = {
    "Outliers: IQR": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "Outliers: Z-score": {"color": "#E69F00", "linestyle": "--", "marker": "s"},
    "Denoising: Wavelet": {"color": "#000000", "linestyle": "-", "marker": "P"},
    "Denoising: Butterworth": {"color": "#D55E00", "linestyle": (0, (6, 2)), "marker": "X"},
    "Denoising: Savitzky-Golay": {"color": "#56B4E9", "linestyle": (0, (3, 2, 1, 2, 1, 2)), "marker": "v"},
    "Denoising: Bandpass": {"color": "#009E73", "linestyle": "-.", "marker": "^"},
    "Denoising: Hampel": {"color": "#CC79A7", "linestyle": ":", "marker": "D"},
}
CURVE_SIZE = {"linewidth": 1.7, "markersize": 5.5}


def read_scores(path, seed, model, key_field):
    scores = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["model"].casefold() != model.casefold() or row["status"] != "ok":
                continue
            key = row[key_field]
            if int(row["seed"]) != seed or key in scores:
                raise ValueError(f"Unexpected seed or duplicate result for {key} in {path}")
            accuracy = float(row["test_acc"])
            if not 0 <= accuracy <= 1:
                raise ValueError(f"Invalid test accuracy for {key} in {path}: {accuracy}")
            scores[key] = (accuracy, row["split_digest"])
    return scores


def mean_accuracy_changes(dataset, full_test_dir, model):
    dataset_dir = ROOT / f"test_{dataset}"
    changes_by_seed = []
    for seed in SEEDS:
        raw_path = dataset_dir / "raw_test" / f"result{seed}" / f"{dataset}_raw_summary.csv"
        raw_scores = read_scores(raw_path, seed, model, "model")
        if len(raw_scores) != 1:
            raise ValueError(f"Expected one successful Raw result for {model} in {raw_path}")
        raw_accuracy, raw_split = next(iter(raw_scores.values()))

        full_path = dataset_dir / full_test_dir / f"result{seed}" / f"{dataset}_full_summary.csv"
        full_scores = read_scores(full_path, seed, model, "combo_name")
        changes = {}
        for combo, (accuracy, split) in full_scores.items():
            if not raw_split or split != raw_split:
                raise ValueError(f"Raw and full-test splits differ for {combo} in {full_path}")
            changes[combo] = 100 * (accuracy - raw_accuracy)
        if not changes or (changes_by_seed and changes.keys() != changes_by_seed[0].keys()):
            raise ValueError(f"Missing or inconsistent combinations for {model} in {full_path}")
        changes_by_seed.append(changes)

    return {
        combo: fmean(changes[combo] for changes in changes_by_seed)
        for combo in changes_by_seed[0]
    }


def build_panels(mean_changes, subcarriers):
    options = {
        "denoise": (
            ("Wavelet", "wavelet"),
            ("Butterworth", "butterworth_o5_c0.3"),
            ("Savitzky-Golay", "savgol_w7_p3"),
            ("Bandpass", "bandpass_0.5-50"),
            ("Hampel", "hampel_w5_s3"),
        ),
        "outliers": (("IQR", "iqr"), ("Z-score", "outlier_z-score")),
        "normalize": (("Z-score", "z-score"), ("Min-max", "min-max")),
        "interpolate": tuple(
            (method.title(), f"{method}{subcarriers}")
            for method in ("linear", "cubic", "nearest", "decimate")
        ),
    }
    expected_combos = {
        "+".join(combo)
        for combo in product(*([key for _, key in methods] for methods in options.values()))
    }
    if expected_combos != mean_changes.keys():
        missing = sorted(expected_combos - mean_changes.keys())
        unexpected = sorted(mean_changes.keys() - expected_combos)
        raise ValueError(f"Full-test combinations differ from plot options; missing={missing}, unexpected={unexpected}")

    reference = {stage: methods[0][1] for stage, methods in options.items()}
    stages = (
        ("denoise", "Denoising", "outliers", "Outliers"),
        ("outliers", "Outlier handling", "denoise", "Denoising"),
        ("normalize", "Normalization", "outliers", "Outliers"),
        ("interpolate", "Subcarrier interpolation", "outliers", "Outliers"),
    )
    panels = []
    for stage, title, context_stage, context_title in stages:
        methods = options[stage]
        curves = []
        for context_name, context_key in options[context_stage]:
            context = {**reference, context_stage: context_key}
            values = []
            for _, method in methods:
                config = {**context, stage: method}
                combo = "+".join(config[step] for step in reference)
                values.append(mean_changes[combo])
            curves.append((f"{context_title}: {context_name}", values))
        panels.append((title, [name for name, _ in methods], curves))
    return panels


def plot_method_changes(axis, reference_method, curves, x_limits):
    axis.set_xlim(*x_limits)
    axis.set_ylim(-0.5, len(curves) + 0.5)
    axis.set_axis_off()
    label_transform = axis.get_yaxis_transform()
    axis.text(
        0, len(curves), f"Δ vs. {reference_method} (pp)", transform=label_transform,
        va="center", fontsize=9, weight="bold", color="#334155",
    )

    for row, (label, values) in enumerate(curves):
        row_y = len(curves) - row - 1
        method = label.split(": ", 1)[1]
        axis.text(0, row_y, method, transform=label_transform, va="center", fontsize=8)
        for column, value in enumerate(values[1:], start=1):
            change = round(value - values[0], 2)
            change_label = f"{change:+.2f}" if change else "0.00"
            axis.text(column, row_y, change_label, ha="center", va="center", fontsize=8)


def plot_panels(title, panels, y_limits, output_stem):
    fig = plt.figure(figsize=(11.6, 8.65))
    layout = fig.add_gridspec(2, 2, left=0.10, right=0.98, bottom=0.055, top=0.82, hspace=0.32, wspace=0.20)
    fig.suptitle(title, x=0.10, y=0.963, ha="left", fontsize=16, weight="bold")
    fig.supylabel("Test accuracy change vs. Raw (pp)", x=0.026, y=0.455, fontsize=10)
    handles = [
        Line2D([], [], label=label, **style, **CURVE_SIZE)
        for label, style in CURVE_STYLES.items()
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.10, 0.918), ncols=4,
        fontsize=9, frameon=False, borderaxespad=0, handlelength=2.7,
    )

    for panel_index, ((panel_title, methods, curves), limits) in enumerate(zip(panels, y_limits)):
        slot = layout[panel_index].subgridspec(
            2, 1, height_ratios=(3.0, 1.66), hspace=0.30,
        )
        axis = fig.add_subplot(slot[0])
        positions = range(len(methods))
        axis.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
        for label, values in curves:
            axis.plot(positions, values, label=label, **CURVE_STYLES[label], **CURVE_SIZE)
        panel_letter = "abcd"[panel_index]
        axis.set_title(f"({panel_letter}) {panel_title}", loc="left", fontsize=10, weight="bold")
        axis.set_xticks(positions, methods)
        axis.set_xlim(-0.52, len(methods) - 0.75)
        axis.set_ylim(*limits)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#e0e7eb", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#a6b5c0")
        plot_method_changes(fig.add_subplot(slot[1]), methods[0], curves, axis.get_xlim())

    for suffix in (".png", ".pdf"):
        output = output_stem.with_suffix(suffix)
        fig.savefig(output, dpi=220)
        print(output)
    plt.close(fig)


def main():
    RESULT_DIR.mkdir(exist_ok=True)
    for dataset, name, subcarriers, runs in DATASETS:
        model_panels = [
            (model, build_panels(mean_accuracy_changes(dataset, directory, model), subcarriers))
            for directory, model in runs
        ]
        # Compare the same step on one scale across models, with Raw always in view.
        y_limits = []
        for panels_for_step in zip(*(panels for _, panels in model_panels)):
            values = [0.0] + [
                value for _, _, curves in panels_for_step for _, curve in curves for value in curve
            ]
            low, high = min(values), max(values)
            margin = max(1, 0.08 * (high - low))
            y_limits.append((low - margin, high + margin))

        for model, panels in model_panels:
            output_stem = RESULT_DIR / f"{dataset}_{model.lower()}_vs_raw"
            plot_panels(f"{name} · {model}", panels, y_limits, output_stem)


if __name__ == "__main__":
    main()
