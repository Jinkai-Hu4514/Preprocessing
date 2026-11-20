"""Generate four individual figures and a combined figure from the ElderAL results."""

import csv
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator, PercentFormatter
import numpy as np

from count_spikes import ACTION_NAMES, RESULTS, SYNC_THRESHOLDS


SEEDS = (42, 727, 914)
MODELS = ("CSITime", "ResNet2D", "Random Forest")
MODEL_COLORS = ("#3D6B9E", "#C76D27", "#2A887C")
SEGMENT_FIELDS = ("segments_1_frame", "segments_2_frames", "segments_3plus_frames")
LOCATION_FEATURES = (
    ("spike_fraction", "Spike proportion", "pp", 100),
    ("mean_amplitude", "Mean spike amplitude", "a.u.", 1),
    ("short_return_fraction", "Short-jump recovery", "pp", 100),
    ("sync_50_fraction", "50% synchrony", "pp", 100),
)
LOCATION_COMPARISONS = (
    ("same_action_different_location_mean_abs_diff", "Same action, different locations", "#3D6B9E"),
    ("same_location_different_action_mean_abs_diff", "Same location, different actions", "#C76D27"),
)


def read_rows(filename):
    with (RESULTS / filename).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save(fig, stem):
    for extension in ("png", "pdf"):
        output = RESULTS / f"{stem}.{extension}"
        fig.savefig(output, dpi=300, facecolor="white")
        print(output)
    plt.close(fig)


def style_axis(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#465262")
    ax.spines[["left", "bottom"]].set_linewidth(1.25)
    ax.tick_params(axis="both", length=0, pad=7)
    plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#E1E6ED", linewidth=0.8)


def plot_macro_f1(panel, rows):
    groups = (("spikes", "Spikes only", MODELS), ("raw", "Raw CSI", MODELS[:2]))
    pairs = {(condition, model) for condition, _, models in groups for model in models}
    selected = [row for row in rows if (row["input"], row["model"]) in pairs]
    scores = {(row["input"], row["model"], int(row["seed"])): float(row["macro_f1"]) for row in selected}
    expected = {(condition, model, seed) for condition, model in pairs for seed in SEEDS}
    if len(scores) != len(selected) or scores.keys() != expected:
        raise ValueError("Spikes only and Raw CSI require one result for each plotted model and seed")
    if any(not 0 <= value <= 1 for value in scores.values()):
        raise ValueError("Macro-F1 values must be between 0 and 1")

    ax = panel.add_axes((0.13, 0.15, 0.85, 0.71))
    width = 0.23
    for index, (model, color) in enumerate(zip(MODELS, MODEL_COLORS)):
        positions, model_scores = [], []
        for group_index, (condition, _, models) in enumerate(groups):
            if model in models:
                positions.append(group_index + (index - 1) * width)
                model_scores.append([scores[condition, model, seed] for seed in SEEDS])
        ax.bar(
            positions, [mean(values) for values in model_scores], width=width,
            yerr=[stdev(values) for values in model_scores], color=color, capsize=4, label=model,
            error_kw={"ecolor": "#24313E", "elinewidth": 1.2, "capthick": 1.2},
        )
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(range(len(groups)), [label for _, label, _ in groups])
    ax.set_xlim(-0.52, len(groups) - 0.48)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=3, frameon=False)
    style_axis(ax)


def plot_segment_lengths(panel, rows):
    counts = np.array([[int(row[field]) for field in SEGMENT_FIELDS] for row in rows])
    totals = counts.sum(axis=1)
    if (totals <= 0).any() or (counts < 0).any():
        raise ValueError("Every action must have valid spike segment counts")
    fractions = counts / totals[:, None]
    ax = panel.add_axes((0.10, 0.15, 0.88, 0.66))
    positions = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    labels = ("1 frame", "2 frames", "≥3 frames")
    text_colors = ("white", "#151A20", MODEL_COLORS[2])
    for index, (values, label, color, text_color) in enumerate(zip(fractions.T, labels, MODEL_COLORS, text_colors)):
        ax.bar(positions, values, bottom=bottom, width=0.64, color=color, label=label)
        for position, base, fraction in zip(positions, bottom, values):
            if fraction == 0:
                continue
            label_position = (position, base + fraction / 2)
            offset = (0, 9) if index == len(labels) - 1 else (0, 0)
            ax.annotate(
                f"{fraction:.1%}", xy=label_position, xytext=offset, textcoords="offset points",
                ha="center", va="center", fontsize=9, fontweight="bold", color=text_color,
                annotation_clip=False, clip_on=False,
            )
        bottom += values

    ax.set_xticks(positions, [row["action"] for row in rows])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of spike segments")
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncols=3, frameon=False)
    style_axis(ax)


def plot_synchrony(panel, rows):
    ax = panel.add_axes((0.11, 0.15, 0.87, 0.72))
    positions = np.arange(len(rows))
    width = 0.23
    for index, (threshold, color) in enumerate(zip(SYNC_THRESHOLDS, MODEL_COLORS)):
        values = [float(row[f"sync_{threshold}_fraction"]) for row in rows]
        ax.bar(positions + (index - 1) * width, values, width, color=color, label=f"≥{threshold}% of subcarriers")

    ax.set_xticks(positions, [row["action"] for row in rows])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Fraction of spike frames")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=3, frameon=False)
    style_axis(ax)


def plot_location_comparison(panel, rows):
    data = {row["metric"]: row for row in rows}
    values = [
        [float(data[field][column]) * multiplier for column, _, _ in LOCATION_COMPARISONS]
        for field, _, _, multiplier in LOCATION_FEATURES
    ]
    pp_limit = 1.23 * max(max(pair) for pair, (_, _, unit, _) in zip(values, LOCATION_FEATURES) if unit == "pp")
    grid = panel.add_gridspec(len(LOCATION_FEATURES), 1, left=0.27, right=0.95, bottom=0.12, top=0.86, hspace=0.75)
    panel.legend(
        handles=[Patch(color=color, label=label) for _, label, color in LOCATION_COMPARISONS],
        loc="upper center", bbox_to_anchor=(0.5, 0.97), ncols=2, frameon=False,
    )
    for index, (pair, (_, label, unit, _)) in enumerate(zip(values, LOCATION_FEATURES)):
        ax = panel.add_subplot(grid[index, 0])
        for position, value, (_, _, color) in zip((0.19, -0.19), pair, LOCATION_COMPARISONS):
            bar = ax.barh(position, value, height=0.30, color=color)
            ax.bar_label(bar, labels=[f"{value:.2f} {unit}"], padding=4, fontsize=9)
        ax.set_yticks([0], [label])
        ax.set_ylim(-0.47, 0.47)
        ax.set_xlim(0, pp_limit if unit == "pp" else 1.23 * max(pair))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        style_axis(ax, grid_axis="x")
    panel.supxlabel("Mean absolute difference", y=0.041, fontsize=11, fontweight="bold")


def main():
    by_action = {int(row["action_id"]): row for row in read_rows("spikes_by_action.csv")}
    actions = [by_action[action_id] for action_id in ACTION_NAMES]
    figures = (
        ("fig1_macro_f1", (6.4, 4.5), plot_macro_f1, read_rows("classification_results.csv")),
        ("fig2_spike_segment_lengths", (8.7, 4.8), plot_segment_lengths, actions),
        ("fig3_subcarrier_synchrony", (8.7, 4.8), plot_synchrony, actions),
        ("fig4_location_vs_action", (8.7, 6.4), plot_location_comparison, read_rows("location_comparison.csv")),
    )
    with plt.rc_context({
        "font.family": "Arial", "font.size": 10, "text.color": "#182334",
        "axes.labelcolor": "#182334", "xtick.color": "#394653", "ytick.color": "#394653",
        "axes.labelweight": "bold", "pdf.fonttype": 42, "pdf.use14corefonts": False,
    }):
        for stem, size, draw, rows in figures:
            fig = plt.figure(figsize=size, layout="none")
            draw(fig, rows)
            save(fig, stem)

        combined = plt.figure(figsize=(17.8, 11.8), layout="none")
        panels = combined.subfigures(2, 2, wspace=0.04, hspace=0.04)
        for letter, panel, (_, _, draw, rows) in zip("abcd", panels.flat, figures):
            draw(panel, rows)
            panel.text(0.025, 0.022, f"({letter})", ha="left", va="bottom", fontsize=12, fontweight="bold")
        save(combined, "fig_combined")


if __name__ == "__main__":
    main()
