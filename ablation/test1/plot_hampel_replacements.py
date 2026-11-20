"""Plot ElderAL frame replacement rates and same-channel run lengths."""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR.parents[1] / "ablation_hampel_noize" / "results" / "spikes_per_sample.csv"
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_STEM = "elder_hampel_replacements_by_activity"
ACTIONS = ("Static", "Walking", "Fall", "Sleeping", "Drinking", "Sitting / TV")
RUN_FIELDS = (
    ("segments_1_frame", "1-frame runs", "#376895"),
    ("segments_2_frames", "2-frame runs", "#dfa15a"),
    ("segments_3plus_frames", "≥3-frame runs", "#168377"),
)


def calculate_rates(csv_path):
    """Sum counts by activity before calculating either proportion."""
    counts = {action: Counter() for action in ACTIONS}
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            action = counts[row["action"]]
            frame_count = int(row["frames"])
            affected_indices = json.loads(row["spike_frame_indices_zero_based"])
            if frame_count <= 0 or affected_indices != sorted(set(affected_indices)) or any(
                type(index) is not int or not 0 <= index < frame_count
                for index in affected_indices
            ):
                raise ValueError(f"Invalid frame counts or indices for {row['sample_id']}")

            action["frames"] += frame_count
            action["affected_frames"] += len(affected_indices)
            for field, _, _ in RUN_FIELDS:
                run_count = int(row[field])
                if run_count < 0:
                    raise ValueError(f"Negative run count for {row['sample_id']} in {field}")
                action[field] += run_count

    frame_rates = []
    run_rates = []
    for activity in ACTIONS:
        action = counts[activity]
        total_runs = sum(action[field] for field, _, _ in RUN_FIELDS)
        if not action["frames"] or not total_runs:
            raise ValueError(f"Missing frames or replacement runs for {activity}")
        frame_rates.append(action["affected_frames"] / action["frames"])
        run_rates.append(tuple(action[field] / total_runs for field, _, _ in RUN_FIELDS))
    return frame_rates, run_rates


def draw_figure(frame_rates, run_rates):
    figure, axis = plt.subplots(figsize=(10.4, 4.75), facecolor="white")
    figure.subplots_adjust(left=0.092, right=0.982, top=0.875, bottom=0.105)

    positions = range(len(ACTIONS))
    left = [position - 0.19 for position in positions]
    right = [position + 0.19 for position in positions]
    affected_bars = axis.bar(left, frame_rates, width=0.33, color="#c1cbd6", label="Affected frames")
    axis.bar_label(
        affected_bars, labels=[f"{rate:.1%}" for rate in frame_rates],
        padding=4, fontsize=9.4, color="#35465a",
    )

    bottoms = [0.0] * len(ACTIONS)
    label_colors = ("white", "#3a2513", "#0f6b60")
    for layer, ((_, label, color), rates) in enumerate(zip(RUN_FIELDS, zip(*run_rates))):
        axis.bar(right, rates, bottom=bottoms, width=0.33, color=color, label=label)
        top_layer = layer == len(RUN_FIELDS) - 1
        for position, bottom, rate in zip(right, bottoms, rates):
            if rate > 0:
                axis.annotate(
                    f"{rate:.1%}", xy=(position, bottom + rate * (1 if top_layer else 0.5)),
                    xytext=(0, 5 if top_layer else 0), textcoords="offset points",
                    ha="center", va="bottom" if top_layer else "center",
                    fontsize=9.4, color=label_colors[layer],
                )
        bottoms = [bottom + rate for bottom, rate in zip(bottoms, rates)]

    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e4eaf0", linewidth=0.8)
    axis.set(ylim=(0, 1.07), xlim=(-0.61, len(ACTIONS) - 0.39), ylabel="Percentage")
    axis.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axis.set_xticks(positions, ACTIONS)
    axis.tick_params(axis="both", length=0, pad=8)
    axis.spines[["top", "left", "right"]].set_visible(False)
    axis.spines["bottom"].set_color("#9dabb8")
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.092, 0.965),
        ncols=4, frameon=False, borderaxespad=0,
        handlelength=1.35, columnspacing=1.9, fontsize=10,
    )
    return figure


def main():
    frame_rates, run_rates = calculate_rates(INPUT_CSV)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": "#172433",
        "axes.labelcolor": "#35465a",
        "xtick.color": "#35465a",
        "ytick.color": "#35465a",
        "pdf.fonttype": 42,
    }):
        figure = draw_figure(frame_rates, run_rates)
        for extension in ("png", "pdf"):
            destination = RESULTS_DIR / f"{OUTPUT_STEM}.{extension}"
            figure.savefig(destination, dpi=300, facecolor="white")
            print(destination)
        plt.close(figure)


if __name__ == "__main__":
    main()
