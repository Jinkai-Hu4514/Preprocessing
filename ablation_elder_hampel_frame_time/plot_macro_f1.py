"""Plot the ElderAL feature ablations and raw baselines."""

from math import ceil
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from common import CONDITIONS, NEURAL_MODELS, RAW_DIR, RESULTS, SEEDS, read_csv


MODELS = (*NEURAL_MODELS, "Random Forest")
COLORS = ("#4477AA", "#EE9944", "#228877")
LABELS = ("Frame count only", "Timing stats only", "Frame count + timing stats", "Raw")


def read_scores():
    source = RESULTS / "comparison.csv"
    rows = read_csv(source)
    by_run = {(row["model"], int(row["seed"])): row for row in rows}
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    if len(rows) != len(expected) or by_run.keys() != expected:
        raise ValueError(f"{source.name} 需要包含三个模型和三个种子的完整结果")

    columns = [f"{condition}_macro_f1" for condition in CONDITIONS]
    try:
        values = {key: tuple(float(row[column]) for column in columns) for key, row in by_run.items()}
    except (KeyError, ValueError) as error:
        raise ValueError("缺少逐种子 Macro-F1，请先运行三份训练脚本") from error
    if any(not 0 <= value <= 1 for scores in values.values() for value in scores):
        raise ValueError("Macro-F1 必须在 0 到 1 之间")

    models = {
        model: [[values[model, seed][column] for seed in SEEDS] for column in range(len(CONDITIONS))]
        for model in MODELS
    }
    raw_scores = {model: [] for model in NEURAL_MODELS}
    for seed in SEEDS:
        summary = RAW_DIR / f"result{seed}" / "elder_raw_summary.csv"
        raw_rows = read_csv(summary)
        for model in NEURAL_MODELS:
            matches = [
                row for row in raw_rows
                if row["model"].lower() == model.lower() and row["seed"] == str(seed) and row["status"] == "ok"
            ]
            if len(matches) != 1 or float(matches[0]["test_acc"]) != float(by_run[model, seed]["raw_accuracy"]):
                raise ValueError(f"{model} seed {seed} 的 Raw 记录与对比表不匹配")
            score = float(matches[0]["macro_f1"])
            if not 0 <= score <= 1:
                raise ValueError(f"{model} seed {seed} 的 Raw Macro-F1 不在 0 到 1 之间")
            raw_scores[model].append(score)
    for model, scores in raw_scores.items():
        models[model].append(scores)
    return models


def plot():
    models = read_scores()
    width = 0.23
    errors = {"ecolor": "#20242A", "elinewidth": 1.15, "capthick": 1.15}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(9.6, 5.0))
        fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.14)

        bars = []
        for index, (model, color) in enumerate(zip(MODELS, COLORS)):
            scores = models[model]
            positions = [position + (index - 1) * width for position in range(len(CONDITIONS))]
            if model in NEURAL_MODELS:
                positions.append(len(CONDITIONS) + (index - 0.5) * width)
            bar = ax.bar(
                positions,
                [mean(group) for group in scores], width=width, yerr=[stdev(group) for group in scores],
                color=color, capsize=3.5, error_kw=errors, label=model, zorder=2,
            )
            bars.append(bar)

        upper = max(mean(group) + stdev(group) for groups in models.values() for group in groups)
        ax.set_ylim(0, min(1.0, max(0.5, ceil((upper + 0.02) * 10) / 10)))
        ax.set_xlim(-0.55, len(LABELS) - 0.45)
        ax.set_ylabel("Macro-F1")
        ax.set_xticks(range(len(LABELS)), LABELS)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.tick_params(axis="x", pad=8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E3E6EA", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            handles=bars, loc="lower center",
            bbox_to_anchor=(0.5, 1.015), ncol=3, fontsize=10, columnspacing=1.2, frameon=False,
        )

        for extension in ("png", "pdf"):
            output = RESULTS / f"elder_macro_f1.{extension}"
            fig.savefig(output, dpi=300, facecolor="white")
            print(output)
        plt.close(fig)


if __name__ == "__main__":
    plot()
