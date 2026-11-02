"""Audit Stage-I model/preset accuracy patterns used in manuscript text."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "paper_figures/derived_data/stage1_19models_6presets_accuracy.csv"

DATASET_ORDER = ["Widar", "Gait", "XRF55", "ElderAL"]
PRESET_ORDER = [
    "high_quality",
    "fast",
    "robust",
    "gesture_recognition",
    "activity_detection",
    "localization",
]


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    data["test_acc_pct"] = pd.to_numeric(data["test_acc_pct"], errors="coerce")
    ok = data[data["status"].eq("ok")].copy()

    print("STATUS COUNTS")
    print(
        data.groupby(["dataset", "status"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(DATASET_ORDER)
        .to_string()
    )

    print("\nPRESET DESCRIPTIVES")
    descriptives = (
        ok.groupby(["dataset", "preset"], sort=False)["test_acc_pct"]
        .agg(n="size", mean="mean", median="median", minimum="min", maximum="max")
        .round(3)
    )
    print(descriptives.to_string())

    print("\nMODEL SENSITIVITY (only models with all six successful presets)")
    complete = (
        ok.groupby(["dataset", "model"])["preset"]
        .nunique()
        .loc[lambda values: values.eq(6)]
        .index
    )
    complete_ok = ok.set_index(["dataset", "model"]).loc[complete].reset_index()
    sensitivity = (
        complete_ok.groupby(["dataset", "model"])["test_acc_pct"]
        .agg(mean="mean", minimum="min", maximum="max", range_pp=np.ptp)
        .reset_index()
    )
    for dataset in DATASET_ORDER:
        subset = sensitivity[sensitivity["dataset"].eq(dataset)].sort_values(
            "range_pp", ascending=False
        )
        print(f"\n{dataset}")
        print(subset.round(3).to_string(index=False))

    print("\nWITHIN-MODEL BEST/WORST PRESET EXAMPLES")
    for dataset in DATASET_ORDER:
        subset = sensitivity[sensitivity["dataset"].eq(dataset)].sort_values(
            "range_pp", ascending=False
        )
        for model in subset.head(3)["model"]:
            values = complete_ok[
                complete_ok["dataset"].eq(dataset) & complete_ok["model"].eq(model)
            ]
            best = values.loc[values["test_acc_pct"].idxmax()]
            worst = values.loc[values["test_acc_pct"].idxmin()]
            print(
                dataset,
                model,
                f"best={best['preset']}:{best['test_acc_pct']:.3f}",
                f"worst={worst['preset']}:{worst['test_acc_pct']:.3f}",
                f"range={best['test_acc_pct'] - worst['test_acc_pct']:.3f} pp",
            )

    print("\nBEST PRESET COUNTS WITHIN COMPLETE MODELS")
    best_rows = complete_ok.loc[
        complete_ok.groupby(["dataset", "model"])["test_acc_pct"].idxmax()
    ]
    worst_rows = complete_ok.loc[
        complete_ok.groupby(["dataset", "model"])["test_acc_pct"].idxmin()
    ]
    for dataset in DATASET_ORDER:
        print(f"\n{dataset} best")
        print(
            best_rows[best_rows["dataset"].eq(dataset)]["preset"]
            .value_counts()
            .reindex(PRESET_ORDER, fill_value=0)
            .to_string()
        )
        print(f"{dataset} worst")
        print(
            worst_rows[worst_rows["dataset"].eq(dataset)]["preset"]
            .value_counts()
            .reindex(PRESET_ORDER, fill_value=0)
            .to_string()
        )


if __name__ == "__main__":
    main()
