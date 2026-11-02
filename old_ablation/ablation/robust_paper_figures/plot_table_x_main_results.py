#!/usr/bin/env python3
"""Render matched Robust calibration results for Widar and Gait."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
WIDAR = ROOT / "SDP/test_wider/result/full_tests_new/widar_320_pipeline_optimized_mlpmodel_summary.csv"
GAIT_MAIN = ROOT / "SDP/test_gait/result/full_tests_new/gait_320_pipeline_optimized_mlpmodel_summary.csv"
GAIT_SHARD = ROOT / "SDP/test_gait/result/full_tests_new_gpu1_290_320/gait_320_pipeline_gpu1_290_320_mlpmodel_summary.csv"

PAIR_KEYS = ["denoise", "outliers", "normalize", "interpolate"]
CALIBRATIONS = ["linear", "polynomial_d3", "stc", "robust"]


def load_ok(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.loc[df["status"].eq("ok")].copy()


def summarize(dataset: str, df: pd.DataFrame, expected_total: int) -> dict[str, object]:
    duplicate_indices = int(df["combo_index"].duplicated().sum())
    if duplicate_indices:
        raise RuntimeError(f"{dataset}: duplicate combo indices: {duplicate_indices}")

    complete_keys = (
        df.groupby(PAIR_KEYS, dropna=False)["calibrate"]
        .agg(lambda s: set(s.dropna()))
        .loc[lambda s: s.apply(lambda values: set(CALIBRATIONS).issubset(values))]
        .index
    )
    complete = (
        df.set_index(PAIR_KEYS)
        .loc[complete_keys]
        .reset_index()
        .loc[lambda x: x["calibrate"].isin(CALIBRATIONS)]
        .copy()
    )
    pivot = complete.pivot(index=PAIR_KEYS, columns="calibrate", values="test_acc")
    if pivot.shape[0] != len(complete_keys) or pivot.isna().any().any():
        raise RuntimeError(f"{dataset}: incomplete matched pivot")

    means = 100.0 * pivot[CALIBRATIONS].mean()
    nonrobust_mean = pivot[["linear", "polynomial_d3", "stc"]].mean(axis=1)
    gap = 100.0 * (pivot["robust"] - nonrobust_mean)
    robust_last = pivot["robust"].lt(
        pivot[["linear", "polynomial_d3", "stc"]].min(axis=1)
    )

    return {
        "Dataset": dataset,
        "Coverage": f"{df['combo_index'].nunique()}/{expected_total}",
        "Matched cells": int(pivot.shape[0]),
        "Linear (%)": float(means["linear"]),
        "Polynomial (%)": float(means["polynomial_d3"]),
        "STC (%)": float(means["stc"]),
        "Robust (%)": float(means["robust"]),
        "Robust vs non-robust mean (pp)": float(gap.mean()),
        "Robust last": f"{int(robust_last.sum())}/{len(robust_last)}",
    }


def build_values() -> pd.DataFrame:
    widar = load_ok(WIDAR)
    gait = pd.concat([load_ok(GAIT_MAIN), load_ok(GAIT_SHARD)], ignore_index=True)
    rows = [summarize("Widar", widar, 320), summarize("Gait", gait, 320)]
    values = pd.DataFrame(rows)

    expected = {
        "Widar": (320, 80, 59.48),
        "Gait": (316, 76, None),
    }
    for row in values.itertuples(index=False):
        minimum_runs, minimum_cells, robust = expected[row[0]]
        completed_runs = int(str(row[1]).split("/")[0])
        assert completed_runs >= minimum_runs
        assert row[2] >= minimum_cells
        if robust is not None:
            assert np.isclose(row[6], robust, atol=0.01)
        assert row[8] == f"{row[2]}/{row[2]}"
    return values


def render(values: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    rows = []
    for row in values.itertuples(index=False):
        rows.append(
            [
                row[0],
                row[1],
                f"{row[2]}",
                f"{row[3]:.2f}",
                f"{row[4]:.2f}",
                f"{row[5]:.2f}",
                f"{row[6]:.2f}",
                f"{row[7]:.2f}",
                row[8],
            ]
        )

    columns = [
        "Dataset",
        "Completed\nruns",
        "Matched\ncells",
        "Linear\n(%)",
        "Poly.\n(d=3, %)",
        "STC\n(%)",
        "Robust\n(%)",
        "Robust − mean\n(Linear, Poly., STC)\n(pp)",
        "Robust\nranked last",
    ]

    fig, ax = plt.subplots(figsize=(8.3, 2.25))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.105, 0.105, 0.105, 0.09, 0.105, 0.085, 0.09, 0.19, 0.125],
        bbox=[0.005, 0.29, 0.99, 0.67],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 1.35)

    header = "#244A73"
    robust_fill = "#F6CFCB"
    alternating = "#F3F5F7"
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.1)
        if r == 0:
            cell.set_facecolor(header)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(alternating if r % 2 == 0 else "white")
            if c in (6, 7, 8):
                cell.set_facecolor(robust_fill)
                cell.get_text().set_weight("bold")

    gait_coverage = values.loc[values["Dataset"].eq("Gait"), "Coverage"].iloc[0]
    gait_missing = 320 - int(gait_coverage.split("/")[0])
    ax.text(
        0.008,
        0.19,
        "Means are computed only over cells containing all four calibration methods; each cell fixes denoising, outlier handling, normalization, and interpolation.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.8,
        color="#333333",
    )
    ax.text(
        0.008,
        0.09,
        f"Widar is complete. Gait currently contains {gait_coverage} successful MLP runs; {gait_missing} incomplete combinations are excluded by matched-cell analysis.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.8,
        color="#333333",
    )

    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT_DIR / f"table_x_robust_main_results.{ext}",
            dpi=400 if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    values.to_csv(OUT_DIR / "table_x_robust_main_results.csv", index=False)


if __name__ == "__main__":
    render(build_values())
