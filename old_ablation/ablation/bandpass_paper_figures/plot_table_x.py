#!/usr/bin/env python3
"""Render the XRF55 80-combination denoiser summary as a paper table."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "SDP/test_xrf55/result/full_tests/xrf55_80_pipeline_resnet1d_summary.csv"
OUT_DIR = Path(__file__).resolve().parent

METHODS = ["wavelet", "butterworth", "savgol", "hampel", "bandpass"]
METHOD_IDS = {
    "wavelet": "wavelet",
    "butterworth": "butterworth_o5_c0.3",
    "savgol": "savgol_w7_p3",
    "hampel": "hampel_w5_s3",
    "bandpass": "bandpass_0.5-50",
}
PAIR_KEYS = ["outliers", "normalize", "interpolate"]


def build_values() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    ok = df.loc[df["status"].eq("ok")].copy()
    if len(ok) != 80:
        raise RuntimeError(f"Expected 80 successful rows, found {len(ok)}")

    rows = []
    bp = ok.loc[
        ok["denoise"].eq(METHOD_IDS["bandpass"]), PAIR_KEYS + ["test_acc"]
    ]
    if len(bp) != 16:
        raise RuntimeError(f"Expected 16 bandpass rows, found {len(bp)}")

    for method in METHODS:
        part = ok.loc[ok["denoise"].eq(METHOD_IDS[method])].copy()
        if len(part) != 16:
            raise RuntimeError(f"Expected 16 rows for {method}, found {len(part)}")
        test_pct = 100.0 * part["test_acc"]

        if method == "bandpass":
            deficit = None
            matched_losses = None
        else:
            paired = bp.merge(
                part[PAIR_KEYS + ["test_acc"]],
                on=PAIR_KEYS,
                suffixes=("_bp", "_ref"),
                validate="one_to_one",
            )
            diff_pp = 100.0 * (paired["test_acc_ref"] - paired["test_acc_bp"])
            deficit = diff_pp.mean()
            matched_losses = int((diff_pp > 0).sum())

        rows.append(
            {
                "Method": method,
                "Mean test accuracy (%)": test_pct.mean(),
                "Range low (%)": test_pct.min(),
                "Range high (%)": test_pct.max(),
                "Bandpass deficit (pp)": deficit,
                "Matched losses": matched_losses,
            }
        )
    return pd.DataFrame(rows)


def render(values: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    cell_text = []
    for row in values.itertuples(index=False):
        deficit = "--" if pd.isna(row[4]) else f"{row[4]:.2f}"
        losses = "--" if pd.isna(row[5]) else f"{int(row[5])}/16"
        cell_text.append(
            [
                row[0],
                f"{row[1]:.2f}",
                f"{row[2]:.2f}–{row[3]:.2f}",
                deficit,
                losses,
            ]
        )

    columns = [
        "Denoiser",
        "Mean test\naccuracy (%)",
        "Test range\n(%)",
        "Bandpass deficit\n(pp)",
        "Bandpass lower\nin matched cells",
    ]

    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.18, 0.19, 0.19, 0.20, 0.24],
        bbox=[0.01, 0.18, 0.98, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.35)

    header_color = "#244A73"
    bandpass_color = "#F6CFCB"
    alternate = "#F3F5F7"
    edge = "#FFFFFF"
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(edge)
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif r == len(METHODS):
            cell.set_facecolor(bandpass_color)
            cell.get_text().set_weight("bold")
        elif r % 2 == 0:
            cell.set_facecolor(alternate)
        else:
            cell.set_facecolor("white")

    ax.text(
        0.01,
        0.105,
        "Five denoisers × 16 matched downstream configurations (80 runs total).",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.2,
        color="#333333",
    )
    ax.text(
        0.01,
        0.045,
        "All 16 bandpass configurations ranked last within their matched cells; all runs used seed 42.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.2,
        color="#333333",
    )

    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT_DIR / f"table_x_xrf55_denoiser_summary.{ext}",
            dpi=400 if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    values.to_csv(OUT_DIR / "table_x_values.csv", index=False)


if __name__ == "__main__":
    render(build_values())
