"""Generate an IEEE-width factor-lattice atlas for Stage-II experiments.

The Stage-II experiment fixes one representative model per dataset:

* Widar: MLP, 320 complete preprocessing configurations.
* ElderAL: CSITime, 80 complete preprocessing configurations.

Every successful configuration is retained in the atlas.  The figure contains
English text only; its Chinese filename is intended for manuscript asset
management.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PACKAGE_DIR / "output"
DERIVED_DIR = PACKAGE_DIR / "derived_data"

WIDAR_FILE = (
    ROOT
    / "SDP/test_wider/result/full_tests_new/"
    "widar_320_pipeline_optimized_mlpmodel_summary.csv"
)
ELDERAL_FILE = (
    ROOT
    / "SDP/test_elderAL/result/full_tests/"
    "elderAL_80_pipeline_csitime_summary.csv"
)

DENOISE_ORDER = [
    "wavelet",
    "butterworth_o5_c0.3",
    "savgol_w7_p3",
    "bandpass_0.5-50",
    "hampel_w5_s3",
]
DENOISE_LABELS = {
    "wavelet": "Wavelet",
    "butterworth_o5_c0.3": "Butterworth",
    "savgol_w7_p3": "Savitzky–Golay",
    "bandpass_0.5-50": "Band-pass",
    "hampel_w5_s3": "Hampel",
}

OUTLIER_ORDER = ["iqr", "outlier_z-score"]
OUTLIER_LABELS = {"iqr": "IQR", "outlier_z-score": "Z-score"}

CALIBRATION_ORDER = ["linear", "polynomial_d3", "stc", "robust"]
CALIBRATION_LABELS = {
    "linear": "Linear",
    "polynomial_d3": "Polynomial",
    "stc": "STC",
    "robust": "Robust",
}

NORMALIZATION_ORDER = ["z-score", "min-max"]
NORMALIZATION_LABELS = {"z-score": "Z-score", "min-max": "Min–max"}

INTERPOLATION_METHOD_ORDER = ["linear", "cubic", "nearest", "decimate"]
INTERPOLATION_LABELS = {
    "linear": "Lin.",
    "cubic": "Cub.",
    "nearest": "Near.",
    "decimate": "Dec.",
}


def configure_matplotlib() -> None:
    """Use IEEE-compatible typography and editable SVG text."""

    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is None:
        raise FileNotFoundError("Arial or Calibri is required for this figure")

    fontManager.addfont(str(font_path))
    font_name = FontProperties(fname=str(font_path)).get_name()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "text.color": "#202020",
            "axes.labelcolor": "#202020",
            "axes.titlecolor": "#202020",
            "xtick.color": "#303030",
            "ytick.color": "#303030",
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _interpolation_method(value: str) -> str:
    text = str(value).strip().lower()
    for method in INTERPOLATION_METHOD_ORDER:
        if text.startswith(method):
            return method
    raise ValueError(f"Unknown interpolation setting: {value}")


def _validate_factorial(
    frame: pd.DataFrame,
    dataset: str,
    expected_model: str,
    expected_rows: int,
    include_calibration: bool,
) -> pd.DataFrame:
    required = {
        "combo_id",
        "model",
        "status",
        "test_acc",
        "denoise",
        "outliers",
        "normalize",
        "interpolate",
    }
    if include_calibration:
        required.add("calibrate")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{dataset} is missing columns: {sorted(missing)}")
    if len(frame) != expected_rows:
        raise ValueError(f"{dataset}: expected {expected_rows} rows, found {len(frame)}")
    if frame["combo_id"].nunique() != expected_rows:
        raise ValueError(f"{dataset}: combo_id is not unique")
    if set(frame["model"]) != {expected_model}:
        raise ValueError(f"{dataset}: unexpected model set {set(frame['model'])}")
    if set(frame["status"].astype(str).str.lower()) != {"ok"}:
        raise ValueError(f"{dataset}: Stage-II table contains unsuccessful rows")

    result = frame.copy()
    result.insert(0, "dataset", dataset)
    result["test_acc"] = pd.to_numeric(result["test_acc"], errors="raise")
    result["test_acc_pct"] = result["test_acc"] * 100.0
    if not result["test_acc_pct"].between(0, 100).all():
        raise ValueError(f"{dataset}: test accuracy outside [0, 100]")
    result["interpolation_method"] = result["interpolate"].map(
        _interpolation_method
    )

    if set(result["denoise"]) != set(DENOISE_ORDER):
        raise ValueError(f"{dataset}: unexpected denoising levels")
    if set(result["outliers"]) != set(OUTLIER_ORDER):
        raise ValueError(f"{dataset}: unexpected outlier-removal levels")
    if set(result["normalize"]) != set(NORMALIZATION_ORDER):
        raise ValueError(f"{dataset}: unexpected normalization levels")
    if set(result["interpolation_method"]) != set(INTERPOLATION_METHOD_ORDER):
        raise ValueError(f"{dataset}: unexpected interpolation levels")
    if include_calibration and set(result["calibrate"]) != set(CALIBRATION_ORDER):
        raise ValueError(f"{dataset}: unexpected calibration levels")

    keys = ["denoise", "outliers", "normalize", "interpolation_method"]
    if include_calibration:
        keys.insert(2, "calibrate")
    if result.duplicated(keys).any():
        raise ValueError(f"{dataset}: duplicated full-factor configuration")

    expected_product = (
        len(DENOISE_ORDER)
        * len(OUTLIER_ORDER)
        * len(NORMALIZATION_ORDER)
        * len(INTERPOLATION_METHOD_ORDER)
        * (len(CALIBRATION_ORDER) if include_calibration else 1)
    )
    if expected_product != expected_rows:
        raise AssertionError("Internal factorial specification is inconsistent")
    return result


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both authoritative Stage-II summaries and write a traceable table."""

    widar = _validate_factorial(
        pd.read_csv(WIDAR_FILE),
        dataset="Widar",
        expected_model="mlpmodel",
        expected_rows=320,
        include_calibration=True,
    )
    elderal = _validate_factorial(
        pd.read_csv(ELDERAL_FILE),
        dataset="ElderAL",
        expected_model="csitime",
        expected_rows=80,
        include_calibration=False,
    )
    elderal["calibrate"] = pd.NA

    for frame, source in [(widar, WIDAR_FILE), (elderal, ELDERAL_FILE)]:
        frame["source_file"] = source.relative_to(ROOT).as_posix()

    combined = pd.concat([widar, elderal], ignore_index=True, sort=False)
    columns = [
        "dataset",
        "combo_id",
        "model",
        "status",
        "test_acc",
        "test_acc_pct",
        "denoise",
        "outliers",
        "calibrate",
        "normalize",
        "interpolate",
        "interpolation_method",
        "source_file",
    ]
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    combined[columns].to_csv(
        DERIVED_DIR / "fine_grained_widar_elderal_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return widar, elderal


def _matrix(
    frame: pd.DataFrame,
    denoiser: str,
    include_calibration: bool,
) -> np.ndarray:
    subset = frame[frame["denoise"].eq(denoiser)]
    columns = pd.MultiIndex.from_product(
        [NORMALIZATION_ORDER, INTERPOLATION_METHOD_ORDER],
        names=["normalize", "interpolation_method"],
    )
    if include_calibration:
        rows = pd.MultiIndex.from_product(
            [CALIBRATION_ORDER, OUTLIER_ORDER],
            names=["calibrate", "outliers"],
        )
        table = subset.pivot(
            index=["calibrate", "outliers"],
            columns=["normalize", "interpolation_method"],
            values="test_acc_pct",
        ).reindex(index=rows, columns=columns)
    else:
        table = subset.pivot(
            index="outliers",
            columns=["normalize", "interpolation_method"],
            values="test_acc_pct",
        ).reindex(index=OUTLIER_ORDER, columns=columns)
    if table.isna().any().any():
        raise ValueError(f"Incomplete matrix for {denoiser}")
    return table.to_numpy(dtype=float)


def _draw_heatmap(
    axis: mpl.axes.Axes,
    values: np.ndarray,
    cmap: mpl.colors.Colormap,
    norm: Normalize,
    show_x_labels: bool,
    y_labels: list[str] | None,
    calibration_groups: bool,
) -> mpl.collections.QuadMesh:
    n_rows, n_cols = values.shape
    image = axis.pcolormesh(
        np.arange(n_cols + 1),
        np.arange(n_rows + 1),
        values,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.38,
        antialiased=True,
    )
    axis.set_aspect("equal")
    axis.set_xlim(0, n_cols)
    axis.set_ylim(n_rows, 0)

    axis.set_xticks(np.arange(n_cols) + 0.5)
    if show_x_labels:
        axis.set_xticklabels(
            [
                INTERPOLATION_LABELS[method]
                for _normalization in NORMALIZATION_ORDER
                for method in INTERPOLATION_METHOD_ORDER
            ],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
    else:
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)

    axis.set_yticks(np.arange(n_rows) + 0.5)
    if y_labels is None:
        axis.set_yticklabels([])
        axis.tick_params(axis="y", length=0)
    else:
        axis.set_yticklabels(y_labels)

    axis.tick_params(which="major", width=0.55, length=2.2, pad=1.5)

    axis.axvline(4.0, color="white", linewidth=1.15)
    if calibration_groups:
        for boundary in [2.0, 4.0, 6.0]:
            axis.axhline(boundary, color="white", linewidth=1.0)

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.55)

    return image


def build_figure(widar: pd.DataFrame, elderal: pd.DataFrame) -> mpl.figure.Figure:
    """Build the complete 400-cell factor-lattice accuracy atlas."""

    figure = plt.figure(figsize=(7.16, 4.25), facecolor="white")
    grid = GridSpec(
        2,
        5,
        figure=figure,
        height_ratios=[4.0, 1.0],
        left=0.135,
        right=0.985,
        top=0.865,
        bottom=0.235,
        wspace=0.10,
        hspace=0.82,
    )

    cmap = mpl.colormaps["cividis"]
    norm = Normalize(vmin=50, vmax=95)
    top_row_labels = [
        f"{CALIBRATION_LABELS[calibration]} · {OUTLIER_LABELS[outlier]}"
        for calibration in CALIBRATION_ORDER
        for outlier in OUTLIER_ORDER
    ]
    bottom_row_labels = [OUTLIER_LABELS[item] for item in OUTLIER_ORDER]

    top_axes: list[mpl.axes.Axes] = []
    bottom_axes: list[mpl.axes.Axes] = []
    last_image: mpl.collections.QuadMesh | None = None

    for column_index, denoiser in enumerate(DENOISE_ORDER):
        top_axis = figure.add_subplot(grid[0, column_index])
        top_axes.append(top_axis)
        last_image = _draw_heatmap(
            top_axis,
            _matrix(widar, denoiser, include_calibration=True),
            cmap,
            norm,
            show_x_labels=False,
            y_labels=top_row_labels if column_index == 0 else None,
            calibration_groups=True,
        )
        top_axis.set_title(
            DENOISE_LABELS[denoiser],
            pad=5,
            fontweight="semibold",
        )
        if column_index == 0:
            top_axis.set_ylabel("Calibration · Outlier Removal", labelpad=7)

        bottom_axis = figure.add_subplot(grid[1, column_index])
        bottom_axes.append(bottom_axis)
        _draw_heatmap(
            bottom_axis,
            _matrix(elderal, denoiser, include_calibration=False),
            cmap,
            norm,
            show_x_labels=True,
            y_labels=bottom_row_labels if column_index == 0 else None,
            calibration_groups=False,
        )
        if column_index == 0:
            bottom_axis.set_ylabel("Outlier Removal", labelpad=7)

        for axis in [top_axis, bottom_axis]:
            label_y = -0.075 if axis is top_axis else 1.08
            vertical_alignment = "top" if axis is top_axis else "bottom"
            axis.text(
                0.25,
                label_y,
                NORMALIZATION_LABELS["z-score"],
                transform=axis.transAxes,
                ha="center",
                va=vertical_alignment,
                fontsize=7.0,
                fontweight="semibold",
            )
            axis.text(
                0.75,
                label_y,
                NORMALIZATION_LABELS["min-max"],
                transform=axis.transAxes,
                ha="center",
                va=vertical_alignment,
                fontsize=7.0,
                fontweight="semibold",
            )

    figure.text(
        0.018,
        0.942,
        "(a) Widar — MLP",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
    )
    figure.text(
        0.018,
        0.405,
        "(b) ElderAL — CSITime",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
    )
    figure.text(
        0.56,
        0.126,
        "Interpolation Method",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    figure.text(
        0.56,
        0.895,
        "Denoising Method",
        ha="center",
        va="center",
        fontsize=8.5,
    )

    if last_image is None:
        raise AssertionError("No heatmap was drawn")
    colorbar_axis = figure.add_axes([0.265, 0.055, 0.58, 0.028])
    # Draw the scale as vector cells rather than Matplotlib's rasterized
    # gradient, so the delivered SVG remains entirely resolution-independent.
    scale_edges = np.linspace(norm.vmin, norm.vmax, 181)
    scale_centers = 0.5 * (scale_edges[:-1] + scale_edges[1:])
    colorbar_axis.pcolormesh(
        scale_edges,
        np.array([0.0, 1.0]),
        scale_centers[np.newaxis, :],
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="none",
        rasterized=False,
    )
    colorbar_axis.set_xlim(norm.vmin, norm.vmax)
    colorbar_axis.set_ylim(0.0, 1.0)
    colorbar_axis.set_yticks([])
    colorbar_axis.set_xticks([50, 60, 70, 80, 90, 95])
    colorbar_axis.set_xlabel("Test Accuracy (%)", labelpad=2)
    colorbar_axis.tick_params(axis="x", width=0.55, length=2.5, labelsize=7.2, pad=1.5)
    for spine in colorbar_axis.spines.values():
        spine.set_linewidth(0.55)
        spine.set_edgecolor("#555555")
    return figure


def save_figure(figure: mpl.figure.Figure) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "Widar与ElderAL细粒度全因子测试准确率图谱"
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    metadata = {
        "Title": "Fine-grained factorial test-accuracy atlas for Widar and ElderAL",
        "Description": (
            "All 320 Widar MLP configurations and all 80 ElderAL CSITime "
            "configurations; shared accuracy scale."
        ),
        "Creator": "Matplotlib; generate_fine_grained_widar_elderal_atlas.py",
    }
    figure.savefig(svg_path, format="svg", metadata=metadata, facecolor="white")
    figure.savefig(png_path, format="png", dpi=400, facecolor="white")
    plt.close(figure)
    return svg_path, png_path


def main() -> None:
    configure_matplotlib()
    widar, elderal = load_data()
    figure = build_figure(widar, elderal)
    svg_path, png_path = save_figure(figure)

    for dataset, frame in [("Widar", widar), ("ElderAL", elderal)]:
        accuracy = frame["test_acc_pct"]
        best = frame.loc[accuracy.idxmax()]
        print(
            f"{dataset}: n={len(frame)}, min={accuracy.min():.2f}, "
            f"median={accuracy.median():.2f}, mean={accuracy.mean():.2f}, "
            f"max={accuracy.max():.2f}, best={best['combo_id']}"
        )
    print(f"SVG: {svg_path}")
    print(f"PNG: {png_path}")
    print(f"Data: {DERIVED_DIR / 'fine_grained_widar_elderal_long.csv'}")


if __name__ == "__main__":
    main()
