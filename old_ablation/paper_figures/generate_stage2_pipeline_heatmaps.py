"""Generate the four-panel Stage-II preprocessing-pipeline heatmap.

The layout follows the supplied reference figure:

* Gait and Widar show all 320 configurations (including calibration).
* XRF55 and ElderAL show all 80 configurations (without calibration).
* Rows encode denoising, outlier removal, and calibration where applicable.
* Columns encode normalization and interpolation.

The delivered PDF uses vector ``QuadMesh`` cells and vector text/markers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PREVIEW_DIR = ROOT / ".codex_tmp"

GAIT_MAIN = (
    ROOT
    / "SDP/test_gait/result/full_tests_new/"
    "gait_320_pipeline_optimized_mlpmodel_summary.csv"
)
GAIT_SHARD = (
    ROOT
    / "SDP/test_gait/result/full_tests_new_gpu1_290_320/"
    "gait_320_pipeline_gpu1_290_320_mlpmodel_summary.csv"
)
WIDAR = (
    ROOT
    / "SDP/test_wider/result/full_tests_new/"
    "widar_320_pipeline_optimized_mlpmodel_summary.csv"
)
XRF55 = (
    ROOT
    / "SDP/test_xrf55/result/full_tests/"
    "xrf55_80_pipeline_resnet1d_summary.csv"
)
ELDERAL = (
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

INTERPOLATION_ORDER = ["linear", "cubic", "nearest", "decimate"]
INTERPOLATION_LABELS = {
    "linear": "Linear",
    "cubic": "Cubic",
    "nearest": "Nearest",
    "decimate": "Decimate",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    model: str
    expected_rows: int
    include_calibration: bool


SPECS = {
    "Gait": DatasetSpec("Gait", "(a) Gait — MLP", "mlpmodel", 320, True),
    "Widar": DatasetSpec("Widar", "(b) Widar — MLP", "mlpmodel", 320, True),
    "XRF55": DatasetSpec("XRF55", "(c) XRF55 — ResNet1D", "resnet1d", 80, False),
    "ElderAL": DatasetSpec("ElderAL", "(d) ElderAL — CSITime", "csitime", 80, False),
}


def configure_matplotlib() -> None:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is not None:
        fontManager.addfont(str(font_path))
        font_name = FontProperties(fname=str(font_path)).get_name()
    else:
        font_name = "DejaVu Sans"

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.3,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.0,
            "text.color": "#202020",
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def interpolation_method(value: object) -> str:
    text = str(value).strip().lower()
    for method in INTERPOLATION_ORDER:
        if text.startswith(method):
            return method
    raise ValueError(f"Unknown interpolation setting: {value!r}")


def validate_factorial(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    required = {
        "combo_index",
        "combo_id",
        "model",
        "status",
        "test_acc",
        "denoise",
        "outliers",
        "normalize",
        "interpolate",
    }
    if spec.include_calibration:
        required.add("calibrate")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{spec.name}: missing columns {sorted(missing)}")
    if len(frame) != spec.expected_rows:
        raise ValueError(
            f"{spec.name}: expected {spec.expected_rows} rows, found {len(frame)}"
        )
    if frame["combo_id"].nunique() != spec.expected_rows:
        raise ValueError(f"{spec.name}: combo_id is not unique")
    expected_indices = set(range(1, spec.expected_rows + 1))
    actual_indices = set(pd.to_numeric(frame["combo_index"], errors="raise").astype(int))
    if actual_indices != expected_indices:
        missing_indices = sorted(expected_indices - actual_indices)
        extra_indices = sorted(actual_indices - expected_indices)
        raise ValueError(
            f"{spec.name}: incomplete combo_index coverage; "
            f"missing={missing_indices}, extra={extra_indices}"
        )
    if set(frame["model"].astype(str)) != {spec.model}:
        raise ValueError(f"{spec.name}: unexpected model values {set(frame['model'])}")
    if set(frame["status"].astype(str).str.lower()) != {"ok"}:
        raise ValueError(f"{spec.name}: unsuccessful rows are present")

    result = frame.copy()
    result["test_acc"] = pd.to_numeric(result["test_acc"], errors="raise")
    if not result["test_acc"].between(0.0, 1.0).all():
        raise ValueError(f"{spec.name}: test_acc lies outside [0, 1]")
    result["test_acc_pct"] = result["test_acc"] * 100.0
    result["interpolation_method"] = result["interpolate"].map(
        interpolation_method
    )

    factor_checks = {
        "denoise": DENOISE_ORDER,
        "outliers": OUTLIER_ORDER,
        "normalize": NORMALIZATION_ORDER,
        "interpolation_method": INTERPOLATION_ORDER,
    }
    if spec.include_calibration:
        factor_checks["calibrate"] = CALIBRATION_ORDER
    for column, expected in factor_checks.items():
        if set(result[column]) != set(expected):
            raise ValueError(
                f"{spec.name}: unexpected {column} levels {sorted(set(result[column]))}"
            )

    keys = ["denoise", "outliers"]
    if spec.include_calibration:
        keys.append("calibrate")
    keys.extend(["normalize", "interpolation_method"])
    if result.duplicated(keys).any():
        duplicates = result.loc[result.duplicated(keys, keep=False), keys]
        raise ValueError(f"{spec.name}: duplicated factorial cells\n{duplicates}")

    expected_product = (
        len(DENOISE_ORDER)
        * len(OUTLIER_ORDER)
        * len(NORMALIZATION_ORDER)
        * len(INTERPOLATION_ORDER)
        * (len(CALIBRATION_ORDER) if spec.include_calibration else 1)
    )
    if expected_product != spec.expected_rows:
        raise AssertionError("Internal factorial specification is inconsistent")
    return result


def load_data() -> dict[str, pd.DataFrame]:
    gait_main = pd.read_csv(GAIT_MAIN)
    gait_shard = pd.read_csv(GAIT_SHARD)
    if set(gait_main["combo_index"]).intersection(gait_shard["combo_index"]):
        raise ValueError("Gait source files overlap in combo_index")
    gait = pd.concat([gait_main, gait_shard], ignore_index=True).sort_values(
        "combo_index"
    )

    raw_frames = {
        "Gait": gait,
        "Widar": pd.read_csv(WIDAR),
        "XRF55": pd.read_csv(XRF55),
        "ElderAL": pd.read_csv(ELDERAL),
    }
    return {
        name: validate_factorial(frame, SPECS[name])
        for name, frame in raw_frames.items()
    }


def accuracy_matrix(frame: pd.DataFrame, spec: DatasetSpec) -> np.ndarray:
    row_names = ["denoise", "outliers"]
    row_levels: list[list[str]] = [DENOISE_ORDER, OUTLIER_ORDER]
    if spec.include_calibration:
        row_names.append("calibrate")
        row_levels.append(CALIBRATION_ORDER)
    rows = pd.MultiIndex.from_product(row_levels, names=row_names)
    columns = pd.MultiIndex.from_product(
        [NORMALIZATION_ORDER, INTERPOLATION_ORDER],
        names=["normalize", "interpolation_method"],
    )
    table = frame.pivot(
        index=row_names,
        columns=["normalize", "interpolation_method"],
        values="test_acc_pct",
    ).reindex(index=rows, columns=columns)
    if table.isna().any().any():
        missing = table.isna().stack(future_stack=True)
        raise ValueError(f"{spec.name}: missing factorial cells\n{missing[missing]}")
    return table.to_numpy(dtype=float)


def add_cell(
    axis: mpl.axes.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    fontsize: float,
    linewidth: float = 0.45,
    fontweight: str = "normal",
) -> None:
    axis.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor="white",
            edgecolor="#777777",
            linewidth=linewidth,
            clip_on=False,
            zorder=4,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        clip_on=False,
        zorder=5,
    )


def draw_column_headers(axis: mpl.axes.Axes, label_left: float) -> None:
    del label_left
    add_cell(
        axis,
        0.0,
        -4.0,
        8.0,
        1.0,
        "Normalization",
        fontsize=7.2,
        fontweight="normal",
    )
    for normalization_index, normalization in enumerate(NORMALIZATION_ORDER):
        add_cell(
            axis,
            4.0 * normalization_index,
            -3.0,
            4.0,
            1.0,
            NORMALIZATION_LABELS[normalization],
            fontsize=7.0,
            fontweight="normal",
        )
    add_cell(
        axis,
        0.0,
        -2.0,
        8.0,
        1.0,
        "Interpolation",
        fontsize=7.0,
        fontweight="normal",
    )
    for column_index, method in enumerate(
        INTERPOLATION_ORDER * len(NORMALIZATION_ORDER)
    ):
        add_cell(
            axis,
            float(column_index),
            -1.0,
            1.0,
            1.0,
            INTERPOLATION_LABELS[method],
            fontsize=6.2,
        )


def draw_row_headers(
    axis: mpl.axes.Axes,
    spec: DatasetSpec,
    label_left: float,
) -> None:
    if spec.include_calibration:
        denoise_x0, denoise_x1 = label_left, -2.15
        outlier_x0, outlier_x1 = denoise_x1, -1.15
        calibration_x0, calibration_x1 = outlier_x1, 0.0
        add_cell(
            axis,
            denoise_x0,
            -1.0,
            denoise_x1 - denoise_x0,
            1.0,
            "Denoising",
            fontsize=6.4,
        )
        add_cell(
            axis,
            outlier_x0,
            -1.0,
            outlier_x1 - outlier_x0,
            1.0,
            "Outlier",
            fontsize=6.4,
        )
        add_cell(
            axis,
            calibration_x0,
            -1.0,
            calibration_x1 - calibration_x0,
            1.0,
            "Calibration",
            fontsize=6.1,
        )

        row = 0
        for denoiser in DENOISE_ORDER:
            add_cell(
                axis,
                denoise_x0,
                float(row),
                denoise_x1 - denoise_x0,
                8.0,
                DENOISE_LABELS[denoiser],
                fontsize=6.5,
                linewidth=0.65,
            )
            for outlier in OUTLIER_ORDER:
                add_cell(
                    axis,
                    outlier_x0,
                    float(row),
                    outlier_x1 - outlier_x0,
                    4.0,
                    OUTLIER_LABELS[outlier],
                    fontsize=6.2,
                    linewidth=0.55,
                )
                for calibration in CALIBRATION_ORDER:
                    add_cell(
                        axis,
                        calibration_x0,
                        float(row),
                        calibration_x1 - calibration_x0,
                        1.0,
                        CALIBRATION_LABELS[calibration],
                        fontsize=5.9,
                    )
                    row += 1
    else:
        denoise_x0, denoise_x1 = label_left, -1.35
        outlier_x0, outlier_x1 = denoise_x1, 0.0
        add_cell(
            axis,
            denoise_x0,
            -1.0,
            denoise_x1 - denoise_x0,
            1.0,
            "Denoising",
            fontsize=6.8,
        )
        add_cell(
            axis,
            outlier_x0,
            -1.0,
            outlier_x1 - outlier_x0,
            1.0,
            "Outlier",
            fontsize=6.8,
        )
        row = 0
        for denoiser in DENOISE_ORDER:
            add_cell(
                axis,
                denoise_x0,
                float(row),
                denoise_x1 - denoise_x0,
                2.0,
                DENOISE_LABELS[denoiser],
                fontsize=6.8,
                linewidth=0.65,
            )
            for outlier in OUTLIER_ORDER:
                add_cell(
                    axis,
                    outlier_x0,
                    float(row),
                    outlier_x1 - outlier_x0,
                    1.0,
                    OUTLIER_LABELS[outlier],
                    fontsize=6.5,
                )
                row += 1


def draw_heatmap(
    axis: mpl.axes.Axes,
    values: np.ndarray,
    spec: DatasetSpec,
    cmap: mpl.colors.Colormap,
    norm: Normalize,
) -> mpl.collections.QuadMesh:
    n_rows, n_columns = values.shape
    if n_columns != 8:
        raise ValueError(f"{spec.name}: expected 8 columns, found {n_columns}")

    image = axis.pcolormesh(
        np.arange(n_columns + 1),
        np.arange(n_rows + 1),
        values,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors=(1.0, 1.0, 1.0, 0.68),
        linewidth=0.28,
        antialiased=True,
        rasterized=False,
        zorder=1,
    )

    label_left = -3.55 if spec.include_calibration else -3.05
    draw_column_headers(axis, label_left)
    draw_row_headers(axis, spec, label_left)

    axis.vlines(4.0, 0.0, float(n_rows), color="#454545", linewidth=0.8, zorder=3)
    if spec.include_calibration:
        for boundary in range(4, n_rows, 4):
            linewidth = 0.85 if boundary % 8 == 0 else 0.5
            axis.axhline(boundary, color="#555555", linewidth=linewidth, zorder=3)
    else:
        for boundary in range(2, n_rows, 2):
            axis.axhline(boundary, color="#555555", linewidth=0.8, zorder=3)

    best_flat = int(np.nanargmax(values))
    best_row, best_column = np.unravel_index(best_flat, values.shape)
    axis.plot(
        best_column + 0.5,
        best_row + 0.5,
        marker="*",
        markersize=6.4 if spec.include_calibration else 7.2,
        markerfacecolor="#111111",
        markeredgecolor="#111111",
        linestyle="none",
        zorder=6,
    )

    axis.set_xlim(label_left, 8.0)
    axis.set_ylim(float(n_rows), -4.1)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(spec.title, loc="left", pad=13.0, fontweight="bold")
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def build_figure(frames: dict[str, pd.DataFrame]) -> mpl.figure.Figure:
    figure = plt.figure(figsize=(14.2, 8.6), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[3.55, 1.35],
        left=0.035,
        right=0.925,
        top=0.935,
        bottom=0.07,
        wspace=0.105,
        hspace=0.18,
    )
    cmap = mpl.colormaps["turbo"]
    norm = Normalize(vmin=0.0, vmax=100.0)

    panel_order = ["Gait", "Widar", "XRF55", "ElderAL"]
    last_image: mpl.collections.QuadMesh | None = None
    for index, name in enumerate(panel_order):
        spec = SPECS[name]
        axis = figure.add_subplot(grid[index // 2, index % 2])
        last_image = draw_heatmap(
            axis,
            accuracy_matrix(frames[name], spec),
            spec,
            cmap,
            norm,
        )

    if last_image is None:
        raise AssertionError("No heatmap was drawn")

    colorbar_axis = figure.add_axes([0.944, 0.13, 0.018, 0.73])
    colorbar = figure.colorbar(last_image, cax=colorbar_axis, orientation="vertical")
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_ticks([0, 20, 40, 60, 80, 100])
    colorbar.set_label("Test accuracy (%)", rotation=90, labelpad=8)
    colorbar.ax.tick_params(labelsize=7.2, width=0.55, length=3.0, pad=2.0)
    colorbar.outline.set_linewidth(0.6)
    colorbar.outline.set_edgecolor("#555555")
    figure.add_artist(
        Line2D(
            [0.946],
            [0.10],
            transform=figure.transFigure,
            marker="*",
            markersize=6.4,
            markerfacecolor="#111111",
            markeredgecolor="#111111",
            linestyle="none",
        )
    )
    figure.text(0.954, 0.10, "Best", ha="left", va="center", fontsize=7.2)
    return figure


def save_figure(figure: mpl.figure.Figure) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "fig_stage2_pipeline_test_accuracy_heatmaps.pdf"
    preview_path = PREVIEW_DIR / "fig_stage2_pipeline_test_accuracy_heatmaps_preview.png"
    metadata = {
        "Title": "Stage-II preprocessing-pipeline test-accuracy heatmaps",
        "Subject": (
            "Gait and Widar 320-configuration lattices; XRF55 and ElderAL "
            "80-configuration lattices"
        ),
        "Author": "SDP benchmark",
        "Creator": "Matplotlib; generate_stage2_pipeline_heatmaps.py",
    }
    figure.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata=metadata,
    )
    figure.savefig(
        preview_path,
        format="png",
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return pdf_path, preview_path


def report_summary(frames: dict[str, pd.DataFrame]) -> None:
    for name in ["Gait", "Widar", "XRF55", "ElderAL"]:
        frame = frames[name]
        best_index = frame["test_acc_pct"].idxmax()
        best = frame.loc[best_index]
        calibration = (
            f", calibrate={best['calibrate']}"
            if SPECS[name].include_calibration
            else ""
        )
        print(
            f"{name}: rows={len(frame)}, min={frame['test_acc_pct'].min():.2f}, "
            f"mean={frame['test_acc_pct'].mean():.2f}, "
            f"max={frame['test_acc_pct'].max():.2f}, best={best['combo_id']}, "
            f"denoise={best['denoise']}, outliers={best['outliers']}"
            f"{calibration}, normalize={best['normalize']}, "
            f"interpolate={best['interpolation_method']}"
        )


def main() -> None:
    configure_matplotlib()
    frames = load_data()
    report_summary(frames)
    figure = build_figure(frames)
    pdf_path, preview_path = save_figure(figure)
    print(f"Vector PDF: {pdf_path}")
    print(f"QA preview: {preview_path}")


if __name__ == "__main__":
    main()
