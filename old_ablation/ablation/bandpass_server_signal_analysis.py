"""Server-side Bandpass signal diagnostics for ElderAL and XRF55.

This script is designed to be copied to and run on the data server.  It never
contains cached/local measurements.  All reported values are computed from the
paths supplied at runtime.

Main outputs
------------
ElderAL
    * raw-length/bypass statistics before any resize or padding;
    * exact source-output identity checks for T < the derived filtfilt limit;
    * publication figures for the global bypass mechanism and deterministic
      before/after examples.
XRF55
    * per-sample negative-value distributions immediately after each denoiser;
    * strict (< 0) and scale-aware meaningful-negative definitions;
    * publication figures and machine-readable summaries.

Examples
--------
Run the complete signal study on the server::

    python ablation/bandpass_server_signal_analysis.py \
      --elder-root sdp_dataset/elderAL \
      --xrf-root sdp_dataset/xrf55/wifi

Only ElderAL or XRF55::

    python ablation/bandpass_server_signal_analysis.py --only elder
    python ablation/bandpass_server_signal_analysis.py --only xrf

The self-test uses synthetic arrays only and does not read a dataset::

    python ablation/bandpass_server_signal_analysis.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import multiprocessing as mp
import platform
import re
import sys
import time
import traceback
import types
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import butter


REPO_ROOT = Path(__file__).resolve().parents[1]
WSDP_ROOT = (
    REPO_ROOT / "SDP" / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
)
WSDP_SRC = WSDP_ROOT / "src"
DEFAULT_ELDER_ROOT = REPO_ROOT / "sdp_dataset" / "elderAL"
DEFAULT_XRF_ROOT = REPO_ROOT / "sdp_dataset" / "xrf55" / "wifi"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "ablation" / "bandpass_server_results" / "signal_analysis"
)
DEFAULT_XRF_USER_COUNT = 3
XRF_ACTION_IDS = tuple(range(1, 56))
XRF_REPETITION_IDS = tuple(range(1, 21))

BANDPASS_ORDER = 4
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 50.0
ELDER_FS_HZ = 1000.0
XRF_LEGACY_FS_HZ = 1000.0
XRF_TRUE_FS_HZ = 200.0
XRF_CACHE_SCHEMA_VERSION = 2

METHOD_ORDER = [
    "raw",
    "wavelet",
    "butterworth_o5_c0.3",
    "savgol_w7_p3",
    "bandpass_fs1000",
    "bandpass_fs200",
    "hampel_w5_s3",
]
METHOD_LABELS = {
    "raw": "Raw",
    "wavelet": "Wavelet",
    "butterworth_o5_c0.3": "Butterworth",
    "savgol_w7_p3": "Savgol",
    "bandpass_fs1000": "Bandpass\n$ f_s=1000$",
    "bandpass_fs200": "Bandpass\n$ f_s=200$",
    "hampel_w5_s3": "Hampel",
}
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#111111",
    "gray": "#7A7A7A",
}

ELDER_FIELDS = [
    "file_name",
    "frames",
    "values",
    "source_min_length",
    "predicted_bypass",
    "output_exactly_equal_to_input",
    "max_absolute_change",
    "mae",
    "nmae",
    "rmse",
    "nrmse",
    "pearson_r",
    "nonfinite_input_count",
    "nonfinite_output_count",
]
XRF_FIELDS = [
    "run_signature",
    "file_name",
    "record_index",
    "frames",
    "features",
    "antennas",
    "input_representation",
    "input_dtype",
    "max_abs_imaginary",
    "raw_is_nonnegative",
    "raw_negative_rate",
    "method",
    "values",
    "finite_values",
    "nonfinite_count",
    "negative_count_strict",
    "negative_rate_strict",
    "negative_count_meaningful",
    "negative_rate_meaningful",
    "negative_energy_fraction",
    "meaningful_negative_tolerance",
    "step_pairs",
    "rising_step_count",
    "rising_step_rate",
    "falling_step_count",
    "falling_step_rate",
    "flat_step_count",
    "flat_step_rate",
    "zero_crossing_count",
    "zero_crossing_rate",
    "negative_negative_pair_count",
    "negative_negative_pair_rate",
    "comparable_abs_slope_count",
    "abs_slope_direction_disagreement_count",
    "abs_slope_direction_disagreement_rate",
    "meaningful_original_slope_count",
    "abs_slope_direction_changed_or_lost_count",
    "abs_slope_direction_changed_or_lost_rate",
    "minimum",
    "p01",
    "p05",
    "median",
    "p95",
    "p99",
    "maximum",
    "mean",
    "std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--elder-root",
        type=Path,
        default=DEFAULT_ELDER_ROOT,
        help="Server ElderAL directory (default: sdp_dataset/elderAL)",
    )
    parser.add_argument(
        "--xrf-root",
        type=Path,
        default=DEFAULT_XRF_ROOT,
        help="Server XRF55 directory (default: sdp_dataset/xrf55/wifi)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Result directory",
    )
    parser.add_argument(
        "--only",
        choices=["all", "elder", "xrf"],
        default="all",
        help="Run both studies or one dataset only",
    )
    parser.add_argument(
        "--max-elder-files",
        type=int,
        default=0,
        help="0 means all files; positive values are smoke-test limits",
    )
    parser.add_argument(
        "--max-xrf-files",
        type=int,
        default=0,
        help="0 means all files from selected users; positive values are smoke-test limits",
    )
    parser.add_argument(
        "--xrf-user-count",
        type=int,
        default=DEFAULT_XRF_USER_COUNT,
        help=(
            "Use the first N numeric XRF55 user IDs; default 3 matches the "
            "classification experiment"
        ),
    )
    parser.add_argument(
        "--meaningful-negative-relative-tol",
        type=float,
        default=1e-6,
        help="Negative threshold = -relative_tol * median nonzero raw magnitude",
    )
    parser.add_argument(
        "--xrf-complex-policy",
        choices=["error", "amplitude"],
        default="error",
        help=(
            "Default 'error' refuses genuinely complex XRF input. "
            "'amplitude' explicitly studies abs(CSI) and records that choice."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume XRF per-sample CSV and skip completed file/record/method rows",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print XRF progress every N files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Spawned CPU workers for XRF55 file-level analysis",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic source/filter checks only; no dataset access",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration and exit before dataset loading",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def source_min_length(
    order: int = BANDPASS_ORDER,
    low_hz: float = BANDPASS_LOW_HZ,
    high_hz: float = BANDPASS_HIGH_HZ,
    fs_hz: float = ELDER_FS_HZ,
) -> tuple[int, np.ndarray, np.ndarray]:
    nyquist = fs_hz / 2.0
    b, a = butter(
        order,
        [low_hz / nyquist, high_hz / nyquist],
        btype="band",
    )
    return 3 * max(len(a), len(b)) + 1, b, a


def import_wsdp_runtime() -> dict[str, Any]:
    """Import only reader/algorithm packages, avoiding the training stack.

    The repository's ``wsdp.__init__`` imports ``core`` and therefore PyTorch.
    Signal statistics do not need that dependency, so a namespace package is
    installed before the exact reader/algorithm submodules are imported.
    """
    if not (WSDP_SRC / "wsdp").is_dir():
        raise FileNotFoundError(f"WSDP source directory not found: {WSDP_SRC}")
    if str(WSDP_SRC) not in sys.path:
        sys.path.insert(0, str(WSDP_SRC))

    if "wsdp" not in sys.modules:
        wsdp_package = types.ModuleType("wsdp")
        wsdp_package.__path__ = [str(WSDP_SRC / "wsdp")]
        wsdp_package.__package__ = "wsdp"
        sys.modules["wsdp"] = wsdp_package
    if "wsdp.algorithms" not in sys.modules:
        algorithms_package = types.ModuleType("wsdp.algorithms")
        algorithms_package.__path__ = [str(WSDP_SRC / "wsdp" / "algorithms")]
        algorithms_package.__package__ = "wsdp.algorithms"
        sys.modules["wsdp.algorithms"] = algorithms_package

    from wsdp import readers
    from wsdp.algorithms.amplitude import hampel_filter
    from wsdp.algorithms.denoising import wavelet_denoise_csi
    from wsdp.algorithms.denoising_butterworth import (
        butterworth_bandpass,
        butterworth_denoise,
        savgol_denoise,
    )

    return {
        "readers": readers,
        "hampel_filter": hampel_filter,
        "wavelet_denoise_csi": wavelet_denoise_csi,
        "butterworth_bandpass": butterworth_bandpass,
        "butterworth_denoise": butterworth_denoise,
        "savgol_denoise": savgol_denoise,
    }


def publication_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 120,
        }
    )


def save_publication_figure(fig: Any, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 600}),
    ):
        path = stem.with_suffix(suffix)
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    return outputs


def panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )


def append_csv_row(
    path: Path,
    row: dict[str, Any],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def append_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    """Append one completely analysed record in a single file transaction."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def iter_csv(path: Path) -> Iterable[dict[str, str]]:
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def csv_header(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def valid_xrf_csv_row(row: dict[str, Any]) -> bool:
    """Reject truncated/corrupt rows so resume recomputes them."""
    if any(row.get(field) in {None, ""} for field in XRF_FIELDS):
        return False
    if str(row.get("method")) not in METHOD_ORDER:
        return False
    try:
        int(row["record_index"])
        int(row["values"])
        int(row["finite_values"])
        int(row["step_pairs"])
        float(row["negative_rate_meaningful"])
        float(row["abs_slope_direction_changed_or_lost_rate"])
    except (TypeError, ValueError):
        return False
    return True


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int = 2000,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean without a large index matrix."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    batch_size = 100
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        indices = rng.integers(
            0,
            values.size,
            size=(stop - start, values.size),
        )
        estimates[start:stop] = np.mean(values[indices], axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def stack_record(record: Any) -> np.ndarray | None:
    frames = sorted(record.frames, key=lambda frame: frame.timestamp)
    if len(frames) < 2:
        return None
    array = np.stack([frame.csi_array for frame in frames], axis=0)
    if array.ndim == 2:
        array = np.expand_dims(array, -1)
    if array.ndim != 3:
        raise ValueError(f"Expected (T,F,A), got {array.shape}: {record.file_name}")
    return array


def numeric_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left).reshape(-1)
    right_flat = np.asarray(right).reshape(-1)
    finite = np.isfinite(left_flat) & np.isfinite(right_flat)
    left_flat = left_flat[finite]
    right_flat = right_flat[finite]
    if left_flat.size < 2 or np.std(left_flat) < 1e-12 or np.std(right_flat) < 1e-12:
        return 1.0 if np.array_equal(left_flat, right_flat) else 0.0
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def elder_row(
    record: Any,
    raw: np.ndarray,
    filtered: np.ndarray,
    min_length: int,
) -> dict[str, Any]:
    difference = filtered - raw
    finite_raw = np.isfinite(raw)
    finite_filtered = np.isfinite(filtered)
    mae = float(np.nanmean(np.abs(difference)))
    rmse = float(np.sqrt(np.nanmean(np.abs(difference) ** 2)))
    mean_abs = float(np.nanmean(np.abs(raw)))
    raw_rms = float(np.sqrt(np.nanmean(np.abs(raw) ** 2)))
    return {
        "file_name": str(record.file_name),
        "frames": int(raw.shape[0]),
        "values": int(raw.size),
        "source_min_length": min_length,
        "predicted_bypass": bool(raw.shape[0] < min_length),
        "output_exactly_equal_to_input": bool(
            np.array_equal(filtered, raw, equal_nan=True)
        ),
        "max_absolute_change": float(np.nanmax(np.abs(difference))),
        "mae": mae,
        "nmae": mae / max(mean_abs, 1e-12),
        "rmse": rmse,
        "nrmse": rmse / max(raw_rms, 1e-12),
        "pearson_r": numeric_correlation(raw, filtered),
        "nonfinite_input_count": int(raw.size - np.count_nonzero(finite_raw)),
        "nonfinite_output_count": int(
            filtered.size - np.count_nonzero(finite_filtered)
        ),
    }


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def choose_representative_row(
    rows: list[dict[str, Any]],
    *,
    bypass: bool,
) -> dict[str, Any]:
    candidates = [row for row in rows if parse_bool(row["predicted_bypass"]) is bypass]
    if not candidates:
        raise RuntimeError(f"No {'bypass' if bypass else 'eligible'} ElderAL sample")
    target = float(np.median([float(row["frames"]) for row in candidates]))
    return min(
        candidates,
        key=lambda row: (
            abs(float(row["frames"]) - target),
            str(row["file_name"]),
        ),
    )


def highest_variance_channel(raw: np.ndarray) -> tuple[int, int]:
    variances = np.nanvar(np.asarray(raw, dtype=np.float64), axis=0)
    index = int(np.nanargmax(variances))
    return tuple(int(value) for value in np.unravel_index(index, variances.shape))


def summarize_elder(
    rows: list[dict[str, Any]],
    min_length: int,
    coefficient_lengths: dict[str, int],
) -> dict[str, Any]:
    bypass = [row for row in rows if parse_bool(row["predicted_bypass"])]
    eligible = [row for row in rows if not parse_bool(row["predicted_bypass"])]
    exact_bypass = [
        row for row in bypass if parse_bool(row["output_exactly_equal_to_input"])
    ]
    total_frames = sum(int(row["frames"]) for row in rows)
    bypass_frames = sum(int(row["frames"]) for row in bypass)
    total_values = sum(int(row["values"]) for row in rows)
    bypass_values = sum(int(row["values"]) for row in bypass)
    return {
        "dataset": "elderAL",
        "valid_samples": len(rows),
        "source_min_length": min_length,
        "source_condition": f"T < {min_length}",
        "coefficient_lengths": coefficient_lengths,
        "bypass_samples": len(bypass),
        "bypass_sample_rate": len(bypass) / max(len(rows), 1),
        "bypass_frames": bypass_frames,
        "bypass_frame_rate": bypass_frames / max(total_frames, 1),
        "bypass_values": bypass_values,
        "bypass_value_rate": bypass_values / max(total_values, 1),
        "bypass_exact_matches": len(exact_bypass),
        "bypass_exact_match_rate": len(exact_bypass) / max(len(bypass), 1),
        "eligible_samples": len(eligible),
        "eligible_changed_samples": sum(
            not parse_bool(row["output_exactly_equal_to_input"]) for row in eligible
        ),
        "bypass_nrmse_median": (
            float(np.median([float(row["nrmse"]) for row in bypass]))
            if bypass
            else None
        ),
        "eligible_nrmse_median": (
            float(np.median([float(row["nrmse"]) for row in eligible]))
            if eligible
            else None
        ),
        "frame_minimum": min(int(row["frames"]) for row in rows),
        "frame_median": float(np.median([int(row["frames"]) for row in rows])),
        "frame_maximum": max(int(row["frames"]) for row in rows),
        "length_histogram": dict(
            sorted(Counter(int(row["frames"]) for row in rows).items())
        ),
    }


def plot_elder_overview(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    publication_style()
    import matplotlib.pyplot as plt

    min_length = int(summary["source_min_length"])
    frames = np.asarray([int(row["frames"]) for row in rows])
    nrmse = np.asarray([float(row["nrmse"]) for row in rows])
    bypass = frames < min_length

    # Keep the two panels comfortably separated: the original compact layout
    # placed the scatter legend and the exact-match annotation in the same
    # upper-right corner, which made the exported figure unreadable.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axes[0]
    max_frame = int(np.max(frames))
    if max_frame - int(np.min(frames)) <= 120:
        bins = np.arange(int(np.min(frames)) - 0.5, max_frame + 1.5)
    else:
        bins = min(50, max(15, int(np.sqrt(len(frames)))))
    ax.hist(
        frames,
        bins=bins,
        color=OKABE_ITO["blue"],
        edgecolor="white",
        linewidth=0.35,
    )
    ax.axvspan(
        ax.get_xlim()[0],
        min_length,
        color=OKABE_ITO["orange"],
        alpha=0.13,
        lw=0,
    )
    ax.axvline(
        min_length,
        color=OKABE_ITO["red"],
        linestyle="--",
        linewidth=1.1,
    )
    ax.set_xlabel("Raw sequence length, $T$ (frames)")
    ax.set_ylabel("Number of samples")
    ax.text(
        0.98,
        0.95,
        (
            f"$T<{min_length}$: {summary['bypass_samples']}/"
            f"{summary['valid_samples']}\n"
            f"({summary['bypass_sample_rate'] * 100:.1f}% of samples)"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.8",
            "linewidth": 0.5,
            "alpha": 0.92,
        },
    )
    panel_label(ax, "a")

    ax = axes[1]
    ax.scatter(
        frames[bypass],
        nrmse[bypass],
        s=10,
        color=OKABE_ITO["orange"],
        alpha=0.65,
        edgecolors="none",
        rasterized=True,
        label=f"Bypassed ($T<{min_length}$)",
    )
    ax.scatter(
        frames[~bypass],
        nrmse[~bypass],
        s=10,
        color=OKABE_ITO["blue"],
        alpha=0.65,
        edgecolors="none",
        rasterized=True,
        label=f"Filtered ($T\\geq{min_length}$)",
    )
    ax.axvline(
        min_length,
        color=OKABE_ITO["red"],
        linestyle="--",
        linewidth=1.1,
    )
    ax.axhline(0, color=OKABE_ITO["black"], linewidth=0.7)
    ax.set_xlabel("Raw sequence length, $T$ (frames)")
    ax.set_ylabel("Normalized RMSE")
    ax.set_ylim(bottom=-0.055)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        edgecolor="0.8",
        borderpad=0.35,
        handletextpad=0.45,
    )
    ax.text(
        0.03,
        0.07,
        (
            "Exact bypass match\n"
            f"{summary['bypass_exact_matches']}/"
            f"{summary['bypass_samples']} "
            f"({summary['bypass_exact_match_rate'] * 100:.1f}%)"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.8",
            "linewidth": 0.5,
            "alpha": 0.92,
        },
    )
    panel_label(ax, "b")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.93, wspace=0.34)
    outputs = save_publication_figure(
        fig, output_dir / "elder_bandpass_bypass_overview"
    )
    plt.close(fig)
    return outputs


def plot_elder_before_after(
    representatives: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    publication_style()
    import matplotlib.pyplot as plt

    if not representatives:
        return []
    column_count = len(representatives)
    fig, axes = plt.subplots(
        2,
        column_count,
        figsize=(3.6 * column_count, 4.5),
        squeeze=False,
    )
    for column, item in enumerate(representatives):
        raw_line = item["raw_line"]
        filtered_line = item["filtered_line"]
        difference = filtered_line - raw_line
        frames = np.arange(len(raw_line))
        metadata = item["metadata"]
        is_bypass = parse_bool(metadata["predicted_bypass"])
        case_label = "Bypassed short sample" if is_bypass else "Eligible sample"

        ax = axes[0, column]
        ax.plot(
            frames,
            filtered_line,
            color=OKABE_ITO["red"],
            linewidth=1.45,
            label="Bandpass output",
            zorder=2,
        )
        ax.plot(
            frames,
            raw_line,
            color=OKABE_ITO["blue"],
            linewidth=0.9,
            marker="o",
            markersize=2.1,
            markevery=max(1, len(frames) // 24),
            markerfacecolor="white",
            markeredgewidth=0.45,
            label="Raw amplitude",
            zorder=3,
        )
        ax.set_xlabel("Frame")
        ax.set_ylabel("Signal value")
        ax.set_title(
            f"{case_label}: $T={len(frames)}$",
            pad=4,
        )
        ax.legend(frameon=False, loc="best")
        ax.text(
            0.02,
            0.04,
            (
                f"NRMSE={float(metadata['nrmse']):.3g}\n"
                f"$r$={float(metadata['pearson_r']):.4f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
        )
        panel_label(ax, chr(ord("a") + column))

        ax = axes[1, column]
        ax.plot(
            frames,
            difference,
            color=(OKABE_ITO["orange"] if is_bypass else OKABE_ITO["purple"]),
        )
        ax.axhline(0, color=OKABE_ITO["black"], linewidth=0.7)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Bandpass $-$ raw")
        ax.text(
            0.98,
            0.94,
            f"max$|\\Delta|$={float(metadata['max_absolute_change']):.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        panel_label(ax, chr(ord("a") + column_count + column))

    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.10,
        top=0.95,
        wspace=0.28,
        hspace=0.34,
    )
    outputs = save_publication_figure(fig, output_dir / "elder_bandpass_before_after")
    plt.close(fig)
    return outputs


def run_elder(
    root: Path,
    output_dir: Path,
    runtime: dict[str, Any],
    max_files: int,
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"ElderAL directory not found: {root}")
    min_length, b, a = source_min_length()
    if min_length != 28:
        raise RuntimeError(
            f"Source-derived Bandpass threshold changed: expected 28, got {min_length}"
        )

    print(f"[ElderAL] data: {root}")
    discovered_files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and "truth" not in path.name.lower()
    ]
    if max_files > 0:
        discovered_files = discovered_files[:max_files]
    reader = runtime["readers"].get_reader_class("elderAL")()
    records = []
    format_mismatch = 0
    read_failures = []
    decoded_files = 0
    for file_index, path in enumerate(discovered_files, start=1):
        try:
            if not reader.sniff(str(path)):
                format_mismatch += 1
                continue
            loaded = reader.read_file(str(path))
            loaded_records = loaded if isinstance(loaded, list) else [loaded]
            loaded_records = [record for record in loaded_records if record is not None]
            records.extend(loaded_records)
            decoded_files += 1
        except Exception as exc:
            read_failures.append(
                {
                    "file": str(path.relative_to(root)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if file_index % 50 == 0 or file_index == len(discovered_files):
            print(f"[ElderAL] read files {file_index}/{len(discovered_files)}")
    records = sorted(records, key=lambda record: str(record.file_name))

    valid_records: dict[str, Any] = {}
    rows = []
    skipped = 0
    processing_failures = []
    for index, record in enumerate(records, start=1):
        try:
            raw = stack_record(record)
            if raw is None:
                skipped += 1
                continue
            filtered = runtime["butterworth_bandpass"](
                raw,
                order=BANDPASS_ORDER,
                low_freq=BANDPASS_LOW_HZ,
                high_freq=BANDPASS_HIGH_HZ,
                fs=ELDER_FS_HZ,
            )
            row = elder_row(record, raw, filtered, min_length)
            rows.append(row)
            valid_records[str(record.file_name)] = record
        except Exception as exc:
            processing_failures.append(
                {
                    "file": str(record.file_name),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if index % 50 == 0 or index == len(records):
            print(f"[ElderAL] processed {index}/{len(records)}")

    if not rows:
        raise RuntimeError("No valid ElderAL samples (requires at least 2 frames)")
    summary = summarize_elder(
        rows,
        min_length,
        {"b": len(b), "a": len(a)},
    )
    summary["loaded_records"] = len(records)
    summary["skipped_invalid_records"] = skipped
    summary["data_path"] = str(root)
    summary["discovery"] = {
        "selected_files": len(discovered_files),
        "successfully_decoded_files": decoded_files,
        "format_mismatch_files": format_mismatch,
        "failed_files": len(read_failures),
        "failures": read_failures,
        "failed_records_during_bandpass_or_metrics": len(processing_failures),
        "record_processing_failures": processing_failures,
        "max_files_limit": max_files,
    }
    if (
        summary["bypass_samples"] > 0
        and summary["bypass_exact_matches"] != summary["bypass_samples"]
    ):
        summary["official_complete"] = False
        summary["completion_reasons"] = [
            "At least one predicted T<28 bypass output was not exactly equal "
            "to its input."
        ]
        json_dump(output_dir / "elder_bandpass_summary.json", summary)
        raise RuntimeError(
            "ElderAL source-integrity failure: predicted bypass outputs were "
            "not 100% exact matches."
        )
    write_csv(output_dir / "elder_bandpass_per_sample.csv", rows, ELDER_FIELDS)

    representatives = []
    representative_rows = []
    for bypass in (True, False):
        if any(parse_bool(row["predicted_bypass"]) is bypass for row in rows):
            representative_rows.append(choose_representative_row(rows, bypass=bypass))
    for row in representative_rows:
        record = valid_records[str(row["file_name"])]
        raw = stack_record(record)
        if raw is None:
            raise AssertionError("Representative record became invalid")
        filtered = runtime["butterworth_bandpass"](
            raw,
            order=BANDPASS_ORDER,
            low_freq=BANDPASS_LOW_HZ,
            high_freq=BANDPASS_HIGH_HZ,
            fs=ELDER_FS_HZ,
        )
        channel = highest_variance_channel(np.abs(raw))
        representatives.append(
            {
                "metadata": row,
                "channel": {
                    "subcarrier": channel[0],
                    "antenna": channel[1],
                    "selection_rule": "maximum raw temporal variance",
                },
                "raw_line": np.asarray(raw[:, channel[0], channel[1]]).real,
                "filtered_line": np.asarray(filtered[:, channel[0], channel[1]]).real,
            }
        )

    figure_files = []
    figure_files.extend(plot_elder_overview(rows, summary, output_dir))
    if representatives:
        figure_files.extend(plot_elder_before_after(representatives, output_dir))
    if not any(
        parse_bool(item["metadata"]["predicted_bypass"]) for item in representatives
    ):
        summary["before_after_figure_skipped_reason"] = (
            "The selected scope did not contain a T<28 bypass sample."
        )
    summary["representatives"] = [
        {
            "file_name": item["metadata"]["file_name"],
            "frames": item["metadata"]["frames"],
            "predicted_bypass": item["metadata"]["predicted_bypass"],
            **item["channel"],
        }
        for item in representatives
    ]
    summary["figure_files"] = figure_files
    json_dump(output_dir / "elder_bandpass_summary.json", summary)
    print(
        "[ElderAL] bypass "
        f"{summary['bypass_samples']}/{summary['valid_samples']} "
        f"({summary['bypass_sample_rate']:.2%}); "
        f"exact-match rate={summary['bypass_exact_match_rate']:.2%}"
    )
    return summary


def parse_xrf_file_ids(path: Path) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)_(\d+)_(\d+)", path.stem)
    if match is None:
        return None
    return tuple(int(match.group(index)) for index in range(1, 4))


def discover_xrf_files(
    root: Path,
    max_files: int,
    user_count: int,
) -> tuple[list[Path], dict[str, Any]]:
    candidates: list[tuple[Path, tuple[int, int, int]]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "truth" in path.name.lower():
            continue
        identifiers = parse_xrf_file_ids(path)
        if identifiers is not None:
            candidates.append((path, identifiers))
    available_users = sorted({identifiers[0] for _, identifiers in candidates})
    selected_users = available_users[:user_count]
    if len(selected_users) != user_count:
        raise RuntimeError(
            f"Requested {user_count} XRF55 users, found "
            f"{len(available_users)}: {available_users}"
        )
    selected_user_set = set(selected_users)
    selected_candidates = [
        (path, identifiers)
        for path, identifiers in candidates
        if identifiers[0] in selected_user_set
    ]
    observed_grid = Counter(identifiers for _, identifiers in selected_candidates)
    expected_grid = {
        (user_id, action_id, repetition_id)
        for user_id in selected_users
        for action_id in XRF_ACTION_IDS
        for repetition_id in XRF_REPETITION_IDS
    }
    missing_grid = sorted(expected_grid - set(observed_grid))
    unexpected_grid = sorted(set(observed_grid) - expected_grid)
    duplicate_grid = sorted(
        (key, count) for key, count in observed_grid.items() if count != 1
    )
    files = [path for path, _ in selected_candidates]
    selected_files_before_limit = len(files)
    if max_files > 0:
        files = files[:max_files]
    return files, {
        "candidate_files_all_users": len(candidates),
        "selected_files_before_limit": selected_files_before_limit,
        "selected_user_ids": selected_users,
        "available_user_ids": available_users,
        "requested_user_count": user_count,
        "expected_selected_files": len(expected_grid),
        "complete_user_action_repetition_grid": not (
            missing_grid or unexpected_grid or duplicate_grid
        ),
        "missing_grid_cells_preview": missing_grid[:10],
        "unexpected_grid_cells_preview": unexpected_grid[:10],
        "duplicate_grid_cells_preview": duplicate_grid[:10],
    }


def prepare_xrf_input(
    array: np.ndarray,
    complex_policy: str,
) -> tuple[np.ndarray, str, float]:
    max_imag = float(np.max(np.abs(np.imag(array)))) if np.iscomplexobj(array) else 0.0
    real_scale = float(np.max(np.abs(np.real(array)))) if array.size else 0.0
    negligible = max_imag <= max(1e-10, real_scale * 1e-10)
    if not np.iscomplexobj(array) or negligible:
        real = np.asarray(np.real(array))
        representation = (
            "real_reader_values_nonnegative"
            if np.all(real >= 0)
            else "real_reader_values_signed"
        )
        return real, representation, max_imag
    if complex_policy == "amplitude":
        return (
            np.asarray(np.abs(array)),
            "explicit_abs_of_complex_csi",
            max_imag,
        )
    raise RuntimeError(
        "Genuinely complex XRF55 input encountered. Negative values are not "
        "defined for complex numbers. Re-run with --xrf-complex-policy amplitude "
        "only if the intended experiment is explicitly on abs(CSI). "
        f"max|imag|={max_imag:.6g}"
    )


def coerce_denoiser_output_real(
    output: np.ndarray,
    method: str,
) -> np.ndarray:
    output = np.asarray(output)
    if np.iscomplexobj(output):
        max_imag = float(np.max(np.abs(np.imag(output)))) if output.size else 0.0
        real_scale = float(np.max(np.abs(np.real(output)))) if output.size else 0.0
        if max_imag > max(1e-10, real_scale * 1e-10):
            raise RuntimeError(
                f"{method} produced genuinely complex output; a signed negative "
                f"rate is undefined (max|imag|={max_imag:.6g})."
            )
        output = np.real(output)
    return np.asarray(output)


def run_denoiser(
    method: str,
    raw: np.ndarray,
    runtime: dict[str, Any],
) -> np.ndarray:
    if method == "raw":
        return raw.copy()
    if method == "wavelet":
        result = runtime["wavelet_denoise_csi"](raw)
    elif method == "butterworth_o5_c0.3":
        result = runtime["butterworth_denoise"](raw, order=5, cutoff=0.3)
    elif method == "savgol_w7_p3":
        result = runtime["savgol_denoise"](raw, window_length=7, polyorder=3)
    elif method == "bandpass_fs1000":
        result = runtime["butterworth_bandpass"](
            raw,
            order=BANDPASS_ORDER,
            low_freq=BANDPASS_LOW_HZ,
            high_freq=BANDPASS_HIGH_HZ,
            fs=XRF_LEGACY_FS_HZ,
        )
    elif method == "bandpass_fs200":
        result = runtime["butterworth_bandpass"](
            raw,
            order=BANDPASS_ORDER,
            low_freq=BANDPASS_LOW_HZ,
            high_freq=BANDPASS_HIGH_HZ,
            fs=XRF_TRUE_FS_HZ,
        )
    elif method == "hampel_w5_s3":
        result = runtime["hampel_filter"](raw, window_size=5, n_sigma=3.0)
    else:
        raise ValueError(f"Unknown method: {method}")
    return coerce_denoiser_output_real(result, method)


def meaningful_tolerance(raw: np.ndarray, relative: float) -> float:
    finite = np.abs(raw[np.isfinite(raw)])
    nonzero = finite[finite > 0]
    scale = float(np.median(nonzero)) if nonzero.size else 1.0
    machine_floor = np.finfo(np.float64).eps * max(scale, 1.0) * 100.0
    return max(relative * scale, machine_floor)


def xrf_stats_row(
    *,
    run_signature: str,
    file_name: str,
    record_index: int,
    raw: np.ndarray,
    output: np.ndarray,
    method: str,
    representation: str,
    raw_dtype: str,
    max_imag: float,
    tolerance: float,
) -> dict[str, Any]:
    flat = np.asarray(output).reshape(-1)
    finite_mask = np.isfinite(flat)
    finite = np.asarray(flat[finite_mask], dtype=np.float64)
    if finite.size == 0:
        raise ValueError(f"No finite {method} values: {file_name}")
    strict = finite < 0
    meaningful = finite < -tolerance
    total_energy = float(np.sum(finite**2))
    quantiles = np.percentile(finite, [1, 5, 50, 95, 99])
    left = np.asarray(output[:-1], dtype=np.float64)
    right = np.asarray(output[1:], dtype=np.float64)
    finite_pairs = np.isfinite(left) & np.isfinite(right)
    delta = right - left
    rising = finite_pairs & (delta > tolerance)
    falling = finite_pairs & (delta < -tolerance)
    flat_steps = finite_pairs & ~(rising | falling)
    zero_crossing = (
        finite_pairs
        & (np.signbit(left) != np.signbit(right))
        & (np.abs(left) > tolerance)
        & (np.abs(right) > tolerance)
    )
    negative_negative = finite_pairs & (left < -tolerance) & (right < -tolerance)
    absolute_delta = np.abs(right) - np.abs(left)
    comparable = (
        finite_pairs
        & (np.abs(delta) > tolerance)
        & (np.abs(absolute_delta) > tolerance)
    )
    slope_disagreement = comparable & (np.signbit(delta) != np.signbit(absolute_delta))
    meaningful_original_slope = finite_pairs & (np.abs(delta) > tolerance)
    slope_changed_or_lost = meaningful_original_slope & (
        (np.abs(absolute_delta) <= tolerance)
        | (np.signbit(delta) != np.signbit(absolute_delta))
    )
    pair_count = int(np.count_nonzero(finite_pairs))
    comparable_count = int(np.count_nonzero(comparable))
    meaningful_original_count = int(np.count_nonzero(meaningful_original_slope))
    return {
        "run_signature": run_signature,
        "file_name": file_name,
        "record_index": record_index,
        "frames": int(raw.shape[0]),
        "features": int(raw.shape[1]),
        "antennas": int(raw.shape[2]),
        "input_representation": representation,
        "input_dtype": raw_dtype,
        "max_abs_imaginary": max_imag,
        "raw_is_nonnegative": bool(np.all(raw >= 0)),
        "raw_negative_rate": float(np.mean(raw < 0)),
        "method": method,
        "values": int(flat.size),
        "finite_values": int(finite.size),
        "nonfinite_count": int(flat.size - finite.size),
        "negative_count_strict": int(np.count_nonzero(strict)),
        "negative_rate_strict": float(np.mean(strict)),
        "negative_count_meaningful": int(np.count_nonzero(meaningful)),
        "negative_rate_meaningful": float(np.mean(meaningful)),
        "negative_energy_fraction": float(
            np.sum(finite[meaningful] ** 2) / max(total_energy, 1e-30)
        ),
        "meaningful_negative_tolerance": tolerance,
        "step_pairs": pair_count,
        "rising_step_count": int(np.count_nonzero(rising)),
        "rising_step_rate": float(np.count_nonzero(rising) / max(pair_count, 1)),
        "falling_step_count": int(np.count_nonzero(falling)),
        "falling_step_rate": float(np.count_nonzero(falling) / max(pair_count, 1)),
        "flat_step_count": int(np.count_nonzero(flat_steps)),
        "flat_step_rate": float(np.count_nonzero(flat_steps) / max(pair_count, 1)),
        "zero_crossing_count": int(np.count_nonzero(zero_crossing)),
        "zero_crossing_rate": float(
            np.count_nonzero(zero_crossing) / max(pair_count, 1)
        ),
        "negative_negative_pair_count": int(np.count_nonzero(negative_negative)),
        "negative_negative_pair_rate": float(
            np.count_nonzero(negative_negative) / max(pair_count, 1)
        ),
        "comparable_abs_slope_count": comparable_count,
        "abs_slope_direction_disagreement_count": int(
            np.count_nonzero(slope_disagreement)
        ),
        "abs_slope_direction_disagreement_rate": float(
            np.count_nonzero(slope_disagreement) / max(comparable_count, 1)
        ),
        "meaningful_original_slope_count": meaningful_original_count,
        "abs_slope_direction_changed_or_lost_count": int(
            np.count_nonzero(slope_changed_or_lost)
        ),
        "abs_slope_direction_changed_or_lost_rate": float(
            np.count_nonzero(slope_changed_or_lost) / max(meaningful_original_count, 1)
        ),
        "minimum": float(np.min(finite)),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def summarize_xrf_rows(
    rows: list[dict[str, Any]],
    discovery: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    method_rows = []
    for method in METHOD_ORDER:
        values = grouped.get(method, [])
        if not values:
            continue
        sample_meaningful = np.asarray(
            [float(row["negative_rate_meaningful"]) for row in values]
        )
        sample_strict = np.asarray(
            [float(row["negative_rate_strict"]) for row in values]
        )
        total_finite = sum(int(row["finite_values"]) for row in values)
        total_strict = sum(int(row["negative_count_strict"]) for row in values)
        total_meaningful = sum(int(row["negative_count_meaningful"]) for row in values)
        total_steps = sum(int(row["step_pairs"]) for row in values)
        total_rising = sum(int(row["rising_step_count"]) for row in values)
        total_falling = sum(int(row["falling_step_count"]) for row in values)
        total_flat = sum(int(row["flat_step_count"]) for row in values)
        total_crossings = sum(int(row["zero_crossing_count"]) for row in values)
        total_negative_pairs = sum(
            int(row["negative_negative_pair_count"]) for row in values
        )
        total_comparable = sum(int(row["comparable_abs_slope_count"]) for row in values)
        total_slope_disagreement = sum(
            int(row["abs_slope_direction_disagreement_count"]) for row in values
        )
        total_meaningful_original_slopes = sum(
            int(row["meaningful_original_slope_count"]) for row in values
        )
        total_slope_changed_or_lost = sum(
            int(row["abs_slope_direction_changed_or_lost_count"]) for row in values
        )
        q1, median, q3 = np.percentile(sample_meaningful, [25, 50, 75])
        ci_low, ci_high = bootstrap_mean_ci(
            sample_meaningful,
            seed=20260725 + METHOD_ORDER.index(method),
        )
        method_rows.append(
            {
                "method": method,
                "samples": len(values),
                "per_sample_strict_mean": float(np.mean(sample_strict)),
                "per_sample_strict_median": float(np.median(sample_strict)),
                "per_sample_meaningful_mean": float(np.mean(sample_meaningful)),
                "per_sample_meaningful_mean_bootstrap_ci_low": ci_low,
                "per_sample_meaningful_mean_bootstrap_ci_high": ci_high,
                "per_sample_meaningful_q1": float(q1),
                "per_sample_meaningful_median": float(median),
                "per_sample_meaningful_q3": float(q3),
                "per_sample_meaningful_min": float(np.min(sample_meaningful)),
                "per_sample_meaningful_max": float(np.max(sample_meaningful)),
                "element_weighted_strict_rate": total_strict / max(total_finite, 1),
                "element_weighted_meaningful_rate": total_meaningful
                / max(total_finite, 1),
                "mean_negative_energy_fraction": float(
                    np.mean([float(row["negative_energy_fraction"]) for row in values])
                ),
                "element_weighted_rising_step_rate": total_rising / max(total_steps, 1),
                "element_weighted_falling_step_rate": total_falling
                / max(total_steps, 1),
                "element_weighted_flat_step_rate": total_flat / max(total_steps, 1),
                "element_weighted_zero_crossing_rate": total_crossings
                / max(total_steps, 1),
                "element_weighted_negative_negative_pair_rate": (
                    total_negative_pairs / max(total_steps, 1)
                ),
                "element_weighted_abs_slope_direction_disagreement_rate": (
                    total_slope_disagreement / max(total_comparable, 1)
                ),
                "comparable_abs_slope_pairs": total_comparable,
                "element_weighted_abs_slope_direction_changed_or_lost_rate": (
                    total_slope_changed_or_lost
                    / max(total_meaningful_original_slopes, 1)
                ),
                "meaningful_original_slope_pairs": (total_meaningful_original_slopes),
            }
        )
    return {
        "dataset": "xrf55",
        "scope": (
            "only records with all configured denoisers completed; "
            "statistics are immediately after denoising, before IQR, "
            "interpolation, padding, split, or normalization"
        ),
        "discovery": discovery,
        "methods": method_rows,
    }


def plot_xrf_negative_distribution(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    publication_style()
    import matplotlib.pyplot as plt

    grouped: dict[str, np.ndarray] = {}
    for method in METHOD_ORDER:
        grouped[method] = np.asarray(
            [
                float(row["negative_rate_meaningful"]) * 100
                for row in rows
                if str(row["method"]) == method
            ]
        )
    methods = [method for method in METHOD_ORDER if grouped[method].size]
    data = [grouped[method] for method in methods]
    labels = [METHOD_LABELS[method] for method in methods]
    method_summary = {row["method"]: row for row in summary["methods"]}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.15),
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )
    ax = axes[0]
    positions = np.arange(1, len(methods) + 1)
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": OKABE_ITO["black"], "linewidth": 1.1},
        whiskerprops={"color": OKABE_ITO["gray"], "linewidth": 0.8},
        capprops={"color": OKABE_ITO["gray"], "linewidth": 0.8},
        boxprops={"color": OKABE_ITO["gray"], "linewidth": 0.8},
    )
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(
            OKABE_ITO["red"] if method.startswith("bandpass") else OKABE_ITO["blue"]
        )
        patch.set_alpha(0.5)

    rng = np.random.default_rng(20260725)
    for position, method, values in zip(positions, methods, data):
        # Plot at most 5000 deterministic points per method to keep PDF usable;
        # box statistics always use the complete per-sample population.
        if len(values) > 5000:
            indices = rng.choice(len(values), size=5000, replace=False)
            plot_values = values[indices]
        else:
            plot_values = values
        jitter = rng.uniform(-0.17, 0.17, size=len(plot_values))
        ax.scatter(
            position + jitter,
            plot_values,
            s=4,
            color=OKABE_ITO["black"],
            alpha=min(0.28, max(0.035, 120 / max(len(plot_values), 1))),
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        method_row = method_summary[method]
        mean = 100 * float(method_row["per_sample_meaningful_mean"])
        low = 100 * float(method_row["per_sample_meaningful_mean_bootstrap_ci_low"])
        high = 100 * float(method_row["per_sample_meaningful_mean_bootstrap_ci_high"])
        ax.errorbar(
            position,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            markersize=3.5,
            capsize=2,
            color=OKABE_ITO["green"],
            zorder=4,
        )
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.set_ylabel("Meaningful negative values per sample (%)")
    ax.set_xlabel("Signal immediately after denoising")
    ax.set_ylim(bottom=min(-1.5, ax.get_ylim()[0]))
    ax.text(
        0.01,
        0.99,
        "Box/CI use all samples; dots show at most 5000 per method",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
    )
    panel_label(ax, "a")

    ax = axes[1]
    strict = [
        method_summary[method]["element_weighted_strict_rate"] * 100
        for method in methods
    ]
    meaningful = [
        method_summary[method]["element_weighted_meaningful_rate"] * 100
        for method in methods
    ]
    y = np.arange(len(methods))
    height = 0.34
    ax.barh(
        y + height / 2,
        strict,
        height=height,
        color=OKABE_ITO["gray"],
        label="Strict $x<0$",
    )
    ax.barh(
        y - height / 2,
        meaningful,
        height=height,
        color=[
            OKABE_ITO["red"] if method.startswith("bandpass") else OKABE_ITO["blue"]
            for method in methods
        ],
        label="Scale-aware $x<-\\epsilon$",
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Element-weighted negative rate (%)")
    ax.legend(frameon=False, loc="lower right")
    for index, value in enumerate(meaningful):
        ax.text(
            value + max(0.25, ax.get_xlim()[1] * 0.01),
            index - height / 2,
            f"{value:.2f}%",
            va="center",
            fontsize=6.7,
        )
    panel_label(ax, "b")
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.26,
        top=0.96,
        wspace=0.34,
    )
    outputs = save_publication_figure(
        fig, output_dir / "xrf55_denoiser_negative_distribution"
    )
    plt.close(fig)
    return outputs


def plot_xrf_direction_effects(
    summary: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    """Show rise/fall balance and how abs changes temporal direction."""
    publication_style()
    import matplotlib.pyplot as plt

    rows = summary["methods"]
    if not rows:
        return []
    methods = [str(row["method"]) for row in rows]
    labels = [METHOD_LABELS[method] for method in methods]
    rising = 100 * np.asarray(
        [float(row["element_weighted_rising_step_rate"]) for row in rows]
    )
    falling = 100 * np.asarray(
        [float(row["element_weighted_falling_step_rate"]) for row in rows]
    )
    flat = 100 * np.asarray(
        [float(row["element_weighted_flat_step_rate"]) for row in rows]
    )
    disagreement = 100 * np.asarray(
        [
            float(row["element_weighted_abs_slope_direction_changed_or_lost_rate"])
            for row in rows
        ]
    )
    crossings = 100 * np.asarray(
        [float(row["element_weighted_zero_crossing_rate"]) for row in rows]
    )
    x = np.arange(len(methods))
    colors = [
        OKABE_ITO["red"] if method.startswith("bandpass") else OKABE_ITO["blue"]
        for method in methods
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    ax = axes[0]
    ax.bar(x, rising, color=OKABE_ITO["green"], label="Rise: $\\Delta x>\\epsilon$")
    ax.bar(
        x,
        falling,
        bottom=rising,
        color=OKABE_ITO["purple"],
        label="Fall: $\\Delta x<-\\epsilon$",
    )
    ax.bar(
        x,
        flat,
        bottom=rising + falling,
        color=OKABE_ITO["gray"],
        label="Near-flat",
    )
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Adjacent time-step pairs (%)")
    ax.set_title("Rise/fall means the sign of $x_t-x_{t-1}$", loc="left")
    ax.legend(frameon=False, fontsize=6.5)
    panel_label(ax, "a")

    ax = axes[1]
    width = 0.38
    ax.bar(
        x - width / 2,
        disagreement,
        width=width,
        color=colors,
        label="Slope reversed or erased by $|x|$",
    )
    ax.bar(
        x + width / 2,
        crossings,
        width=width,
        color=OKABE_ITO["orange"],
        label="Zero-crossing pairs",
    )
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Metric-specific adjacent-pair rate (%)")
    ax.set_title("Potential downstream sign-folding effect", loc="left")
    ax.legend(frameon=False, fontsize=6.5)
    panel_label(ax, "b")
    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.27,
        top=0.94,
        wspace=0.31,
    )
    outputs = save_publication_figure(
        fig,
        output_dir / "xrf55_rise_fall_and_abs_effect",
    )
    plt.close(fig)
    return outputs


def analyze_xrf_file_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Read and fully analyse one XRF55 file in a CPU-only spawn worker."""
    path = Path(task["path"])
    relative_name = str(task["relative_name"])
    try:
        runtime = import_wsdp_runtime()
        reader = runtime["readers"].get_reader_class("xrf55")()
        if not reader.sniff(str(path)):
            return {
                "status": "format_mismatch",
                "relative_name": relative_name,
            }
        loaded = reader.read_file(str(path))
        records = loaded if isinstance(loaded, list) else [loaded]
        records = [record for record in records if record is not None]
        rows: list[dict[str, Any]] = []
        valid_records = 0
        skipped_short = 0
        record_keys: list[tuple[str, int]] = []
        representation_counts: Counter[str] = Counter()
        raw_nonnegative = 0
        raw_signed = 0
        completed_by_record = {
            int(key): set(value) for key, value in task["completed_by_record"].items()
        }
        for record_index, record in enumerate(records):
            raw_original = stack_record(record)
            if raw_original is None:
                skipped_short += 1
                continue
            raw, representation, max_imag = prepare_xrf_input(
                raw_original,
                task["complex_policy"],
            )
            representation_counts[representation] += 1
            if np.all(raw >= 0):
                raw_nonnegative += 1
            else:
                raw_signed += 1
            valid_records += 1
            record_keys.append((relative_name, record_index))
            tolerance = meaningful_tolerance(
                raw,
                float(task["relative_tolerance"]),
            )
            already_done = completed_by_record.get(record_index, set())
            record_rows = []
            for method in METHOD_ORDER:
                if method in already_done:
                    continue
                output = run_denoiser(method, raw, runtime)
                record_rows.append(
                    xrf_stats_row(
                        run_signature=str(task["run_signature"]),
                        file_name=relative_name,
                        record_index=record_index,
                        raw=raw,
                        output=output,
                        method=method,
                        representation=representation,
                        raw_dtype=str(raw_original.dtype),
                        max_imag=max_imag,
                        tolerance=tolerance,
                    )
                )
            # Keep the file result in memory until all methods/records succeed.
            rows.extend(record_rows)
        return {
            "status": "ok",
            "relative_name": relative_name,
            "rows": rows,
            "valid_records": valid_records,
            "skipped_short_records": skipped_short,
            "record_keys": record_keys,
            "representation_counts": dict(representation_counts),
            "raw_nonnegative_records": raw_nonnegative,
            "raw_signed_records": raw_signed,
        }
    except Exception as exc:
        return {
            "status": "error",
            "relative_name": relative_name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "fatal_complex": "Genuinely complex XRF55 input" in str(exc),
        }


def run_xrf(
    root: Path,
    output_dir: Path,
    runtime: dict[str, Any],
    *,
    max_files: int,
    user_count: int,
    relative_tolerance: float,
    complex_policy: str,
    resume: bool,
    progress_every: int,
    workers: int,
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"XRF55 directory not found: {root}")
    files, selection = discover_xrf_files(root, max_files, user_count)
    selected_users = list(selection["selected_user_ids"])
    available_users = list(selection["available_user_ids"])
    selected_files_before_limit = int(selection["selected_files_before_limit"])
    if not files:
        raise RuntimeError(f"No XRF55 files discovered under {root}")
    run_signature = stable_hash(
        {
            "data_path": str(root),
            "selected_user_ids": selected_users,
            "available_user_ids": available_users,
            "requested_user_count": user_count,
            "files": [
                {
                    "relative_path": str(path.relative_to(root)),
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in files
            ],
            "methods": METHOD_ORDER,
            "relative_tolerance": relative_tolerance,
            "complex_policy": complex_policy,
            "cache_schema_version": XRF_CACHE_SCHEMA_VERSION,
            "source_files": {
                relative: hashlib.sha256((WSDP_SRC / relative).read_bytes()).hexdigest()
                for relative in (
                    "wsdp/algorithms/denoising.py",
                    "wsdp/algorithms/denoising_butterworth.py",
                    "wsdp/algorithms/amplitude.py",
                    "wsdp/readers/xrf_reader.py",
                    "wsdp/readers/base.py",
                    "wsdp/structure/CSIData.py",
                    "wsdp/structure/CSIFrame.py",
                )
            },
            "analysis_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
    )

    csv_path = output_dir / "xrf55_negative_per_sample.csv"
    if csv_path.exists() and not resume:
        csv_path.unlink()
    if resume and csv_path.exists() and "run_signature" not in csv_header(csv_path):
        raise RuntimeError(
            "Existing XRF CSV predates configuration signatures and cannot be "
            "safely resumed. Re-run without --resume to start a clean result."
        )
    current_existing: dict[tuple[str, int, str], dict[str, str]] = {}
    rejected_resume_rows = 0
    if resume:
        for row in iter_csv(csv_path):
            if row.get("run_signature") != run_signature:
                continue
            if not valid_xrf_csv_row(row):
                rejected_resume_rows += 1
                continue
            key = (
                str(row["file_name"]),
                int(row["record_index"]),
                str(row["method"]),
            )
            current_existing[key] = row
    completed = set(current_existing)

    discovery = {
        "data_path": str(root),
        "discovered_files": len(files),
        **selection,
        "supported_files": 0,
        "successfully_decoded_files": 0,
        "format_mismatch_files": 0,
        "failed_files": 0,
        "valid_records": 0,
        "skipped_short_records": 0,
        "complex_policy": complex_policy,
        "relative_tolerance": relative_tolerance,
        "run_signature": run_signature,
        "raw_nonnegative_records": 0,
        "raw_signed_records": 0,
        "input_representation_counts": {},
        "rejected_corrupt_resume_rows": rejected_resume_rows,
        "failures": [],
    }
    representation_counts: Counter[str] = Counter()
    seen_record_keys: set[tuple[str, int]] = set()
    print(f"[XRF55] data: {root}")
    print(
        f"[XRF55] selected users: {selected_users}; "
        f"files: {len(files)}/{selected_files_before_limit} "
        "(selected scope)"
    )
    completed_by_file: dict[str, dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for file_name, record_index, method in completed:
        completed_by_file[file_name][record_index].append(method)
    tasks = [
        {
            "path": str(path),
            "relative_name": str(path.relative_to(root)),
            "relative_tolerance": relative_tolerance,
            "complex_policy": complex_policy,
            "run_signature": run_signature,
            "completed_by_record": dict(
                completed_by_file.get(str(path.relative_to(root)), {})
            ),
        }
        for path in files
    ]
    worker = analyze_xrf_file_worker
    if workers == 1:
        results = map(worker, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        )
        results = executor.map(worker, tasks, chunksize=1)
    fatal_error: RuntimeError | None = None
    try:
        for file_index, result in enumerate(results, start=1):
            status = result["status"]
            relative_name = str(result["relative_name"])
            if status == "format_mismatch":
                discovery["format_mismatch_files"] += 1
            elif status == "error":
                discovery["supported_files"] += 1
                discovery["failed_files"] += 1
                failure = {
                    "file": relative_name,
                    "error": result["error"],
                    "traceback": result["traceback"],
                }
                discovery["failures"].append(failure)
                print(f"[XRF55] failed {relative_name}: {failure['error']}")
                if result.get("fatal_complex"):
                    fatal_error = RuntimeError(result["error"])
                    break
            else:
                discovery["supported_files"] += 1
                discovery["successfully_decoded_files"] += 1
                discovery["valid_records"] += int(result["valid_records"])
                discovery["skipped_short_records"] += int(
                    result["skipped_short_records"]
                )
                discovery["raw_nonnegative_records"] += int(
                    result["raw_nonnegative_records"]
                )
                discovery["raw_signed_records"] += int(result["raw_signed_records"])
                representation_counts.update(result["representation_counts"])
                seen_record_keys.update(
                    (str(file_name), int(record_index))
                    for file_name, record_index in result["record_keys"]
                )
                pending_rows = result["rows"]
                append_csv_rows(csv_path, pending_rows, XRF_FIELDS)
                for row in pending_rows:
                    completed.add(
                        (
                            str(row["file_name"]),
                            int(row["record_index"]),
                            str(row["method"]),
                        )
                    )
            if file_index % max(progress_every, 1) == 0 or file_index == len(files):
                print(
                    f"[XRF55] files {file_index}/{len(files)} | "
                    f"valid records={discovery['valid_records']} | "
                    f"failed={discovery['failed_files']}"
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if fatal_error is not None:
        discovery["input_representation_counts"] = dict(representation_counts)
        json_dump(output_dir / "xrf55_discovery_partial.json", discovery)
        raise fatal_error

    deduplicated_rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    rejected_final_rows = 0
    for row in iter_csv(csv_path):
        if row.get("run_signature") != run_signature:
            continue
        if not valid_xrf_csv_row(row):
            rejected_final_rows += 1
            continue
        key = (
            str(row["file_name"]),
            int(row["record_index"]),
            str(row["method"]),
        )
        deduplicated_rows[key] = dict(row)
    current_rows = list(deduplicated_rows.values())
    methods_by_record: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in current_rows:
        methods_by_record[(str(row["file_name"]), int(row["record_index"]))].add(
            str(row["method"])
        )
    complete_record_keys = {
        key
        for key, methods in methods_by_record.items()
        if methods == set(METHOD_ORDER)
    }
    rows = [
        row
        for row in current_rows
        if (str(row["file_name"]), int(row["record_index"])) in complete_record_keys
    ]
    if not rows:
        raise RuntimeError("No XRF55 statistics were generated")
    discovery["fully_analyzed_records"] = len(complete_record_keys)
    discovery["rejected_corrupt_rows_at_summary"] = rejected_final_rows
    discovery["partially_analyzed_records"] = len(
        set(methods_by_record) - complete_record_keys
    )
    discovery["seen_valid_records_this_invocation"] = len(seen_record_keys)
    complete_files = {file_name for file_name, _ in complete_record_keys}
    discovery["files_with_at_least_one_fully_analyzed_record"] = len(complete_files)
    discovery["input_representation_counts"] = dict(representation_counts)
    summary = summarize_xrf_rows(rows, discovery)
    summary["figure_files"] = plot_xrf_negative_distribution(rows, summary, output_dir)
    summary["figure_files"].extend(plot_xrf_direction_effects(summary, output_dir))
    if summary["methods"]:
        method_fields = list(summary["methods"][0])
        write_csv(
            output_dir / "xrf55_negative_method_summary.csv",
            summary["methods"],
            method_fields,
        )
    json_dump(output_dir / "xrf55_negative_summary.json", summary)
    method_summary = {row["method"]: row for row in summary["methods"]}
    print("[XRF55] element-weighted meaningful negative rates:")
    for method in METHOD_ORDER:
        if method in method_summary:
            print(
                f"  {method}: "
                f"{method_summary[method]['element_weighted_meaningful_rate']:.4%}"
            )
    return summary


def synthetic_self_test() -> None:
    """Test the exact Bandpass source file without importing top-level wsdp."""
    assert DEFAULT_OUTPUT_ROOT == (
        REPO_ROOT / "ablation" / "bandpass_server_results" / "signal_analysis"
    )
    assert parse_xrf_file_ids(Path("03_55_20.npy")) == (3, 55, 20)
    module_path = WSDP_SRC / "wsdp" / "algorithms" / "denoising_butterworth.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location(
        "_bandpass_server_selftest", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    min_length, b, a = source_min_length()
    assert min_length == 28
    assert len(a) == len(b) == 9
    t27 = np.arange(27, dtype=np.float64)
    short = (10 + np.sin(2 * np.pi * t27 / 8))[:, None, None]
    short_output = module.butterworth_bandpass(
        short,
        order=BANDPASS_ORDER,
        low_freq=BANDPASS_LOW_HZ,
        high_freq=BANDPASS_HIGH_HZ,
        fs=ELDER_FS_HZ,
    )
    assert np.array_equal(short, short_output)

    t28 = np.arange(28, dtype=np.float64)
    eligible = (10 + np.sin(2 * np.pi * t28 / 8))[:, None, None]
    eligible_output = module.butterworth_bandpass(
        eligible,
        order=BANDPASS_ORDER,
        low_freq=BANDPASS_LOW_HZ,
        high_freq=BANDPASS_HIGH_HZ,
        fs=ELDER_FS_HZ,
    )
    assert not np.array_equal(eligible, eligible_output)

    t = np.arange(1000, dtype=np.float64)
    positive = (
        10
        + 1.5 * np.sin(2 * np.pi * 2.0 * t / XRF_TRUE_FS_HZ)
        + 0.3 * np.sin(2 * np.pi * 20.0 * t / XRF_TRUE_FS_HZ)
    )[:, None, None]
    signed = module.butterworth_bandpass(
        positive,
        order=BANDPASS_ORDER,
        low_freq=BANDPASS_LOW_HZ,
        high_freq=BANDPASS_HIGH_HZ,
        fs=XRF_TRUE_FS_HZ,
    )
    assert np.any(signed < 0) and np.any(signed > 0)
    test_row = xrf_stats_row(
        run_signature="synthetic",
        file_name="synthetic",
        record_index=0,
        raw=positive,
        output=signed,
        method="bandpass_fs200",
        representation="synthetic_positive_amplitude",
        raw_dtype=str(positive.dtype),
        max_imag=0.0,
        tolerance=meaningful_tolerance(positive, 1e-6),
    )
    assert test_row["negative_rate_meaningful"] > 0
    assert (
        test_row["rising_step_count"]
        + test_row["falling_step_count"]
        + test_row["flat_step_count"]
        == test_row["step_pairs"]
    )
    assert test_row["abs_slope_direction_disagreement_rate"] > 0
    complete_config = {
        "only": "all",
        "max_elder_files": 0,
        "max_xrf_files": 0,
        "xrf_user_count": DEFAULT_XRF_USER_COUNT,
    }
    complete_elder = {
        "valid_samples": 10,
        "loaded_records": 10,
        "skipped_invalid_records": 0,
        "bypass_samples": 8,
        "bypass_exact_matches": 8,
        "discovery": {
            "failed_files": 0,
            "failed_records_during_bandpass_or_metrics": 0,
        },
    }
    complete_xrf = {
        "discovery": {
            "failed_files": 0,
            "skipped_short_records": 0,
            "valid_records": 10,
            "fully_analyzed_records": 10,
            "partially_analyzed_records": 0,
            "successfully_decoded_files": 10,
            "files_with_at_least_one_fully_analyzed_record": 10,
            "selected_user_ids": [1, 2, 3],
            "available_user_ids": [1, 2, 3, 4],
            "requested_user_count": DEFAULT_XRF_USER_COUNT,
            "selected_files_before_limit": 3300,
            "expected_selected_files": 3300,
            "complete_user_action_repetition_grid": True,
        },
        "methods": [{"method": method, "samples": 10} for method in METHOD_ORDER],
    }
    assert not evaluate_signal_completion(
        complete_config,
        complete_elder,
        complete_xrf,
    )
    partial_reasons = evaluate_signal_completion(
        {**complete_config, "only": "elder"},
        complete_elder,
        None,
    )
    assert partial_reasons
    assert any("--only=all" in reason for reason in partial_reasons)
    failed_reasons = evaluate_signal_completion(
        complete_config,
        complete_elder,
        {
            **complete_xrf,
            "discovery": {
                **complete_xrf["discovery"],
                "failed_files": 1,
            },
        },
    )
    assert any("failed" in reason.lower() for reason in failed_reasons)
    print(
        "Synthetic self-test passed: threshold=28, T=27 bypasses, "
        "T=28 filters, positive input produces signed Bandpass output, "
        "rise/fall/abs-direction metrics are internally consistent, and "
        "official-completion gates reject partial/failed runs."
    )


def evaluate_signal_completion(
    config: dict[str, Any],
    elder: dict[str, Any] | None,
    xrf: dict[str, Any] | None,
) -> list[str]:
    """Return reasons that prevent this run from being an official full study."""
    reasons: list[str] = []
    if config.get("only") != "all":
        reasons.append("Official completion requires --only=all.")
    if int(config.get("max_elder_files", 0)) != 0:
        reasons.append("Official completion requires --max-elder-files=0.")
    if int(config.get("max_xrf_files", 0)) != 0:
        reasons.append("Official completion requires --max-xrf-files=0.")
    if int(config.get("xrf_user_count", 0)) != DEFAULT_XRF_USER_COUNT:
        reasons.append(
            "Official completion requires the first 3 XRF55 users (--xrf-user-count=3)."
        )

    if elder is None:
        reasons.append("ElderAL results are missing.")
    else:
        discovery = elder.get("discovery", {})
        if int(discovery.get("failed_files", 0)) != 0:
            reasons.append("One or more supported ElderAL files failed to load.")
        if int(discovery.get("failed_records_during_bandpass_or_metrics", 0)) != 0:
            reasons.append("One or more ElderAL records failed processing.")
        if int(elder.get("skipped_invalid_records", 0)) != 0:
            reasons.append("One or more decoded ElderAL records were invalid.")
        if int(elder.get("valid_samples", 0)) != int(elder.get("loaded_records", 0)):
            reasons.append("ElderAL valid-record coverage is below 100%.")
        bypass_samples = int(elder.get("bypass_samples", 0))
        if bypass_samples == 0:
            reasons.append(
                "No T<28 ElderAL bypass sample was available for the required "
                "before/after comparison."
            )
        elif int(elder.get("bypass_exact_matches", 0)) != bypass_samples:
            reasons.append("ElderAL bypass exact-match coverage is below 100%.")

    if xrf is None:
        reasons.append("XRF55 results are missing.")
    else:
        discovery = xrf.get("discovery", {})
        selected_users = [
            int(value) for value in discovery.get("selected_user_ids", [])
        ]
        available_users = [
            int(value) for value in discovery.get("available_user_ids", [])
        ]
        if selected_users != available_users[:DEFAULT_XRF_USER_COUNT]:
            reasons.append("XRF55 selection is not the first 3 numeric user IDs.")
        if int(discovery.get("requested_user_count", 0)) != (DEFAULT_XRF_USER_COUNT):
            reasons.append("XRF55 discovery does not record user_count=3.")
        if not bool(discovery.get("complete_user_action_repetition_grid", False)):
            reasons.append(
                "XRF55 does not contain exactly one file for every selected "
                "user × action(1..55) × repetition(1..20)."
            )
        if int(discovery.get("selected_files_before_limit", 0)) != int(
            discovery.get("expected_selected_files", -1)
        ):
            reasons.append("XRF55 selected-file count is not the official 3300.")
        if int(discovery.get("failed_files", 0)) != 0:
            reasons.append("One or more supported XRF55 files failed.")
        if int(discovery.get("skipped_short_records", 0)) != 0:
            reasons.append("One or more decoded XRF55 records were too short.")
        valid_records = int(discovery.get("valid_records", 0))
        complete_records = int(discovery.get("fully_analyzed_records", 0))
        if complete_records != valid_records:
            reasons.append("XRF55 seven-method record coverage is below 100%.")
        if int(discovery.get("partially_analyzed_records", 0)) != 0:
            reasons.append("One or more XRF55 records have partial method output.")
        if int(
            discovery.get(
                "files_with_at_least_one_fully_analyzed_record",
                0,
            )
        ) != int(discovery.get("successfully_decoded_files", 0)):
            reasons.append("XRF55 decoded-file coverage is below 100%.")
        methods = {
            str(row.get("method")): int(row.get("samples", 0))
            for row in xrf.get("methods", [])
        }
        if set(methods) != set(METHOD_ORDER):
            reasons.append("XRF55 output does not contain all seven methods.")
        elif any(count != complete_records for count in methods.values()):
            reasons.append("XRF55 method sample counts are not fully matched.")
    return reasons


def signal_coverage(
    elder: dict[str, Any] | None,
    xrf: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build compact coverage metadata for reports and machine checks."""
    elder_discovery = elder.get("discovery", {}) if elder else {}
    xrf_discovery = xrf.get("discovery", {}) if xrf else {}
    return {
        "elderAL": {
            "loaded_records": int(elder.get("loaded_records", 0)) if elder else 0,
            "valid_records": int(elder.get("valid_samples", 0)) if elder else 0,
            "failed_files": int(elder_discovery.get("failed_files", 0)),
            "failed_records": int(
                elder_discovery.get(
                    "failed_records_during_bandpass_or_metrics",
                    0,
                )
            ),
        },
        "xrf55": {
            "discovered_files": int(xrf_discovery.get("discovered_files", 0)),
            "selected_user_ids": list(xrf_discovery.get("selected_user_ids", [])),
            "available_user_ids": list(xrf_discovery.get("available_user_ids", [])),
            "selected_files_before_limit": int(
                xrf_discovery.get("selected_files_before_limit", 0)
            ),
            "expected_selected_files": int(
                xrf_discovery.get("expected_selected_files", 0)
            ),
            "complete_user_action_repetition_grid": bool(
                xrf_discovery.get(
                    "complete_user_action_repetition_grid",
                    False,
                )
            ),
            "successfully_decoded_files": int(
                xrf_discovery.get("successfully_decoded_files", 0)
            ),
            "failed_files": int(xrf_discovery.get("failed_files", 0)),
            "valid_records": int(xrf_discovery.get("valid_records", 0)),
            "fully_analyzed_records": int(
                xrf_discovery.get("fully_analyzed_records", 0)
            ),
            "partially_analyzed_records": int(
                xrf_discovery.get("partially_analyzed_records", 0)
            ),
            "expected_methods": len(METHOD_ORDER),
        },
    }


def write_signal_plain_language_report(
    overall: dict[str, Any],
    output_dir: Path,
    report_name: str = "signal_report_summary.md",
) -> None:
    official_complete = bool(overall.get("official_complete", False))
    reasons = list(overall.get("completion_reasons", []))
    lines = [
        (
            "# Bandpass 信号层实验：正式汇报摘要"
            if official_complete
            else "# [PRELIMINARY / INCOMPLETE] Bandpass 信号层实验摘要"
        ),
        "",
        (
            "> 状态：OFFICIAL COMPLETE。数据范围与方法覆盖通过正式完成门。"
            if official_complete
            else "> 状态：PRELIMINARY / INCOMPLETE。以下数值只能描述本次"
            "已成功处理的范围，不能作为完整数据集结论。"
        ),
        "",
    ]
    if reasons:
        lines.extend(
            ["## 未完成原因", ""] + [f"- {reason}" for reason in reasons] + [""]
        )
    lines.extend(
        [
            "## 概念",
            "",
            "- **基线**是静态环境、设备增益和很慢漂移形成的信号底座。"
            "Bandpass 不只去基线，还同时去掉 50 Hz 以上部分，因此"
            "`raw−Bandpass` 不能全部叫基线。",
            "- **正/负**表示 Bandpass 零中心的两侧；它不等于上升/下降。",
            "- **上升/下降**看相邻帧差值：`x[t]−x[t−1]>0` 才是上升，小于 0 才是下降。",
            "",
        ]
    )
    elder = overall.get("elderAL")
    if elder:
        bypass_rate = float(elder["bypass_sample_rate"])
        lines.extend(
            [
                "## ElderAL",
                "",
                f"- 源码条件是严格的 `T < {elder['source_min_length']}` "
                "直接返回原始数据。",
                f"- 本次服务器数据中，{elder['bypass_samples']}/"
                f"{elder['valid_samples']} 个有效样本被旁路，"
                f"占 {100 * elder['bypass_sample_rate']:.2f}%。",
                f"- 被旁路样本与输入逐值完全相同的比例为 "
                f"{100 * elder['bypass_exact_match_rate']:.2f}%。",
            ]
        )
        if official_complete and bypass_rate >= 0.5:
            lines.append(
                "- 多数 ElderAL 样本没有真正执行 Bandpass；这强烈支持"
                "“旁路使该方法没有伤害多数样本”是其表现尚可的主要机制。"
            )
        elif official_complete and bypass_rate > 0:
            lines.append(
                "- 确有样本被旁路，但比例未过半；它只能解释部分现象，"
                "不能单独称为主要原因。"
            )
        elif official_complete:
            lines.append(
                "- 本次完整数据中没有样本触发旁路，因此不支持预期的ElderAL旁路解释。"
            )
        else:
            lines.append(
                "- 当前运行不完整，只能确认已处理范围内的旁路现象；"
                "不据此判断它是否解释完整 ElderAL 的分类表现。"
            )
        lines.append("")
    xrf = overall.get("xrf55")
    if xrf:
        discovery = xrf["discovery"]
        lines.extend(
            [
                "## XRF55",
                "",
                f"- 为与分类实验一致，只统计数值排序后的前三个用户："
                f"{discovery.get('selected_user_ids', [])}。",
                f"- 完整分析记录数：{discovery.get('fully_analyzed_records', 0)}；"
                f"其中原始值全非负的记录数："
                f"{discovery.get('raw_nonnegative_records', 0)}。",
                "- 下表统计位置严格在“去噪后、IQR/插值/padding/归一化前”。",
                "",
                "| 方法 | 有意义负值率 | 负值能量占比 | abs后斜率反向或被抹平率 |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in xrf["methods"]:
            lines.append(
                f"| {row['method']} | "
                f"{100 * row['element_weighted_meaningful_rate']:.3f}% | "
                f"{100 * row['mean_negative_energy_fraction']:.3f}% | "
                f"{100 * row['element_weighted_abs_slope_direction_changed_or_lost_rate']:.3f}% |"
            )
        lines.extend(
            [
                "",
                "如果 Bandpass 的负值率、负值能量和 `abs` 后方向改变率明显高于"
                "其他去噪方法，说明它更容易受后续 IQR/`abs` 符号折叠影响。"
                "但这仍是信号机制证据；是否真正导致识别率下降，要以独立的"
                "2×2 分类消融结果为准。",
            ]
        )
    (output_dir / report_name).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    elder_root = resolve_path(args.elder_root)
    xrf_root = resolve_path(args.xrf_root)
    output_dir = resolve_path(args.output_dir)
    min_length, b, a = source_min_length()

    if args.meaningful_negative_relative_tol <= 0:
        raise ValueError("--meaningful-negative-relative-tol must be > 0")
    if args.max_elder_files < 0 or args.max_xrf_files < 0:
        raise ValueError("file limits must be >= 0")
    if args.xrf_user_count < 1:
        raise ValueError("--xrf-user-count must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    config = {
        "repo_root": str(REPO_ROOT),
        "wsdp_source": str(WSDP_SRC),
        "elder_root": str(elder_root),
        "xrf_root": str(xrf_root),
        "output_dir": str(output_dir),
        "only": args.only,
        "source_min_length": min_length,
        "coefficient_lengths": {"b": len(b), "a": len(a)},
        "xrf_complex_policy": args.xrf_complex_policy,
        "max_elder_files": args.max_elder_files,
        "max_xrf_files": args.max_xrf_files,
        "xrf_user_count": args.xrf_user_count,
        "workers": args.workers,
        "meaningful_negative_relative_tolerance": (
            args.meaningful_negative_relative_tol
        ),
        "path_checks": {
            "wsdp_source_exists": WSDP_SRC.is_dir(),
            "elder_root_exists": elder_root.is_dir(),
            "xrf_root_exists": xrf_root.is_dir(),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "created_unix": time.time(),
    }
    print(json.dumps(config, ensure_ascii=False, indent=2))

    if args.self_test:
        synthetic_self_test()
        return
    if args.dry_run:
        print("Dry run complete; no dataset was loaded.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    json_dump(output_dir / "run_configuration.json", config)
    runtime = import_wsdp_runtime()
    overall: dict[str, Any] = {"configuration": config}
    if args.only in {"all", "elder"}:
        overall["elderAL"] = run_elder(
            elder_root,
            output_dir,
            runtime,
            args.max_elder_files,
        )
    if args.only in {"all", "xrf"}:
        overall["xrf55"] = run_xrf(
            xrf_root,
            output_dir,
            runtime,
            max_files=args.max_xrf_files,
            user_count=args.xrf_user_count,
            relative_tolerance=args.meaningful_negative_relative_tol,
            complex_policy=args.xrf_complex_policy,
            resume=args.resume,
            progress_every=args.progress_every,
            workers=args.workers,
        )
    completion_reasons = evaluate_signal_completion(
        config,
        overall.get("elderAL"),
        overall.get("xrf55"),
    )
    overall["official_complete"] = not completion_reasons
    overall["completion_reasons"] = completion_reasons
    overall["coverage"] = signal_coverage(
        overall.get("elderAL"),
        overall.get("xrf55"),
    )
    for dataset_key, summary_name in (
        ("elderAL", "elder_bandpass_summary.json"),
        ("xrf55", "xrf55_negative_summary.json"),
    ):
        if dataset_key in overall:
            overall[dataset_key]["official_complete"] = overall["official_complete"]
            overall[dataset_key]["completion_reasons"] = completion_reasons
            json_dump(output_dir / summary_name, overall[dataset_key])
    if args.only == "all":
        summary_name = "signal_study_summary.json"
        report_name = "signal_report_summary.md"
    else:
        summary_name = f"signal_study_summary_{args.only}_partial.json"
        report_name = f"signal_report_summary_{args.only}_partial.md"
    json_dump(output_dir / summary_name, overall)
    write_signal_plain_language_report(
        overall,
        output_dir,
        report_name=report_name,
    )
    print(f"Signal study complete: {output_dir}")


if __name__ == "__main__":
    main()
