"""Locate the causal failure mode of WSDP's ``robust`` phase sanitizer.

This experiment combines two independent kinds of evidence:

1. Pair every completed ``robust`` row in the existing 320-combination Widar
   and Gait tables with the otherwise-identical ``linear`` row.
2. Split ``robust_phase_sanitization`` into its common-mode and temporal
   detrending parts on real CSI.  The temporal branch is tested as implemented
   (fit the first 50 frames and extrapolate), with the extrapolation stopped at
   frame 49, and with the same 50-point estimator spread over the full sample.

The default output is deliberately inside ``ablation/``.  The source package
and the historical experiment outputs are read-only inputs.

Examples
--------
Run the local sampled datasets and one deterministic probe seed::

    python ablation/robust_rootcause_ablation.py --dataset all --seed 42

Only audit the existing full-training result tables::

    python ablation/robust_rootcause_ablation.py --summary-only

Fast implementation checks without loading CSI files::

    python ablation/robust_rootcause_ablation.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import interp1d


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = SCRIPT_DIR / "robust_rootcause_results"


def _find_wsdp_src() -> Path:
    """Support both the local outer repo and the server's inner SDP layout."""
    candidates = (
        REPO_ROOT
        / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
        / "src",
        REPO_ROOT
        / "SDP"
        / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
        / "src",
        REPO_ROOT / "src",
    )
    for candidate in candidates:
        if (candidate / "wsdp").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find the WSDP source tree. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _find_summary_path(dataset: str, filename: str) -> Path:
    test_dir = "test_wider" if dataset == "widar" else "test_gait"
    candidates = (
        REPO_ROOT / test_dir / "result" / "full_tests_new" / filename,
        REPO_ROOT / "SDP" / test_dir / "result" / "full_tests_new" / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


WSDP_SRC = _find_wsdp_src()
sys.path.insert(0, str(WSDP_SRC))
sys.modules.setdefault("kagglehub", types.ModuleType("kagglehub"))

from wsdp import readers  # noqa: E402
from wsdp.algorithms import execute_pipeline  # noqa: E402
from wsdp.algorithms.phase import robust_phase_sanitization  # noqa: E402
from wsdp.algorithms.subcarrier_mapping import get_subcarrier_indices  # noqa: E402
from wsdp.processors.base_processor import (  # noqa: E402
    _parse_file_info_from_filename,
    _selector,
)


VARIANTS = (
    "no_calibration",
    "linear_reference",
    "common_only",
    "detrend_first50_only",
    "detrend_fullspan50_only",
    "robust_shared_first50",
    "robust_window_limited",
    "robust_fullspan50",
    "robust_first50",
)

SUMMARY_PATHS = {
    "widar": _find_summary_path(
        "widar", "widar_320_pipeline_optimized_mlpmodel_summary.csv"
    ),
    "gait": _find_summary_path(
        "gait", "gait_320_pipeline_optimized_mlpmodel_summary.csv"
    ),
}

PREFIX_STEPS = {
    "none": {},
    "savgol_iqr": {
        "denoise": {"method": "savgol", "window_length": 7, "polyorder": 3},
        "outliers": {"method": "iqr", "factor": 1.5},
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(array)) if array.size else float("nan")


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _condition_key(row: dict[str, str], omit: str) -> tuple[str, ...]:
    fields = ("denoise", "outliers", "calibrate", "normalize", "interpolate")
    return tuple(row.get(field, "") for field in fields if field != omit)


def audit_accuracy_tables(output_dir: Path) -> dict[str, Any]:
    """Pair robust and linear rows while holding every other factor fixed."""
    all_pair_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {}

    for dataset, path in SUMMARY_PATHS.items():
        if not path.exists():
            report[dataset] = {"status": "missing", "path": str(path)}
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        valid = [
            row
            for row in rows
            if row.get("status") == "ok"
            and row.get("calibrate") in {"linear", "robust"}
            and safe_float(row.get("test_acc")) is not None
        ]
        linear = {
            _condition_key(row, "calibrate"): row
            for row in valid
            if row["calibrate"] == "linear"
        }
        pair_rows: list[dict[str, Any]] = []
        for robust in (row for row in valid if row["calibrate"] == "robust"):
            reference = linear.get(_condition_key(robust, "calibrate"))
            if reference is None:
                continue
            robust_acc = float(robust["test_acc"])
            linear_acc = float(reference["test_acc"])
            pair = {
                "dataset": dataset,
                "denoise": robust["denoise"],
                "outliers": robust["outliers"],
                "normalize": robust["normalize"],
                "interpolate": robust["interpolate"],
                "linear_test_acc": linear_acc,
                "robust_test_acc": robust_acc,
                "robust_minus_linear": robust_acc - linear_acc,
                "linear_combo_id": reference.get("combo_id", ""),
                "robust_combo_id": robust.get("combo_id", ""),
            }
            pair_rows.append(pair)
            all_pair_rows.append(pair)

        by_normalization: dict[str, Any] = {}
        for normalization in sorted({row["normalize"] for row in pair_rows}):
            values = [
                row["robust_minus_linear"]
                for row in pair_rows
                if row["normalize"] == normalization
            ]
            by_normalization[normalization] = summarize_values(values)

        by_interpolation: dict[str, Any] = {}
        for interpolation in sorted({row["interpolate"] for row in pair_rows}):
            values = [
                row["robust_minus_linear"]
                for row in pair_rows
                if row["interpolate"] == interpolation
            ]
            by_interpolation[interpolation] = summarize_values(values)

        # Matched 2x2 interaction: how much more robust loses under min-max.
        norm_lookup = {
            (
                row["denoise"],
                row["outliers"],
                row["interpolate"],
                row["normalize"],
            ): row["robust_minus_linear"]
            for row in pair_rows
        }
        norm_interactions = []
        norm_bases = {
            (denoise, outliers, interpolation)
            for denoise, outliers, interpolation, _ in norm_lookup
        }
        for denoise, outliers, interpolation in norm_bases:
            z_key = (denoise, outliers, interpolation, "z-score")
            m_key = (denoise, outliers, interpolation, "min-max")
            if z_key in norm_lookup and m_key in norm_lookup:
                norm_interactions.append(norm_lookup[m_key] - norm_lookup[z_key])

        # Matched interpolation interaction relative to nearest.  A negative
        # value means complex mixing adds damage beyond robust itself.
        interpolation_interactions: dict[str, list[float]] = defaultdict(list)
        raw_lookup = {
            (
                row["denoise"],
                row["outliers"],
                row["normalize"],
                row["interpolate"],
                calibration,
            ): accuracy
            for row in valid
            for calibration, accuracy in [(row["calibrate"], float(row["test_acc"]))]
        }
        bases = {
            (row["denoise"], row["outliers"], row["normalize"])
            for row in valid
        }
        for denoise, outliers, normalization in bases:
            for method in ("linear15", "cubic15", "decimate15"):
                keys = {
                    "rn": (denoise, outliers, normalization, "nearest15", "robust"),
                    "ln": (denoise, outliers, normalization, "nearest15", "linear"),
                    "rm": (denoise, outliers, normalization, method, "robust"),
                    "lm": (denoise, outliers, normalization, method, "linear"),
                }
                if all(key in raw_lookup for key in keys.values()):
                    value = (
                        raw_lookup[keys["rm"]]
                        - raw_lookup[keys["rn"]]
                        - raw_lookup[keys["lm"]]
                        + raw_lookup[keys["ln"]]
                    )
                    interpolation_interactions[method].append(value)

        report[dataset] = {
            "status": "ok",
            "path": str(path),
            "completed_table_rows": len(rows),
            "paired_rows": len(pair_rows),
            "overall_robust_minus_linear": summarize_values(
                row["robust_minus_linear"] for row in pair_rows
            ),
            "by_normalization": by_normalization,
            "by_interpolation": by_interpolation,
            "minmax_x_robust_interaction": summarize_values(norm_interactions),
            "interpolation_x_robust_interaction_vs_nearest": {
                method: summarize_values(values)
                for method, values in sorted(interpolation_interactions.items())
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "accuracy_effects.json", report)
    if all_pair_rows:
        fieldnames = list(all_pair_rows[0])
        with (output_dir / "accuracy_pairs.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_pair_rows)
    return report


def infer_data_path(dataset: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = {
        "gait": (
            REPO_ROOT / "sdp_dataset" / "Gait_Dataset" / "CSI_Gait",
            REPO_ROOT / "sdp_dataset" / "Gait_Dataset",
            REPO_ROOT / "sdp_dataset" / "gait",
            REPO_ROOT / "SDP" / "sdp_dataset" / "Gait_Dataset" / "CSI_Gait",
            REPO_ROOT.parent / "sdp_dataset" / "Gait_Dataset" / "CSI_Gait",
            REPO_ROOT.parent / "sdp_dataset" / "Gait_Dataset",
            REPO_ROOT.parent / "sdp_dataset" / "gait",
        ),
        "widar": (
            REPO_ROOT / "sdp_dataset" / "widar_common3",
            REPO_ROOT / "sdp_dataset" / "widar",
            REPO_ROOT / "SDP" / "sdp_dataset" / "widar_common3",
            REPO_ROOT.parent / "sdp_dataset" / "widar_common3",
            REPO_ROOT.parent / "sdp_dataset" / "widar",
        ),
    }[dataset]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def parse_item_metadata(item: Any, dataset: str) -> tuple[Any, Any] | None:
    parsed = _parse_file_info_from_filename(item.file_name, dataset)
    if parsed is None:
        return None
    return _selector(parsed, dataset)


def select_balanced_items(
    items: list[Any], dataset: str, max_samples: int | None, seed: int
) -> list[Any]:
    if max_samples is None or len(items) <= max_samples:
        return items
    buckets: dict[Any, list[Any]] = defaultdict(list)
    for item in items:
        metadata = parse_item_metadata(item, dataset)
        if metadata is not None:
            buckets[metadata[0]].append(item)
    rng = np.random.default_rng(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[Any] = []
    while len(selected) < max_samples and buckets:
        progressed = False
        for label in sorted(buckets, key=str):
            if buckets[label] and len(selected) < max_samples:
                selected.append(buckets[label].pop())
                progressed = True
        if not progressed:
            break
    return selected


def select_balanced_files(
    files: list[Path], dataset: str, max_samples: int | None, seed: int
) -> list[Path]:
    """Select files before parsing, avoiding WSDP's fixed 16-reader pool."""
    valid: list[tuple[Path, Any]] = []
    for path in files:
        parsed = _parse_file_info_from_filename(path.name, dataset)
        if parsed is None:
            continue
        label, _ = _selector(parsed, dataset)
        valid.append((path, label))
    if max_samples is None or len(valid) <= max_samples:
        return [path for path, _ in valid]
    buckets: dict[Any, list[Path]] = defaultdict(list)
    for path, label in valid:
        buckets[label].append(path)
    rng = np.random.default_rng(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[Path] = []
    while len(selected) < max_samples:
        progressed = False
        for label in sorted(buckets, key=str):
            if buckets[label] and len(selected) < max_samples:
                selected.append(buckets[label].pop())
                progressed = True
        if not progressed:
            break
    return selected


def item_to_array(
    item: Any, dataset: str
) -> tuple[np.ndarray, Any, Any, dict[str, float]] | None:
    metadata = parse_item_metadata(item, dataset)
    if metadata is None:
        return None
    label, group = metadata
    frames = sorted(item.frames, key=lambda frame: frame.timestamp)
    if len(frames) < 2:
        return None
    csi = np.stack([frame.csi_array for frame in frames], axis=0)
    if csi.ndim == 2:
        csi = csi[..., None]
    if csi.ndim != 3 or np.isrealobj(csi):
        return None
    timestamps = np.asarray([float(frame.timestamp) for frame in frames])
    positive_deltas = np.diff(timestamps)
    positive_deltas = positive_deltas[positive_deltas > 0]
    timing = {
        "median_frame_interval_ticks": float(np.median(positive_deltas))
        if positive_deltas.size
        else 0.0,
        "first50_span_ticks": float(timestamps[min(49, len(timestamps) - 1)] - timestamps[0]),
        "full_span_ticks": float(timestamps[-1] - timestamps[0]),
    }
    return csi, label, group, timing


def load_samples(
    dataset: str, path: Path, max_samples: int | None, seed: int
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{dataset} data path does not exist: {path}")
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and "truth" not in candidate.name
    )
    selected_files = select_balanced_files(files, dataset, max_samples, seed)
    reader = readers.get_reader_class(dataset)()
    arrays: list[np.ndarray] = []
    raw_labels: list[Any] = []
    raw_groups: list[Any] = []
    timing_rows: list[dict[str, float]] = []
    for index, file_path in enumerate(selected_files, 1):
        if not reader.sniff(str(file_path)):
            continue
        loaded = reader.read_file(str(file_path))
        items = loaded if isinstance(loaded, list) else [loaded]
        for item in items:
            parsed = item_to_array(item, dataset)
            if parsed is None:
                continue
            csi, label, group, timing = parsed
            arrays.append(csi)
            raw_labels.append(label)
            raw_groups.append(group)
            timing_rows.append(timing)
        if index % 25 == 0 or index == len(selected_files):
            print(f"  read {dataset}: {index}/{len(selected_files)}", flush=True)
    if not arrays:
        raise RuntimeError(f"No valid complex {dataset} samples were parsed from {path}")
    unique_labels = sorted(set(raw_labels), key=str)
    unique_groups = sorted(set(raw_groups), key=str)
    label_map = {label: index for index, label in enumerate(unique_labels)}
    group_map = {group: index for index, group in enumerate(unique_groups)}
    labels = np.asarray([label_map[item] for item in raw_labels], dtype=np.int64)
    groups = np.asarray([group_map[item] for item in raw_groups], dtype=np.int64)
    metadata = {
        "dataset": dataset,
        "path": str(path),
        "available_files": len(files),
        "selected_files": len(selected_files),
        "selected_valid_samples": len(arrays),
        "labels": [str(value) for value in unique_labels],
        "label_distribution": {
            str(key): int(value) for key, value in Counter(raw_labels).items()
        },
        "groups": [str(value) for value in unique_groups],
        "group_distribution": {
            str(key): int(value) for key, value in Counter(raw_groups).items()
        },
        "length": summarize_values(array.shape[0] for array in arrays),
        "first_shape": list(arrays[0].shape),
        "timestamp_note": (
            "Intel 5300 timestamp_low ticks are treated as microseconds; the robust "
            "implementation itself discards timestamps and uses frame indices."
        ),
        "timing": {
            key: summarize_values(row[key] for row in timing_rows)
            for key in timing_rows[0]
        },
    }
    return arrays, labels, groups, metadata


def apply_prefix(samples: list[np.ndarray], dataset: str, prefix: str) -> list[np.ndarray]:
    steps = PREFIX_STEPS[prefix]
    if not steps:
        return [sample.copy() for sample in samples]
    result = []
    for index, sample in enumerate(samples, 1):
        result.append(execute_pipeline(sample, steps, dataset=dataset))
        if index % 25 == 0 or index == len(samples):
            print(f"  prefix {dataset}: {index}/{len(samples)}", flush=True)
    return result


def theil_sen_slope(phases: np.ndarray, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size < 2:
        return np.zeros(phases.shape[1:], dtype=np.float64)
    left, right = np.triu_indices(indices.size, k=1)
    i = indices[left]
    j = indices[right]
    denominator = (j - i).astype(np.float64)[:, None, None]
    slopes = (phases[j] - phases[i]) / denominator
    return np.median(slopes, axis=0)


def linear_reference(csi: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of the repository's packet-wise linear fit."""
    _, carriers, _ = csi.shape
    x = np.asarray(get_subcarrier_indices(num_subcarriers=carriers), dtype=np.float64)
    centered = x - np.mean(x)
    denominator = float(np.sum(centered**2))
    phase = np.unwrap(np.angle(csi), axis=1)
    slope = np.sum(phase * centered[None, :, None], axis=1) / denominator
    intercept = np.mean(phase, axis=1) - slope * float(np.mean(x))
    fitted = intercept[:, None, :] + slope[:, None, :] * x[None, :, None]
    corrected = phase - fitted
    return (np.abs(csi) * np.exp(1j * corrected)).astype(csi.dtype, copy=False)


def phase_parts(csi: np.ndarray) -> dict[str, np.ndarray | int]:
    phase = np.unwrap(np.angle(csi), axis=0).astype(np.float64, copy=False)
    common = np.median(phase, axis=1, keepdims=True)
    centered = phase - common
    count = min(50, phase.shape[0])
    first_indices = np.arange(count, dtype=np.int64)
    full_indices = np.unique(
        np.linspace(0, phase.shape[0] - 1, count, dtype=np.int64)
    )
    return {
        "phase": phase,
        "common": common,
        "centered": centered,
        "first_slope": theil_sen_slope(centered, first_indices),
        "raw_first_slope": theil_sen_slope(phase, first_indices),
        "full_slope": theil_sen_slope(centered, full_indices),
        "fit_count": count,
    }


def reconstruct(csi: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return (np.abs(csi) * np.exp(1j * phase)).astype(csi.dtype, copy=False)


def apply_variant(
    csi: np.ndarray, variant: str, parts: dict[str, np.ndarray | int] | None = None
) -> np.ndarray:
    if variant == "no_calibration":
        return csi.copy()
    if variant == "linear_reference":
        return linear_reference(csi)
    if parts is None:
        parts = phase_parts(csi)
    phase = np.asarray(parts["phase"])
    centered = np.asarray(parts["centered"])
    first_slope = np.asarray(parts["first_slope"])
    raw_first_slope = np.asarray(parts["raw_first_slope"])
    full_slope = np.asarray(parts["full_slope"])
    fit_count = int(parts["fit_count"])
    times = np.arange(csi.shape[0], dtype=np.float64)[:, None, None]
    if variant == "common_only":
        corrected = centered
    elif variant == "detrend_first50_only":
        corrected = phase - times * raw_first_slope[None, :, :]
    elif variant == "detrend_fullspan50_only":
        raw_full_slope = theil_sen_slope(
            phase,
            np.unique(
                np.linspace(0, csi.shape[0] - 1, min(50, csi.shape[0]), dtype=int)
            ),
        )
        corrected = phase - times * raw_full_slope[None, :, :]
    elif variant == "robust_shared_first50":
        # One slope per antenna is a global rotation.  Unlike the source's
        # independent F x A slopes, it cannot scramble inter-carrier phase.
        shared_slope = np.median(first_slope, axis=0, keepdims=True)
        corrected = centered - times * shared_slope[None, :, :]
    elif variant == "robust_first50":
        corrected = centered - times * first_slope[None, :, :]
    elif variant == "robust_window_limited":
        limited = np.minimum(times, max(fit_count - 1, 0))
        corrected = centered - limited * first_slope[None, :, :]
    elif variant == "robust_fullspan50":
        corrected = centered - times * full_slope[None, :, :]
    else:
        raise ValueError(f"Unknown phase variant: {variant}")
    return reconstruct(csi, corrected)


def circular_abs(values: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * values)))


def adjacent_phase(csi: np.ndarray) -> np.ndarray:
    return np.angle(csi[:, 1:, :] * np.conj(csi[:, :-1, :]))


def fast_interpolate(csi: np.ndarray, target_k: int, method: str) -> np.ndarray:
    """Axis-vectorized equivalent of WSDP's per-packet interpolation loop."""
    carriers = csi.shape[1]
    if carriers == target_k:
        return csi.copy()
    source_x = np.asarray(
        get_subcarrier_indices(num_subcarriers=carriers), dtype=np.float64
    )
    target_x = np.linspace(source_x[0], source_x[-1], target_k)
    if np.iscomplexobj(csi):
        real = interp1d(
            source_x,
            np.real(csi),
            kind=method,
            axis=1,
            bounds_error=False,
            fill_value="extrapolate",
        )(target_x)
        imaginary = interp1d(
            source_x,
            np.imag(csi),
            kind=method,
            axis=1,
            bounds_error=False,
            fill_value="extrapolate",
        )(target_x)
        return real + 1j * imaginary
    return interp1d(
        source_x,
        csi,
        kind=method,
        axis=1,
        bounds_error=False,
        fill_value="extrapolate",
    )(target_x)


def interpolation_cancellation(csi: np.ndarray, method: str) -> np.ndarray:
    cartesian = fast_interpolate(csi, target_k=15, method=method)
    amplitude_reference = fast_interpolate(np.abs(csi), target_k=15, method=method)
    denominator = np.maximum(np.abs(amplitude_reference), 1e-8)
    return np.abs(cartesian) / denominator


def sample_slope_metrics(parts: dict[str, np.ndarray | int], length: int) -> dict[str, float]:
    first = np.asarray(parts["first_slope"])
    full = np.asarray(parts["full_slope"])
    fit_count = int(parts["fit_count"])
    horizon = min(length - 1, 1499)
    adjacent = np.diff(first, axis=0)
    drift = np.abs(adjacent) * horizon
    wrapped_drift = circular_abs(adjacent * horizon)
    first_flat = first.reshape(-1)
    full_flat = full.reshape(-1)
    if np.std(first_flat) > 0 and np.std(full_flat) > 0:
        correlation = float(np.corrcoef(first_flat, full_flat)[0, 1])
    else:
        correlation = 0.0
    return {
        "length": float(length),
        "fit_context_fraction": float(fit_count / length),
        "extrapolation_factor_to_model_horizon": float(
            horizon / max(fit_count - 1, 1)
        ),
        "first_slope_abs_median_rad_per_frame": float(np.median(np.abs(first))),
        "first_vs_full_slope_correlation": correlation,
        "first_vs_full_slope_mae_rad_per_frame": float(np.mean(np.abs(first - full))),
        "adjacent_drift_at_model_horizon_mean_rad": float(np.mean(drift)),
        "adjacent_drift_at_model_horizon_p90_rad": float(np.percentile(drift, 90)),
        "adjacent_rotation_at_model_horizon_mean_rad": float(np.mean(wrapped_drift)),
        "adjacent_rotation_at_model_horizon_gt_pi_over_2": float(
            np.mean(wrapped_drift > np.pi / 2)
        ),
    }


def sample_variant_metrics(
    baseline: np.ndarray, result: np.ndarray, variant: str
) -> dict[str, Any]:
    length = result.shape[0]
    quarter = max(1, length // 4)
    before_adjacent = adjacent_phase(baseline)
    after_adjacent = adjacent_phase(result)
    relative_rotation = circular_abs(after_adjacent - before_adjacent)
    row: dict[str, Any] = {
        "variant": variant,
        "amplitude_max_abs_error": float(
            np.max(np.abs(np.abs(result) - np.abs(baseline)))
        ),
        "adjacent_phase_jump_early_mean_rad": float(
            np.mean(np.abs(after_adjacent[:quarter]))
        ),
        "adjacent_phase_jump_late_mean_rad": float(
            np.mean(np.abs(after_adjacent[-quarter:]))
        ),
        "late_relative_rotation_mean_rad": float(
            np.mean(relative_rotation[-quarter:])
        ),
        "late_relative_rotation_gt_pi_over_2": float(
            np.mean(relative_rotation[-quarter:] > np.pi / 2)
        ),
        "late_relative_rotation_gt_3pi_over_4": float(
            np.mean(relative_rotation[-quarter:] > 3 * np.pi / 4)
        ),
    }
    nearest = fast_interpolate(result, target_k=15, method="nearest")
    amplitude = np.abs(nearest).astype(np.float64)
    amp_mean = np.mean(amplitude, axis=0, keepdims=True)
    amp_std = np.std(amplitude, axis=0, keepdims=True)
    zscore_amplitude = (amplitude - amp_mean) / np.where(amp_std < 1e-8, 1.0, amp_std)
    amp_min = np.min(amplitude, axis=0, keepdims=True)
    amp_range = np.max(amplitude, axis=0, keepdims=True) - amp_min
    minmax_amplitude = (amplitude - amp_min) / np.where(
        amp_range < 1e-8, 1.0, amp_range
    )
    phase_std = float(np.std(np.angle(nearest)))
    zscore_std = float(np.std(zscore_amplitude))
    minmax_std = float(np.std(minmax_amplitude))
    row.update(
        {
            "phase_channel_std": phase_std,
            "zscore_amplitude_channel_std": zscore_std,
            "zscore_phase_to_amplitude_std_ratio": phase_std
            / max(zscore_std, 1e-8),
            "minmax_amplitude_channel_std": minmax_std,
            "minmax_phase_to_amplitude_std_ratio": phase_std
            / max(minmax_std, 1e-8),
            "minmax_exact_zero_fraction": float(np.mean(minmax_amplitude == 0.0)),
        }
    )
    for method in ("linear", "cubic"):
        ratio = interpolation_cancellation(result, method)
        row[f"{method}_cancellation_ratio_mean"] = float(np.mean(ratio))
        row[f"{method}_cancellation_ratio_p10"] = float(np.percentile(ratio, 10))
        row[f"{method}_cancellation_ratio_lt_0_5"] = float(np.mean(ratio < 0.5))
        row[f"{method}_late_minus_early_ratio"] = float(
            np.mean(ratio[-quarter:]) - np.mean(ratio[:quarter])
        )
    return row


def temporal_bin_edges(length: int, bins: int) -> np.ndarray:
    return np.linspace(0, length, bins + 1, dtype=np.int64)


def probe_features(csi: np.ndarray, bins: int = 6) -> np.ndarray:
    """Compact amplitude/phase features for a one-seed causal probe."""
    if csi.shape[1] != 15:
        csi = fast_interpolate(csi, target_k=15, method="nearest")
    csi = csi[:1500]
    amplitude = np.abs(csi).astype(np.float64)
    amp_mean = np.mean(amplitude, axis=0, keepdims=True)
    amp_std = np.std(amplitude, axis=0, keepdims=True)
    amplitude = (amplitude - amp_mean) / np.where(amp_std < 1e-8, 1.0, amp_std)
    phase = np.angle(csi)
    adjacent = adjacent_phase(csi)
    temporal_step = np.angle(csi[1:] * np.conj(csi[:-1]))
    features: list[np.ndarray] = []
    edges = temporal_bin_edges(len(csi), bins)
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        amp_block = amplitude[left:right]
        phase_block = phase[left:right]
        adjacent_block = adjacent[left:right]
        step_left = min(left, max(len(temporal_step) - 1, 0))
        step_right = min(max(right - 1, step_left + 1), len(temporal_step))
        step_block = temporal_step[step_left:step_right]
        features.extend(
            [
                np.mean(amp_block, axis=0).reshape(-1),
                np.std(amp_block, axis=0).reshape(-1),
                np.mean(np.sin(phase_block), axis=0).reshape(-1),
                np.mean(np.cos(phase_block), axis=0).reshape(-1),
                np.mean(np.sin(adjacent_block), axis=0).reshape(-1),
                np.mean(np.cos(adjacent_block), axis=0).reshape(-1),
                np.mean(np.sin(step_block), axis=0).reshape(-1),
                np.mean(np.cos(step_block), axis=0).reshape(-1),
            ]
        )
    return np.concatenate(features).astype(np.float32, copy=False)


def fisher_ratio(features: np.ndarray, labels: np.ndarray) -> float:
    global_mean = np.mean(features, axis=0)
    between = np.zeros(features.shape[1], dtype=np.float64)
    within = np.zeros(features.shape[1], dtype=np.float64)
    for label in np.unique(labels):
        block = features[labels == label]
        class_mean = np.mean(block, axis=0)
        between += len(block) * (class_mean - global_mean) ** 2
        within += np.sum((block - class_mean) ** 2, axis=0)
    ratio = between / np.maximum(within, 1e-8)
    return float(np.median(ratio[np.isfinite(ratio)]))


def run_probe(
    feature_sets: dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_index, test_index = next(splitter.split(labels, labels, groups))
    result: dict[str, Any] = {
        "seed": seed,
        "split": "GroupShuffleSplit(test_size=0.30)",
        "train_size": int(len(train_index)),
        "test_size": int(len(test_index)),
        "train_groups": int(len(np.unique(groups[train_index]))),
        "test_groups": int(len(np.unique(groups[test_index]))),
        "train_classes": np.unique(labels[train_index]).astype(int).tolist(),
        "test_classes": np.unique(labels[test_index]).astype(int).tolist(),
        "conditions": {},
    }
    for variant, features in feature_sets.items():
        classifier = ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        classifier.fit(features[train_index], labels[train_index])
        predictions = classifier.predict(features[test_index])
        result["conditions"][variant] = {
            "test_acc": float(accuracy_score(labels[test_index], predictions)),
            "balanced_test_acc": float(
                balanced_accuracy_score(labels[test_index], predictions)
            ),
            "feature_fisher_ratio_median": fisher_ratio(features, labels),
            "feature_count": int(features.shape[1]),
        }
    return result


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    result: dict[str, Any] = {}
    for variant, items in grouped.items():
        metrics = {}
        keys = sorted(set().union(*(item.keys() for item in items)) - {"variant"})
        for key in keys:
            values = [safe_float(item.get(key)) for item in items]
            metrics[key] = summarize_values(value for value in values if value is not None)
        result[variant] = metrics
    return result


def source_equivalence(samples: list[np.ndarray], limit: int) -> dict[str, Any]:
    absolute_errors = []
    relative_errors = []
    for sample in samples[:limit]:
        ours = apply_variant(sample, "robust_first50")
        source = robust_phase_sanitization(sample)
        error = np.abs(ours.astype(np.complex128) - source.astype(np.complex128))
        absolute_errors.append(float(np.max(error)))
        relative_errors.append(
            float(np.max(error / np.maximum(np.abs(source), 1e-8)))
        )
    return {
        "samples": min(limit, len(samples)),
        "max_complex_absolute_error": max(absolute_errors, default=0.0),
        "max_complex_relative_error": max(relative_errors, default=0.0),
    }


def run_dataset(
    dataset: str,
    path: Path,
    output_dir: Path,
    max_samples: int | None,
    prefix: str,
    seed: int,
    diagnostic_limit: int,
    equivalence_limit: int,
    skip_probe: bool,
) -> dict[str, Any]:
    started = time.time()
    samples, labels, groups, metadata = load_samples(dataset, path, max_samples, seed)
    print(
        f"Loaded {dataset}: n={len(samples)}, classes={len(np.unique(labels))}, "
        f"groups={len(np.unique(groups))}, path={path}",
        flush=True,
    )
    samples = apply_prefix(samples, dataset, prefix)
    dataset_dir = output_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata["prefix"] = prefix
    metadata["prefix_steps"] = PREFIX_STEPS[prefix]
    write_json(dataset_dir / "dataset_metadata.json", metadata)

    equivalence = source_equivalence(samples, equivalence_limit)
    write_json(dataset_dir / "source_equivalence.json", equivalence)

    metric_rows: list[dict[str, Any]] = []
    slope_rows: list[dict[str, Any]] = []
    feature_rows: dict[str, list[np.ndarray]] = {variant: [] for variant in VARIANTS}
    for sample_index, sample in enumerate(samples):
        parts = phase_parts(sample)
        slope = sample_slope_metrics(parts, sample.shape[0])
        slope["sample_index"] = float(sample_index)
        slope_rows.append(slope)
        baseline = apply_variant(sample, "common_only", parts)
        for variant in VARIANTS:
            result = apply_variant(sample, variant, parts)
            if sample_index < min(diagnostic_limit, len(samples)):
                row = sample_variant_metrics(baseline, result, variant)
                row["sample_index"] = sample_index
                row["length"] = sample.shape[0]
                metric_rows.append(row)
            if not skip_probe:
                feature_rows[variant].append(probe_features(result))
        if (sample_index + 1) % 10 == 0 or sample_index + 1 == len(samples):
            print(f"  variants {dataset}: {sample_index + 1}/{len(samples)}", flush=True)

    slope_aggregate = {
        key: summarize_values(row[key] for row in slope_rows)
        for key in slope_rows[0]
        if key != "sample_index"
    }
    metric_aggregate = aggregate_metric_rows(metric_rows)
    write_json(dataset_dir / "slope_diagnostics.json", slope_aggregate)
    write_json(dataset_dir / "variant_diagnostics.json", metric_aggregate)

    if metric_rows:
        with (dataset_dir / "variant_diagnostics_per_sample.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)

    probe = None
    if not skip_probe and len(np.unique(labels)) >= 2 and len(np.unique(groups)) >= 2:
        feature_sets = {
            variant: np.stack(rows, axis=0) for variant, rows in feature_rows.items()
        }
        probe = run_probe(feature_sets, labels, groups, seed)
        write_json(dataset_dir / "probe_accuracy.json", probe)

    completion = {
        "dataset": dataset,
        "status": "ok",
        "duration_sec": time.time() - started,
        "samples": len(samples),
        "diagnostic_samples": min(diagnostic_limit, len(samples)),
        "probe_run": probe is not None,
        "source_equivalence": equivalence,
    }
    write_json(dataset_dir / "completion.json", completion)
    return completion


def format_pp(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def render_report(
    output_dir: Path, accuracy: dict[str, Any], completions: dict[str, Any]
) -> None:
    lines = [
        "# Robust phase sanitization root-cause ablation",
        "",
        "This file is generated from the committed 320-combination accuracy tables "
        "and the real-CSI component diagnostics in this directory.",
        "",
        "## Paired full-training evidence",
        "",
        "| Dataset | Pairs | Robust − linear | z-score | min-max | min-max interaction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("widar", "gait"):
        item = accuracy.get(dataset, {})
        if item.get("status") != "ok":
            continue
        overall = item["overall_robust_minus_linear"]["mean"]
        zscore = item["by_normalization"].get("z-score", {}).get("mean", float("nan"))
        minmax = item["by_normalization"].get("min-max", {}).get("mean", float("nan"))
        interaction = item["minmax_x_robust_interaction"].get("mean", float("nan"))
        lines.append(
            f"| {dataset} | {item['paired_rows']} | {format_pp(overall)} | "
            f"{format_pp(zscore)} | {format_pp(minmax)} | {format_pp(interaction)} |"
        )
    lines.extend(
        [
            "",
            "A negative interaction means min-max amplifies the loss caused by robust.",
            "",
            "## Signal/component runs",
            "",
        ]
    )
    for dataset, completion in completions.items():
        lines.append(
            f"- {dataset}: {completion.get('status')}, samples={completion.get('samples')}, "
            f"probe={completion.get('probe_run')}, duration={completion.get('duration_sec', 0):.1f}s"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "- A large loss for nearest15 proves interpolation cancellation is not required.",
            "- Recovery from robust_first50 to robust_window_limited identifies long-horizon "
            "extrapolation; recovery to robust_fullspan50 identifies the first-50 fit window.",
            "- A remaining loss in common_only identifies removal of label-bearing common phase.",
            "- Lower Cartesian cancellation ratios for robust_first50 identify complex "
            "interpolation as an amplifier, not the sole root cause.",
            "",
            "See `accuracy_effects.json`, each dataset's `slope_diagnostics.json`, "
            "`variant_diagnostics.json`, and `probe_accuracy.json` for the numeric evidence.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    rng = np.random.default_rng(42)
    time_count, carriers, antennas = 160, 30, 3
    times = np.arange(time_count, dtype=np.float64)[:, None, None]
    base = rng.uniform(-np.pi, np.pi, size=(1, carriers, antennas))
    common = 0.02 * times + 0.2 * np.sin(times / 13.0)
    slopes = rng.normal(0.0, 0.003, size=(1, carriers, antennas))
    phase = base + common + times * slopes
    amplitude = 2.0 + rng.random((time_count, carriers, antennas))
    csi = (amplitude * np.exp(1j * phase)).astype(np.complex64)
    parts = phase_parts(csi)
    robust = apply_variant(csi, "robust_first50", parts)
    source = robust_phase_sanitization(csi)
    relative = np.max(
        np.abs(robust.astype(np.complex128) - source.astype(np.complex128))
        / np.maximum(np.abs(source), 1e-8)
    )
    assert relative < 5e-5, relative
    limited = apply_variant(csi, "robust_window_limited", parts)
    assert np.max(np.abs(np.abs(limited) - np.abs(csi))) < 1e-5
    linear = linear_reference(csi)
    assert linear.shape == csi.shape and np.iscomplexobj(linear)
    print("self-test: ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "gait", "widar"), default="all")
    parser.add_argument("--gait-path", default=None)
    parser.add_argument("--widar-path", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-samples-gait", type=int, default=180)
    parser.add_argument("--max-samples-widar", type=int, default=180)
    parser.add_argument("--diagnostic-samples", type=int, default=64)
    parser.add_argument("--equivalence-samples", type=int, default=2)
    parser.add_argument("--prefix", choices=tuple(PREFIX_STEPS), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "seed": args.seed,
        "dataset": args.dataset,
        "prefix": args.prefix,
        "max_samples_gait": args.max_samples_gait,
        "max_samples_widar": args.max_samples_widar,
        "diagnostic_samples": args.diagnostic_samples,
        "equivalence_samples": args.equivalence_samples,
        "variants": list(VARIANTS),
        "probe": "ExtraTreesClassifier, one GroupShuffleSplit seed",
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "settings.json", settings)
    accuracy = audit_accuracy_tables(output_dir)
    completions: dict[str, Any] = {}
    if not args.summary_only:
        datasets = ("gait", "widar") if args.dataset == "all" else (args.dataset,)
        for dataset in datasets:
            path = infer_data_path(dataset, getattr(args, f"{dataset}_path"))
            max_samples = getattr(args, f"max_samples_{dataset}")
            try:
                completions[dataset] = run_dataset(
                    dataset=dataset,
                    path=path,
                    output_dir=output_dir,
                    max_samples=max_samples,
                    prefix=args.prefix,
                    seed=args.seed,
                    diagnostic_limit=args.diagnostic_samples,
                    equivalence_limit=args.equivalence_samples,
                    skip_probe=args.skip_probe,
                )
            except Exception as error:
                completions[dataset] = {
                    "dataset": dataset,
                    "status": "failed",
                    "error": repr(error),
                }
                write_json(output_dir / dataset / "completion.json", completions[dataset])
                raise
    render_report(output_dir, accuracy, completions)
    write_json(
        output_dir / "completion.json",
        {"status": "ok", "datasets": completions, "summary_audit": True},
    )
    print(f"Robust root-cause ablation complete: {output_dir}")


if __name__ == "__main__":
    main()
