"""Server-only XRF55 Bandpass sampling-rate/sign-preservation ablation.

Research question
-----------------
The legacy XRF55 pipeline applies ``abs`` inside IQR clipping and applies
``abs`` again immediately before train-split z-score normalization.  A
zero-centred Bandpass signal therefore loses its positive/negative half-cycle
structure.  This script separates that representation effect from the sampling
rate effect with the controlled 2 x 2 experiment

    fs in {1000, 200} x representation in {legacy_abs, signed}

and adds three mechanism/reference cases:

    * fs=200, signed-IQR followed by legacy abs normalization;
    * fs=200, signed representation without IQR;
    * Savgol(7, 3) with the legacy IQR/normalization path.

The exact processing order is

    raw real XRF55
    -> denoise
    -> IQR (or no IQR)
    -> cubic interpolation to 15 subcarriers
    -> resize/pad to 1000 frames
    -> repetition split (1-12 / 13-16 / 17-20)
    -> legacy abs or signed values
    -> train-only global z-score
    -> ResNet1D

This file is intended to be copied to and run on the data server.  ``--help``,
``--dry-run`` and ``--self-test`` do not load a dataset.

Examples
--------
Full official experiment (7 cases x 5 model seeds)::

    python ablation/bandpass_server_sign_ablation.py \
      --data-path sdp_dataset/xrf55/wifi --resume

Fast server smoke test::

    python ablation/bandpass_server_sign_ablation.py \
      --cases bp_fs200_legacy_abs bp_fs200_signed \
      --seeds 42 --epochs 1 --user-count 1

Signal/preprocessing diagnostics only::

    python ablation/bandpass_server_sign_ablation.py --diagnostics-only
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import multiprocessing as mp
import os
import platform
import random
import re
import sys
import tempfile
import time
import traceback
import types
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WSDP_SRC = (
    REPO_ROOT
    / "SDP"
    / "SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing-main"
    / "src"
)
DEFAULT_DATA_PATH = REPO_ROOT / "sdp_dataset" / "xrf55" / "wifi"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "ablation" / "bandpass_server_results" / "sign_ablation"
)

DATASET_NAME = "xrf55"
MODEL_NAME = "resnet1d"
PADDING_LENGTH = 1000
TARGET_SUBCARRIERS = 15
IQR_FACTOR = 1.5
DEFAULT_SEEDS = (42, 49, 514, 654, 886)
DEFAULT_EPOCHS = 50
DEFAULT_SPLIT_SEED = 42
EXPECTED_ACTION_IDS = tuple(range(1, 56))

CASES: tuple[dict[str, Any], ...] = (
    {
        "case": "bp_fs1000_legacy_abs",
        "denoiser": "bandpass",
        "fs_hz": 1000.0,
        "iqr_mode": "legacy_abs",
        "normalization_input": "abs",
        "role": "main_2x2",
    },
    {
        "case": "bp_fs1000_signed",
        "denoiser": "bandpass",
        "fs_hz": 1000.0,
        "iqr_mode": "signed",
        "normalization_input": "signed",
        "role": "main_2x2",
    },
    {
        "case": "bp_fs200_legacy_abs",
        "denoiser": "bandpass",
        "fs_hz": 200.0,
        "iqr_mode": "legacy_abs",
        "normalization_input": "abs",
        "role": "main_2x2",
    },
    {
        "case": "bp_fs200_signed",
        "denoiser": "bandpass",
        "fs_hz": 200.0,
        "iqr_mode": "signed",
        "normalization_input": "signed",
        "role": "main_2x2",
    },
    {
        "case": "bp_fs200_signed_iqr_absnorm",
        "denoiser": "bandpass",
        "fs_hz": 200.0,
        "iqr_mode": "signed",
        "normalization_input": "abs",
        "role": "mechanism",
    },
    {
        "case": "bp_fs200_signed_no_iqr",
        "denoiser": "bandpass",
        "fs_hz": 200.0,
        "iqr_mode": "none",
        "normalization_input": "signed",
        "role": "mechanism",
    },
    {
        "case": "savgol_reference",
        "denoiser": "savgol",
        "fs_hz": None,
        "iqr_mode": "legacy_abs",
        "normalization_input": "abs",
        "role": "reference",
    },
)
CASE_BY_ID = {case["case"]: case for case in CASES}
OFFICIAL_CASE_IDS = tuple(CASE_BY_ID)
OFFICIAL_USER_COUNT = 3
OFFICIAL_EFFECT_IDS = (
    "sampling_effect_legacy",
    "sampling_effect_signed",
    "sign_effect_fs1000",
    "sign_effect_fs200",
    "sampling_x_sign_interaction",
    "signed_iqr_then_absnorm_vs_legacy_fs200",
    "signed_norm_vs_signed_iqr_absnorm",
    "remove_iqr_effect_signed_fs200",
    "signed_fs200_vs_savgol",
)

TRAINING_FIELDS = [
    "case",
    "role",
    "denoiser",
    "fs_hz",
    "iqr_mode",
    "normalization_input",
    "model",
    "model_seed",
    "split_seed",
    "epochs",
    "config_hash",
    "study_context_hash",
    "status",
    "best_val_acc",
    "test_acc",
    "train_size",
    "val_size",
    "test_size",
    "input_shape",
    "checkpoint",
    "predictions_file",
    "test_sample_manifest",
    "duration_sec",
    "error",
]

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#111111",
    "gray": "#777777",
}

_WORKER_RUNTIME: dict[str, Any] | None = None
_PREPROCESS_RUNTIME: dict[str, Any] | None = None


class Tee:
    def __init__(self, *files: Any) -> None:
        self.files = files

    def write(self, data: str) -> None:
        for file in self.files:
            file.write(data)

    def flush(self) -> None:
        for file in self.files:
            file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Server XRF55 directory (default: sdp_dataset/xrf55/wifi)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Result directory",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=list(CASE_BY_ID),
        default=list(CASE_BY_ID),
        help="Cases to run; default is the complete seven-case experiment",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Model initialization/DataLoader seeds",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--user-count",
        type=int,
        default=3,
        help="Use the first N distinct user IDs after numeric sorting",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--gpu",
        default="1",
        help="Physical CUDA device exposed to this script; use '' for default",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip only successful case/seed rows with the same config hash",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Preprocess selected cases and write stage diagnostics; do not train",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate aggregate tables/figures from current result CSV",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic pure-numpy checks only; no dataset access",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths/cases/settings and exit before importing WSDP or torch",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(
    path: Path,
    row: dict[str, Any],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            existing_reader = csv.DictReader(handle)
            existing_fields = list(existing_reader.fieldnames or [])
            existing_rows = list(existing_reader)
        if existing_fields != fields:
            migration_path = path.with_name(f"{path.name}.schema_migration.tmp")
            with migration_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fields,
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(existing_rows)
            os.replace(migration_path, path)
    header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if header:
            writer.writeheader()
        writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def audit_result_artifacts(
    row: dict[str, Any],
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate that a nominally successful result is complete and readable."""
    reasons: list[str] = []
    raw_checkpoint = str(row.get("checkpoint", "")).strip()
    raw_predictions = str(row.get("predictions_file", "")).strip()
    raw_manifest = str(row.get("test_sample_manifest", "")).strip()
    checkpoint = Path(raw_checkpoint) if raw_checkpoint else None
    predictions_file = Path(raw_predictions) if raw_predictions else None
    manifest = Path(raw_manifest) if raw_manifest else None

    if checkpoint is None or not checkpoint.is_file():
        reasons.append("checkpoint missing")
    elif checkpoint.stat().st_size <= 0:
        reasons.append("checkpoint is empty")
    elif runtime is not None and "torch" in runtime:
        try:
            try:
                loaded_checkpoint = runtime["torch"].load(
                    checkpoint,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                loaded_checkpoint = runtime["torch"].load(
                    checkpoint,
                    map_location="cpu",
                )
            del loaded_checkpoint
        except Exception as error:
            reasons.append(
                f"checkpoint is not loadable: {type(error).__name__}: {error}"
            )

    manifest_count: int | None = None
    if manifest is None or not manifest.is_file():
        reasons.append("test_sample_manifest missing")
    elif manifest.stat().st_size <= 0:
        reasons.append("test_sample_manifest is empty")
    else:
        try:
            manifest_count = len(read_csv(manifest))
            if manifest_count <= 0:
                reasons.append("test_sample_manifest has no data rows")
        except Exception as error:
            reasons.append(
                f"test_sample_manifest is not readable: {type(error).__name__}: {error}"
            )

    prediction_count: int | None = None
    if predictions_file is None or not predictions_file.is_file():
        reasons.append("predictions artifact missing")
    elif predictions_file.stat().st_size <= 0:
        reasons.append("predictions artifact is empty")
    else:
        try:
            with np.load(predictions_file, allow_pickle=False) as payload:
                if "predictions" not in payload or "targets" not in payload:
                    raise ValueError("npz must contain predictions and targets arrays")
                predictions = np.asarray(payload["predictions"])
                targets = np.asarray(payload["targets"])
            if predictions.ndim != 1 or targets.ndim != 1:
                reasons.append("predictions and targets must both be 1-D")
            elif len(predictions) != len(targets):
                reasons.append("predictions and targets have different lengths")
            else:
                prediction_count = len(predictions)
                if prediction_count <= 0:
                    reasons.append("predictions and targets are empty")
                if manifest_count is not None and prediction_count != manifest_count:
                    reasons.append(
                        "prediction length does not match manifest rows: "
                        f"{prediction_count} != {manifest_count}"
                    )
        except Exception as error:
            reasons.append(
                f"predictions artifact is not loadable: {type(error).__name__}: {error}"
            )

    return {
        "valid": not reasons,
        "case": row.get("case"),
        "model_seed": row.get("model_seed"),
        "config_hash": row.get("config_hash"),
        "study_context_hash": row.get("study_context_hash"),
        "checkpoint": raw_checkpoint,
        "predictions_file": raw_predictions,
        "test_sample_manifest": raw_manifest,
        "manifest_rows": manifest_count,
        "prediction_rows": prediction_count,
        "checkpoint_load_checked": runtime is not None and "torch" in runtime,
        "reasons": reasons,
    }


def import_runtime(gpu: str = "1") -> dict[str, Any]:
    """Import the repository training runtime only for a real server run."""
    global _WORKER_RUNTIME
    if _WORKER_RUNTIME is not None:
        return _WORKER_RUNTIME
    if not WSDP_SRC.is_dir():
        raise FileNotFoundError(f"WSDP source directory not found: {WSDP_SRC}")
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "wsdp_bandpass_sign_mpl"),
    )
    if str(WSDP_SRC) not in sys.path:
        sys.path.insert(0, str(WSDP_SRC))
    sys.modules.setdefault("kagglehub", types.ModuleType("kagglehub"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader
    from wsdp import readers
    from wsdp.algorithms.amplitude import remove_outliers
    from wsdp.algorithms.denoising_butterworth import (
        butterworth_bandpass,
        savgol_denoise,
    )
    from wsdp.algorithms.interpolation import interpolate_grid
    from wsdp.core import _create_data_split, _evaluate_model
    from wsdp.datasets import CSIDataset
    from wsdp.models import create_model
    from wsdp.processors.base_processor import (
        _parse_file_info_from_filename,
        _selector,
    )
    from wsdp.utils import (
        load_params,
        resize_csi_to_fixed_length,
        train_model,
    )

    _WORKER_RUNTIME = {
        "plt": plt,
        "torch": torch,
        "nn": nn,
        "ReduceLROnPlateau": ReduceLROnPlateau,
        "DataLoader": DataLoader,
        "readers": readers,
        "remove_outliers": remove_outliers,
        "butterworth_bandpass": butterworth_bandpass,
        "savgol_denoise": savgol_denoise,
        "interpolate_grid": interpolate_grid,
        "_create_data_split": _create_data_split,
        "_evaluate_model": _evaluate_model,
        "CSIDataset": CSIDataset,
        "create_model": create_model,
        "_parse_file_info_from_filename": _parse_file_info_from_filename,
        "_selector": _selector,
        "load_params": load_params,
        "resize_csi_to_fixed_length": resize_csi_to_fixed_length,
        "train_model": train_model,
    }
    return _WORKER_RUNTIME


def import_preprocess_runtime() -> dict[str, Any]:
    """Import reader and signal algorithms without importing torch in workers."""
    global _PREPROCESS_RUNTIME
    if _PREPROCESS_RUNTIME is not None:
        return _PREPROCESS_RUNTIME
    if _WORKER_RUNTIME is not None:
        _PREPROCESS_RUNTIME = {
            key: _WORKER_RUNTIME[key]
            for key in (
                "readers",
                "remove_outliers",
                "butterworth_bandpass",
                "savgol_denoise",
                "interpolate_grid",
            )
        }
        return _PREPROCESS_RUNTIME
    if not WSDP_SRC.is_dir():
        raise FileNotFoundError(f"WSDP source directory not found: {WSDP_SRC}")
    if str(WSDP_SRC) not in sys.path:
        sys.path.insert(0, str(WSDP_SRC))
    if "wsdp" not in sys.modules:
        package = types.ModuleType("wsdp")
        package.__path__ = [str(WSDP_SRC / "wsdp")]
        package.__package__ = "wsdp"
        sys.modules["wsdp"] = package
    if "wsdp.algorithms" not in sys.modules:
        package = types.ModuleType("wsdp.algorithms")
        package.__path__ = [str(WSDP_SRC / "wsdp" / "algorithms")]
        package.__package__ = "wsdp.algorithms"
        sys.modules["wsdp.algorithms"] = package

    from wsdp import readers
    from wsdp.algorithms.amplitude import remove_outliers
    from wsdp.algorithms.denoising_butterworth import (
        butterworth_bandpass,
        savgol_denoise,
    )
    from wsdp.algorithms.interpolation import interpolate_grid

    _PREPROCESS_RUNTIME = {
        "readers": readers,
        "remove_outliers": remove_outliers,
        "butterworth_bandpass": butterworth_bandpass,
        "savgol_denoise": savgol_denoise,
        "interpolate_grid": interpolate_grid,
    }
    return _PREPROCESS_RUNTIME


def validate_case_definitions() -> None:
    ids = [case["case"] for case in CASES]
    if len(ids) != len(set(ids)):
        raise AssertionError("Case IDs are not unique")
    matrix = {
        (float(case["fs_hz"]), case["normalization_input"])
        for case in CASES
        if case["role"] == "main_2x2"
    }
    expected = {
        (1000.0, "abs"),
        (1000.0, "signed"),
        (200.0, "abs"),
        (200.0, "signed"),
    }
    if matrix != expected:
        raise AssertionError(f"Invalid 2x2 matrix: {matrix}")


def parse_positive_unique(values: Iterable[int], name: str) -> list[int]:
    result = sorted(set(int(value) for value in values))
    if not result or result[0] < 1:
        raise ValueError(f"{name} must contain positive integers")
    return result


def legacy_iqr_numpy(values: np.ndarray, factor: float = IQR_FACTOR) -> np.ndarray:
    amplitude = np.abs(values)
    q1 = np.percentile(amplitude, 25, axis=0, keepdims=True)
    q3 = np.percentile(amplitude, 75, axis=0, keepdims=True)
    iqr = q3 - q1
    return np.clip(amplitude, q1 - factor * iqr, q3 + factor * iqr)


def restore_sign(filtered: np.ndarray, clipped_magnitude: np.ndarray) -> np.ndarray:
    signed = np.where(np.signbit(filtered), -clipped_magnitude, clipped_magnitude)
    if not np.allclose(
        np.abs(signed),
        clipped_magnitude,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ):
        raise AssertionError("signed-IQR magnitude equivalence failed")
    return signed


def zscore_three_way(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    use_abs: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train_values = np.abs(train) if use_abs else np.asarray(train)
    val_values = np.abs(val) if use_abs else np.asarray(val)
    test_values = np.abs(test) if use_abs else np.asarray(test)
    axes = (0, 1) if train_values.ndim >= 2 else (0,)
    mean = np.mean(train_values, axis=axes, keepdims=True)
    std = np.std(train_values, axis=axes, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)
    normalized = (
        (train_values - mean) / std,
        (val_values - mean) / std,
        (test_values - mean) / std,
    )
    metadata = {
        "use_abs_before_zscore": use_abs,
        "stat_axes": list(axes),
        "mean_shape": list(mean.shape),
        "std_shape": list(std.shape),
        "train_mean_min": float(np.min(mean)),
        "train_mean_max": float(np.max(mean)),
        "train_std_min": float(np.min(std)),
        "train_std_max": float(np.max(std)),
    }
    return *normalized, metadata


def synthetic_self_test() -> None:
    """Pure-numpy checks; never imports the dataset reader or torch."""
    assert DEFAULT_OUTPUT_DIR == (
        REPO_ROOT / "ablation" / "bandpass_server_results" / "sign_ablation"
    )
    validate_case_definitions()
    values = np.asarray(
        [
            [[-8.0, -0.0], [1.0, 2.0]],
            [[-2.0, 0.0], [2.0, 3.0]],
            [[-1.0, -3.0], [3.0, 4.0]],
            [[9.0, 4.0], [4.0, 100.0]],
        ]
    )
    legacy = legacy_iqr_numpy(values)
    signed = restore_sign(values, legacy)
    assert np.all(legacy >= 0)
    assert np.allclose(np.abs(signed), legacy)
    assert np.any(signed < 0)
    assert np.signbit(signed[0, 0, 1])

    train = np.arange(24.0).reshape(2, 3, 4) - 8.0
    val = np.ones_like(train) * 10.0
    test = np.ones_like(train) * -20.0
    signed_result = zscore_three_way(train, val, test, use_abs=False)
    abs_result = zscore_three_way(train, val, test, use_abs=True)
    assert not np.allclose(signed_result[0], abs_result[0])
    altered_val = val * 100000.0
    repeated = zscore_three_way(train, altered_val, test, use_abs=False)
    assert np.allclose(signed_result[0], repeated[0])

    repetitions = np.arange(1, 21)
    assert int(np.sum((repetitions >= 1) & (repetitions <= 12))) == 12
    assert int(np.sum((repetitions >= 13) & (repetitions <= 16))) == 4
    assert int(np.sum((repetitions >= 17) & (repetitions <= 20))) == 4
    synthetic_grid = {
        (user, action, repetition)
        for user in (1, 2, 3)
        for action in EXPECTED_ACTION_IDS
        for repetition in range(1, 21)
    }
    assert len(synthetic_grid) == 3 * 55 * 20
    assert set(DEFAULT_SEEDS) == {42, 49, 514, 654, 886}

    positive = np.asarray([1.0, 2.0, 3.0])
    negative = -positive
    assert np.all(np.diff(positive) > 0)
    assert np.all(np.diff(negative) < 0)

    with tempfile.TemporaryDirectory(prefix="bandpass_completion_selftest_") as tmp:
        root = Path(tmp)
        study_hash = "synthetic-official-study"
        hashes = {
            case_id: f"synthetic-{index:02d}"
            for index, case_id in enumerate(OFFICIAL_CASE_IDS)
        }
        rows: list[dict[str, Any]] = []
        prediction_paths: dict[tuple[str, int], Path] = {}
        checkpoint_paths: dict[tuple[str, int], Path] = {}
        for case in CASES:
            case_id = case["case"]
            manifest = root / case_id / "test_sample_manifest.csv"
            write_csv(
                manifest,
                [
                    {"sample_index": 0, "official_split": "test"},
                    {"sample_index": 1, "official_split": "test"},
                ],
            )
            for seed_index, seed in enumerate(DEFAULT_SEEDS):
                run_dir = root / "runs" / case_id / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = run_dir / "best_checkpoint.pth"
                checkpoint.write_bytes(b"synthetic non-empty checkpoint")
                checkpoint_paths[(case_id, seed)] = checkpoint
                predictions_file = run_dir / "test_predictions.npz"
                np.savez_compressed(
                    predictions_file,
                    predictions=np.asarray([0, 1]),
                    targets=np.asarray([0, 1]),
                )
                prediction_paths[(case_id, seed)] = predictions_file
                rows.append(
                    {
                        **case,
                        "model": MODEL_NAME,
                        "model_seed": seed,
                        "split_seed": DEFAULT_SPLIT_SEED,
                        "epochs": DEFAULT_EPOCHS,
                        "config_hash": hashes[case_id],
                        "study_context_hash": study_hash,
                        "status": "ok",
                        "best_val_acc": 0.5 + 0.001 * seed_index,
                        "test_acc": 0.4 + 0.001 * seed_index,
                        "checkpoint": str(checkpoint),
                        "predictions_file": str(predictions_file),
                        "test_sample_manifest": str(manifest),
                    }
                )
        training_path = root / "training_summary.csv"
        write_csv(training_path, rows)
        settings = {
            "case_hashes": hashes,
            "case_study_context_hashes": {
                case_id: study_hash for case_id in OFFICIAL_CASE_IDS
            },
            "study_context_hash": study_hash,
            "epochs": DEFAULT_EPOCHS,
            "user_count": OFFICIAL_USER_COUNT,
            "selected_user_ids": [1, 2, 3],
            "effective_training_params": {"epochs": DEFAULT_EPOCHS},
        }
        aggregates, paired, successful, artifact_audit = aggregate_results(
            training_path,
            hashes,
            root / "official",
            study_context_hash=study_hash,
        )
        official = evaluate_official_completion(
            settings,
            hashes,
            successful,
            paired,
            artifact_audit,
        )
        assert official["official_complete"]

        smoke_cases = set(OFFICIAL_CASE_IDS[:2])
        smoke_hashes = {
            case_id: case_hash
            for case_id, case_hash in hashes.items()
            if case_id in smoke_cases
        }
        smoke_settings = {
            **settings,
            "case_hashes": smoke_hashes,
            "case_study_context_hashes": {
                case_id: study_hash for case_id in smoke_cases
            },
            "epochs": 1,
            "user_count": 1,
            "selected_user_ids": [1],
            "effective_training_params": {"epochs": 1},
        }
        smoke_audit = {
            **artifact_audit,
            "valid_result_keys": [
                item
                for item in artifact_audit["valid_result_keys"]
                if item["case"] in smoke_cases
            ],
        }
        smoke = evaluate_official_completion(
            smoke_settings,
            smoke_hashes,
            [row for row in successful if row["case"] in smoke_cases],
            paired,
            smoke_audit,
        )
        assert not smoke["official_complete"] and smoke["status"] == "INCOMPLETE"
        smoke_report_dir = root / "smoke_report"
        write_plain_language_report(
            aggregates,
            paired,
            smoke_report_dir,
            smoke,
        )
        smoke_report = (smoke_report_dir / "report_summary.md").read_text(
            encoding="utf-8"
        )
        assert "# INCOMPLETE" in smoke_report
        assert "禁止据此下强因果结论" in smoke_report
        assert "强烈支持" not in smoke_report

        missing_case = OFFICIAL_CASE_IDS[-1]
        missing_hashes = {
            case_id: case_hash
            for case_id, case_hash in hashes.items()
            if case_id != missing_case
        }
        missing_audit = {
            **artifact_audit,
            "valid_result_keys": [
                item
                for item in artifact_audit["valid_result_keys"]
                if item["case"] != missing_case
            ],
        }
        missing = evaluate_official_completion(
            {
                **settings,
                "case_hashes": missing_hashes,
                "case_study_context_hashes": {
                    case_id: study_hash for case_id in missing_hashes
                },
            },
            missing_hashes,
            [row for row in successful if row["case"] != missing_case],
            paired,
            missing_audit,
        )
        assert not missing["official_complete"]

        broken_key = (OFFICIAL_CASE_IDS[0], DEFAULT_SEEDS[0])
        np.savez_compressed(
            prediction_paths[broken_key],
            predictions=np.asarray([0, 1]),
            targets=np.asarray([0]),
        )
        empty_checkpoint_key = (OFFICIAL_CASE_IDS[1], DEFAULT_SEEDS[1])
        checkpoint_paths[empty_checkpoint_key].write_bytes(b"")
        (
            _,
            bad_paired,
            bad_successful,
            bad_artifact_audit,
        ) = aggregate_results(
            training_path,
            hashes,
            root / "bad_artifact",
            study_context_hash=study_hash,
        )
        bad_artifact = evaluate_official_completion(
            settings,
            hashes,
            bad_successful,
            bad_paired,
            bad_artifact_audit,
        )
        assert not bad_artifact["official_complete"]
        assert any(
            "different lengths" in reason
            for record in bad_artifact_audit["records"]
            for reason in record["reasons"]
        )
        assert any(
            "checkpoint is empty" in reason
            for record in bad_artifact_audit["records"]
            for reason in record["reasons"]
        )
    print(
        "Synthetic self-test passed: complete 2x2 matrix, signed-IQR "
        "magnitude equivalence, train-only z-score, 12/4/4 split, and "
        "rise/fall=sign(diff), plus official/smoke/missing-case/bad-artifact "
        "completion gates."
    )


def source_equivalence_gate(runtime: dict[str, Any]) -> dict[str, Any]:
    """Check custom branch logic against the exact repository functions."""
    rng = np.random.default_rng(20260725)
    raw = 3.0 + rng.normal(size=(80, 30, 2))
    bp = runtime["butterworth_bandpass"](
        raw,
        order=4,
        low_freq=0.5,
        high_freq=50.0,
        fs=200.0,
    )
    source_legacy = runtime["remove_outliers"](
        bp,
        method="iqr",
        factor=IQR_FACTOR,
    )
    numpy_legacy = legacy_iqr_numpy(bp)
    if not np.allclose(source_legacy, numpy_legacy, rtol=1e-10, atol=1e-12):
        raise AssertionError("Local legacy-IQR formula differs from WSDP source")
    signed = restore_sign(bp, source_legacy)

    synthetic = rng.normal(size=(20, 8, 3, 2))
    labels = np.arange(20) % 5
    repetitions = np.arange(1, 21)
    source_split = runtime["_create_data_split"](
        synthetic,
        labels,
        repetitions,
        test_split=0.3,
        val_split=0.5,
        seed=42,
        use_simple_split=False,
        dataset=DATASET_NAME,
        pipeline_steps={"normalize": {"method": "z-score"}},
    )
    raw_split = runtime["_create_data_split"](
        synthetic,
        labels,
        repetitions,
        test_split=0.3,
        val_split=0.5,
        seed=42,
        use_simple_split=False,
        dataset=DATASET_NAME,
        pipeline_steps={},
    )
    manual = zscore_three_way(
        raw_split[0],
        raw_split[1],
        raw_split[2],
        use_abs=True,
    )
    for source_array, manual_array in zip(source_split[:3], manual[:3]):
        if not np.allclose(source_array, manual_array, rtol=1e-10, atol=1e-12):
            raise AssertionError("Manual legacy normalization differs from core.py")

    dataset = runtime["CSIDataset"](
        signed[np.newaxis, ...],
        np.asarray([0]),
        dataset_name=DATASET_NAME,
        pipeline_steps={"normalize": {"method": "z-score"}},
    )
    preserved = dataset.data_list.detach().cpu().numpy()[0]
    if not np.allclose(preserved, signed.astype(np.float32)):
        raise AssertionError("CSIDataset did not preserve signed real XRF55 input")
    return {
        "status": "passed",
        "legacy_iqr_matches_source": True,
        "signed_iqr_abs_matches_legacy": True,
        "manual_legacy_normalization_matches_core": True,
        "csi_dataset_preserves_signed_real_values": True,
    }


def parse_user_id(path: Path) -> int | None:
    match = re.search(r"(\d+)_(\d+)_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def discover_selected_files(
    data_path: Path,
    user_count: int,
) -> tuple[list[Path], list[int], str]:
    if not data_path.is_dir():
        raise FileNotFoundError(f"XRF55 data directory not found: {data_path}")
    candidates: list[tuple[Path, int]] = []
    for path in sorted(data_path.rglob("*")):
        if not path.is_file() or "truth" in path.name.lower():
            continue
        user_id = parse_user_id(path)
        if user_id is not None:
            candidates.append((path, user_id))
    users = sorted({user_id for _, user_id in candidates})[:user_count]
    if len(users) < user_count:
        raise RuntimeError(
            f"Requested {user_count} users, found only {len(users)}: {users}"
        )
    files = [path for path, user_id in candidates if user_id in set(users)]
    if not files:
        raise RuntimeError(f"No selected XRF55 files under {data_path}")
    signature_rows = []
    for path in files:
        stat = path.stat()
        signature_rows.append(
            {
                "relative_path": str(path.relative_to(data_path)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return files, users, stable_hash(signature_rows)


def stack_real_record(record: Any) -> np.ndarray:
    frames = sorted(record.frames, key=lambda frame: frame.timestamp)
    if not frames:
        raise ValueError(f"No frames: {record.file_name}")
    array = np.stack([frame.csi_array for frame in frames], axis=0)
    if array.ndim == 2:
        array = np.expand_dims(array, -1)
    if array.ndim != 3 or array.shape[0] < 2:
        raise ValueError(
            f"Expected real (T,F,A), T>=2; got {array.shape}: {record.file_name}"
        )
    if np.iscomplexobj(array):
        maximum_imaginary = float(np.max(np.abs(np.imag(array))))
        scale = max(float(np.max(np.abs(array))), 1.0)
        if maximum_imaginary > 1e-10 * scale:
            raise ValueError(
                "Signed Bandpass ablation is undefined for genuinely complex "
                f"CSI: {record.file_name}, max|imag|={maximum_imaginary:.6g}. "
                "Use the legacy amplitude experiment for complex .dat data; "
                "do not silently call it a signed-value experiment."
            )
        array = np.real(array)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Non-finite raw values: {record.file_name}")
    return np.asarray(array)


def rate_negative(values: np.ndarray) -> float:
    return float(np.count_nonzero(values < 0) / max(values.size, 1))


def sign_agreement(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.median(np.abs(left))), 1.0)
    mask = (np.abs(left) > 1e-10 * scale) & (np.abs(right) > 1e-10 * scale)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.signbit(left[mask]) == np.signbit(right[mask])))


def slope_agreement(left: np.ndarray, right: np.ndarray) -> float:
    left_delta = np.diff(left, axis=0)
    right_delta = np.diff(right, axis=0)
    scale = max(float(np.median(np.abs(left_delta))), 1.0)
    mask = (np.abs(left_delta) > 1e-10 * scale) & (np.abs(right_delta) > 1e-10 * scale)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.signbit(left_delta[mask]) == np.signbit(right_delta[mask])))


def transform_real_array(
    raw: np.ndarray,
    case: dict[str, Any],
    runtime: dict[str, Any],
    file_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one controlled branch before fixed-length resize."""
    if case["denoiser"] == "bandpass":
        denoised = runtime["butterworth_bandpass"](
            raw,
            order=4,
            low_freq=0.5,
            high_freq=50.0,
            fs=float(case["fs_hz"]),
        )
    elif case["denoiser"] == "savgol":
        denoised = runtime["savgol_denoise"](
            raw,
            window_length=7,
            polyorder=3,
        )
    else:
        raise ValueError(f"Unknown denoiser: {case['denoiser']}")

    legacy_magnitude: np.ndarray | None = None
    if case["iqr_mode"] == "legacy_abs":
        outlier_output = runtime["remove_outliers"](
            denoised,
            method="iqr",
            factor=IQR_FACTOR,
        )
    elif case["iqr_mode"] == "signed":
        legacy_magnitude = runtime["remove_outliers"](
            denoised,
            method="iqr",
            factor=IQR_FACTOR,
        )
        outlier_output = restore_sign(denoised, legacy_magnitude)
    elif case["iqr_mode"] == "none":
        outlier_output = denoised.copy()
    else:
        raise ValueError(f"Unknown IQR mode: {case['iqr_mode']}")

    interpolated = runtime["interpolate_grid"](
        outlier_output,
        target_K=TARGET_SUBCARRIERS,
        method="cubic",
        dataset=DATASET_NAME,
    )
    if not np.all(np.isfinite(interpolated)):
        raise ValueError(f"Non-finite processed values: {file_name}")
    diagnostics = {
        "file_name": file_name,
        "frames": int(raw.shape[0]),
        "input_dtype": str(raw.dtype),
        "raw_negative_rate": rate_negative(raw),
        "denoised_negative_rate": rate_negative(denoised),
        "after_iqr_negative_rate": rate_negative(outlier_output),
        "after_cubic_negative_rate": rate_negative(interpolated),
        "denoise_to_iqr_sign_agreement": sign_agreement(
            denoised,
            outlier_output,
        ),
        "denoise_to_iqr_slope_agreement": slope_agreement(
            denoised,
            outlier_output,
        ),
        "denoised_zero_crossing_rate": float(
            np.mean(np.signbit(denoised[1:]) != np.signbit(denoised[:-1]))
        ),
        "iqr_zero_crossing_rate": float(
            np.mean(np.signbit(outlier_output[1:]) != np.signbit(outlier_output[:-1]))
        ),
        "signed_iqr_magnitude_gate": (
            ""
            if legacy_magnitude is None
            else bool(
                np.allclose(
                    np.abs(outlier_output),
                    legacy_magnitude,
                    rtol=1e-10,
                    atol=1e-12,
                )
            )
        ),
    }
    return interpolated, diagnostics


def process_xrf_file_worker(
    path_string: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    """Read and transform one file inside a CPU-only spawn worker."""
    try:
        runtime = import_preprocess_runtime()
        path = Path(path_string)
        reader = runtime["readers"].get_reader_class(DATASET_NAME)()
        if not reader.sniff(str(path)):
            return {"status": "format_mismatch", "file": path_string}
        loaded = reader.read_file(str(path))
        records = loaded if isinstance(loaded, list) else [loaded]
        records = [record for record in records if record is not None]
        if len(records) != 1:
            raise RuntimeError(
                "XRF55 reader must return exactly one record per selected "
                f"file; got {len(records)} for {path}"
            )
        record = records[0]
        match = re.search(
            r"(\d+)_(\d+)_(\d+)",
            Path(str(record.file_name)).stem,
        )
        if match is None:
            raise ValueError(f"Invalid XRF55 file name: {record.file_name}")
        label = int(match.group(2))
        repetition = int(match.group(3))
        raw = stack_real_record(record)
        interpolated, diagnostics = transform_real_array(
            raw,
            case,
            runtime,
            str(record.file_name),
        )
        resized = np.zeros(
            (
                PADDING_LENGTH,
                interpolated.shape[1],
                interpolated.shape[2],
            ),
            dtype=interpolated.dtype,
        )
        copy_length = min(PADDING_LENGTH, interpolated.shape[0])
        resized[:copy_length] = interpolated[:copy_length]
        return {
            "status": "ok",
            "file": path_string,
            "array": resized,
            "label": label,
            "repetition": repetition,
            "diagnostics": {"case": case["case"], **diagnostics},
        }
    except Exception:
        error = traceback.format_exc()
        return {
            "status": "error",
            "file": path_string,
            "error": error,
            "fatal_complex": (
                "Signed Bandpass ablation is undefined for genuinely complex" in error
            ),
        }


def process_record_worker(
    record: Any,
    case: dict[str, Any],
) -> tuple[np.ndarray | None, Any, Any, dict[str, Any] | None]:
    runtime = import_runtime(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parsed = runtime["_parse_file_info_from_filename"](
        record.file_name,
        DATASET_NAME,
    )
    if parsed is None:
        return None, None, None, None
    label, repetition = runtime["_selector"](parsed, DATASET_NAME)
    raw = stack_real_record(record)

    if case["denoiser"] == "bandpass":
        denoised = runtime["butterworth_bandpass"](
            raw,
            order=4,
            low_freq=0.5,
            high_freq=50.0,
            fs=float(case["fs_hz"]),
        )
    elif case["denoiser"] == "savgol":
        denoised = runtime["savgol_denoise"](
            raw,
            window_length=7,
            polyorder=3,
        )
    else:
        raise ValueError(f"Unknown denoiser: {case['denoiser']}")

    legacy_magnitude: np.ndarray | None = None
    if case["iqr_mode"] == "legacy_abs":
        outlier_output = runtime["remove_outliers"](
            denoised,
            method="iqr",
            factor=IQR_FACTOR,
        )
    elif case["iqr_mode"] == "signed":
        legacy_magnitude = runtime["remove_outliers"](
            denoised,
            method="iqr",
            factor=IQR_FACTOR,
        )
        outlier_output = restore_sign(denoised, legacy_magnitude)
    elif case["iqr_mode"] == "none":
        outlier_output = denoised.copy()
    else:
        raise ValueError(f"Unknown IQR mode: {case['iqr_mode']}")

    interpolated = runtime["interpolate_grid"](
        outlier_output,
        target_K=TARGET_SUBCARRIERS,
        method="cubic",
        dataset=DATASET_NAME,
    )
    if not np.all(np.isfinite(interpolated)):
        raise ValueError(f"Non-finite processed values: {record.file_name}")
    diagnostics = {
        "file_name": str(record.file_name),
        "frames": int(raw.shape[0]),
        "raw_negative_rate": rate_negative(raw),
        "denoised_negative_rate": rate_negative(denoised),
        "after_iqr_negative_rate": rate_negative(outlier_output),
        "after_cubic_negative_rate": rate_negative(interpolated),
        "denoise_to_iqr_sign_agreement": sign_agreement(
            denoised,
            outlier_output,
        ),
        "denoise_to_iqr_slope_agreement": slope_agreement(
            denoised,
            outlier_output,
        ),
        "denoised_zero_crossing_rate": float(
            np.mean(np.signbit(denoised[1:]) != np.signbit(denoised[:-1]))
        ),
        "iqr_zero_crossing_rate": float(
            np.mean(np.signbit(outlier_output[1:]) != np.signbit(outlier_output[:-1]))
        ),
        "signed_iqr_magnitude_gate": (
            ""
            if legacy_magnitude is None
            else bool(
                np.allclose(
                    np.abs(outlier_output),
                    legacy_magnitude,
                    rtol=1e-10,
                    atol=1e-12,
                )
            )
        ),
    }
    return interpolated, label, repetition, diagnostics


def preprocess_case(
    runtime: dict[str, Any],
    records: list[Any],
    case: dict[str, Any],
    workers: int,
    case_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Any], dict[str, Any]]:
    arrays: list[np.ndarray] = []
    labels_raw: list[Any] = []
    repetitions: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    worker = partial(process_record_worker, case=case)

    if workers == 1:
        results = map(worker, records)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(worker, records, chunksize=1)
    try:
        for index, (array, label, repetition, row) in enumerate(results, start=1):
            if array is not None:
                arrays.append(array)
                labels_raw.append(label)
                repetitions.append(int(repetition))
                diagnostics.append({"case": case["case"], **(row or {})})
            if index % 100 == 0 or index == len(records):
                print(f"Preprocess {case['case']}: {index}/{len(records)}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    if not arrays:
        raise RuntimeError(f"No valid samples for {case['case']}")
    invalid_repetitions = sorted(
        {value for value in repetitions if value < 1 or value > 20}
    )
    if invalid_repetitions:
        raise ValueError(f"Invalid repetition IDs: {invalid_repetitions}")
    if set(repetitions) != set(range(1, 21)):
        missing = sorted(set(range(1, 21)) - set(repetitions))
        raise RuntimeError(
            "Official XRF55 experiment requires repetitions 1..20; "
            f"missing repetition IDs: {missing}"
        )
    selected_users = sorted(
        {
            user
            for row in diagnostics
            if (user := parse_user_id(Path(str(row["file_name"])))) is not None
        }
    )
    grid_users: dict[tuple[Any, int], set[int]] = defaultdict(set)
    for label, repetition, row in zip(labels_raw, repetitions, diagnostics):
        user = parse_user_id(Path(str(row["file_name"])))
        if user is not None:
            grid_users[(label, int(repetition))].add(user)
    missing_cells = [
        (str(label), repetition)
        for label in sorted(set(labels_raw))
        for repetition in range(1, 21)
        if grid_users.get((label, repetition), set()) != set(selected_users)
    ]
    if missing_cells:
        raise RuntimeError(
            "Incomplete user/action/repetition grid; first missing cells: "
            f"{missing_cells[:10]}"
        )

    resized = runtime["resize_csi_to_fixed_length"](
        arrays,
        target_length=PADDING_LENGTH,
    )
    unique_labels = sorted(set(labels_raw))
    label_map = {label: index for index, label in enumerate(unique_labels)}
    processed = np.asarray(resized)
    labels = np.asarray(
        [label_map[label] for label in labels_raw],
        dtype=np.int64,
    )
    groups = np.asarray(repetitions, dtype=np.int64)
    write_csv(case_dir / "preprocessing_diagnostics_per_sample.csv", diagnostics)
    sample_manifest = [
        {
            "sample_index": index,
            "file_name": diagnostics[index].get("file_name", ""),
            "mapped_label": int(labels[index]),
            "raw_label": str(labels_raw[index]),
            "repetition": int(groups[index]),
            "official_split": (
                "train"
                if 1 <= int(groups[index]) <= 12
                else "validation"
                if 13 <= int(groups[index]) <= 16
                else "test"
            ),
        }
        for index in range(len(labels))
    ]
    write_csv(case_dir / "sample_manifest.csv", sample_manifest)
    write_csv(
        case_dir / "test_sample_manifest.csv",
        [row for row in sample_manifest if row["official_split"] == "test"],
    )
    metadata = {
        "case": case,
        "samples": len(processed),
        "sample_shape": list(processed.shape[1:]),
        "unique_raw_labels": [str(value) for value in unique_labels],
        "class_count": len(unique_labels),
        "repetition_counts": {
            str(key): int(value) for key, value in sorted(Counter(repetitions).items())
        },
        "selected_user_ids": selected_users,
        "complete_user_action_repetition_grid": True,
        "sample_order_hash": stable_hash(
            [
                (
                    row.get("file_name"),
                    int(labels[index]),
                    int(groups[index]),
                )
                for index, row in enumerate(diagnostics)
            ]
        ),
    }
    write_json(case_dir / "preprocessing_metadata.json", metadata)
    return processed, labels, groups, unique_labels, metadata


def preprocess_case_from_files(
    files: list[Path],
    case: dict[str, Any],
    workers: int,
    case_dir: Path,
    case_hash: str,
    dataset_signature: str,
    expected_users: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Any], dict[str, Any]]:
    """Stream file results into a dtype-preserving disk-backed numpy array."""
    cache_path = case_dir / "processed_source_dtype.npy"
    labels_path = case_dir / "labels.npy"
    groups_path = case_dir / "repetitions.npy"
    metadata_path = case_dir / "preprocessing_metadata.json"
    diagnostics_path = case_dir / "preprocessing_diagnostics_per_sample.csv"
    sample_manifest_path = case_dir / "sample_manifest.csv"
    test_manifest_path = case_dir / "test_sample_manifest.csv"
    if all(
        path.exists()
        for path in (
            cache_path,
            labels_path,
            groups_path,
            metadata_path,
            diagnostics_path,
            sample_manifest_path,
            test_manifest_path,
        )
    ):
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            cached_metadata.get("config_hash") == case_hash
            and cached_metadata.get("dataset_signature") == dataset_signature
            and cached_metadata.get("cache_complete") is True
        ):
            full_cache = np.load(cache_path, mmap_mode="r")
            valid_count = int(cached_metadata["samples"])
            labels = np.load(labels_path)
            groups = np.load(groups_path)
            if not (
                valid_count == len(labels) == len(groups)
                and valid_count <= len(full_cache)
            ):
                raise RuntimeError("Preprocessing cache metadata is inconsistent")
            print(f"Reuse preprocessing cache: {case['case']} ({valid_count} samples)")
            return (
                full_cache[:valid_count],
                labels,
                groups,
                list(cached_metadata["unique_raw_labels"]),
                cached_metadata,
            )

    worker = partial(process_xrf_file_worker, case=case)
    if workers == 1:
        results = map(worker, map(str, files))
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
        )
        results = executor.map(worker, map(str, files), chunksize=1)

    memmap: np.memmap | None = None
    diagnostics: list[dict[str, Any]] = []
    raw_labels: list[int] = []
    repetitions: list[int] = []
    format_mismatch = 0
    failures: list[dict[str, str]] = []
    fatal_failure = False
    try:
        for file_index, result in enumerate(results, start=1):
            status = result["status"]
            if status == "format_mismatch":
                format_mismatch += 1
            elif status == "error":
                failures.append(
                    {
                        "file": result["file"],
                        "error": result["error"],
                    }
                )
                if result.get("fatal_complex"):
                    fatal_failure = True
                    break
            else:
                array = np.asarray(result["array"])
                if memmap is None:
                    memmap = np.lib.format.open_memmap(
                        cache_path,
                        mode="w+",
                        dtype=array.dtype,
                        shape=(len(files), *array.shape),
                    )
                if array.shape != memmap.shape[1:]:
                    raise RuntimeError(
                        f"Inconsistent sample shape {array.shape}; "
                        f"expected {memmap.shape[1:]}"
                    )
                write_index = len(raw_labels)
                memmap[write_index] = array
                raw_labels.append(int(result["label"]))
                repetitions.append(int(result["repetition"]))
                diagnostics.append(result["diagnostics"])
            if file_index % 100 == 0 or file_index == len(files):
                print(f"Stream preprocess {case['case']}: {file_index}/{len(files)}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=fatal_failure)
    if failures:
        write_json(case_dir / "preprocessing_failures.json", failures)
        raise RuntimeError(
            f"{len(failures)} XRF55 file(s) failed in {case['case']}; "
            f"first failure:\n{failures[0]['error']}"
        )
    if memmap is None or not raw_labels:
        raise RuntimeError(f"No valid samples for {case['case']}")
    memmap.flush()

    if set(repetitions) != set(range(1, 21)):
        missing = sorted(set(range(1, 21)) - set(repetitions))
        raise RuntimeError(
            "Official XRF55 experiment requires repetitions 1..20; "
            f"missing repetition IDs: {missing}"
        )
    selected_users = sorted(
        {
            user
            for row in diagnostics
            if (user := parse_user_id(Path(str(row["file_name"])))) is not None
        }
    )
    if selected_users != sorted(expected_users):
        raise RuntimeError(
            f"Expected users {sorted(expected_users)}, decoded {selected_users}"
        )
    observed_grid: Counter[tuple[int, int, int]] = Counter()
    for label, repetition, row in zip(raw_labels, repetitions, diagnostics):
        user = parse_user_id(Path(str(row["file_name"])))
        if user is not None:
            observed_grid[(user, label, repetition)] += 1
    expected_grid = {
        (user, action, repetition)
        for user in expected_users
        for action in EXPECTED_ACTION_IDS
        for repetition in range(1, 21)
    }
    observed_keys = set(observed_grid)
    missing_cells = sorted(expected_grid - observed_keys)
    unexpected_cells = sorted(observed_keys - expected_grid)
    duplicate_cells = sorted(
        (key, count) for key, count in observed_grid.items() if count != 1
    )
    if missing_cells or unexpected_cells or duplicate_cells:
        raise RuntimeError(
            "XRF55 grid must contain exactly one file for every selected "
            "user × action(1..55) × repetition(1..20). "
            f"missing={missing_cells[:10]}, unexpected={unexpected_cells[:10]}, "
            f"duplicate_or_nonunit={duplicate_cells[:10]}"
        )

    unique_labels = sorted(set(raw_labels))
    label_map = {label: index for index, label in enumerate(unique_labels)}
    labels = np.asarray([label_map[label] for label in raw_labels], dtype=np.int64)
    groups = np.asarray(repetitions, dtype=np.int64)
    np.save(labels_path, labels)
    np.save(groups_path, groups)
    write_csv(diagnostics_path, diagnostics)
    sample_manifest = [
        {
            "sample_index": index,
            "file_name": diagnostics[index]["file_name"],
            "mapped_label": int(labels[index]),
            "raw_label": int(raw_labels[index]),
            "repetition": int(groups[index]),
            "official_split": (
                "train"
                if 1 <= int(groups[index]) <= 12
                else "validation"
                if 13 <= int(groups[index]) <= 16
                else "test"
            ),
        }
        for index in range(len(labels))
    ]
    write_csv(sample_manifest_path, sample_manifest)
    write_csv(
        test_manifest_path,
        [row for row in sample_manifest if row["official_split"] == "test"],
    )
    metadata = {
        "case": case,
        "config_hash": case_hash,
        "dataset_signature": dataset_signature,
        "cache_complete": True,
        "cache_file": str(cache_path),
        "cache_capacity": len(files),
        "samples": len(labels),
        "sample_shape": list(memmap.shape[1:]),
        "source_dtype": str(memmap.dtype),
        "unique_raw_labels": unique_labels,
        "class_count": len(unique_labels),
        "selected_user_ids": selected_users,
        "complete_user_action_repetition_grid": True,
        "expected_action_ids": list(EXPECTED_ACTION_IDS),
        "expected_samples": len(expected_grid),
        "format_mismatch_files": format_mismatch,
        "repetition_counts": {
            str(key): int(value) for key, value in sorted(Counter(repetitions).items())
        },
        "sample_order_hash": stable_hash(
            [
                (
                    diagnostics[index]["file_name"],
                    int(labels[index]),
                    int(groups[index]),
                )
                for index in range(len(labels))
            ]
        ),
        "memory_strategy": (
            "file paths sent to spawn workers; dtype-preserving results streamed "
            "into numpy open_memmap; raw CSIData records are not retained"
        ),
    }
    write_json(metadata_path, metadata)
    return memmap[: len(labels)], labels, groups, unique_labels, metadata


def split_and_normalize(
    runtime: dict[str, Any],
    processed: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    case: dict[str, Any],
    split_seed: int,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    common = {
        "test_split": 0.3,
        "val_split": 0.5,
        "seed": split_seed,
        "use_simple_split": False,
        "dataset": DATASET_NAME,
    }
    if case["normalization_input"] == "abs":
        split = runtime["_create_data_split"](
            processed,
            labels,
            groups,
            pipeline_steps={"normalize": {"method": "z-score"}},
            **common,
        )
        normalization_metadata = {
            "implementation": "exact wsdp.core._create_xrf55_repetition_split",
            "use_abs_before_zscore": True,
            "statistics": "training repetitions only, axes=(sample,time)",
        }
    else:
        raw_split = runtime["_create_data_split"](
            processed,
            labels,
            groups,
            pipeline_steps={},
            **common,
        )
        train, val, test, normalization_metadata = zscore_three_way(
            raw_split[0],
            raw_split[1],
            raw_split[2],
            use_abs=False,
        )
        split = (
            train,
            val,
            test,
            raw_split[3],
            raw_split[4],
            raw_split[5],
        )
        normalization_metadata["implementation"] = "manual signed train-only zscore"

    if any(len(array) == 0 for array in split[:3]):
        raise RuntimeError("Official XRF55 split produced an empty subset")
    split = tuple(
        np.asarray(array, dtype=np.float32) if index < 3 else np.asarray(array)
        for index, array in enumerate(split)
    )
    normalization_metadata.update(
        {
            "train_size": len(split[0]),
            "val_size": len(split[1]),
            "test_size": len(split[2]),
            "post_normalization_negative_rate": {
                "train": rate_negative(split[0]),
                "validation": rate_negative(split[1]),
                "test": rate_negative(split[2]),
            },
        }
    )
    return split, normalization_metadata


def set_seed(runtime: dict[str, Any], seed: int) -> None:
    torch = runtime["torch"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_loaders(
    runtime: dict[str, Any],
    split: tuple[np.ndarray, ...],
    batch_size: int,
    seed: int,
) -> tuple[Any, Any, Any]:
    dataset_pipeline = {"normalize": {"method": "z-score"}}
    train, val, test, train_y, val_y, test_y = split

    def make(data: np.ndarray, labels: np.ndarray, shuffle: bool) -> Any:
        generator = None
        if shuffle:
            generator = runtime["torch"].Generator()
            generator.manual_seed(seed)
        return runtime["DataLoader"](
            runtime["CSIDataset"](
                data,
                labels,
                dataset_name=DATASET_NAME,
                pipeline_steps=dataset_pipeline,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            generator=generator,
        )

    return (
        make(train, train_y, True),
        make(val, val_y, False),
        make(
            test,
            test_y,
            False,
        ),
    )


def save_history(path: Path, history: dict[str, list[Any]]) -> None:
    keys = list(history)
    count = max((len(history[key]) for key in keys), default=0)
    write_csv(
        path,
        [
            {
                "epoch": index + 1,
                **{
                    key: history[key][index] if index < len(history[key]) else ""
                    for key in keys
                },
            }
            for index in range(count)
        ],
    )


def train_one_seed(
    runtime: dict[str, Any],
    split: tuple[np.ndarray, ...],
    class_count: int,
    case: dict[str, Any],
    case_hash: str,
    study_context_hash: str,
    model_seed: int,
    split_seed: int,
    epochs: int,
    params: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    run_dir = (
        output_dir
        / "runs"
        / case["case"]
        / f"config_{case_hash}"
        / f"seed_{model_seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_checkpoint.pth"
    predictions_file = run_dir / "test_predictions.npz"
    test_sample_manifest = (
        output_dir
        / "cases"
        / case["case"]
        / f"config_{case_hash}"
        / "test_sample_manifest.csv"
    )
    started = time.time()
    row = {
        **case,
        "model": MODEL_NAME,
        "model_seed": model_seed,
        "split_seed": split_seed,
        "epochs": epochs,
        "config_hash": case_hash,
        "study_context_hash": study_context_hash,
        "status": "failed",
        "best_val_acc": "",
        "test_acc": "",
        "train_size": len(split[0]),
        "val_size": len(split[1]),
        "test_size": len(split[2]),
        "input_shape": "",
        "checkpoint": str(checkpoint),
        "predictions_file": str(predictions_file),
        "test_sample_manifest": str(test_sample_manifest),
        "duration_sec": "",
        "error": "",
    }
    # A failed rerun must never fall through to a checkpoint from an older run
    # with the same configuration.
    for stale_path in (checkpoint, predictions_file):
        if stale_path.exists():
            stale_path.unlink()
    with (run_dir / "train_process.txt").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            try:
                set_seed(runtime, model_seed)
                loaders = build_loaders(
                    runtime,
                    split,
                    int(params.get("batch", 32)),
                    model_seed,
                )
                input_shape = tuple(loaders[0].dataset.data_list.shape[1:])
                device = runtime["torch"].device(
                    "cuda" if runtime["torch"].cuda.is_available() else "cpu"
                )
                model = runtime["create_model"](
                    MODEL_NAME,
                    num_classes=class_count,
                    input_shape=input_shape,
                ).to(device)
                criterion = runtime["nn"].CrossEntropyLoss()
                optimizer = runtime["torch"].optim.AdamW(
                    model.parameters(),
                    lr=float(params.get("lr", 3e-4)),
                    weight_decay=float(params.get("wd", 1e-3)),
                )
                scheduler = runtime["ReduceLROnPlateau"](
                    optimizer,
                    mode="min",
                    factor=0.1,
                    patience=5,
                )
                print("=" * 78)
                print(f"case={case['case']}; seed={model_seed}; hash={case_hash}")
                print(f"input_shape={input_shape}; device={device}")
                print(json.dumps(case, ensure_ascii=False, indent=2))
                print("=" * 78)
                history = runtime["train_model"](
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    loaders[0],
                    loaders[1],
                    epochs,
                    device,
                    checkpoint,
                    PADDING_LENGTH,
                )
                save_history(run_dir / "training_history.csv", history)
                if not checkpoint.exists():
                    raise RuntimeError(f"Checkpoint was not created: {checkpoint}")
                try:
                    saved = runtime["torch"].load(
                        checkpoint,
                        map_location=device,
                        weights_only=False,
                    )
                except TypeError:
                    saved = runtime["torch"].load(checkpoint, map_location=device)
                model.load_state_dict(saved["model_state_dict"])
                predictions, targets, test_acc = runtime["_evaluate_model"](
                    model,
                    loaders[2],
                    device,
                )
                np.savez_compressed(
                    predictions_file,
                    predictions=np.asarray(predictions, dtype=np.int64),
                    targets=np.asarray(targets, dtype=np.int64),
                )
                row.update(
                    {
                        "status": "ok",
                        "best_val_acc": float(saved.get("best_val_acc", 0.0)) / 100.0,
                        "test_acc": float(test_acc),
                        "input_shape": json.dumps(input_shape),
                    }
                )
                print(
                    f"Completed: val={row['best_val_acc']:.4f}, "
                    f"test={row['test_acc']:.4f}"
                )
                del model, loaders
            except Exception:
                row["error"] = traceback.format_exc()
                print(row["error"])
    row["duration_sec"] = f"{time.time() - started:.2f}"
    gc.collect()
    if runtime["torch"].cuda.is_available():
        runtime["torch"].cuda.empty_cache()
    return row


def current_successes(
    path: Path,
    hashes: dict[str, str],
    runtime: dict[str, Any] | None = None,
    study_context_hash: str | None = None,
    audit_records: list[dict[str, Any]] | None = None,
) -> dict[tuple[str, int], dict[str, str]]:
    result: dict[tuple[str, int], dict[str, str]] = {}
    for row in read_csv(path):
        case_id = row.get("case", "")
        if not (
            row.get("status") == "ok"
            and case_id in hashes
            and row.get("config_hash") == hashes[case_id]
        ):
            continue
        artifact = audit_result_artifacts(row, runtime)
        if (
            study_context_hash is not None
            and row.get("study_context_hash") != study_context_hash
        ):
            artifact["reasons"].append(
                "result study_context_hash does not match current study"
            )
        try:
            model_seed = int(row["model_seed"])
            test_acc = float(row["test_acc"])
            best_val_acc = float(row["best_val_acc"])
            if not np.isfinite(test_acc) or not np.isfinite(best_val_acc):
                raise ValueError("accuracy is non-finite")
        except (KeyError, TypeError, ValueError) as error:
            artifact["reasons"].append(
                f"invalid result metadata: {type(error).__name__}: {error}"
            )
            model_seed = -1
        artifact["valid"] = not artifact["reasons"]
        if audit_records is not None:
            audit_records.append(artifact)
        if artifact["valid"]:
            result[(case_id, model_seed)] = row
    return result


def sync_case_results_to_global(
    output_dir: Path,
    hashes: dict[str, str],
    training_path: Path,
) -> int:
    """Recover valid per-case result rows into the global append-only CSV."""
    global_keys = {
        (
            row.get("case"),
            row.get("model_seed"),
            row.get("config_hash"),
            row.get("status"),
        )
        for row in read_csv(training_path)
    }
    recovered = 0
    for case_id, case_hash in hashes.items():
        case_results = (
            output_dir / "cases" / case_id / f"config_{case_hash}" / "case_results.csv"
        )
        for saved_row in read_csv(case_results):
            if (
                saved_row.get("case") != case_id
                or saved_row.get("config_hash") != case_hash
            ):
                continue
            key = (
                saved_row.get("case"),
                saved_row.get("model_seed"),
                saved_row.get("config_hash"),
                saved_row.get("status"),
            )
            if key not in global_keys:
                append_csv(training_path, saved_row, TRAINING_FIELDS)
                global_keys.add(key)
                recovered += 1
    if recovered:
        print(f"Recovered {recovered} per-case result row(s) into {training_path}")
    return recovered


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int = 20260725,
    repetitions: int = 10000,
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    estimates = np.mean(values[indices], axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def aggregate_results(
    training_path: Path,
    hashes: dict[str, str],
    output_dir: Path,
    runtime: dict[str, Any] | None = None,
    study_context_hash: str | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, str]],
    dict[str, Any],
]:
    audit_records: list[dict[str, Any]] = []
    all_successful = current_successes(
        training_path,
        hashes,
        runtime=runtime,
        study_context_hash=study_context_hash,
        audit_records=audit_records,
    )
    successful = [
        row
        for row in all_successful.values()
        if int(row["model_seed"]) in set(DEFAULT_SEEDS)
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in successful:
        grouped[row["case"]].append(row)
    aggregates: list[dict[str, Any]] = []
    for case_id in CASE_BY_ID:
        rows = grouped.get(case_id, [])
        if not rows:
            continue
        tests = np.asarray([float(row["test_acc"]) for row in rows])
        vals = np.asarray([float(row["best_val_acc"]) for row in rows])
        low, high = bootstrap_mean_ci(tests)
        aggregates.append(
            {
                "case": case_id,
                "n_model_seeds": len(rows),
                "model_seeds": ",".join(
                    str(int(row["model_seed"]))
                    for row in sorted(rows, key=lambda item: int(item["model_seed"]))
                ),
                "mean_best_val_acc": float(np.mean(vals)),
                "std_best_val_acc": (
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                ),
                "mean_test_acc": float(np.mean(tests)),
                "std_test_acc": (
                    float(np.std(tests, ddof=1)) if len(tests) > 1 else 0.0
                ),
                "bootstrap_mean_test_ci_low": low,
                "bootstrap_mean_test_ci_high": high,
                "ci_scope": "training-randomness seeds only; not dataset-level CI",
                **CASE_BY_ID[case_id],
            }
        )

    by_case_seed = {
        (row["case"], int(row["model_seed"])): float(row["test_acc"])
        for row in successful
    }
    effects = [
        ("sampling_effect_legacy", "bp_fs200_legacy_abs", "bp_fs1000_legacy_abs"),
        ("sampling_effect_signed", "bp_fs200_signed", "bp_fs1000_signed"),
        ("sign_effect_fs1000", "bp_fs1000_signed", "bp_fs1000_legacy_abs"),
        ("sign_effect_fs200", "bp_fs200_signed", "bp_fs200_legacy_abs"),
        (
            "signed_iqr_then_absnorm_vs_legacy_fs200",
            "bp_fs200_signed_iqr_absnorm",
            "bp_fs200_legacy_abs",
        ),
        (
            "signed_norm_vs_signed_iqr_absnorm",
            "bp_fs200_signed",
            "bp_fs200_signed_iqr_absnorm",
        ),
        (
            "remove_iqr_effect_signed_fs200",
            "bp_fs200_signed_no_iqr",
            "bp_fs200_signed",
        ),
        ("signed_fs200_vs_savgol", "bp_fs200_signed", "savgol_reference"),
    ]
    paired_rows: list[dict[str, Any]] = []
    paired_seed_rows: list[dict[str, Any]] = []
    paired_values: dict[str, dict[int, float]] = {}
    for effect, minuend, subtrahend in effects:
        seeds = sorted(
            {seed for case_id, seed in by_case_seed if case_id == minuend}
            & {seed for case_id, seed in by_case_seed if case_id == subtrahend}
        )
        differences = {
            seed: by_case_seed[(minuend, seed)] - by_case_seed[(subtrahend, seed)]
            for seed in seeds
        }
        paired_seed_rows.extend(
            {
                "effect": effect,
                "definition": f"{minuend} - {subtrahend}",
                "model_seed": seed,
                "component_a_name": minuend,
                "component_a_value": by_case_seed[(minuend, seed)],
                "component_b_name": subtrahend,
                "component_b_value": by_case_seed[(subtrahend, seed)],
                "delta_test_acc": difference,
            }
            for seed, difference in differences.items()
        )
        paired_values[effect] = differences
        values = np.asarray(list(differences.values()), dtype=float)
        if not len(values):
            continue
        low, high = bootstrap_mean_ci(values)
        paired_rows.append(
            {
                "effect": effect,
                "definition": f"{minuend} - {subtrahend}",
                "n_paired_seeds": len(values),
                "seeds": ",".join(map(str, seeds)),
                "mean_delta_test_acc": float(np.mean(values)),
                "std_delta_test_acc": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "positive_seed_fraction": float(np.mean(values > 0)),
                "ci_scope": "training-randomness seeds only; not dataset-level CI",
            }
        )
    if "sign_effect_fs200" in paired_values and "sign_effect_fs1000" in paired_values:
        common = sorted(
            set(paired_values["sign_effect_fs200"])
            & set(paired_values["sign_effect_fs1000"])
        )
        interaction = np.asarray(
            [
                paired_values["sign_effect_fs200"][seed]
                - paired_values["sign_effect_fs1000"][seed]
                for seed in common
            ]
        )
        if len(interaction):
            paired_seed_rows.extend(
                {
                    "effect": "sampling_x_sign_interaction",
                    "definition": "sign_effect_fs200 - sign_effect_fs1000",
                    "model_seed": seed,
                    "component_a_name": "sign_effect_fs200",
                    "component_a_value": paired_values["sign_effect_fs200"][seed],
                    "component_b_name": "sign_effect_fs1000",
                    "component_b_value": paired_values["sign_effect_fs1000"][seed],
                    "delta_test_acc": (
                        paired_values["sign_effect_fs200"][seed]
                        - paired_values["sign_effect_fs1000"][seed]
                    ),
                }
                for seed in common
            )
            low, high = bootstrap_mean_ci(interaction)
            paired_rows.append(
                {
                    "effect": "sampling_x_sign_interaction",
                    "definition": "sign_effect_fs200 - sign_effect_fs1000",
                    "n_paired_seeds": len(interaction),
                    "seeds": ",".join(map(str, common)),
                    "mean_delta_test_acc": float(np.mean(interaction)),
                    "std_delta_test_acc": (
                        float(np.std(interaction, ddof=1))
                        if len(interaction) > 1
                        else 0.0
                    ),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "positive_seed_fraction": float(np.mean(interaction > 0)),
                    "ci_scope": "training-randomness seeds only; not dataset-level CI",
                }
            )
    write_csv(output_dir / "training_aggregate.csv", aggregates)
    write_csv(output_dir / "paired_effects.csv", paired_rows)
    write_csv(output_dir / "paired_seed_effects.csv", paired_seed_rows)
    artifact_audit = {
        "checkpoint_load_checked": runtime is not None and "torch" in runtime,
        "candidate_rows": len(audit_records),
        "valid_unique_results": len(all_successful),
        "valid_result_keys": [
            {"case": case_id, "model_seed": seed}
            for case_id, seed in sorted(all_successful)
        ],
        "records": audit_records,
    }
    write_json(output_dir / "artifact_audit.json", artifact_audit)
    return aggregates, paired_rows, successful, artifact_audit


def evaluate_official_completion(
    settings: dict[str, Any],
    hashes: dict[str, str],
    successful: list[dict[str, str]],
    paired_rows: list[dict[str, Any]],
    artifact_audit: dict[str, Any],
) -> dict[str, Any]:
    """Apply the single strict gate for an official causal interpretation."""
    reasons: list[str] = []
    official_cases = set(OFFICIAL_CASE_IDS)
    default_seeds = set(DEFAULT_SEEDS)
    current_study_hash = str(settings.get("study_context_hash", "")).strip()

    settings_hashes = settings.get("case_hashes", {})
    if settings_hashes != hashes:
        reasons.append(
            "evaluated case hashes do not exactly match experiment_settings.json"
        )
    if set(hashes) != official_cases:
        missing = sorted(official_cases - set(hashes))
        extra = sorted(set(hashes) - official_cases)
        reasons.append(
            f"official case hashes incomplete: missing={missing}, extra={extra}"
        )
    try:
        configured_epochs = int(settings.get("epochs"))
    except (TypeError, ValueError):
        configured_epochs = -1
    if configured_epochs != DEFAULT_EPOCHS:
        reasons.append(f"epochs must be {DEFAULT_EPOCHS}, observed {configured_epochs}")
    try:
        configured_users = int(settings.get("user_count"))
    except (TypeError, ValueError):
        configured_users = -1
    selected_user_ids = settings.get("selected_user_ids", [])
    if (
        configured_users != OFFICIAL_USER_COUNT
        or not isinstance(selected_user_ids, list)
        or len(selected_user_ids) != OFFICIAL_USER_COUNT
        or len(set(selected_user_ids)) != OFFICIAL_USER_COUNT
    ):
        reasons.append(
            "official run requires user_count=3 and exactly 3 selected user IDs"
        )
    effective = settings.get("effective_training_params", {})
    try:
        effective_epochs = int(effective.get("epochs"))
    except (AttributeError, TypeError, ValueError):
        effective_epochs = -1
    if effective_epochs != DEFAULT_EPOCHS:
        reasons.append("effective training parameters do not record epochs=50")
    if not current_study_hash:
        reasons.append("study_context_hash is missing")
    context_by_case = settings.get("case_study_context_hashes", {})
    if not isinstance(context_by_case, dict):
        context_by_case = {}
    if set(context_by_case) != official_cases or any(
        context_by_case.get(case_id) != current_study_hash
        for case_id in OFFICIAL_CASE_IDS
    ):
        reasons.append(
            "all seven case hashes are not tied to the current study_context_hash"
        )

    valid_keys: set[tuple[str, int]] = set()
    for item in artifact_audit.get("valid_result_keys", []):
        try:
            valid_keys.add((str(item.get("case")), int(item["model_seed"])))
        except (KeyError, TypeError, ValueError):
            reasons.append(f"malformed valid-result key in artifact audit: {item}")
    expected_keys = {
        (case_id, seed) for case_id in OFFICIAL_CASE_IDS for seed in DEFAULT_SEEDS
    }
    missing_results = sorted(expected_keys - valid_keys)
    unexpected_results = sorted(valid_keys - expected_keys)
    if missing_results or unexpected_results:
        reasons.append(
            "valid artifact-backed results must be exactly the official 35: "
            f"missing={missing_results}, unexpected={unexpected_results}"
        )
    invalid_current = [
        record
        for record in artifact_audit.get("records", [])
        if not record.get("valid")
    ]
    if missing_results and invalid_current:
        missing_result_set = set(missing_results)
        compact = []
        for record in invalid_current:
            try:
                key = (
                    str(record.get("case")),
                    int(record.get("model_seed", -1)),
                )
            except (TypeError, ValueError):
                continue
            if key in missing_result_set:
                compact.append(
                    {
                        "case": record.get("case"),
                        "model_seed": record.get("model_seed"),
                        "reasons": record.get("reasons"),
                    }
                )
        if compact:
            reasons.append(f"invalid result artifacts: {compact}")

    result_epochs: set[int] = set()
    for row in successful:
        try:
            result_epochs.add(int(row.get("epochs", -1)))
        except (TypeError, ValueError):
            result_epochs.add(-1)
    if valid_keys == expected_keys and result_epochs != {DEFAULT_EPOCHS}:
        reasons.append(
            f"all 35 result rows must record epochs=50, observed {result_epochs}"
        )
    if valid_keys == expected_keys and any(
        row.get("study_context_hash") != current_study_hash for row in successful
    ):
        reasons.append("result rows do not all match the current study hash")

    effect_by_name = {row["effect"]: row for row in paired_rows}
    effect_seed_counts: dict[str, int] = {}
    for effect_id in OFFICIAL_EFFECT_IDS:
        row = effect_by_name.get(effect_id)
        if row is None:
            effect_seed_counts[effect_id] = 0
            reasons.append(f"key effect missing: {effect_id}")
            continue
        seeds = {
            int(value)
            for value in str(row.get("seeds", "")).split(",")
            if value.strip()
        }
        effect_seed_counts[effect_id] = len(seeds)
        if seeds != default_seeds or int(row.get("n_paired_seeds", 0)) != len(
            DEFAULT_SEEDS
        ):
            reasons.append(
                f"key effect {effect_id} is not paired on all 5 official seeds"
            )

    case_seed_counts = Counter(case_id for case_id, _ in valid_keys)
    official_complete = not reasons
    completion = {
        "status": "OFFICIAL_COMPLETE" if official_complete else "INCOMPLETE",
        "official_complete": official_complete,
        "reasons": reasons,
        "study_context_hash": current_study_hash or None,
        "expected": {
            "case_ids": list(OFFICIAL_CASE_IDS),
            "seeds": list(DEFAULT_SEEDS),
            "epochs": DEFAULT_EPOCHS,
            "user_count": OFFICIAL_USER_COUNT,
            "result_count": len(expected_keys),
            "key_effect_ids": list(OFFICIAL_EFFECT_IDS),
        },
        "observed": {
            "case_hash_ids": sorted(hashes),
            "valid_result_count": len(valid_keys),
            "case_seed_counts": {
                case_id: case_seed_counts.get(case_id, 0)
                for case_id in OFFICIAL_CASE_IDS
            },
            "effect_seed_counts": effect_seed_counts,
            "epochs": configured_epochs,
            "effective_epochs": effective_epochs,
            "user_count": configured_users,
            "selected_user_ids": selected_user_ids,
            "checkpoint_load_checked": artifact_audit.get(
                "checkpoint_load_checked",
                False,
            ),
        },
    }
    return completion


def write_completion_status(
    output_dir: Path,
    completion: dict[str, Any],
) -> None:
    write_json(output_dir / "completion_status.json", completion)


def publication_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: Any, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")


def plot_results(
    runtime: dict[str, Any],
    aggregates: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    successful: list[dict[str, str]],
    output_dir: Path,
    completion: dict[str, Any],
) -> None:
    plt = runtime["plt"]
    publication_style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1), constrained_layout=True)
    if not completion["official_complete"]:
        fig.suptitle(
            "INCOMPLETE — exploratory output only; no causal conclusion",
            color=OKABE_ITO["red"],
            fontsize=13,
            fontweight="bold",
        )
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "INCOMPLETE",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color=OKABE_ITO["red"],
                alpha=0.12,
                fontsize=28,
                fontweight="bold",
                rotation=20,
                zorder=0,
            )

    main_order = [
        "bp_fs1000_legacy_abs",
        "bp_fs1000_signed",
        "bp_fs200_legacy_abs",
        "bp_fs200_signed",
    ]
    positions = np.asarray([0.0, 1.0, 2.5, 3.5])
    by_case_seed = {
        (row["case"], int(row["model_seed"])): 100.0 * float(row["test_acc"])
        for row in successful
    }
    all_seeds = sorted({seed for _, seed in by_case_seed})
    for seed in all_seeds:
        for pair in ((0, 1), (2, 3)):
            if all((main_order[index], seed) in by_case_seed for index in pair):
                axes[0].plot(
                    positions[list(pair)],
                    [by_case_seed[(main_order[index], seed)] for index in pair],
                    color=OKABE_ITO["gray"],
                    alpha=0.45,
                    linewidth=0.9,
                    zorder=1,
                )
        for index, case_id in enumerate(main_order):
            if (case_id, seed) in by_case_seed:
                axes[0].scatter(
                    positions[index],
                    by_case_seed[(case_id, seed)],
                    color=OKABE_ITO["orange"],
                    edgecolor="white",
                    linewidth=0.4,
                    s=24,
                    zorder=3,
                )
    aggregate_by_case = {row["case"]: row for row in aggregates}
    for index, case_id in enumerate(main_order):
        if case_id not in aggregate_by_case:
            continue
        row = aggregate_by_case[case_id]
        axes[0].errorbar(
            positions[index],
            100.0 * float(row["mean_test_acc"]),
            yerr=100.0 * float(row["std_test_acc"]),
            fmt="o",
            color=OKABE_ITO["blue"],
            capsize=3,
            markersize=6,
            zorder=4,
        )
    axes[0].set_xticks(
        positions,
        ["Legacy\nabs", "Signed", "Legacy\nabs", "Signed"],
    )
    axes[0].text(
        0.5,
        -0.20,
        "$f_s=1000$ Hz",
        ha="center",
        transform=axes[0].get_xaxis_transform(),
    )
    axes[0].text(
        3.0, -0.20, "$f_s=200$ Hz", ha="center", transform=axes[0].get_xaxis_transform()
    )
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title(
        "A  Controlled 2 × 2 ablation (mean ± SD; dots = seeds)",
        loc="left",
    )
    axes[0].grid(True, axis="y")

    displayed = [
        row
        for row in paired_rows
        if row["effect"]
        in {
            "sampling_effect_legacy",
            "sampling_effect_signed",
            "sign_effect_fs1000",
            "sign_effect_fs200",
            "sampling_x_sign_interaction",
            "remove_iqr_effect_signed_fs200",
            "signed_iqr_then_absnorm_vs_legacy_fs200",
            "signed_norm_vs_signed_iqr_absnorm",
            "signed_fs200_vs_savgol",
        }
    ]
    labels = {
        "sampling_effect_legacy": "$f_s$ effect | legacy",
        "sampling_effect_signed": "$f_s$ effect | signed",
        "sign_effect_fs1000": "sign effect | 1000 Hz",
        "sign_effect_fs200": "sign effect | 200 Hz",
        "sampling_x_sign_interaction": "sampling × sign",
        "remove_iqr_effect_signed_fs200": "remove IQR | signed, 200 Hz",
        "signed_iqr_then_absnorm_vs_legacy_fs200": ("signed-IQR+absnorm − legacy"),
        "signed_norm_vs_signed_iqr_absnorm": "signed norm − abs norm",
        "signed_fs200_vs_savgol": "signed 200 Hz − Savgol",
    }
    y = np.arange(len(displayed))[::-1]
    for y_value, row in zip(y, displayed):
        mean = 100.0 * float(row["mean_delta_test_acc"])
        low = 100.0 * float(row["bootstrap_ci_low"])
        high = 100.0 * float(row["bootstrap_ci_high"])
        axes[1].errorbar(
            mean,
            y_value,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color=OKABE_ITO["green"],
            capsize=3,
        )
    axes[1].axvline(0.0, color=OKABE_ITO["black"], linewidth=0.8)
    axes[1].set_yticks(y, [labels[row["effect"]] for row in displayed])
    axes[1].set_xlabel("Paired test-accuracy difference (percentage points)")
    axes[1].set_title(
        "B  Paired effects (95% bootstrap CI over model seeds)",
        loc="left",
    )
    axes[1].grid(True, axis="x")
    save_figure(
        fig,
        output_dir / "figures" / "bandpass_sampling_sign_ablation",
    )
    plt.close(fig)


def write_plain_language_report(
    aggregates: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    output_dir: Path,
    completion: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_by_case = {row["case"]: row for row in aggregates}
    effect_by_name = {row["effect"]: row for row in paired_rows}
    status = completion["status"]
    lines = [
        "# Bandpass 在 XRF55 上效果不佳：服务器实验汇报摘要",
        "",
        f"# {status}",
        "",
    ]
    if not completion["official_complete"]:
        lines.extend(
            [
                "**当前输出不是正式实验结论。所有差值只能当烟测/探索结果，"
                "禁止据此下强因果结论。**",
                "",
                "未完成原因：",
                *[f"- {reason}" for reason in completion["reasons"]],
                "",
            ]
        )
    lines.extend(
        [
            "## 先把三个概念说清楚",
            "",
            "- **基线**：CSI 中由静态环境、设备增益和很慢漂移造成的整体底座。"
            "Bandpass 去掉低于 0.5 Hz 的慢变化，同时也去掉高于 50 Hz 的部分，"
            "所以“原始值减 Bandpass”不能全部叫基线。",
            "- **正值/负值**：Bandpass 把信号拉到零附近后，正负只表示位于零中心的"
            "哪一侧，不直接表示上升或下降。",
            "- **上升/下降**：看相邻帧差值 `x[t]-x[t-1]`。差值为正才叫上升，"
            "差值为负才叫下降。",
            "- **z-score 后的负数**：只表示该数值低于训练集均值，已经不是 "
            "Bandpass 输出的负半轴，二者不能混为一谈。",
            "",
            "## 分类结果",
            "",
        ]
    )
    lines.extend(
        [
            "| 实验分支 | 完成度 | 配置 | 测试准确率（均值 ± SD） |",
            "|---|---:|---|---:|",
        ]
    )
    epochs = completion["observed"]["epochs"]
    users = completion["observed"]["user_count"]
    for case in CASES:
        row = aggregate_by_case.get(case["case"])
        count = int(row["n_model_seeds"]) if row else 0
        accuracy = (
            f"{100 * float(row['mean_test_acc']):.2f}% ± "
            f"{100 * float(row['std_test_acc']):.2f}%"
            if row
            else "—"
        )
        fs_text = f"{case['fs_hz']:g} Hz" if case["fs_hz"] is not None else "N/A"
        configuration = (
            f"{case['denoiser']}; fs={fs_text}; IQR={case['iqr_mode']}; "
            f"norm={case['normalization_input']}; epochs={epochs}; users={users}"
        )
        lines.append(
            f"| {case['case']} | {count}/{len(DEFAULT_SEEDS)} | "
            f"{configuration} | {accuracy} |"
        )
    lines.extend(["", "## 因果判定", ""])
    sign_200 = effect_by_name.get("sign_effect_fs200")
    sampling_signed = effect_by_name.get("sampling_effect_signed")
    if not completion["official_complete"]:
        lines.append(
            "- **INCOMPLETE：不进行正式因果判定。** 下列数值如存在，"
            "只用于检查代码和决定是否继续跑满。"
        )
        if sign_200:
            lines.append(
                "- 探索性符号效应（200 Hz）："
                f"{100.0 * float(sign_200['mean_delta_test_acc']):+.2f} "
                f"个百分点（{int(sign_200['n_paired_seeds'])}/5 对 seed）。"
            )
        if sampling_signed:
            lines.append(
                "- 探索性采样率效应（保留符号，200−1000 Hz）："
                f"{100.0 * float(sampling_signed['mean_delta_test_acc']):+.2f} "
                f"个百分点（{int(sampling_signed['n_paired_seeds'])}/5 对 seed）。"
            )
    elif sign_200:
        delta = 100.0 * float(sign_200["mean_delta_test_acc"])
        positive = float(sign_200["positive_seed_fraction"])
        paired_count = int(sign_200["n_paired_seeds"])
        completed_seed_set = {
            int(value) for value in str(sign_200["seeds"]).split(",") if value.strip()
        }
        ci_low = float(sign_200["bootstrap_ci_low"])
        if completed_seed_set != set(DEFAULT_SEEDS):
            verdict = (
                f"目前只完成 {paired_count}/{len(DEFAULT_SEEDS)} 个预注册 seed，"
                "只能当烟测或初步结果，不能下因果结论。"
            )
        elif delta > 0 and positive >= 0.8 and ci_low > 0:
            verdict = (
                "在固定 200 Hz 时，保留 Bandpass 正负号后多数 seed 都提高，"
                "且训练随机性 bootstrap 区间未跨 0；强烈支持“后续 abs/IQR "
                "折叠符号”是性能下降原因之一。"
            )
        elif delta > 0:
            verdict = (
                "保留符号的平均结果有提高，但不同 seed 不够一致；目前只能说"
                "有部分证据，不能下强因果结论。"
            )
        else:
            verdict = (
                "固定 200 Hz 后保留符号没有提高，因此本实验不支持“符号折叠是主要原因”。"
            )
        lines.append(
            f"- 符号效应（200 Hz）：{delta:+.2f} 个百分点，"
            f"{positive:.0%} 的配对 seed 为正。{verdict}"
        )
    elif completion["official_complete"]:
        lines.append("- 符号效应：结果尚不完整。")
    if completion["official_complete"] and sampling_signed:
        delta = 100.0 * float(sampling_signed["mean_delta_test_acc"])
        paired_count = int(sampling_signed["n_paired_seeds"])
        completed_seed_set = {
            int(value)
            for value in str(sampling_signed["seeds"]).split(",")
            if value.strip()
        }
        if completed_seed_set == set(DEFAULT_SEEDS):
            completeness = "预注册的 5 个配对 seed 已全部完成。"
        else:
            completeness = (
                f"目前只完成 {paired_count}/{len(DEFAULT_SEEDS)} 个预注册 "
                "seed，只能当初步结果。"
            )
        lines.append(
            f"- 采样率效应（保留符号后，200−1000 Hz）：{delta:+.2f} 个百分点。"
            "它衡量改正符号问题后，采样率设置本身还剩多少影响。"
            f"{completeness}"
        )
    lines.extend(
        [
            "",
            "## 汇报时必须加的一句话",
            "",
            "正式结论要求预注册的 5 个模型 seed 全部完成。这里的重复只改变模型"
            "随机种子，所以误差条和 bootstrap 区间只反映训练随机性，不能当成"
            "对整个 XRF55 总体的统计置信区间。",
        ]
    )
    (output_dir / "report_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    write_json(
        output_dir / "report_summary.json",
        {
            "status": completion["status"],
            "official_complete": completion["official_complete"],
            "completion_reasons": completion["reasons"],
            "completion": completion,
            "training_aggregate": aggregates,
            "paired_effects": paired_rows,
        },
    )


def build_case_hashes(
    selected_cases: list[dict[str, Any]],
    args: argparse.Namespace,
    data_path: Path,
    users: list[int],
    dataset_signature: str,
    params: dict[str, Any],
    source_signature: str,
    runtime_fingerprint: dict[str, Any],
) -> tuple[dict[str, str], str]:
    common = {
        "data_path": str(data_path),
        "selected_user_ids": users,
        "dataset_signature": dataset_signature,
        "padding_length": PADDING_LENGTH,
        "target_subcarriers": TARGET_SUBCARRIERS,
        "iqr_factor": IQR_FACTOR,
        "split": "repetition_1-12_train_13-16_val_17-20_test",
        "split_seed": args.split_seed,
        "model": MODEL_NAME,
        "epochs": args.epochs,
        "optimizer": {
            "name": "AdamW",
            "batch": int(params.get("batch", 32)),
            "learning_rate": float(params.get("lr", 3e-4)),
            "weight_decay": float(params.get("wd", 1e-3)),
        },
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.1,
            "patience": 5,
        },
        "source_content_signature": source_signature,
        "runtime_fingerprint": runtime_fingerprint,
    }
    hashes = {
        case["case"]: stable_hash({"common": common, "case": case})
        for case in selected_cases
    }
    return hashes, stable_hash(common)


def source_content_signature() -> str:
    """Hash the exact code/config files that define this experiment."""
    relative_paths = [
        "wsdp/algorithms/denoising_butterworth.py",
        "wsdp/algorithms/amplitude.py",
        "wsdp/algorithms/interpolation.py",
        "wsdp/algorithms/subcarrier_mapping.py",
        "wsdp/algorithms/registry.py",
        "wsdp/core.py",
        "wsdp/dataset_policy.py",
        "wsdp/datasets/CSIDataset.py",
        "wsdp/processors/base_processor.py",
        "wsdp/processors/configurable_processor.py",
        "wsdp/readers/__init__.py",
        "wsdp/readers/base.py",
        "wsdp/readers/xrf_reader.py",
        "wsdp/structure/__init__.py",
        "wsdp/structure/CSIData.py",
        "wsdp/structure/CSIFrame.py",
        "wsdp/models/__init__.py",
        "wsdp/models/mainstream.py",
        "wsdp/models/registry.py",
        "wsdp/utils/__init__.py",
        "wsdp/utils/train_func.py",
        "wsdp/utils/resize.py",
        "wsdp/configs/model_params.json",
    ]
    rows = []
    for relative in relative_paths:
        path = WSDP_SRC / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source file missing: {path}")
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    rows.append(
        {
            "path": str(Path(__file__).name),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    return stable_hash(rows)


def main() -> None:
    args = parse_args()
    validate_case_definitions()
    if args.self_test:
        synthetic_self_test()
        return

    data_path = resolve_path(args.data_path)
    output_dir = resolve_path(args.output_dir)
    selected_cases = [CASE_BY_ID[case_id] for case_id in dict.fromkeys(args.cases)]
    args.seeds = parse_positive_unique(args.seeds, "--seeds")
    if args.epochs < 1 or args.user_count < 1 or args.workers < 1:
        raise ValueError("--epochs, --user-count and --workers must be >= 1")

    dry_payload = {
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "cases": selected_cases,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "user_count": args.user_count,
        "workers": args.workers,
        "split": "repetition 1-12 train / 13-16 val / 17-20 test",
        "pipeline_order": (
            "raw -> denoise -> IQR/no-IQR -> cubic15 -> resize1000 -> "
            "split -> abs/signed -> train-only global z-score -> ResNet1D"
        ),
    }
    if args.dry_run:
        print(json.dumps(dry_payload, ensure_ascii=False, indent=2))
        return

    runtime = import_runtime(args.gpu)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "training_summary.csv"

    if args.plot_only:
        settings = json.loads(
            (output_dir / "experiment_settings.json").read_text(encoding="utf-8")
        )
        hashes = settings["case_hashes"]
        sync_case_results_to_global(output_dir, hashes, training_path)
        aggregates, paired, successful, artifact_audit = aggregate_results(
            training_path,
            hashes,
            output_dir,
            runtime=runtime,
            study_context_hash=settings.get("study_context_hash"),
        )
        completion = evaluate_official_completion(
            settings,
            hashes,
            successful,
            paired,
            artifact_audit,
        )
        write_completion_status(output_dir, completion)
        plot_results(
            runtime,
            aggregates,
            paired,
            successful,
            output_dir,
            completion,
        )
        write_plain_language_report(
            aggregates,
            paired,
            output_dir,
            completion,
        )
        return

    gate = source_equivalence_gate(runtime)
    files, users, dataset_signature = discover_selected_files(
        data_path,
        args.user_count,
    )
    params = runtime["load_params"](DATASET_NAME)
    source_signature = source_content_signature()
    import scipy

    runtime_fingerprint = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": runtime["torch"].__version__,
        "scipy": scipy.__version__,
        "torch_cuda": runtime["torch"].version.cuda,
        "cudnn": (
            runtime["torch"].backends.cudnn.version()
            if hasattr(runtime["torch"].backends, "cudnn")
            else None
        ),
    }
    hashes, study_context_hash = build_case_hashes(
        selected_cases,
        args,
        data_path,
        users,
        dataset_signature,
        params,
        source_signature,
        runtime_fingerprint,
    )
    settings_path = output_dir / "experiment_settings.json"
    manifest_hashes: dict[str, str] = {}
    manifest_context_hashes: dict[str, str] = {}
    previous_settings: dict[str, Any] = {}
    if settings_path.exists():
        previous_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if previous_settings.get("study_context_hash") == study_context_hash:
            manifest_hashes.update(previous_settings.get("case_hashes", {}))
            manifest_context_hashes.update(
                previous_settings.get("case_study_context_hashes", {})
            )
    manifest_hashes.update(hashes)
    manifest_context_hashes.update({case_id: study_context_hash for case_id in hashes})
    settings = {
        **dry_payload,
        "selected_user_ids": users,
        "selected_files": len(files),
        "dataset_signature": dataset_signature,
        "case_hashes": manifest_hashes,
        "case_study_context_hashes": manifest_context_hashes,
        "study_context_hash": study_context_hash,
        "source_content_signature": source_signature,
        "runtime_fingerprint": runtime_fingerprint,
        "effective_training_params": {
            "batch": int(params.get("batch", 32)),
            "learning_rate": float(params.get("lr", 3e-4)),
            "weight_decay": float(params.get("wd", 1e-3)),
            "epochs": args.epochs,
        },
        "source_equivalence_gate": gate,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": runtime["torch"].__version__,
        "cuda_available": runtime["torch"].cuda.is_available(),
        "cuda_device_name": (
            runtime["torch"].cuda.get_device_name(0)
            if runtime["torch"].cuda.is_available()
            else None
        ),
        "created_unix": time.time(),
    }
    if previous_settings.get("study_context_hash") == study_context_hash:
        if previous_settings.get("sample_order_hash") is not None:
            settings["sample_order_hash"] = previous_settings["sample_order_hash"]
    write_json(settings_path, settings)

    write_json(
        output_dir / "dataset_metadata.json",
        {
            "discovered_selected_files": len(files),
            "selected_user_ids": users,
            "dataset_signature": dataset_signature,
            "input_policy": (
                "genuinely complex CSI is rejected because a signed-value "
                "ablation is undefined for complex numbers"
            ),
            "loading_strategy": (
                "workers receive paths and stream processed arrays to memmap; "
                "the full raw dataset is never retained in the parent process"
            ),
        },
    )

    sync_case_results_to_global(output_dir, manifest_hashes, training_path)
    baseline_order_hash: str | None = None
    for case_index, case in enumerate(selected_cases, start=1):
        case_dir = (
            output_dir / "cases" / case["case"] / f"config_{hashes[case['case']]}"
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        successful = current_successes(
            training_path,
            hashes,
            study_context_hash=study_context_hash,
        )
        pending = [
            seed for seed in args.seeds if (case["case"], seed) not in successful
        ]
        if args.resume and not pending and not args.diagnostics_only:
            print(f"Skip completed case with matching hash: {case['case']}")
            continue
        if not args.resume:
            pending = args.seeds

        print("\n" + "=" * 78)
        print(f"Case {case_index}/{len(selected_cases)}: {case['case']}")
        print("=" * 78)
        processed, labels, groups, unique_labels, metadata = preprocess_case_from_files(
            files,
            case,
            args.workers,
            case_dir,
            hashes[case["case"]],
            dataset_signature,
            users,
        )
        if baseline_order_hash is None:
            baseline_order_hash = metadata["sample_order_hash"]
            previous_order = previous_settings.get("sample_order_hash")
            if (
                previous_settings.get("study_context_hash") == study_context_hash
                and previous_order is not None
                and previous_order != baseline_order_hash
            ):
                raise AssertionError(
                    "Sample order differs from prior cases in the same study "
                    f"context: {previous_order} != {baseline_order_hash}"
                )
            settings["sample_order_hash"] = baseline_order_hash
            write_json(settings_path, settings)
        elif metadata["sample_order_hash"] != baseline_order_hash:
            raise AssertionError(
                f"Sample order/split inputs changed in case {case['case']}"
            )
        split, normalization_metadata = split_and_normalize(
            runtime,
            processed,
            labels,
            groups,
            case,
            args.split_seed,
        )
        write_json(
            case_dir / "normalization_and_split_metadata.json",
            normalization_metadata,
        )
        del processed
        gc.collect()

        if not args.diagnostics_only:
            for seed in pending:
                row = train_one_seed(
                    runtime,
                    split,
                    len(unique_labels),
                    case,
                    hashes[case["case"]],
                    study_context_hash,
                    seed,
                    args.split_seed,
                    args.epochs,
                    params,
                    output_dir,
                )
                append_csv(training_path, row, TRAINING_FIELDS)
                append_csv(
                    case_dir / "case_results.csv",
                    row,
                    TRAINING_FIELDS,
                )
        del split
        gc.collect()

    aggregates, paired, successful, artifact_audit = aggregate_results(
        training_path,
        manifest_hashes,
        output_dir,
        runtime=runtime,
        study_context_hash=study_context_hash,
    )
    completion = evaluate_official_completion(
        settings,
        manifest_hashes,
        successful,
        paired,
        artifact_audit,
    )
    write_completion_status(output_dir, completion)
    plot_results(
        runtime,
        aggregates,
        paired,
        successful,
        output_dir,
        completion,
    )
    write_plain_language_report(
        aggregates,
        paired,
        output_dir,
        completion,
    )
    print(f"Finished [{completion['status']}]. Results: {output_dir}")


if __name__ == "__main__":
    main()
