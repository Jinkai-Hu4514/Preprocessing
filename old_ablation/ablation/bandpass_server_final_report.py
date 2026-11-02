"""Audit and merge the two server-side Bandpass result directories.

This script reads result artifacts only.  It never imports WSDP and never
opens ElderAL or XRF55 dataset files.

Official use::

    python ablation/bandpass_server_final_report.py

An incomplete run is reported as ``PRELIMINARY`` and exits non-zero.  Use
``--allow-incomplete`` only when an explicitly preliminary report is useful.
The synthetic self-test creates result-shaped temporary files and covers a
complete strongly-supporting result, a complete non-supporting result, and a
missing-seed result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = REPO_ROOT / "ablation" / "bandpass_server_results"
DEFAULT_SIGNAL_DIR = DEFAULT_RESULT_ROOT / "signal_analysis"
DEFAULT_SIGN_DIR = DEFAULT_RESULT_ROOT / "sign_ablation"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_ROOT / "final_report"

EXPECTED_SEEDS = (42, 49, 514, 654, 886)
EXPECTED_METHODS = (
    "raw",
    "wavelet",
    "butterworth_o5_c0.3",
    "savgol_w7_p3",
    "bandpass_fs1000",
    "bandpass_fs200",
    "hampel_w5_s3",
)
EXPECTED_CASES = (
    "bp_fs1000_legacy_abs",
    "bp_fs1000_signed",
    "bp_fs200_legacy_abs",
    "bp_fs200_signed",
    "bp_fs200_signed_iqr_absnorm",
    "bp_fs200_signed_no_iqr",
    "savgol_reference",
)
EFFECT_DEFINITIONS = {
    "sampling_effect_legacy": (
        "bp_fs200_legacy_abs",
        "bp_fs1000_legacy_abs",
    ),
    "sampling_effect_signed": (
        "bp_fs200_signed",
        "bp_fs1000_signed",
    ),
    "sign_effect_fs1000": (
        "bp_fs1000_signed",
        "bp_fs1000_legacy_abs",
    ),
    "sign_effect_fs200": (
        "bp_fs200_signed",
        "bp_fs200_legacy_abs",
    ),
    "signed_iqr_then_absnorm_vs_legacy_fs200": (
        "bp_fs200_signed_iqr_absnorm",
        "bp_fs200_legacy_abs",
    ),
    "signed_norm_vs_signed_iqr_absnorm": (
        "bp_fs200_signed",
        "bp_fs200_signed_iqr_absnorm",
    ),
    "remove_iqr_effect_signed_fs200": (
        "bp_fs200_signed_no_iqr",
        "bp_fs200_signed",
    ),
    "signed_fs200_vs_savgol": (
        "bp_fs200_signed",
        "savgol_reference",
    ),
}
INTERACTION_EFFECT = "sampling_x_sign_interaction"
ALL_EFFECTS = (*EFFECT_DEFINITIONS, INTERACTION_EFFECT)

# Exactly four report figure groups are mandatory.  The rise/fall figure is
# audited as a supplemental figure because it was added after preregistration.
REQUIRED_FIGURES = (
    (
        "elder_bypass_overview",
        "signal",
        Path("elder_bandpass_bypass_overview"),
    ),
    (
        "elder_before_after",
        "signal",
        Path("elder_bandpass_before_after"),
    ),
    (
        "xrf_negative_distribution",
        "signal",
        Path("xrf55_denoiser_negative_distribution"),
    ),
    (
        "sampling_sign_ablation",
        "sign",
        Path("figures") / "bandpass_sampling_sign_ablation",
    ),
)
SUPPLEMENTAL_FIGURE = Path("xrf55_rise_fall_and_abs_effect")
FIGURE_SUFFIXES = (".png", ".pdf", ".svg")


class AuditLog:
    """Collect machine-readable audit findings."""

    def __init__(self) -> None:
        self.errors: list[dict[str, str]] = []
        self.warnings: list[dict[str, str]] = []

    def error(
        self,
        code: str,
        message: str,
        path: Path | None = None,
    ) -> None:
        item = {"code": code, "message": message}
        if path is not None:
            item["path"] = str(path)
        self.errors.append(item)

    def warning(
        self,
        code: str,
        message: str,
        path: Path | None = None,
    ) -> None:
        item = {"code": code, "message": message}
        if path is not None:
            item["path"] = str(path)
        self.warnings.append(item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signal-dir",
        type=Path,
        default=DEFAULT_SIGNAL_DIR,
        help="Existing signal_analysis result directory",
    )
    parser.add_argument(
        "--sign-dir",
        type=Path,
        default=DEFAULT_SIGN_DIR,
        help="Existing sign_ablation result directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for bandpass_result_audit.json and final Markdown",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a PRELIMINARY report and return zero despite missing evidence",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run three synthetic result-audit scenarios; no real results are read",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def read_json(
    path: Path,
    log: AuditLog,
    code: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        log.error(code, "缺少 JSON 结果文件。", path)
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.error(code, f"JSON 无法读取：{exc}", path)
        return None
    if not isinstance(value, dict):
        log.error(code, "JSON 顶层必须是对象。", path)
        return None
    return value


def read_csv_rows(
    path: Path,
    log: AuditLog,
    code: str,
) -> list[dict[str, str]]:
    if not path.is_file():
        log.error(code, "缺少 CSV 结果文件。", path)
        return []
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        log.error(code, f"CSV 无法读取：{exc}", path)
        return []
    if not rows:
        log.error(code, "CSV 没有数据行。", path)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def as_float(
    value: Any,
    *,
    name: str,
    log: AuditLog,
    code: str,
) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        log.error(code, f"{name} 不是有效数字：{value!r}")
        return None
    if not math.isfinite(number):
        log.error(code, f"{name} 不是有限数字：{value!r}")
        return None
    return number


def as_int(
    value: Any,
    *,
    name: str,
    log: AuditLog,
    code: str,
) -> int | None:
    number = as_float(value, name=name, log=log, code=code)
    if number is None:
        return None
    integer = int(number)
    if number != integer:
        log.error(code, f"{name} 必须是整数：{value!r}")
        return None
    return integer


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def official_flag(payload: dict[str, Any] | None) -> bool | None:
    if not payload:
        return None
    direct = as_bool(payload.get("official_complete"))
    if direct is not None:
        return direct
    nested = payload.get("completion_status")
    if isinstance(nested, dict):
        return as_bool(nested.get("official_complete"))
    return None


def figure_file_valid(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    try:
        if path.stat().st_size == 0:
            return False, "empty"
        with path.open("rb") as handle:
            prefix = handle.read(512)
    except OSError as exc:
        return False, f"read_error: {exc}"
    suffix = path.suffix.lower()
    if suffix == ".png" and not prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return False, "invalid_png_signature"
    if suffix == ".pdf" and not prefix.startswith(b"%PDF"):
        return False, "invalid_pdf_signature"
    if suffix == ".svg" and b"<svg" not in prefix.lower():
        return False, "invalid_svg_header"
    return True, "ok"


def audit_figures(
    signal_dir: Path,
    sign_dir: Path,
    log: AuditLog,
) -> dict[str, Any]:
    roots = {"signal": signal_dir, "sign": sign_dir}
    groups = []
    for group_id, root_name, relative_stem in REQUIRED_FIGURES:
        files = []
        group_ok = True
        for suffix in FIGURE_SUFFIXES:
            path = (roots[root_name] / relative_stem).with_suffix(suffix)
            valid, reason = figure_file_valid(path)
            files.append(
                {
                    "format": suffix[1:],
                    "path": str(path),
                    "valid": valid,
                    "reason": reason,
                }
            )
            if not valid:
                group_ok = False
                log.error(
                    "figure_incomplete",
                    f"科研图组 {group_id} 的 {suffix} 文件缺失或无效：{reason}",
                    path,
                )
        groups.append({"group": group_id, "complete": group_ok, "files": files})

    supplemental_files = []
    supplemental_present = False
    supplemental_complete = True
    for suffix in FIGURE_SUFFIXES:
        path = (signal_dir / SUPPLEMENTAL_FIGURE).with_suffix(suffix)
        present = path.exists()
        supplemental_present = supplemental_present or present
        valid, reason = figure_file_valid(path)
        supplemental_files.append(
            {
                "format": suffix[1:],
                "path": str(path),
                "valid": valid,
                "reason": reason,
            }
        )
        if present and not valid:
            supplemental_complete = False
    if supplemental_present and not all(item["valid"] for item in supplemental_files):
        supplemental_complete = False
        log.warning(
            "supplemental_figure_incomplete",
            "补充的上升/下降科研图没有同时具备 PNG、PDF、SVG。",
            signal_dir / SUPPLEMENTAL_FIGURE,
        )
    return {
        "required_group_count": len(REQUIRED_FIGURES),
        "required_groups": groups,
        "all_required_complete": all(item["complete"] for item in groups),
        "supplemental_direction_figure": {
            "present": supplemental_present,
            "complete_if_present": supplemental_complete,
            "files": supplemental_files,
        },
    }


def audit_elder(signal_dir: Path, log: AuditLog) -> dict[str, Any]:
    summary_path = signal_dir / "elder_bandpass_summary.json"
    sample_path = signal_dir / "elder_bandpass_per_sample.csv"
    summary = read_json(summary_path, log, "elder_summary_missing")
    rows = read_csv_rows(sample_path, log, "elder_samples_missing")
    result: dict[str, Any] = {
        "summary_path": str(summary_path),
        "sample_csv_path": str(sample_path),
        "complete": False,
        "mechanism_support": "not_evaluable",
    }
    if summary is None:
        return result
    if official_flag(summary) is not True:
        log.error(
            "elder_not_official_complete",
            "Elder 摘要没有明确记录 official_complete=true。",
            summary_path,
        )

    threshold = as_int(
        summary.get("source_min_length"),
        name="Elder source_min_length",
        log=log,
        code="elder_threshold_invalid",
    )
    valid_samples = as_int(
        summary.get("valid_samples"),
        name="Elder valid_samples",
        log=log,
        code="elder_counts_invalid",
    )
    bypass_samples = as_int(
        summary.get("bypass_samples"),
        name="Elder bypass_samples",
        log=log,
        code="elder_counts_invalid",
    )
    exact_matches = as_int(
        summary.get("bypass_exact_matches"),
        name="Elder bypass_exact_matches",
        log=log,
        code="elder_counts_invalid",
    )
    bypass_rate = as_float(
        summary.get("bypass_sample_rate"),
        name="Elder bypass_sample_rate",
        log=log,
        code="elder_rate_invalid",
    )
    exact_rate = as_float(
        summary.get("bypass_exact_match_rate"),
        name="Elder bypass_exact_match_rate",
        log=log,
        code="elder_rate_invalid",
    )

    if threshold != 28:
        log.error(
            "elder_threshold_wrong",
            f"源码旁路阈值应为严格 T<28，结果记录为 {threshold!r}。",
            summary_path,
        )
    if valid_samples is not None and valid_samples <= 0:
        log.error(
            "elder_no_samples",
            "Elder 没有有效样本。",
            summary_path,
        )
    if (
        valid_samples is not None
        and bypass_samples is not None
        and not 0 <= bypass_samples <= valid_samples
    ):
        log.error(
            "elder_counts_inconsistent",
            "Elder 旁路样本数超出有效样本数。",
            summary_path,
        )
    if (
        valid_samples
        and bypass_samples is not None
        and bypass_rate is not None
        and not close(bypass_rate, bypass_samples / valid_samples)
    ):
        log.error(
            "elder_rate_inconsistent",
            "Elder 旁路比例与样本计数不一致。",
            summary_path,
        )
    if bypass_samples is not None and exact_matches != bypass_samples:
        log.error(
            "elder_bypass_not_identical",
            "并非所有 T<28 样本都与 Bandpass 输入逐值完全相同。",
            summary_path,
        )
    if exact_rate is None or not close(exact_rate, 1.0):
        log.error(
            "elder_exact_rate_not_one",
            "Elder 旁路样本的完全相同比例不是 100%。",
            summary_path,
        )

    discovery = summary.get("discovery")
    if isinstance(discovery, dict):
        failed_files = as_int(
            discovery.get("failed_files", 0),
            name="Elder discovery.failed_files",
            log=log,
            code="elder_counts_invalid",
        )
        if failed_files:
            log.error(
                "elder_file_failures",
                f"Elder 有 {failed_files} 个文件读取失败。",
                summary_path,
            )

    if valid_samples is not None and len(rows) != valid_samples:
        log.error(
            "elder_csv_count_mismatch",
            f"Elder CSV 有 {len(rows)} 行，但摘要记录 {valid_samples} 个样本。",
            sample_path,
        )
    row_errors = 0
    for index, row in enumerate(rows, start=2):
        try:
            frames = int(row["frames"])
            predicted = as_bool(row["predicted_bypass"])
            identical = as_bool(row["output_exactly_equal_to_input"])
            max_change = float(row["max_absolute_change"])
        except (KeyError, TypeError, ValueError):
            row_errors += 1
            continue
        if predicted != (frames < 28):
            row_errors += 1
        if predicted and (identical is not True or not close(max_change, 0.0)):
            row_errors += 1
    if row_errors:
        log.error(
            "elder_sample_rows_invalid",
            f"Elder 逐样本 CSV 有 {row_errors} 行不符合 T<28/完全相同规则。",
            sample_path,
        )

    bypass_fraction = bypass_rate if bypass_rate is not None else 0.0
    mechanism = "strong_support" if bypass_fraction >= 0.50 else "not_supported"
    current_error_codes = {
        "elder_threshold_invalid",
        "elder_threshold_wrong",
        "elder_counts_invalid",
        "elder_no_samples",
        "elder_counts_inconsistent",
        "elder_rate_invalid",
        "elder_rate_inconsistent",
        "elder_bypass_not_identical",
        "elder_exact_rate_not_one",
        "elder_file_failures",
        "elder_csv_count_mismatch",
        "elder_sample_rows_invalid",
        "elder_summary_missing",
        "elder_samples_missing",
        "elder_not_official_complete",
    }
    complete = not any(item["code"] in current_error_codes for item in log.errors)
    result.update(
        {
            "complete": complete,
            "source_min_length": threshold,
            "valid_samples": valid_samples,
            "bypass_samples": bypass_samples,
            "bypass_sample_rate": bypass_rate,
            "bypass_exact_matches": exact_matches,
            "bypass_exact_match_rate": exact_rate,
            "high_bypass_support_threshold": 0.50,
            "mechanism_support": mechanism if complete else "not_evaluable",
        }
    )
    return result


def audit_xrf_signal(signal_dir: Path, log: AuditLog) -> dict[str, Any]:
    summary_path = signal_dir / "xrf55_negative_summary.json"
    method_csv_path = signal_dir / "xrf55_negative_method_summary.csv"
    summary = read_json(summary_path, log, "xrf_summary_missing")
    csv_rows = read_csv_rows(
        method_csv_path,
        log,
        "xrf_method_csv_missing",
    )
    result: dict[str, Any] = {
        "summary_path": str(summary_path),
        "method_csv_path": str(method_csv_path),
        "complete": False,
        "mechanism_support": "not_evaluable",
        "methods": [],
    }
    if summary is None:
        return result
    if official_flag(summary) is not True:
        log.error(
            "xrf_not_official_complete",
            "XRF 摘要没有明确记录 official_complete=true。",
            summary_path,
        )
    raw_methods = summary.get("methods")
    if not isinstance(raw_methods, list):
        log.error(
            "xrf_methods_invalid",
            "XRF 摘要中的 methods 不是列表。",
            summary_path,
        )
        return result
    methods = [row for row in raw_methods if isinstance(row, dict)]
    by_method: dict[str, dict[str, Any]] = {}
    for row in methods:
        method = str(row.get("method", ""))
        if method in by_method:
            log.error(
                "xrf_method_duplicate",
                f"XRF 方法 {method!r} 重复。",
                summary_path,
            )
        by_method[method] = row
    missing = sorted(set(EXPECTED_METHODS) - set(by_method))
    extra = sorted(set(by_method) - set(EXPECTED_METHODS))
    if missing:
        log.error(
            "xrf_methods_missing",
            f"XRF 缺少方法：{', '.join(missing)}。",
            summary_path,
        )
    if extra:
        log.error(
            "xrf_methods_extra",
            f"XRF 出现未预注册方法：{', '.join(extra)}。",
            summary_path,
        )

    audited_methods = []
    sample_counts = set()
    required_rates = (
        "element_weighted_meaningful_rate",
        "mean_negative_energy_fraction",
        "element_weighted_abs_slope_direction_changed_or_lost_rate",
    )
    for method in EXPECTED_METHODS:
        row = by_method.get(method)
        if row is None:
            continue
        samples = as_int(
            row.get("samples"),
            name=f"{method}.samples",
            log=log,
            code="xrf_method_value_invalid",
        )
        if samples is not None:
            sample_counts.add(samples)
            if samples <= 0:
                log.error(
                    "xrf_method_no_samples",
                    f"{method} 没有有效样本。",
                    summary_path,
                )
        values: dict[str, float | None] = {}
        for field in required_rates:
            value = as_float(
                row.get(field),
                name=f"{method}.{field}",
                log=log,
                code="xrf_method_value_invalid",
            )
            values[field] = value
            if value is not None and not 0.0 <= value <= 1.0:
                log.error(
                    "xrf_method_rate_out_of_range",
                    f"{method}.{field} 不在 [0,1]。",
                    summary_path,
                )
        audited_methods.append(
            {
                "method": method,
                "samples": samples,
                **values,
            }
        )
    if len(sample_counts) > 1:
        log.error(
            "xrf_method_sample_counts_differ",
            f"七种方法的样本数不一致：{sorted(sample_counts)}。",
            summary_path,
        )

    discovery = summary.get("discovery")
    fully_analyzed = None
    raw_nonnegative = None
    selected_user_ids: list[int] = []
    if not isinstance(discovery, dict):
        log.error(
            "xrf_discovery_missing",
            "XRF 摘要缺少 discovery。",
            summary_path,
        )
    else:
        fully_analyzed = as_int(
            discovery.get("fully_analyzed_records"),
            name="fully_analyzed_records",
            log=log,
            code="xrf_discovery_invalid",
        )
        raw_nonnegative = as_int(
            discovery.get("raw_nonnegative_records"),
            name="raw_nonnegative_records",
            log=log,
            code="xrf_discovery_invalid",
        )
        try:
            selected_user_ids = [
                int(value) for value in discovery.get("selected_user_ids", [])
            ]
            available_user_ids = [
                int(value) for value in discovery.get("available_user_ids", [])
            ]
        except (TypeError, ValueError):
            selected_user_ids = []
            available_user_ids = []
        if (
            len(selected_user_ids) != 3
            or selected_user_ids != available_user_ids[:3]
            or as_int(
                discovery.get("requested_user_count"),
                name="XRF discovery.requested_user_count",
                log=log,
                code="xrf_discovery_invalid",
            )
            != 3
        ):
            log.error(
                "xrf_user_scope_invalid",
                "XRF 信号统计必须使用数值排序后的前三个用户。",
                summary_path,
            )
        selected_files = as_int(
            discovery.get("selected_files_before_limit"),
            name="XRF discovery.selected_files_before_limit",
            log=log,
            code="xrf_discovery_invalid",
        )
        expected_files = as_int(
            discovery.get("expected_selected_files"),
            name="XRF discovery.expected_selected_files",
            log=log,
            code="xrf_discovery_invalid",
        )
        if (
            as_bool(discovery.get("complete_user_action_repetition_grid")) is not True
            or selected_files != 3300
            or expected_files != 3300
        ):
            log.error(
                "xrf_grid_incomplete",
                "XRF 信号统计不是完整的 3用户×55动作×20重复=3300 文件。",
                summary_path,
            )
        for field in (
            "failed_files",
            "partially_analyzed_records",
            "rejected_corrupt_rows_at_summary",
        ):
            count = as_int(
                discovery.get(field, 0),
                name=f"XRF discovery.{field}",
                log=log,
                code="xrf_discovery_invalid",
            )
            if count is None:
                continue
            if count:
                log.error(
                    "xrf_incomplete_records",
                    f"XRF discovery.{field}={count}，不是完整官方结果。",
                    summary_path,
                )
        if fully_analyzed is not None and fully_analyzed <= 0:
            log.error(
                "xrf_no_complete_records",
                "XRF 没有完整分析记录。",
                summary_path,
            )
        if (
            fully_analyzed is not None
            and sample_counts
            and sample_counts != {fully_analyzed}
        ):
            log.error(
                "xrf_discovery_count_mismatch",
                "XRF 方法样本数与 fully_analyzed_records 不一致。",
                summary_path,
            )
        if (
            fully_analyzed is not None
            and raw_nonnegative is not None
            and raw_nonnegative != fully_analyzed
        ):
            log.warning(
                "xrf_raw_not_all_nonnegative",
                "并非所有 XRF 输入记录都全非负；正幅度机制结论需降级。",
                summary_path,
            )

    csv_by_method = {row.get("method", ""): row for row in csv_rows}
    if set(csv_by_method) != set(EXPECTED_METHODS):
        log.error(
            "xrf_method_csv_set_mismatch",
            "XRF 方法级 CSV 不是完整七方法集合。",
            method_csv_path,
        )
    for item in audited_methods:
        method = str(item["method"])
        csv_row = csv_by_method.get(method)
        if csv_row is None:
            continue
        for field in required_rates:
            left = item[field]
            right = as_float(
                csv_row.get(field),
                name=f"CSV {method}.{field}",
                log=log,
                code="xrf_method_csv_invalid",
            )
            if isinstance(left, float) and right is not None and not close(left, right):
                log.error(
                    "xrf_method_csv_mismatch",
                    f"XRF JSON 与 CSV 的 {method}.{field} 不一致。",
                    method_csv_path,
                )

    error_codes = {
        code
        for code in (
            "xrf_summary_missing",
            "xrf_method_csv_missing",
            "xrf_not_official_complete",
            "xrf_methods_invalid",
            "xrf_method_duplicate",
            "xrf_methods_missing",
            "xrf_methods_extra",
            "xrf_method_value_invalid",
            "xrf_method_no_samples",
            "xrf_method_rate_out_of_range",
            "xrf_method_sample_counts_differ",
            "xrf_discovery_missing",
            "xrf_discovery_invalid",
            "xrf_incomplete_records",
            "xrf_no_complete_records",
            "xrf_discovery_count_mismatch",
            "xrf_user_scope_invalid",
            "xrf_grid_incomplete",
            "xrf_method_csv_set_mismatch",
            "xrf_method_csv_invalid",
            "xrf_method_csv_mismatch",
        )
    }
    complete = not any(item["code"] in error_codes for item in log.errors)
    meaningful = {
        str(row["method"]): row["element_weighted_meaningful_rate"]
        for row in audited_methods
    }
    changed = {
        str(row["method"]): (
            row["element_weighted_abs_slope_direction_changed_or_lost_rate"]
        )
        for row in audited_methods
    }
    mechanism = "not_evaluable"
    separation = None
    if complete and all(
        isinstance(meaningful.get(method), float) for method in EXPECTED_METHODS
    ):
        bandpass_rates = [
            float(meaningful["bandpass_fs1000"]),
            float(meaningful["bandpass_fs200"]),
        ]
        other_rates = [
            float(meaningful[method])
            for method in EXPECTED_METHODS
            if not method.startswith("bandpass")
        ]
        separation = min(bandpass_rates) - max(other_rates)
        all_nonnegative = (
            fully_analyzed is not None and raw_nonnegative == fully_analyzed
        )
        direction_present = all(
            isinstance(changed.get(method), float) and float(changed[method]) > 0
            for method in ("bandpass_fs1000", "bandpass_fs200")
        )
        if all_nonnegative and min(bandpass_rates) > 0 and direction_present:
            mechanism = "strong_support"
        else:
            mechanism = "not_supported"
    result.update(
        {
            "complete": complete,
            "fully_analyzed_records": fully_analyzed,
            "raw_nonnegative_records": raw_nonnegative,
            "selected_user_ids": selected_user_ids,
            "methods": audited_methods,
            "bandpass_vs_other_negative_rate_separation": separation,
            "mechanism_support": mechanism,
        }
    )
    return result


def referenced_result_file(
    value: str,
    sign_dir: Path,
) -> Path | None:
    if not value.strip():
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = (REPO_ROOT / path, sign_dir / path)
    return next(
        (candidate for candidate in candidates if candidate.exists()), candidates[0]
    )


def sample_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def recompute_effects(
    by_case_seed: dict[tuple[str, int], float],
) -> dict[str, dict[str, Any]]:
    effects: dict[str, dict[str, Any]] = {}
    per_seed: dict[str, dict[int, float]] = {}
    for effect, (minuend, subtrahend) in EFFECT_DEFINITIONS.items():
        values = {
            seed: (by_case_seed[(minuend, seed)] - by_case_seed[(subtrahend, seed)])
            for seed in EXPECTED_SEEDS
            if (minuend, seed) in by_case_seed and (subtrahend, seed) in by_case_seed
        }
        per_seed[effect] = values
    interaction = {
        seed: (
            per_seed["sign_effect_fs200"][seed] - per_seed["sign_effect_fs1000"][seed]
        )
        for seed in EXPECTED_SEEDS
        if seed in per_seed["sign_effect_fs200"]
        and seed in per_seed["sign_effect_fs1000"]
    }
    per_seed[INTERACTION_EFFECT] = interaction
    for effect, seed_values in per_seed.items():
        values = list(seed_values.values())
        effects[effect] = {
            "effect": effect,
            "seeds": sorted(seed_values),
            "n_paired_seeds": len(values),
            "per_seed_delta_test_acc": {
                str(seed): seed_values[seed] for seed in sorted(seed_values)
            },
            "mean_delta_test_acc": (statistics.fmean(values) if values else None),
            "std_delta_test_acc": sample_stdev(values) if values else None,
            "positive_seed_fraction": (
                sum(value > 0 for value in values) / len(values) if values else None
            ),
        }
    return effects


def training_completion_payload(
    sign_dir: Path,
    settings: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    for name in (
        "completion_status.json",
        "experiment_completion.json",
        "training_completion_status.json",
    ):
        path = sign_dir / name
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None, path
            if isinstance(value, dict):
                return value, path
            return None, path
    if settings is not None and (
        "official_complete" in settings or "completion_status" in settings
    ):
        return settings, sign_dir / "experiment_settings.json"
    return None, None


def audit_training(sign_dir: Path, log: AuditLog) -> dict[str, Any]:
    settings_path = sign_dir / "experiment_settings.json"
    training_path = sign_dir / "training_summary.csv"
    aggregate_path = sign_dir / "training_aggregate.csv"
    paired_path = sign_dir / "paired_effects.csv"
    paired_seed_path = sign_dir / "paired_seed_effects.csv"
    settings = read_json(settings_path, log, "training_settings_missing")
    training_rows = read_csv_rows(
        training_path,
        log,
        "training_summary_missing",
    )
    aggregate_rows = read_csv_rows(
        aggregate_path,
        log,
        "training_aggregate_missing",
    )
    paired_rows = read_csv_rows(
        paired_path,
        log,
        "paired_effects_missing",
    )
    paired_seed_rows = read_csv_rows(
        paired_seed_path,
        log,
        "paired_seed_effects_missing",
    )
    result: dict[str, Any] = {
        "complete": False,
        "settings_path": str(settings_path),
        "training_summary_path": str(training_path),
        "aggregate_path": str(aggregate_path),
        "paired_effects_path": str(paired_path),
        "paired_seed_effects_path": str(paired_seed_path),
        "expected_cases": list(EXPECTED_CASES),
        "expected_seeds": list(EXPECTED_SEEDS),
        "cases": [],
        "effects": [],
    }
    if settings is None:
        return result

    completion_payload, completion_path = training_completion_payload(
        sign_dir,
        settings,
    )
    result["completion_status_path"] = (
        str(completion_path) if completion_path is not None else None
    )
    if official_flag(completion_payload) is not True:
        log.error(
            "training_not_official_complete",
            "训练完成状态没有明确记录 official_complete=true。",
            completion_path or settings_path,
        )

    case_hashes = settings.get("case_hashes")
    if not isinstance(case_hashes, dict):
        log.error(
            "training_case_hashes_missing",
            "experiment_settings.json 缺少 case_hashes。",
            settings_path,
        )
        case_hashes = {}
    missing_hashes = sorted(set(EXPECTED_CASES) - set(case_hashes))
    if missing_hashes:
        log.error(
            "training_case_hashes_incomplete",
            f"缺少 case hash：{', '.join(missing_hashes)}。",
            settings_path,
        )
    epochs = settings.get("epochs")
    user_count = settings.get("user_count")
    if (
        as_int(
            epochs,
            name="settings.epochs",
            log=log,
            code="training_protocol_invalid",
        )
        != 50
    ):
        log.error(
            "training_epochs_not_official",
            "官方实验必须使用 epochs=50。",
            settings_path,
        )
    if (
        as_int(
            user_count,
            name="settings.user_count",
            log=log,
            code="training_protocol_invalid",
        )
        != 3
    ):
        log.error(
            "training_user_count_not_official",
            "官方实验必须使用 user_count=3。",
            settings_path,
        )

    candidates: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in training_rows:
        case = row.get("case", "")
        try:
            seed = int(row.get("model_seed", ""))
        except ValueError:
            continue
        if case not in EXPECTED_CASES or seed not in EXPECTED_SEEDS:
            continue
        if row.get("status") != "ok":
            continue
        if row.get("config_hash") != str(case_hashes.get(case, "")):
            continue
        candidates.setdefault((case, seed), []).append(row)

    selected: dict[tuple[str, int], dict[str, str]] = {}
    missing_pairs = []
    artifact_failures = []
    for case in EXPECTED_CASES:
        for seed in EXPECTED_SEEDS:
            key = (case, seed)
            rows = candidates.get(key, [])
            if not rows:
                missing_pairs.append({"case": case, "seed": seed})
                continue
            if len(rows) > 1:
                log.warning(
                    "training_duplicate_success",
                    f"{case}/seed={seed} 有 {len(rows)} 条当前配置成功记录；"
                    "审计采用最后一条。",
                    training_path,
                )
            row = rows[-1]
            selected[key] = row
            test_acc = as_float(
                row.get("test_acc"),
                name=f"{case}/seed={seed}.test_acc",
                log=log,
                code="training_accuracy_invalid",
            )
            if test_acc is not None and not 0.0 <= test_acc <= 1.0:
                log.error(
                    "training_accuracy_out_of_range",
                    f"{case}/seed={seed} 的 test_acc 不在 [0,1]。",
                    training_path,
                )
            row_epochs = as_int(
                row.get("epochs"),
                name=f"{case}/seed={seed}.epochs",
                log=log,
                code="training_protocol_invalid",
            )
            if row_epochs != 50:
                log.error(
                    "training_row_epochs_not_official",
                    f"{case}/seed={seed} 不是 50 epochs。",
                    training_path,
                )
            for field in (
                "checkpoint",
                "predictions_file",
                "test_sample_manifest",
            ):
                path = referenced_result_file(row.get(field, ""), sign_dir)
                if path is None or not path.is_file() or path.stat().st_size == 0:
                    artifact_failures.append(
                        {
                            "case": case,
                            "seed": seed,
                            "field": field,
                            "path": str(path) if path is not None else "",
                        }
                    )
    if missing_pairs:
        preview = ", ".join(
            f"{item['case']}/seed={item['seed']}" for item in missing_pairs[:8]
        )
        suffix = " …" if len(missing_pairs) > 8 else ""
        log.error(
            "training_seeds_incomplete",
            f"官方 7×5 缺少 {len(missing_pairs)} 组：{preview}{suffix}",
            training_path,
        )
    if artifact_failures:
        log.error(
            "training_artifacts_missing",
            f"{len(artifact_failures)} 个 seed 结果引用文件缺失或为空。",
            training_path,
        )

    by_case_seed: dict[tuple[str, int], float] = {}
    for key, row in selected.items():
        try:
            by_case_seed[key] = float(row["test_acc"])
        except (KeyError, ValueError):
            continue
    recomputed = recompute_effects(by_case_seed)

    case_results = []
    aggregate_by_case = {row.get("case", ""): row for row in aggregate_rows}
    for case in EXPECTED_CASES:
        values = [
            by_case_seed[(case, seed)]
            for seed in EXPECTED_SEEDS
            if (case, seed) in by_case_seed
        ]
        item = {
            "case": case,
            "completed_seeds": [
                seed for seed in EXPECTED_SEEDS if (case, seed) in by_case_seed
            ],
            "n_model_seeds": len(values),
            "mean_test_acc": statistics.fmean(values) if values else None,
            "std_test_acc": sample_stdev(values) if values else None,
        }
        aggregate = aggregate_by_case.get(case)
        if aggregate is None:
            log.error(
                "training_aggregate_case_missing",
                f"training_aggregate.csv 缺少 {case}。",
                aggregate_path,
            )
        elif values:
            reported_n = as_int(
                aggregate.get("n_model_seeds"),
                name=f"aggregate {case}.n_model_seeds",
                log=log,
                code="training_aggregate_invalid",
            )
            reported_mean = as_float(
                aggregate.get("mean_test_acc"),
                name=f"aggregate {case}.mean_test_acc",
                log=log,
                code="training_aggregate_invalid",
            )
            if reported_n != len(values) or (
                reported_mean is not None
                and not close(reported_mean, statistics.fmean(values))
            ):
                log.error(
                    "training_aggregate_mismatch",
                    f"{case} 的 aggregate 与逐 seed 结果不一致。",
                    aggregate_path,
                )
        case_results.append(item)

    reported_effects = {row.get("effect", ""): row for row in paired_rows}
    reported_seed_effects: dict[tuple[str, int], dict[str, str]] = {}
    for row in paired_seed_rows:
        effect = row.get("effect", "")
        try:
            seed = int(row.get("model_seed", ""))
        except ValueError:
            continue
        reported_seed_effects[(effect, seed)] = row

    for effect in ALL_EFFECTS:
        computed = recomputed[effect]
        reported = reported_effects.get(effect)
        if reported is None:
            log.error(
                "paired_effect_missing",
                f"paired_effects.csv 缺少 {effect}。",
                paired_path,
            )
            continue
        try:
            reported_seeds = {
                int(value)
                for value in str(reported.get("seeds", "")).split(",")
                if value.strip()
            }
        except ValueError:
            reported_seeds = set()
            log.error(
                "paired_effect_invalid",
                f"{effect}.seeds 不是合法整数列表。",
                paired_path,
            )
        if reported_seeds != set(EXPECTED_SEEDS):
            log.error(
                "paired_effect_seeds_incomplete",
                f"{effect} 的配对 seed 不是预注册五个。",
                paired_path,
            )
        reported_n = as_int(
            reported.get("n_paired_seeds"),
            name=f"{effect}.n_paired_seeds",
            log=log,
            code="paired_effect_invalid",
        )
        reported_mean = as_float(
            reported.get("mean_delta_test_acc"),
            name=f"{effect}.mean_delta_test_acc",
            log=log,
            code="paired_effect_invalid",
        )
        if reported_n != computed["n_paired_seeds"] or (
            reported_mean is not None
            and computed["mean_delta_test_acc"] is not None
            and not close(reported_mean, computed["mean_delta_test_acc"])
        ):
            log.error(
                "paired_effect_mismatch",
                f"{effect} 与逐 seed 重新计算结果不一致。",
                paired_path,
            )
        for seed in EXPECTED_SEEDS:
            seed_row = reported_seed_effects.get((effect, seed))
            expected_delta = computed["per_seed_delta_test_acc"].get(str(seed))
            if seed_row is None:
                log.error(
                    "paired_seed_effect_missing",
                    f"paired_seed_effects.csv 缺少 {effect}/seed={seed}。",
                    paired_seed_path,
                )
                continue
            reported_delta = as_float(
                seed_row.get("delta_test_acc"),
                name=f"{effect}/seed={seed}.delta_test_acc",
                log=log,
                code="paired_seed_effect_invalid",
            )
            if (
                reported_delta is not None
                and expected_delta is not None
                and not close(reported_delta, expected_delta)
            ):
                log.error(
                    "paired_seed_effect_mismatch",
                    f"{effect}/seed={seed} 与训练结果不一致。",
                    paired_seed_path,
                )
        computed["bootstrap_ci_low"] = as_float(
            reported.get("bootstrap_ci_low"),
            name=f"{effect}.bootstrap_ci_low",
            log=log,
            code="paired_effect_invalid",
        )
        computed["bootstrap_ci_high"] = as_float(
            reported.get("bootstrap_ci_high"),
            name=f"{effect}.bootstrap_ci_high",
            log=log,
            code="paired_effect_invalid",
        )

    training_error_codes = {
        "training_settings_missing",
        "training_summary_missing",
        "training_aggregate_missing",
        "paired_effects_missing",
        "paired_seed_effects_missing",
        "training_not_official_complete",
        "training_case_hashes_missing",
        "training_case_hashes_incomplete",
        "training_protocol_invalid",
        "training_epochs_not_official",
        "training_user_count_not_official",
        "training_accuracy_invalid",
        "training_accuracy_out_of_range",
        "training_row_epochs_not_official",
        "training_seeds_incomplete",
        "training_artifacts_missing",
        "training_aggregate_case_missing",
        "training_aggregate_invalid",
        "training_aggregate_mismatch",
        "paired_effect_missing",
        "paired_effect_seeds_incomplete",
        "paired_effect_invalid",
        "paired_effect_mismatch",
        "paired_seed_effect_missing",
        "paired_seed_effect_invalid",
        "paired_seed_effect_mismatch",
    }
    complete = not any(item["code"] in training_error_codes for item in log.errors)
    result.update(
        {
            "complete": complete,
            "official_complete": official_flag(completion_payload),
            "selected_user_ids": settings.get("selected_user_ids", []),
            "missing_case_seed_pairs": missing_pairs,
            "artifact_failures": artifact_failures,
            "cases": case_results,
            "effects": [recomputed[effect] for effect in ALL_EFFECTS],
        }
    )
    return result


def find_signal_official_payload(
    signal_dir: Path,
    overall: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    for name in (
        "completion_status.json",
        "signal_completion_status.json",
    ):
        path = signal_dir / name
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None, path
            if isinstance(value, dict):
                return value, path
            return None, path
    if overall is not None and (
        "official_complete" in overall or "completion_status" in overall
    ):
        return overall, signal_dir / "signal_study_summary.json"
    return None, None


def verdict_from_evidence(
    complete: bool,
    elder: dict[str, Any],
    xrf: dict[str, Any],
    training: dict[str, Any],
) -> str:
    if not complete:
        return "incomplete"
    effects = {str(row["effect"]): row for row in training.get("effects", [])}
    sign = effects.get("sign_effect_fs200")
    if not sign:
        return "incomplete"
    mean = sign.get("mean_delta_test_acc")
    positive = sign.get("positive_seed_fraction")
    ci_low = sign.get("bootstrap_ci_low")
    if not all(isinstance(value, (int, float)) for value in (mean, positive, ci_low)):
        return "incomplete"
    if (
        float(mean) > 0
        and float(positive) >= 0.8
        and float(ci_low) > 0
        and elder.get("mechanism_support") == "strong_support"
        and xrf.get("mechanism_support") == "strong_support"
    ):
        return "strong_support"
    if float(mean) <= 0:
        return "not_supported"
    return "partial_support"


def audit_results(
    signal_dir: Path,
    sign_dir: Path,
) -> dict[str, Any]:
    log = AuditLog()
    overall_path = signal_dir / "signal_study_summary.json"
    overall = read_json(overall_path, log, "signal_overall_missing")
    official_payload, official_path = find_signal_official_payload(
        signal_dir,
        overall,
    )
    if official_flag(official_payload) is not True:
        log.error(
            "signal_not_official_complete",
            "信号分析完成状态没有明确记录 official_complete=true。",
            official_path or overall_path,
        )

    figures = audit_figures(signal_dir, sign_dir, log)
    elder = audit_elder(signal_dir, log)
    xrf = audit_xrf_signal(signal_dir, log)
    training = audit_training(sign_dir, log)
    if xrf.get("selected_user_ids") != training.get("selected_user_ids"):
        log.error(
            "xrf_user_scope_mismatch",
            "信号统计与分类消融使用的 XRF55 用户不一致。",
        )
    complete = not log.errors
    verdict = verdict_from_evidence(complete, elder, xrf, training)
    return {
        "audit_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "signal_dir": str(signal_dir),
            "sign_dir": str(sign_dir),
            "dataset_files_read": False,
        },
        "status": "COMPLETE" if complete else "PRELIMINARY",
        "official_complete": complete,
        "scientific_verdict": verdict,
        "signal_official_status_path": (
            str(official_path) if official_path is not None else None
        ),
        "figures": figures,
        "elderAL": elder,
        "xrf55_signal": xrf,
        "training_ablation": training,
        "errors": log.errors,
        "warnings": log.warnings,
    }


def percentage(value: Any, digits: int = 2) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "N/A"
    return f"{100.0 * float(value):.{digits}f}%"


def percentage_points(value: Any, digits: int = 2) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "N/A"
    return f"{100.0 * float(value):+.{digits}f}"


def verdict_chinese(verdict: str) -> str:
    return {
        "strong_support": "完整，且强支持符号折叠是第二个性能原因",
        "partial_support": "完整，但只得到部分支持，不能下强因果结论",
        "not_supported": "完整，但分类结果不支持符号折叠导致性能下降",
        "incomplete": "证据未完成，只能作为初步结果",
    }.get(verdict, verdict)


def render_markdown(audit: dict[str, Any]) -> str:
    status = str(audit["status"])
    verdict = str(audit["scientific_verdict"])
    elder = audit["elderAL"]
    xrf = audit["xrf55_signal"]
    training = audit["training_ablation"]
    effects = {str(row["effect"]): row for row in training.get("effects", [])}
    cases = training.get("cases", [])
    lines = [
        "# Bandpass 服务器实验最终审计与汇报",
        "",
        f"> 状态：**{status}**",
        "",
        f"> 科学判定：**{verdict_chinese(verdict)}**",
        "",
    ]
    if status == "PRELIMINARY":
        lines.extend(
            [
                "当前报告不能当作正式结论。缺失项列在文末；"
                "只有官方范围全部跑完后才能去掉 PRELIMINARY。",
                "",
            ]
        )

    lines.extend(
        [
            "## 一句话结论",
            "",
        ]
    )
    if verdict == "strong_support":
        lines.append(
            "ElderAL 上 Bandpass 看起来还可以，主要因为大量短样本根本没有"
            "执行滤波；XRF55 上 Bandpass 会产生明显正负波动，后续 abs 会丢掉"
            "符号和部分变化方向，而且固定 200 Hz 后保留符号能稳定提高分类结果。"
            "因此，采样率错误和符号折叠是两个叠加原因。"
        )
    elif verdict == "not_supported":
        lines.append(
            "信号层可以看到 Bandpass 的正负波动和 abs 折叠，但固定 200 Hz 后"
            "保留符号没有提高分类准确率。因此不能把符号折叠说成性能下降原因；"
            "它只是一个真实存在的信号处理现象。"
        )
    elif verdict == "partial_support":
        lines.append(
            "保留符号后的平均准确率有提高，但五个 seed 的一致性或置信区间"
            "不够强。可以汇报为“可能原因”，不能汇报为已经确认的因果结论。"
        )
    else:
        lines.append(
            "实验文件或预注册 seed 尚未全部完成，现在只能汇报信号机制，"
            "不能下最终性能归因结论。"
        )

    lines.extend(
        [
            "",
            "## 1. ElderAL：为什么 Bandpass 看起来还可以",
            "",
            f"- 源码阈值：严格 `T < {elder.get('source_min_length', 'N/A')}` "
            "直接返回原始数据。",
            f"- 有效样本：{elder.get('valid_samples', 'N/A')}；旁路样本："
            f"{elder.get('bypass_samples', 'N/A')}，占 "
            f"{percentage(elder.get('bypass_sample_rate'))}。",
            "- 旁路样本处理前后逐值完全相同的比例："
            f"{percentage(elder.get('bypass_exact_match_rate'))}。",
            "",
            "大白话：这部分样本没有真正经过 Bandpass，所以它们不会被滤坏。"
            "因此 ElderAL 上的结果不能直接解释成“Bandpass 特别适合该数据集”。",
            "",
            "## 2. XRF55：去噪后负值和 abs 方向折叠",
            "",
            f"- 与分类实验一致，信号统计使用前三个用户："
            f"{xrf.get('selected_user_ids', [])}。",
            "",
            "| 方法 | 有意义负值率 | 负值能量占比 | abs后方向改变或丢失率 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in xrf.get("methods", []):
        lines.append(
            f"| {row['method']} | "
            f"{percentage(row.get('element_weighted_meaningful_rate'), 3)} | "
            f"{percentage(row.get('mean_negative_energy_fraction'), 3)} | "
            f"{percentage(row.get('element_weighted_abs_slope_direction_changed_or_lost_rate'), 3)} |"
        )
    lines.extend(
        [
            "",
            "- **基线**是静态环境、设备增益和很慢漂移形成的信号底座。"
            "Bandpass 还会去掉 50 Hz 以上成分，所以 `raw−Bandpass` 不能全部"
            "叫基线。",
            "- **正/负**只表示 Bandpass 零中心两侧；**上升/下降**看相邻帧"
            "差值 `x[t]−x[t−1]`。",
            "- 取绝对值会把 `+a` 和 `−a` 变成同一个 `a`；负半轴上的局部"
            "斜率还可能反向，某些跨零变化会被压平。",
            "",
            "## 3. 官方 7 case × 5 seed 分类消融",
            "",
            "| case | 完成 seed | 测试准确率均值 ± SD |",
            "|---|---:|---:|",
        ]
    )
    for row in cases:
        lines.append(
            f"| {row['case']} | {row['n_model_seeds']}/5 | "
            f"{percentage(row.get('mean_test_acc'))} ± "
            f"{percentage(row.get('std_test_acc'))} |"
        )
    lines.extend(
        [
            "",
            "## 4. 关键配对效应",
            "",
            "正数表示定义左边的分支准确率更高。单位为测试准确率百分点。",
            "",
            "| 效应 | 配对 seed | 平均差值 | 正向 seed | 95% bootstrap CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for effect in ALL_EFFECTS:
        row = effects.get(effect, {})
        interval = (
            f"[{percentage_points(row.get('bootstrap_ci_low'))}, "
            f"{percentage_points(row.get('bootstrap_ci_high'))}]"
        )
        lines.append(
            f"| {effect} | {row.get('n_paired_seeds', 0)}/5 | "
            f"{percentage_points(row.get('mean_delta_test_acc'))} | "
            f"{percentage(row.get('positive_seed_fraction'), 0)} | "
            f"{interval} |"
        )

    sign = effects.get("sign_effect_fs200", {})
    sampling = effects.get("sampling_effect_signed", {})
    lines.extend(
        [
            "",
            "最关键的两个数：",
            "",
            "- 固定 200 Hz 后的符号效应（signed−legacy abs）："
            f"{percentage_points(sign.get('mean_delta_test_acc'))} 个百分点。",
            "- 保留符号后的采样率效应（200−1000 Hz）："
            f"{percentage_points(sampling.get('mean_delta_test_acc'))} 个百分点。",
            "",
            "误差条和 bootstrap 区间只反映五个模型训练 seed 的随机性，"
            "不是对整个 XRF55 总体的置信区间。",
            "",
            "## 5. 文件与完整性审计",
            "",
            "| 科研图组 | PNG | PDF | SVG |",
            "|---|---:|---:|---:|",
        ]
    )
    for group in audit["figures"]["required_groups"]:
        status_by_format = {
            item["format"]: "✓" if item["valid"] else "✗" for item in group["files"]
        }
        lines.append(
            f"| {group['group']} | {status_by_format['png']} | "
            f"{status_by_format['pdf']} | {status_by_format['svg']} |"
        )
    lines.extend(["", "### 未完成或不一致项", ""])
    if audit["errors"]:
        for item in audit["errors"]:
            location = f"（{item.get('path')}）" if item.get("path") else ""
            lines.append(f"- **{item['code']}**：{item['message']}{location}")
    else:
        lines.append("- 无。官方完整性门全部通过。")
    if audit["warnings"]:
        lines.extend(["", "### 警告", ""])
        for item in audit["warnings"]:
            location = f"（{item.get('path')}）" if item.get("path") else ""
            lines.append(f"- **{item['code']}**：{item['message']}{location}")
    return "\n".join(lines) + "\n"


def write_outputs(audit: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "bandpass_result_audit.json", audit)
    (output_dir / "bandpass_final_report.md").write_text(
        render_markdown(audit),
        encoding="utf-8",
    )


def write_csv_file(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_fake_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".png":
        path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-audit-figure")
    elif path.suffix == ".pdf":
        path.write_bytes(b"%PDF-1.4\n% synthetic audit figure\n%%EOF\n")
    else:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"></svg>\n',
            encoding="utf-8",
        )


def synthetic_effect_rows(
    accuracies: dict[tuple[str, int], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recomputed = recompute_effects(accuracies)
    aggregate_rows = []
    seed_rows = []
    for effect in ALL_EFFECTS:
        row = recomputed[effect]
        values = list(row["per_seed_delta_test_acc"].values())
        mean = statistics.fmean(values) if values else 0.0
        spread = min(abs(value) for value in values) if values else 0.0
        if values and all(value > 0 for value in values):
            low, high = spread / 2.0, max(values) * 1.1
        elif values and all(value < 0 for value in values):
            low, high = min(values) * 1.1, -spread / 2.0
        else:
            low, high = min(values, default=0.0), max(values, default=0.0)
        aggregate_rows.append(
            {
                "effect": effect,
                "definition": "synthetic",
                "n_paired_seeds": len(values),
                "seeds": ",".join(str(seed) for seed in row["seeds"]),
                "mean_delta_test_acc": mean,
                "std_delta_test_acc": sample_stdev(values),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "positive_seed_fraction": (
                    sum(value > 0 for value in values) / len(values) if values else 0.0
                ),
            }
        )
        for seed_text, delta in row["per_seed_delta_test_acc"].items():
            seed_rows.append(
                {
                    "effect": effect,
                    "definition": "synthetic",
                    "model_seed": int(seed_text),
                    "component_a_name": "synthetic_a",
                    "component_a_value": 0.0,
                    "component_b_name": "synthetic_b",
                    "component_b_value": 0.0,
                    "delta_test_acc": delta,
                }
            )
    return aggregate_rows, seed_rows


def build_synthetic_results(
    root: Path,
    *,
    sign_support: bool,
    missing_seed: bool,
) -> tuple[Path, Path]:
    signal_dir = root / "signal"
    sign_dir = root / "sign"
    signal_dir.mkdir(parents=True)
    sign_dir.mkdir(parents=True)
    for _, root_name, stem in REQUIRED_FIGURES:
        base = signal_dir if root_name == "signal" else sign_dir
        for suffix in FIGURE_SUFFIXES:
            create_fake_figure((base / stem).with_suffix(suffix))

    elder_rows = []
    for index in range(100):
        bypass = index < 80
        elder_rows.append(
            {
                "file_name": f"elder_{index:03d}.csv",
                "frames": 20 if bypass else 40,
                "values": 200,
                "source_min_length": 28,
                "predicted_bypass": bypass,
                "output_exactly_equal_to_input": bypass,
                "max_absolute_change": 0.0 if bypass else 1.0,
                "mae": 0.0 if bypass else 0.5,
                "nmae": 0.0 if bypass else 0.5,
                "rmse": 0.0 if bypass else 0.5,
                "nrmse": 0.0 if bypass else 0.5,
                "pearson_r": 1.0 if bypass else 0.5,
                "nonfinite_input_count": 0,
                "nonfinite_output_count": 0,
            }
        )
    write_csv_file(signal_dir / "elder_bandpass_per_sample.csv", elder_rows)
    write_json(
        signal_dir / "elder_bandpass_summary.json",
        {
            "dataset": "elderAL",
            "official_complete": True,
            "valid_samples": 100,
            "source_min_length": 28,
            "source_condition": "T < 28",
            "coefficient_lengths": {"a": 9, "b": 9},
            "bypass_samples": 80,
            "bypass_sample_rate": 0.8,
            "bypass_exact_matches": 80,
            "bypass_exact_match_rate": 1.0,
            "discovery": {"failed_files": 0},
        },
    )
    xrf_methods = []
    for method in EXPECTED_METHODS:
        bandpass = method.startswith("bandpass")
        xrf_methods.append(
            {
                "method": method,
                "samples": 10,
                "per_sample_strict_mean": 0.5 if bandpass else 0.0,
                "per_sample_strict_median": 0.5 if bandpass else 0.0,
                "per_sample_meaningful_mean": 0.5 if bandpass else 0.0,
                "per_sample_meaningful_mean_bootstrap_ci_low": (
                    0.49 if bandpass else 0.0
                ),
                "per_sample_meaningful_mean_bootstrap_ci_high": (
                    0.51 if bandpass else 0.0
                ),
                "per_sample_meaningful_q1": 0.49 if bandpass else 0.0,
                "per_sample_meaningful_median": 0.5 if bandpass else 0.0,
                "per_sample_meaningful_q3": 0.51 if bandpass else 0.0,
                "per_sample_meaningful_min": 0.48 if bandpass else 0.0,
                "per_sample_meaningful_max": 0.52 if bandpass else 0.0,
                "element_weighted_strict_rate": 0.5 if bandpass else 0.0,
                "element_weighted_meaningful_rate": (0.5 if bandpass else 0.0),
                "mean_negative_energy_fraction": (0.45 if bandpass else 0.0),
                "element_weighted_rising_step_rate": 0.48,
                "element_weighted_falling_step_rate": 0.48,
                "element_weighted_flat_step_rate": 0.04,
                "element_weighted_zero_crossing_rate": (0.08 if bandpass else 0.0),
                "element_weighted_negative_negative_pair_rate": (
                    0.4 if bandpass else 0.0
                ),
                "element_weighted_abs_slope_direction_disagreement_rate": (
                    0.45 if bandpass else 0.0
                ),
                "comparable_abs_slope_pairs": 1000,
                "element_weighted_abs_slope_direction_changed_or_lost_rate": (
                    0.5 if bandpass else 0.0
                ),
                "meaningful_original_slope_pairs": 1000,
            }
        )
    write_json(
        signal_dir / "xrf55_negative_summary.json",
        {
            "dataset": "xrf55",
            "official_complete": True,
            "discovery": {
                "fully_analyzed_records": 10,
                "raw_nonnegative_records": 10,
                "selected_user_ids": [1, 2, 3],
                "available_user_ids": [1, 2, 3, 4],
                "requested_user_count": 3,
                "selected_files_before_limit": 3300,
                "expected_selected_files": 3300,
                "complete_user_action_repetition_grid": True,
                "failed_files": 0,
                "partially_analyzed_records": 0,
                "rejected_corrupt_rows_at_summary": 0,
            },
            "methods": xrf_methods,
        },
    )
    write_csv_file(
        signal_dir / "xrf55_negative_method_summary.csv",
        xrf_methods,
    )
    write_json(
        signal_dir / "signal_study_summary.json",
        {"official_complete": True},
    )

    hashes = {case: f"hash_{case}" for case in EXPECTED_CASES}
    write_json(
        sign_dir / "experiment_settings.json",
        {
            "official_complete": True,
            "epochs": 50,
            "user_count": 3,
            "selected_user_ids": [1, 2, 3],
            "case_hashes": hashes,
        },
    )
    write_json(sign_dir / "completion_status.json", {"official_complete": True})
    training_rows = []
    accuracies: dict[tuple[str, int], float] = {}
    for case_index, case in enumerate(EXPECTED_CASES):
        for seed_index, seed in enumerate(EXPECTED_SEEDS):
            if missing_seed and case == "bp_fs200_signed" and seed == 886:
                continue
            base = 0.50 + 0.004 * case_index + 0.001 * seed_index
            if case == "bp_fs200_legacy_abs":
                base = 0.60 + 0.001 * seed_index
            if case == "bp_fs200_signed":
                delta = 0.05 if sign_support else -0.03
                base = 0.60 + delta + 0.001 * seed_index
            if case == "bp_fs1000_legacy_abs":
                base = 0.45 + 0.001 * seed_index
            if case == "bp_fs1000_signed":
                base = 0.47 + 0.001 * seed_index
            artifact_dir = sign_dir / "artifacts" / case / str(seed)
            checkpoint = artifact_dir / "best_checkpoint.pth"
            predictions = artifact_dir / "test_predictions.npz"
            manifest = artifact_dir / "test_sample_manifest.csv"
            for artifact in (checkpoint, predictions, manifest):
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(b"synthetic result artifact\n")
            training_rows.append(
                {
                    "case": case,
                    "role": "synthetic",
                    "denoiser": "synthetic",
                    "fs_hz": "",
                    "iqr_mode": "synthetic",
                    "normalization_input": "synthetic",
                    "model": "resnet1d",
                    "model_seed": seed,
                    "split_seed": 42,
                    "epochs": 50,
                    "config_hash": hashes[case],
                    "status": "ok",
                    "best_val_acc": base,
                    "test_acc": base,
                    "train_size": 100,
                    "val_size": 20,
                    "test_size": 20,
                    "input_shape": "[1000,15,9]",
                    "checkpoint": str(checkpoint),
                    "predictions_file": str(predictions),
                    "test_sample_manifest": str(manifest),
                    "duration_sec": 1.0,
                    "error": "",
                }
            )
            accuracies[(case, seed)] = base
    write_csv_file(sign_dir / "training_summary.csv", training_rows)
    aggregate_rows = []
    for case in EXPECTED_CASES:
        values = [
            accuracies[(case, seed)]
            for seed in EXPECTED_SEEDS
            if (case, seed) in accuracies
        ]
        aggregate_rows.append(
            {
                "case": case,
                "n_model_seeds": len(values),
                "model_seeds": ",".join(
                    str(seed) for seed in EXPECTED_SEEDS if (case, seed) in accuracies
                ),
                "mean_test_acc": statistics.fmean(values),
                "std_test_acc": sample_stdev(values),
            }
        )
    write_csv_file(sign_dir / "training_aggregate.csv", aggregate_rows)
    paired_rows, paired_seed_rows = synthetic_effect_rows(accuracies)
    write_csv_file(sign_dir / "paired_effects.csv", paired_rows)
    write_csv_file(sign_dir / "paired_seed_effects.csv", paired_seed_rows)
    return signal_dir, sign_dir


def synthetic_self_test() -> None:
    assert DEFAULT_RESULT_ROOT == (REPO_ROOT / "ablation" / "bandpass_server_results")
    assert DEFAULT_SIGNAL_DIR == DEFAULT_RESULT_ROOT / "signal_analysis"
    assert DEFAULT_SIGN_DIR == DEFAULT_RESULT_ROOT / "sign_ablation"
    assert DEFAULT_OUTPUT_DIR == DEFAULT_RESULT_ROOT / "final_report"
    with tempfile.TemporaryDirectory(prefix="bandpass_final_audit_") as temp:
        root = Path(temp)
        strong_signal, strong_sign = build_synthetic_results(
            root / "strong",
            sign_support=True,
            missing_seed=False,
        )
        strong = audit_results(strong_signal, strong_sign)
        assert strong["status"] == "COMPLETE", strong["errors"]
        assert strong["scientific_verdict"] == "strong_support"

        no_signal, no_sign = build_synthetic_results(
            root / "not_supported",
            sign_support=False,
            missing_seed=False,
        )
        not_supported = audit_results(no_signal, no_sign)
        assert not_supported["status"] == "COMPLETE", not_supported["errors"]
        assert not_supported["scientific_verdict"] == "not_supported"

        missing_signal, missing_sign = build_synthetic_results(
            root / "missing_seed",
            sign_support=True,
            missing_seed=True,
        )
        incomplete = audit_results(missing_signal, missing_sign)
        assert incomplete["status"] == "PRELIMINARY"
        assert incomplete["scientific_verdict"] == "incomplete"
        assert any(
            item["code"] == "training_seeds_incomplete" for item in incomplete["errors"]
        )
    print(
        "Synthetic audit self-test passed: complete strong support, "
        "complete non-support, and missing-seed PRELIMINARY."
    )


def main() -> int:
    args = parse_args()
    if args.self_test:
        synthetic_self_test()
        return 0
    signal_dir = resolve_path(args.signal_dir)
    sign_dir = resolve_path(args.sign_dir)
    output_dir = resolve_path(args.output_dir)
    audit = audit_results(signal_dir, sign_dir)
    write_outputs(audit, output_dir)
    print(f"Audit status: {audit['status']}")
    print(f"Scientific verdict: {audit['scientific_verdict']}")
    print(f"JSON: {output_dir / 'bandpass_result_audit.json'}")
    print(f"Report: {output_dir / 'bandpass_final_report.md'}")
    if audit["status"] != "COMPLETE" and not args.allow_incomplete:
        print(
            "Official evidence is incomplete. Re-run with --allow-incomplete "
            "only to accept a PRELIMINARY report.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
