"""Select a current XRF55 amplitude trace and compare six Hampel windows."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "datasets" / "xrf55"
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_STEM = "xrf55_hampel_waveforms"
RAW_SHAPE = (3, 30, 3, 1000)  # receiver, subcarrier, antenna, time
SAMPLES_PER_USER = 10
SELECTION_SEED = 42
SUBCARRIER_INDEX = 4
FLAT_ANTENNA_INDEX = 8
SAMPLE_RATE_HZ = 200
N_SIGMA = 3.0
WINDOWS = (3, 5, 11, 21, 51, 101)
DEFAULT_WINDOW = 11
RAW_COLOR = "#7f7f7f"
HAMPEL_COLOR = "#0072b2"
REPLACED_COLOR = "#d55e00"
DEFAULT_BACKGROUND = "#fff7de"
DEFAULT_BORDER = "#e69f00"


def sample_candidates(data_dir: Path) -> list[Path]:
    """Draw the same number of available candidate files per user."""
    by_user = defaultdict(list)
    for path in sorted(data_dir.rglob("*.npy")):
        fields = path.stem.split("_")
        if len(fields) == 3 and all(field.isdigit() for field in fields):
            by_user[fields[0]].append(path)

    rng = np.random.default_rng(SELECTION_SEED)
    candidates = []
    for paths in by_user.values():
        count = min(SAMPLES_PER_USER, len(paths))
        indices = sorted(rng.choice(len(paths), size=count, replace=False))
        candidates.extend(paths[index] for index in indices)
    return candidates


def load_amplitudes(sample_path: Path) -> np.ndarray:
    """Return (time, subcarrier, receiver-major antenna) amplitudes."""
    data = np.load(sample_path, allow_pickle=False)
    if data.size != np.prod(RAW_SHAPE):
        raise ValueError(f"Unexpected XRF55 amplitude shape {data.shape} in {sample_path}")
    if np.iscomplexobj(data) and np.any(np.abs(data.imag) > 1e-10):
        raise ValueError(f"Expected real XRF55 amplitudes in {sample_path}")
    amplitudes = np.asarray(data.real, dtype=float)
    if not np.isfinite(amplitudes).all():
        raise ValueError(f"Non-finite amplitudes in {sample_path}")
    receivers, subcarriers, antennas, frames = RAW_SHAPE
    return amplitudes.reshape(RAW_SHAPE).transpose(3, 1, 0, 2).reshape(
        frames, subcarriers, receivers * antennas,
    )


def hampel(values: np.ndarray, window_frames: int) -> np.ndarray:
    """Replace points that exceed the local median plus or minus 3 scaled MADs."""
    radius = window_frames // 2
    filtered = values.copy()
    for index, value in enumerate(values):
        window = values[max(0, index - radius):index + radius + 1]
        median = np.median(window, axis=0)
        mad = np.median(np.abs(window - median), axis=0)
        changed = np.abs(value - median) > N_SIGMA * 1.4826 * mad
        filtered[index] = np.where(changed, median, value)
    return filtered


def select_trace(candidates: list[Path]):
    """Choose the trace closest to the candidate median replacement curve."""
    traces = []
    counts = []
    for path in candidates:
        amplitudes = load_amplitudes(path)
        raw = amplitudes[:, SUBCARRIER_INDEX, FLAT_ANTENNA_INDEX].copy()
        filtered_by_window = {window: hampel(raw, window) for window in WINDOWS}
        traces.append((path, raw, filtered_by_window))
        counts.append([np.count_nonzero(raw != filtered) for filtered in filtered_by_window.values()])

    counts = np.asarray(counts)
    medians = np.median(counts, axis=0)
    distances = np.abs(counts - medians).sum(axis=1)
    nonincreasing = int(np.count_nonzero((np.diff(counts, axis=1) <= 0).all(axis=1)))
    return traces[int(distances.argmin())], medians, nonincreasing


def draw_waveforms(raw: np.ndarray, filtered_by_window: dict[int, np.ndarray]):
    figure, axes = plt.subplots(3, 2, figsize=(8.0, 5.95), sharex=True, sharey=True)
    figure.subplots_adjust(
        left=0.125, right=0.975, bottom=0.12, top=0.89, wspace=0.14, hspace=0.32,
    )
    seconds = np.arange(raw.size) / SAMPLE_RATE_HZ
    center = raw.mean()
    scale = raw.std() or 1.0
    normalized = (raw - center) / scale

    for axis, window_frames in zip(axes.flat, WINDOWS, strict=True):
        filtered = filtered_by_window[window_frames]
        changed = raw != filtered
        axis.plot(seconds, normalized, color=RAW_COLOR, alpha=0.55, linewidth=0.7)
        axis.plot(seconds, (filtered - center) / scale, color=HAMPEL_COLOR, linewidth=1.0)
        axis.scatter(
            seconds[changed], normalized[changed], s=9, facecolors="none",
            edgecolors=REPLACED_COLOR, alpha=0.8, linewidths=0.65, zorder=3,
        )

        nominal_ms = window_frames * 1000 / SAMPLE_RATE_HZ
        axis.set_title(f"{window_frames} frames ({nominal_ms:.0f} ms)", loc="left", pad=3)
        axis.set_title(
            f"{changed.mean():.1%} replaced", loc="right", pad=3,
            fontsize=8, color="#65432d",
        )
        axis.set_axisbelow(True)
        axis.grid(color="#d9d9d9", alpha=0.75, linewidth=0.5)
        axis.tick_params(direction="out", length=2.7, width=0.7, pad=2)
        if window_frames == DEFAULT_WINDOW:
            axis.set_facecolor(DEFAULT_BACKGROUND)
            for spine in axis.spines.values():
                spine.set_color(DEFAULT_BORDER)

    axes[0, 0].xaxis.set_major_locator(MultipleLocator(1.0))
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axes[0, 0].set_xlim(-0.08, seconds[-1] + 0.08)
    figure.legend(
        handles=[
            Line2D([], [], color=RAW_COLOR, alpha=0.55, label="Raw"),
            Line2D([], [], color=HAMPEL_COLOR, label="Hampel"),
            Line2D(
                [], [], linestyle="none", marker="o", markersize=4,
                markerfacecolor="none", markeredgecolor=REPLACED_COLOR, label="Replaced",
            ),
        ],
        loc="upper center", bbox_to_anchor=(0.54, 0.985), ncols=3,
        frameon=False, handlelength=1.8,
    )
    figure.supxlabel("Time (s)", y=0.037)
    figure.supylabel("Normalized amplitude", x=0.032)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, help="Use one specified XRF55 NPY as the candidate")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Figure output directory")
    args = parser.parse_args()

    if args.sample is not None and not args.sample.is_file():
        parser.error(f"Sample file does not exist: {args.sample}")
    candidates = [args.sample.resolve()] if args.sample is not None else sample_candidates(DATA_DIR)
    if not candidates:
        parser.error(f"No XRF55 NPY files found in {DATA_DIR}")

    selected, median_counts, nonincreasing = select_trace(candidates)
    sample, signal, filtered_by_window = selected
    receiver, receiver_antenna = divmod(FLAT_ANTENNA_INDEX, RAW_SHAPE[2])
    print(
        f"Selected {sample.name} from {len(candidates)} candidates, "
        f"subcarrier={SUBCARRIER_INDEX}, antenna={FLAT_ANTENNA_INDEX}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.75,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
    }):
        figure = draw_waveforms(signal, filtered_by_window)
        for extension in ("png", "pdf"):
            path = args.output_dir / f"{OUTPUT_STEM}.{extension}"
            figure.savefig(path, dpi=300, facecolor="white")
            print(path)
        plt.close(figure)

    metadata = {
        "sample": sample.relative_to(DATA_DIR).as_posix() if sample.is_relative_to(DATA_DIR) else str(sample),
        "candidate_count": len(candidates),
        "selection_seed": SELECTION_SEED if args.sample is None else None,
        "selection_rule": "Closest to the candidate median replacement counts across all six windows",
        "subcarrier_index": SUBCARRIER_INDEX,
        "flat_antenna_index": FLAT_ANTENNA_INDEX,
        "receiver_index": receiver,
        "receiver_antenna_index": receiver_antenna,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "n_sigma": N_SIGMA,
        "nonincreasing_candidate_count": nonincreasing,
        "median_candidate_replaced_points_by_window": {
            window: float(count) for window, count in zip(WINDOWS, median_counts, strict=True)
        },
        "replaced_points_by_window": {
            window: int(np.count_nonzero(signal != filtered))
            for window, filtered in filtered_by_window.items()
        },
    }
    manifest = args.output_dir / "xrf55_hampel_waveform_sample.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
