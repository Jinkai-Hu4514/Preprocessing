"""Plot Hampel comparisons for six fixed ElderAL samples and CSI channels."""

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "datasets" / "elder"
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_STEM = "elder_hampel_waveforms"
SAMPLES = (
    ("Static", "action0_static_new/user0_position1_activity0/20250813=204917_mimo5s_part0.csv", 257, 2),
    ("Walking", "action1_walk_new/user0_position8_activity1/20250813=122048_mimo5s_part0.csv", 322, 2),
    ("Fall", "action2_fall_new/user0_position1_activity2/20250813=230850_mimo5s_part0.csv", 193, 2),
    ("Sleep", "action3_sleep_new/user0_position2_activity3/20250813=183850_mimo5s_part0.csv", 141, 2),
    ("Drinking", "action5_drink_new/user0_position4_activity5/20250813=161044_mimo5s_part0.csv", 141, 2),
    ("Watching TV", "action6_sit_TV_new/user0_position2_activity6/20250813=171349_mimo5s_part0.csv", 139, 2),
)
HALF_WINDOW = 5
N_SIGMA = 3.0
FULL_WINDOW = 2 * HALF_WINDOW + 1
RAW_COLOR = "#7f7f7f"
HAMPEL_COLOR = "#0072b2"
REPLACED_COLOR = "#d55e00"


@dataclass(frozen=True)
class Waveform:
    seconds: np.ndarray
    raw: np.ndarray
    filtered: np.ndarray


def apply_hampel(raw: np.ndarray) -> np.ndarray:
    """Apply the time-window median and MAD threshold to one CSI channel."""
    filtered = raw.copy()
    for frame, value in enumerate(raw):
        window = raw[max(0, frame - HALF_WINDOW):frame + HALF_WINDOW + 1]
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        if np.abs(value - median) > N_SIGMA * 1.4826 * mad:
            filtered[frame] = median
    return filtered


def load_waveform(path: Path, subcarrier: int, receive_link: int) -> Waveform:
    """Read timestamps and one TX 0 amplitude column, then filter it."""
    channel = f"amp_tx0_rx{receive_link}_sub{subcarrier}"
    with path.open(encoding="utf-8-sig", newline="") as source:
        header = [name.strip() for name in next(csv.reader(source), ())]
        if "timestamp" not in header or channel not in header:
            raise ValueError(f"Missing timestamp or {channel} in {path}")
        columns = (header.index("timestamp"), header.index(channel))
        data = np.loadtxt(source, delimiter=",", usecols=columns, ndmin=2)

    if len(data) < 3 or not np.isfinite(data).all():
        raise ValueError(f"Expected at least three finite CSI frames in {path}")
    timestamps, raw = data[np.argsort(data[:, 0], kind="stable")].T
    seconds = (timestamps - timestamps[0]) * 1e-6
    return Waveform(seconds, raw, apply_hampel(raw))


def draw_waveforms(waveforms: dict[str, Waveform]):
    figure, axes = plt.subplots(3, 2, figsize=(8.15, 6.25))
    figure.subplots_adjust(
        left=0.13, right=0.976, top=0.895, bottom=0.12, wspace=0.24, hspace=0.52,
    )
    for axis, (activity, waveform) in zip(axes.flat, waveforms.items(), strict=True):
        center = waveform.raw.mean()
        scale = waveform.raw.std() or 1.0
        raw = (waveform.raw - center) / scale
        filtered = (waveform.filtered - center) / scale
        changed = waveform.raw != waveform.filtered

        axis.plot(waveform.seconds, raw, color=RAW_COLOR, linewidth=0.9, alpha=0.67)
        axis.plot(waveform.seconds, filtered, color=HAMPEL_COLOR, linewidth=1.3)
        axis.scatter(
            waveform.seconds[changed], raw[changed], s=15, facecolors="none",
            edgecolors=REPLACED_COLOR, linewidths=0.8, zorder=3,
        )
        axis.set_title(activity, loc="left", pad=3)
        axis.set_axisbelow(True)
        axis.grid(color="#d9d9d9", linewidth=0.55, alpha=0.75)
        axis.tick_params(direction="out", length=2.8, width=0.7, pad=2)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.margins(x=0.04, y=0.12)

    figure.legend(
        handles=[
            Line2D([], [], color=RAW_COLOR, alpha=0.67, label="Raw"),
            Line2D([], [], color=HAMPEL_COLOR, label=f"Hampel ({FULL_WINDOW} frames)"),
            Line2D(
                [], [], linestyle="none", marker="o", markersize=4.5,
                markerfacecolor="none", markeredgecolor=REPLACED_COLOR, label="Replaced",
            ),
        ],
        loc="upper center", bbox_to_anchor=(0.54, 0.984), ncols=3,
        frameon=False, handlelength=1.8,
    )
    figure.supxlabel("Time from sample start (s)", y=0.037)
    figure.supylabel("Normalized amplitude", x=0.032)
    return figure


def main():
    waveforms = {}
    selections = []
    for activity, relative_csv, subcarrier, receive_link in SAMPLES:
        path = DATA_DIR / relative_csv
        waveform = load_waveform(path, subcarrier, receive_link)
        waveforms[activity] = waveform
        selections.append({
            "activity": activity,
            "sample_csv": relative_csv,
            "frames": len(waveform.raw),
            "span_seconds": float(waveform.seconds[-1]),
            "subcarrier": subcarrier,
            "receive_link": receive_link,
            "replaced_points": int(np.count_nonzero(waveform.raw != waveform.filtered)),
            "sum_absolute_change": float(np.abs(waveform.raw - waveform.filtered).sum()),
        })
        print(
            f"{activity}: {path.name}, subcarrier={subcarrier}, rx={receive_link}",
            flush=True,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.75,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
    }):
        figure = draw_waveforms(waveforms)
        for extension in ("png", "pdf"):
            destination = RESULTS_DIR / f"{OUTPUT_STEM}.{extension}"
            figure.savefig(destination, dpi=300, facecolor="white")
            print(destination)
        plt.close(figure)

    metadata = {
        "full_window_frames": FULL_WINDOW,
        "n_sigma": N_SIGMA,
        "transmit_antenna": 0,
        "sample_selection": "Fixed sample and CSI channel per activity",
        "samples": selections,
    }
    manifest = RESULTS_DIR / "elder_hampel_waveform_samples.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
