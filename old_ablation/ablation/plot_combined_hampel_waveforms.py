"""Assemble the archived ElderAL and XRF55 Hampel waveform figures.

The local checkout contains the six publication PDFs used for ElderAL, but the
raw CSV archive currently contains only the static action.  To keep the new
comparison faithful to all six previously generated figures, this script reads
the plotted line paths directly from those vector PDFs and redraws them with a
single, compact publication style.

Outputs (PNG, PDF, and SVG) are written to
``ablation/combined_hampel_waveforms``:

* ``figure_elderal_six_actions``: six ElderAL actions in a 3 x 2 layout.
* ``figure_elderal_xrf55_hampel_waveforms``: the ElderAL 3 x 2 block on the
  left and the XRF55 3 x 2 block on the right.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pypdf import PdfReader
from pypdf.generic import ContentStream


SCRIPT_DIR = Path(__file__).resolve().parent
ELDER_FIGURE_DIR = SCRIPT_DIR / "ablation_elder_hampel_result" / "figures"
XRF55_FIGURE_DIR = SCRIPT_DIR / "ablation_xrf55_hampel_result" / "figures"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "combined_hampel_waveforms"

RAW_COLOR = "#7F7F7F"
HAMPEL_COLOR = "#0072B2"
REPLACED_COLOR = "#D55E00"
GRID_COLOR = "#D9D9D9"
DEFAULT_FACE = "#FFF7DE"

ELDER_SOURCES = (
    (
        "Static",
        ELDER_FIGURE_DIR / "figure_5_action_0_static_representative_waveforms.pdf",
    ),
    ("Walking", ELDER_FIGURE_DIR / "figure_5_representative_waveforms.pdf"),
    (
        "Fall",
        ELDER_FIGURE_DIR / "figure_5_action_2_fall_representative_waveforms.pdf",
    ),
    (
        "Sleep",
        ELDER_FIGURE_DIR / "figure_5_action_3_sleep_representative_waveforms.pdf",
    ),
    (
        "Drinking",
        ELDER_FIGURE_DIR / "figure_5_action_5_drink_representative_waveforms.pdf",
    ),
    (
        "Watching TV",
        ELDER_FIGURE_DIR / "figure_5_action_6_sit_TV_representative_waveforms.pdf",
    ),
)

XRF55_SOURCE = XRF55_FIGURE_DIR / "figure_3_representative_waveforms.pdf"
XRF55_WINDOWS = (
    (1, "3 frames, 15 ms"),
    (2, "5 frames, 25 ms"),
    (5, "11 frames, 55 ms (default)"),
    (10, "21 frames, 105 ms"),
    (25, "51 frames, 255 ms"),
    (50, "101 frames, 505 ms"),
)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def x1(self) -> float:
        return self.x + self.width

    @property
    def y1(self) -> float:
        return self.y + self.height

    def key(self) -> tuple[float, float, float, float]:
        return tuple(round(value, 6) for value in (self.x, self.y, self.width, self.height))


@dataclass
class GraphicsState:
    clip: Rect | None = None
    stroke: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linewidth: float = 1.0


@dataclass
class StrokePath:
    vertices: np.ndarray
    clip: Rect | None
    stroke: tuple[float, float, float]
    linewidth: float


@dataclass
class PanelData:
    raw_x: np.ndarray
    raw_y: np.ndarray
    hampel_x: np.ndarray
    hampel_y: np.ndarray
    replaced_x: np.ndarray
    replaced_y: np.ndarray


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.75,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 8.0,
            "lines.linewidth": 1.15,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.55,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def as_float(value: object) -> float:
    return float(value)


def path_rect(vertices: list[tuple[float, float]]) -> Rect | None:
    if len(vertices) < 4:
        return None
    points = np.asarray(vertices, dtype=float)
    x0, y0 = np.min(points, axis=0)
    x1, y1 = np.max(points, axis=0)
    if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
        return None
    return Rect(float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def parse_stroke_paths(pdf_path: Path) -> tuple[object, list[StrokePath]]:
    """Parse simple stroked paths and their Matplotlib axes clip rectangles."""
    reader = PdfReader(str(pdf_path))
    page = reader.pages[0]
    operations = ContentStream(page.get_contents(), page.pdf).operations

    state = GraphicsState()
    stack: list[GraphicsState] = []
    strokes: list[StrokePath] = []
    current_subpaths: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    pending_clip = False

    def finish_subpath() -> None:
        nonlocal current
        if current:
            current_subpaths.append(current)
            current = []

    def clear_path() -> None:
        nonlocal current_subpaths, current
        current_subpaths = []
        current = []

    def save_strokes() -> None:
        finish_subpath()
        for subpath in current_subpaths:
            if len(subpath) >= 2:
                strokes.append(
                    StrokePath(
                        vertices=np.asarray(subpath, dtype=float),
                        clip=state.clip,
                        stroke=state.stroke,
                        linewidth=state.linewidth,
                    )
                )
        clear_path()

    for operands, raw_operator in operations:
        operator = raw_operator.decode("latin1")
        if operator == "q":
            stack.append(copy.deepcopy(state))
        elif operator == "Q":
            state = stack.pop() if stack else GraphicsState()
            clear_path()
            pending_clip = False
        elif operator == "w":
            state.linewidth = as_float(operands[0])
        elif operator == "G":
            grey = as_float(operands[0])
            state.stroke = (grey, grey, grey)
        elif operator == "RG":
            state.stroke = tuple(as_float(item) for item in operands[:3])
        elif operator == "m":
            finish_subpath()
            current = [(as_float(operands[0]), as_float(operands[1]))]
        elif operator == "l":
            current.append((as_float(operands[0]), as_float(operands[1])))
        elif operator == "c":
            # Curves are used for glyphs/markers, not the waveform polylines.
            current.append((as_float(operands[4]), as_float(operands[5])))
        elif operator == "h":
            if current:
                current.append(current[0])
        elif operator == "re":
            finish_subpath()
            x, y, width, height = (as_float(item) for item in operands[:4])
            current = [
                (x, y),
                (x + width, y),
                (x + width, y + height),
                (x, y + height),
                (x, y),
            ]
        elif operator in {"W", "W*"}:
            pending_clip = True
        elif operator == "n":
            finish_subpath()
            if pending_clip and current_subpaths:
                rect = path_rect(current_subpaths[-1])
                if rect is not None:
                    state.clip = rect
            clear_path()
            pending_clip = False
        elif operator in {"S", "s"}:
            save_strokes()
        elif operator in {"B", "B*", "b", "b*"}:
            save_strokes()
        elif operator in {"f", "f*"}:
            clear_path()

    return page, strokes


def unique_axes_rectangles(strokes: Iterable[StrokePath]) -> list[Rect]:
    seen: dict[tuple[float, float, float, float], Rect] = {}
    for stroke in strokes:
        if stroke.clip is None:
            continue
        rect = stroke.clip
        if rect.width > 50 and rect.height > 40:
            seen.setdefault(rect.key(), rect)
    return sorted(seen.values(), key=lambda rect: (-rect.y, rect.x))


def colour_distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def rgb(hex_colour: str) -> tuple[float, float, float]:
    value = hex_colour.lstrip("#")
    return tuple(int(value[index : index + 2], 16) / 255 for index in (0, 2, 4))


def candidate_waveforms(strokes: Iterable[StrokePath], rect: Rect) -> list[StrokePath]:
    candidates = [
        stroke
        for stroke in strokes
        if stroke.clip is not None
        and stroke.clip.key() == rect.key()
        and stroke.linewidth >= 1.2
        and stroke.vertices.shape[0] >= 8
    ]
    if len(candidates) < 2:
        raise RuntimeError(
            f"Expected two waveform paths in axes {rect.key()}, found {len(candidates)}"
        )
    return candidates


def select_raw_and_hampel(
    candidates: list[StrokePath],
    source: str,
) -> tuple[StrokePath, StrokePath]:
    if source == "elder":
        raw_target = rgb("#7F7F7F")
        hampel_target = rgb("#D55E00")
    elif source == "xrf55":
        raw_target = rgb("#000000")
        hampel_target = rgb("#0072B2")
    else:
        raise ValueError(source)

    raw_path = min(candidates, key=lambda item: colour_distance(item.stroke, raw_target))
    remaining = [item for item in candidates if item is not raw_path]
    hampel_path = min(
        remaining,
        key=lambda item: colour_distance(item.stroke, hampel_target),
    )
    return raw_path, hampel_path


NUMERIC_TEXT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def extract_numeric_text(page: object) -> list[tuple[float, float, float]]:
    entries: list[tuple[float, float, float]] = []

    def visitor(
        text: str,
        current_matrix: list[float],
        text_matrix: list[float],
        font_dictionary: object,
        font_size: float,
    ) -> None:
        del text_matrix, font_dictionary, font_size
        value = text.strip().replace("−", "-")
        if NUMERIC_TEXT.fullmatch(value):
            entries.append(
                (float(current_matrix[4]), float(current_matrix[5]), float(value))
            )

    page.extract_text(visitor_text=visitor)
    return entries


def vertical_grid_positions(strokes: Iterable[StrokePath], rect: Rect) -> list[float]:
    positions: list[float] = []
    tolerance = 0.15
    for stroke in strokes:
        if stroke.clip is None or stroke.clip.key() != rect.key():
            continue
        vertices = stroke.vertices
        if vertices.shape[0] != 2 or stroke.linewidth > 0.75:
            continue
        if abs(vertices[0, 0] - vertices[1, 0]) > tolerance:
            continue
        span = abs(vertices[1, 1] - vertices[0, 1])
        if span >= 0.95 * rect.height:
            positions.append(float(np.mean(vertices[:, 0])))
    return sorted(set(round(value, 6) for value in positions))


def x_coordinate_transform(
    page: object,
    strokes: Iterable[StrokePath],
    rect: Rect,
) -> tuple[float, float]:
    """Return coefficients mapping PDF x positions to seconds."""
    grids = vertical_grid_positions(strokes, rect)
    labels = [
        (x, value)
        for x, y, value in extract_numeric_text(page)
        if rect.x - 12 <= x <= rect.x1 + 2
        and rect.y - 22 <= y <= rect.y - 2
    ]
    labels.sort(key=lambda item: item[0])
    values = [value for _, value in labels]
    if len(grids) == len(values) and len(grids) >= 2:
        slope, intercept = np.polyfit(np.asarray(grids), np.asarray(values), 1)
        return float(slope), float(intercept)

    # The XRF55 PDF suppresses upper-row x tick labels because axes share x.
    # Its archived signal has 1000 samples at 200 Hz, spanning 4.995 seconds.
    return 4.995 / rect.width, -4.995 * rect.x / rect.width


def map_and_normalize_panel(
    raw_path: StrokePath,
    hampel_path: StrokePath,
    x_slope: float,
    x_intercept: float,
) -> PanelData:
    raw_x = raw_path.vertices[:, 0] * x_slope + x_intercept
    hampel_x = hampel_path.vertices[:, 0] * x_slope + x_intercept

    raw_pdf_y = raw_path.vertices[:, 1]
    hampel_pdf_y = hampel_path.vertices[:, 1]
    center = float(np.mean(raw_pdf_y))
    scale = max(float(np.std(raw_pdf_y)), 1e-12)
    raw_y = (raw_pdf_y - center) / scale
    hampel_y = (hampel_pdf_y - center) / scale

    order_raw = np.argsort(raw_x, kind="stable")
    order_hampel = np.argsort(hampel_x, kind="stable")
    raw_x = raw_x[order_raw]
    raw_y = raw_y[order_raw]
    hampel_x = hampel_x[order_hampel]
    hampel_y = hampel_y[order_hampel]

    raw_at_hampel = np.interp(hampel_x, raw_x, raw_y)
    changed = np.abs(hampel_y - raw_at_hampel) > 1e-3
    return PanelData(
        raw_x=raw_x,
        raw_y=raw_y,
        hampel_x=hampel_x,
        hampel_y=hampel_y,
        replaced_x=hampel_x[changed],
        replaced_y=raw_at_hampel[changed],
    )


def extract_elder_panels() -> list[tuple[str, PanelData]]:
    panels: list[tuple[str, PanelData]] = []
    for label, pdf_path in ELDER_SOURCES:
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        page, strokes = parse_stroke_paths(pdf_path)
        rectangles = unique_axes_rectangles(strokes)
        if len(rectangles) != 5:
            raise RuntimeError(f"Expected five ElderAL axes in {pdf_path}, found {len(rectangles)}")
        default_rect = min(rectangles, key=lambda rect: rect.y)
        candidates = candidate_waveforms(strokes, default_rect)
        raw_path, hampel_path = select_raw_and_hampel(candidates, "elder")
        x_slope, x_intercept = x_coordinate_transform(page, strokes, default_rect)
        panels.append(
            (
                label,
                map_and_normalize_panel(
                    raw_path,
                    hampel_path,
                    x_slope,
                    x_intercept,
                ),
            )
        )
    return panels


def extract_xrf55_panels() -> list[tuple[str, int, PanelData]]:
    if not XRF55_SOURCE.exists():
        raise FileNotFoundError(XRF55_SOURCE)
    page, strokes = parse_stroke_paths(XRF55_SOURCE)
    rectangles = unique_axes_rectangles(strokes)
    if len(rectangles) != 6:
        raise RuntimeError(f"Expected six XRF55 axes, found {len(rectangles)}")

    # ``unique_axes_rectangles`` is already row-major from top to bottom.
    panels: list[tuple[str, int, PanelData]] = []
    bottom_by_column = {
        0: min((rect for rect in rectangles if rect.x < np.median([r.x for r in rectangles])), key=lambda rect: rect.y),
        1: min((rect for rect in rectangles if rect.x > np.median([r.x for r in rectangles])), key=lambda rect: rect.y),
    }
    transforms = {
        column: x_coordinate_transform(page, strokes, rect)
        for column, rect in bottom_by_column.items()
    }

    for index, ((half_window, label), rect) in enumerate(zip(XRF55_WINDOWS, rectangles)):
        candidates = candidate_waveforms(strokes, rect)
        raw_path, hampel_path = select_raw_and_hampel(candidates, "xrf55")
        x_slope, x_intercept = transforms[index % 2]
        panels.append(
            (
                label,
                half_window,
                map_and_normalize_panel(
                    raw_path,
                    hampel_path,
                    x_slope,
                    x_intercept,
                ),
            )
        )
    return panels


def legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=RAW_COLOR, alpha=0.65, label="Raw"),
        Line2D([0], [0], color=HAMPEL_COLOR, label="Hampel"),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=4.0,
            markerfacecolor="none",
            markeredgecolor=REPLACED_COLOR,
            markeredgewidth=0.8,
            label="Replaced",
        ),
    ]


def draw_panel(
    ax: plt.Axes,
    panel: PanelData,
    title: str,
    *,
    show_xlabels: bool,
    show_ylabels: bool,
    highlight: bool = False,
) -> None:
    ax.plot(panel.raw_x, panel.raw_y, color=RAW_COLOR, alpha=0.62, linewidth=0.95)
    ax.plot(panel.hampel_x, panel.hampel_y, color=HAMPEL_COLOR, linewidth=1.15)
    if panel.replaced_x.size:
        # Path simplification may leave several adjacent vertices around one
        # replacement.  Keep markers readable by retaining unique x positions.
        rounded_x = np.round(panel.replaced_x, 6)
        _, unique_indices = np.unique(rounded_x, return_index=True)
        unique_indices.sort()
        ax.scatter(
            panel.replaced_x[unique_indices],
            panel.replaced_y[unique_indices],
            s=10,
            facecolors="none",
            edgecolors=REPLACED_COLOR,
            linewidths=0.65,
            zorder=3,
        )
    ax.set_title(title, loc="left", pad=3.0)
    ax.grid(True, color=GRID_COLOR)
    ax.tick_params(direction="out", length=2.5, width=0.65)
    if not show_xlabels:
        ax.tick_params(labelbottom=False)
    if not show_ylabels:
        ax.tick_params(labelleft=False)
    if highlight:
        ax.set_facecolor(DEFAULT_FACE)
        for spine in ax.spines.values():
            spine.set_color("#E69F00")
            spine.set_linewidth(1.0)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def plot_elder_only(
    elder_panels: list[tuple[str, PanelData]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.35), constrained_layout=False)
    for index, (label, panel) in enumerate(elder_panels):
        row, column = divmod(index, 2)
        draw_panel(
            axes[row, column],
            panel,
            f"({chr(97 + index)}) {label}",
            show_xlabels=row == 2,
            show_ylabels=column == 0,
        )
    fig.text(0.5, 0.945, "ElderAL — 11-frame Hampel (default)", ha="center", va="center", fontsize=10)
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.922),
        ncol=3,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.5,
    )
    fig.supxlabel("Time (s)", x=0.51, y=0.025, fontsize=9)
    fig.supylabel("Normalized amplitude", x=0.02, y=0.49, fontsize=9)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.085, top=0.855, wspace=0.12, hspace=0.25)
    save_figure(fig, output_dir, "figure_elderal_six_actions")


def plot_combined(
    elder_panels: list[tuple[str, PanelData]],
    xrf55_panels: list[tuple[str, int, PanelData]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(14.4, 6.45), constrained_layout=False)

    for index, (label, panel) in enumerate(elder_panels):
        row, column = divmod(index, 2)
        draw_panel(
            axes[row, column],
            panel,
            f"({chr(97 + index)}) {label}",
            show_xlabels=row == 2,
            show_ylabels=column == 0,
        )

    for index, (label, half_window, panel) in enumerate(xrf55_panels):
        row, local_column = divmod(index, 2)
        column = local_column + 2
        draw_panel(
            axes[row, column],
            panel,
            f"({chr(103 + index)}) {label}",
            show_xlabels=row == 2,
            show_ylabels=local_column == 0,
            highlight=half_window == 5,
        )

    fig.text(0.262, 0.945, "ElderAL — six actions, 11-frame Hampel", ha="center", va="center", fontsize=10)
    fig.text(0.750, 0.945, "XRF55 — six Hampel windows", ha="center", va="center", fontsize=10)
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.506, 0.918),
        ncol=3,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.5,
    )
    fig.supxlabel("Time (s)", x=0.505, y=0.025, fontsize=9)
    fig.supylabel("Normalized amplitude", x=0.012, y=0.49, fontsize=9)
    fig.subplots_adjust(left=0.055, right=0.992, bottom=0.085, top=0.85, wspace=0.12, hspace=0.25)

    # A quiet separator makes the two 3 x 2 blocks legible without adding more text.
    fig.add_artist(
        Line2D(
            [0.505, 0.505],
            [0.082, 0.852],
            transform=fig.transFigure,
            color="#BFBFBF",
            linewidth=0.65,
        )
    )
    save_figure(fig, output_dir, "figure_elderal_xrf55_hampel_waveforms")


def write_manifest(output_dir: Path) -> None:
    manifest = {
        "elderal": {
            "panels": [label for label, _ in ELDER_SOURCES],
            "condition": "11-frame Hampel (half-window=5, n_sigma=3)",
            "sources": [str(path.resolve()) for _, path in ELDER_SOURCES],
        },
        "xrf55": {
            "panels": [label for _, label in XRF55_WINDOWS],
            "sources": [str(XRF55_SOURCE.resolve())],
        },
        "rendering": {
            "normalization": "Each channel uses the mean and standard deviation of its archived raw waveform path.",
            "legend": ["Raw", "Hampel", "Replaced"],
            "formats": ["png", "pdf", "svg"],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG, PDF, SVG, and the source manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    configure_style()
    elder_panels = extract_elder_panels()
    xrf55_panels = extract_xrf55_panels()
    plot_elder_only(elder_panels, output_dir)
    plot_combined(elder_panels, xrf55_panels, output_dir)
    write_manifest(output_dir)
    print(f"Saved combined Hampel figures to {output_dir}")


if __name__ == "__main__":
    main()
