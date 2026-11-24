"""Combine the four preset heatmaps into 2×2 PNG and PDF figures."""

from pathlib import Path

from PIL import Image
from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parent
HEATMAPS = (
    ROOT / "test_elder/plot/preset/elder_preset_heatmap.png",
    ROOT / "test_xrf55/plot/preset/xrf55_preset_heatmap.png",
    ROOT / "test_widar3.0/plot/preset/widar_preset_heatmap.png",
    ROOT / "test_gait/plot/preset/gait_preset_heatmap.png",
)


def combine_preset(output_path: Path = ROOT / "combined_preset.png") -> None:
    panels = []
    for path in HEATMAPS:
        with Image.open(path) as source:
            panels.append(source.convert("RGBA"))

    cell_width = max(panel.width for panel in panels)
    cell_height = max(panel.height for panel in panels)
    with Image.new("RGB", (2 * cell_width, 2 * cell_height), "white") as combined:
        for index, panel in enumerate(panels):
            row, column = divmod(index, 2)
            x = column * cell_width + (cell_width - panel.width) // 2
            y = row * cell_height + (cell_height - panel.height) // 2
            combined.paste(panel, (x, y), panel)

        combined.save(output_path, dpi=(300, 300))


def combine_pdf(output_path: Path = ROOT / "combined_preset.pdf") -> None:
    panels = [PdfReader(path.with_suffix(".pdf")).pages[0] for path in HEATMAPS]
    cell_width = max(float(panel.cropbox.width) for panel in panels)
    cell_height = max(float(panel.cropbox.height) for panel in panels)
    writer = PdfWriter()
    combined = writer.add_blank_page(width=2 * cell_width, height=2 * cell_height)

    for index, panel in enumerate(panels):
        row, column = divmod(index, 2)
        box = panel.cropbox
        x = column * cell_width + (cell_width - float(box.width)) / 2 - float(box.left)
        # PDF coordinates start at the bottom-left corner.
        y = (1 - row) * cell_height + (cell_height - float(box.height)) / 2 - float(box.bottom)
        combined.merge_translated_page(panel, tx=x, ty=y)

    with output_path.open("wb") as output:
        writer.write(output)


if __name__ == "__main__":
    combine_preset()
    combine_pdf()
