"""Measure individual algorithms on Widar raw samples."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _benchmark import main


if __name__ == "__main__":
    main("widar3.0", "widar", target_k=15, output_folder="widar")
