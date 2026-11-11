"""Measure individual algorithms on Gait raw samples."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _benchmark import main


if __name__ == "__main__":
    main("gait", "gait", target_k=15)
