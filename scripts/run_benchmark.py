"""Run the validated ColdDDI P4 training, prediction and diagnostic pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coldddi.benchmark import main


if __name__ == "__main__":
    sys.exit(main())
