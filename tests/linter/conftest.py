"""Put the linter source directory on sys.path so tests import `linter`."""

import sys
from pathlib import Path

LINTER_DIR = Path(__file__).resolve().parents[2] / "src" / "linter"

if str(LINTER_DIR) not in sys.path:
    sys.path.insert(0, str(LINTER_DIR))
