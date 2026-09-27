"""Allow scripts to run from a source checkout before editable installation."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> None:
    source = Path(__file__).resolve().parents[1] / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
