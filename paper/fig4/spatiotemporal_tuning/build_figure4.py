#!/usr/bin/env python3
"""Build Figure 4 in the locked manuscript layout from audited artifacts."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning._figure4_renderer import main


if __name__ == "__main__":
    raise SystemExit(main())
