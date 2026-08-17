#!/usr/bin/env python3
"""Production entry point for the native-240 M77 response-subspace analysis."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.model_selection._m77_response_subspace_impl import main


if __name__ == "__main__":
    raise SystemExit(main())
