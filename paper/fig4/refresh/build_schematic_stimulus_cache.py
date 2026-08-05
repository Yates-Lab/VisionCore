#!/usr/bin/env python3
"""Build the cached stimulus payload used by Fig. 4 schematic panels.

This is the only Fig. 4 compose input that needs the raw BackImage data package:
it reconstructs one full stimulus canvas and its crops, then stores those arrays
under ``outputs/cache`` so figure composition itself remains cache-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _fig4_imports  # noqa: F401
import _fig4_contour_schematic as schematic
import _fig4_paths as _paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=_paths.CACHE_DIR)
    parser.add_argument("--image-index", type=int, default=schematic.SCHEMATIC_NEW_BANK_IMAGE_INDEX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out_dir / _paths.SCHEMATIC_STIMULUS_PAYLOAD_NPZ.name
    path = schematic.write_new_bank_stimulus_cache(out_path, image_index=args.image_index)
    print(path)


if __name__ == "__main__":
    main()
