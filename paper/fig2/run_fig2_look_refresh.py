"""Fast Figure 2 refresh for visual inspection.

This keeps the normal inclusion logic but writes separate look-pass caches and
uses a small shuffle count. It is for checking the figure shape before launching
the slower publication-grade refresh.
"""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIG2_DIR = ROOT / "ryan" / "fig2"
if str(FIG2_DIR) not in sys.path:
    sys.path.insert(0, str(FIG2_DIR))

import compute_fig2_data as c  # noqa: E402
import generate_figure2 as g  # noqa: E402


def main():
    c.N_SHUFFLES = 3
    c.N_STAGE1_WORKERS = 4
    c.N_STAGE1_GPUS = 1
    c.DECOMP_CACHE = c.CACHE_DIR / "fig2_decomposition_yates_rowley_look.pkl"
    c.DERIVED_CACHE = c.CACHE_DIR / "fig2_derived_yates_rowley_look.pkl"

    g.compose(refresh=True)
    print(f"LOOK_CACHE {c.DECOMP_CACHE}")
    print(f"LOOK_DERIVED {c.DERIVED_CACHE}")
    print(f"LOOK_FIG {g.FIG_DIR / 'fig2_composite.pdf'}")


if __name__ == "__main__":
    main()
