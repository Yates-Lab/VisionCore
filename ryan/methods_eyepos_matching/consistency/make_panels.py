"""Render PROD | NAIVE | FULL 3-up comparisons for fig2 C/E and fig3 B/C/D by
calling the *existing* production panel functions three times with three data
dicts (production, methods-naive adapter, methods-full adapter). Zero
plotting-code divergence: any visible difference is in the data, not the drawing.

Figures saved under figures/consistency/.

Run: uv run python consistency/make_panels.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONS_DIR = Path(__file__).resolve().parent
METHODS_DIR = CONS_DIR.parent
RYAN = METHODS_DIR.parent
FIG2, FIG3 = RYAN / "fig2", RYAN / "fig3"
OUT = METHODS_DIR / "figures" / "consistency"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CONS_DIR))
from adapter import methods_to_fig2_schema, load_methods_bundle  # noqa: E402


def panel_func(panel_dir: Path, module: str, func: str):
    """Import `module.func` so its `_panel_common` resolves to `panel_dir`'s copy
    while `compute_fig2_data` stays importable from fig2. Returns the function."""
    sys.modules.pop("_panel_common", None)
    sys.modules.pop(module, None)
    for p in (str(FIG3), str(FIG2), str(panel_dir)):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(FIG2))        # compute_fig2_data (fig2-only)
    sys.path.insert(0, str(panel_dir))   # _panel_common resolves here first
    mod = importlib.import_module(module)
    return getattr(mod, func)


# ---- build the three data dicts ----
from compute_fig2_data import load_fig2_data  # noqa: E402
md = load_methods_bundle()
DATA = {
    "PROD": load_fig2_data(refresh=False),
    "NAIVE": methods_to_fig2_schema(md, "naive"),
    # FULL and CENTRAL borrow the naive eye-shuffle null for panel 3D's reference
    # band (the pipeline only runs shuffles for naive); observed values stay
    # each target's own.
    "FULL": methods_to_fig2_schema(md, "full", null_from="naive"),
    "CENTRAL": methods_to_fig2_schema(md, "central", null_from="naive"),
}
COLS = ["PROD", "NAIVE", "FULL", "CENTRAL"]
SUB = {
    "PROD": "production fig2/fig3 pipeline",
    "NAIVE": "methods target='naive' (must match PROD)",
    "FULL": "methods target='full' (new default)",
    "CENTRAL": "methods target='central'",
}

PANELS = [
    ("fig2c", FIG2, "generate_fig2c", "plot_panel_c", "Fig 2C  -  1-alpha histogram (8.3 ms)"),
    ("fig2e", FIG2, "generate_fig2e", "plot_panel_e", "Fig 2E  -  population Fano slope vs window"),
    ("fig3b", FIG3, "generate_fig3b", "plot_panel_b", "Fig 3B  -  noise-corr scatter (8.3 ms)"),
    ("fig3c", FIG3, "generate_fig3c", "plot_panel_c", "Fig 3C  -  mean Fisher-z NC vs window"),
    ("fig3d", FIG3, "generate_fig3d", "plot_panel_d", "Fig 3D  -  delta-z vs shuffle null"),
]

for tag, pdir, module, func, title in PANELS:
    plot = panel_func(pdir, module, func)
    # gridspec + per-column fig.text titles: some panels (fig3b) call ax.remove()
    # and re-add sub-axes inside their subplotspec, so we never title `ax` itself.
    ncol = len(COLS)
    fig = plt.figure(figsize=(5.0 * ncol, 4.8))
    gs = fig.add_gridspec(1, ncol, wspace=0.32)
    for i, col in enumerate(COLS):
        ax = fig.add_subplot(gs[0, i])
        try:
            plot(ax=ax, data=DATA[col])
        except Exception as e:
            ax.text(0.5, 0.5, f"{col} ERROR:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=8, wrap=True)
            import traceback
            print(f"!! {tag} {col}: {e}")
            traceback.print_exc()
        fig.text((i + 0.5) / ncol, 1.0, f"{col} - {SUB[col]}",
                 ha="center", va="bottom", fontsize=9)
    fig.suptitle(title, fontsize=12, y=1.08)
    out = OUT / f"cmp_{tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

print("DONE.")
