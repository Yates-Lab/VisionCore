"""Render 4-up across-radii comparisons BASELINE | r=1.0 | r=0.75 | r=0.5 for
fig2 C/E and fig3 B/C/D, at target='full' (production default), by calling the
*existing* production panel functions four times with four per-radius data dicts.

Reuses the consistency infrastructure wholesale: the adapter
(methods_to_fig2_schema) and the panel_func import-hygiene helper. Only the
columns change (radii instead of targets). Each radius's panel-3D shuffle band
comes from that radius's OWN naive null (null_from='naive' within its bundle).

A second 4-up at target='naive' is emitted as a cross-check.

Figures saved under figures/eyepos_masking/.
Run: uv run python eyepos_masking/make_panels.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import dill
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
METHODS_DIR = THIS_DIR.parent
CONS_DIR = METHODS_DIR / "consistency"
RYAN = METHODS_DIR.parent
FIG2, FIG3 = RYAN / "fig2", RYAN / "fig3"
CACHE = METHODS_DIR / "cache"
OUT = METHODS_DIR / "figures" / "eyepos_masking"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CONS_DIR))
sys.path.insert(0, str(METHODS_DIR))
from adapter import methods_to_fig2_schema  # noqa: E402

RADIUS_TAGS = ["base", "1.0", "0.75", "0.5"]
COL_LABEL = {"base": "BASELINE", "1.0": "r=1.0", "0.75": "r=0.75", "0.5": "r=0.5"}
SUB = {"base": "no extra mask (production window)",
       "1.0": "fixation disk r=1.0 deg",
       "0.75": "fixation disk r=0.75 deg",
       "0.5": "fixation disk r=0.5 deg"}


def panel_func(panel_dir: Path, module: str, func: str):
    """Import `module.func` so its `_panel_common` resolves to `panel_dir`'s copy
    while `compute_fig2_data` stays importable from fig2 (verbatim from
    consistency/make_panels.py)."""
    sys.modules.pop("_panel_common", None)
    sys.modules.pop(module, None)
    for p in (str(FIG3), str(FIG2), str(panel_dir)):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(FIG2))
    sys.path.insert(0, str(panel_dir))
    mod = importlib.import_module(module)
    return getattr(mod, func)


def load_radius_bundle(tag):
    with open(CACHE / f"methods_derived_r{tag}.pkl", "rb") as f:
        return dill.load(f)


def build_data(target):
    bundles = {tag: load_radius_bundle(tag) for tag in RADIUS_TAGS}
    # each radius: its own naive null as the panel-3D reference band
    return {tag: methods_to_fig2_schema(bundles[tag], target, null_from="naive")
            for tag in RADIUS_TAGS}


PANELS = [
    ("fig2c", FIG2, "generate_fig2c", "plot_panel_c", "Fig 2C  -  1-alpha histogram (8.3 ms)"),
    ("fig2e", FIG2, "generate_fig2e", "plot_panel_e", "Fig 2E  -  population Fano slope vs window"),
    ("fig3b", FIG3, "generate_fig3b", "plot_panel_b", "Fig 3B  -  noise-corr scatter (8.3 ms)"),
    ("fig3c", FIG3, "generate_fig3c", "plot_panel_c", "Fig 3C  -  mean Fisher-z NC vs window"),
    ("fig3d", FIG3, "generate_fig3d", "plot_panel_d", "Fig 3D  -  delta-z vs shuffle null"),
]


def render(target, suffix):
    DATA = build_data(target)
    ncol = len(RADIUS_TAGS)
    for tag_p, pdir, module, func, title in PANELS:
        plot = panel_func(pdir, module, func)
        fig = plt.figure(figsize=(5.0 * ncol, 4.8))
        gs = fig.add_gridspec(1, ncol, wspace=0.32)
        for i, tag in enumerate(RADIUS_TAGS):
            ax = fig.add_subplot(gs[0, i])
            try:
                plot(ax=ax, data=DATA[tag])
            except Exception as e:
                ax.text(0.5, 0.5, f"{COL_LABEL[tag]} ERROR:\n{e}", ha="center",
                        va="center", transform=ax.transAxes, color="red",
                        fontsize=8, wrap=True)
                import traceback
                print(f"!! {tag_p} {tag}: {e}")
                traceback.print_exc()
            fig.text((i + 0.5) / ncol, 1.0,
                     f"{COL_LABEL[tag]} - {SUB[tag]}",
                     ha="center", va="bottom", fontsize=9)
        fig.suptitle(f"{title}   [target={target}]", fontsize=12, y=1.08)
        out = OUT / f"mask_{tag_p}_{suffix}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    render("full", "full")
    render("naive", "naive")
    print("DONE.")
