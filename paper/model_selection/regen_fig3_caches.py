"""Regenerate the fig3 inference caches from the pinned checkpoint.

Both caches are built by GPU forward passes over the pinned checkpoint, so any
change to the fig3 inference population requires rebuilding them (~51 min).

Written for the `MIN_TOTAL_SPIKES` 200 -> 0 experiment on 2026-08-04. **That
change was reverted and the threshold stays at 200** -- lowering it costs panel
D up to half its scored windows in 8 of 24 sessions, because the base window
mask is a conjunction over every unit in the session. Do not re-run this
expecting the 0-spike population; see `MODEL_CARD.md`. The live caches are the
200-spike ones, backed up as `*.pre_spikethresh.bak`; the 0-spike caches this
script produced are kept as `*.spikethresh0.bak`.

    FIG3_GPU=0 uv run python paper/model_selection/regen_fig3_caches.py

``FIG3_GPU`` is optional.  Set it when another long-running analysis occupies
one of the host GPUs; otherwise the DataYates helper chooses the least-used
device automatically.
"""
import sys
import time
from pathlib import Path

VISIONCORE_ROOT = Path(__file__).resolve().parent.parent.parent
for p in (VISIONCORE_ROOT, VISIONCORE_ROOT / "paper" / "fig3"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def main():
    t0 = time.time()

    print("=" * 70)
    print("1/2  fig3_digitaltwin.pkl")
    print("=" * 70)
    from _fig3_data import load_fig3_data
    data = load_fig3_data(recompute=True)
    print(f"  -> {len(data['session_results'])} sessions "
          f"({time.time() - t0:.0f}s)")

    print("=" * 70)
    print("2/2  fig3_ablation_inference.pkl")
    print("=" * 70)
    from _fig3_ablation_data import load_ablation_data
    abl = load_ablation_data(recompute=True)
    n = len(abl.get("cd_population", []))
    print(f"  -> {n} neurons in the aggregate ({time.time() - t0:.0f}s)")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
