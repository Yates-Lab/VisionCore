"""FEM-modulation fraction (1 - alpha) for figure 3 panel E.

Panel E is now a primary result: the per-cell FEM modulation fraction for the
neurons vs. each within-model twin condition, on the fig2 frame and the fig2 x
twin intersection population -- the same quantity fig2 panel C reports
("fraction of rate modulation due to FEM"). This module lets fig3 own that
computation rather than borrow it from the replication supplement.

For every cell we compute 1 - alpha with the SAME matched close-pair estimator
fig2 uses (`decompose_model_session`, estimator B, target='full') on the fig2
fixation frame (|eye| < 0.5 deg):

  B_obs   : the neurons (real spikes)        -- condition-independent.
  B_model : the twin rates for a condition   -- intact / zeroed / stabilized.

Both are returned UNCLIPPED (`*_uncl`); callers apply fig2's [0, 1] inclusion
(exclude out-of-range cells rather than clip them onto the boundary).

Inputs come entirely from on-disk caches -- the fig2-frame twin inference cache
(`supp_twin_fig2frame_conditions.pkl`) and the aligned covariance cache -- so no
GPU/model pass runs here. Results are cached per condition; a one-time migration
reuses the legacy `supp_panel_c_{cond}.pkl` cache bit-for-bit if present.
"""
from __future__ import annotations

import sys

import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR

# Shared covariance-decomposition machinery (neutral package; not the supplement).
_COVDECOMP = str(VISIONCORE_ROOT / "paper" / "covariance_decomposition")
if _COVDECOMP not in sys.path:
    sys.path.insert(0, _COVDECOMP)

import derive  # noqa: E402
from data_loading import load_cache as load_aligned_cache, FIXATION_RADIUS  # noqa: E402
from model_decompose import decompose_model_session  # noqa: E402
from fig3_windows import FIG2_REPORTED_WINDOW_BINS  # noqa: E402

# Within-model conditions (fig3 naming); 'intact' == the full twin.
CONDITIONS = ("intact", "zeroed", "stabilized")

# fig2 inclusion (rate > 2 Hz & split-half PSTH R^2 > 0.10) and fig2 frame.
MIN_RATE_HZ = derive.MIN_RATE_HZ            # 2.0
MIN_PSTH_R2 = derive.MIN_PSTH_R2            # 0.05
FIG2_FIXATION_RADIUS = FIXATION_RADIUS      # 0.5 deg

# Twin inference caches (fig2 frame, |eye| < 0.5). Preferred: the 3-condition
# build; fall back to the legacy intact-only build, then the 1.0-deg fig3 cache.
_TWIN_CONDITIONS_CACHE = CACHE_DIR / "supp_twin_fig2frame_conditions.pkl"
_TWIN_INTACT_CACHE = CACHE_DIR / "supp_twin_fig2frame.pkl"
_TWIN_FIG3_CACHE = CACHE_DIR / "fig3_digitaltwin.pkl"

# Legacy cache (identical computation) reused on first run to skip recompute.
_LEGACY_CACHE = {c: CACHE_DIR / f"supp_panel_c_{c}.pkl" for c in CONDITIONS}


def _cache_path(condition, count_bins=None):
    """Cache path, keyed by the counting window the f_FEM was estimated on.

    The window is in the filename so the 25 ms rebuild cannot silently overwrite
    the legacy instantaneous-estimator caches, and so a future window change
    invalidates by name rather than by memory.
    """
    if count_bins is None:
        count_bins = FIG2_REPORTED_WINDOW_BINS
    return CACHE_DIR / f"fig3_femfraction_{condition}_w{count_bins}.pkl"


def _fig2_included(aligned_rec):
    """orig-neuron-index -> (rate_hz, psth_r2) for cells passing fig2 inclusion."""
    nm = np.asarray(aligned_rec["neuron_mask"])
    rate = np.asarray(aligned_rec["rate_hz"], float)
    psth = np.asarray(aligned_rec["psth_r2"], float)
    keep = (np.isfinite(rate) & (rate > MIN_RATE_HZ)
            & np.isfinite(psth) & (psth > MIN_PSTH_R2))
    return {int(o) for o, k in zip(nm, keep) if k}


def _twin_source_path():
    if _TWIN_CONDITIONS_CACHE.exists():
        return _TWIN_CONDITIONS_CACHE
    if _TWIN_INTACT_CACHE.exists():
        return _TWIN_INTACT_CACHE
    return _TWIN_FIG3_CACHE


def _load_twin_source(context=""):
    src = _twin_source_path()
    note = ""
    if src is _TWIN_FIG3_CACHE:
        note = "  (fig2-frame cache missing; run supp _supp_inference.py to match fig2)"
    elif src is _TWIN_INTACT_CACHE:
        note = "  (intact-only legacy cache; only condition='intact' available)"
    print(f"[fig3.femfrac] {context}inference source: {src.name}{note}")
    with open(src, "rb") as f:
        return dill.load(f)


def _select_rhat(sr, condition):
    """Per-session twin rates for one within-model condition, handling both the
    {cond: array} 3-condition cache and the legacy single-array intact cache."""
    rhat = sr["rhat_used"]
    if isinstance(rhat, dict):
        if condition not in rhat:
            raise KeyError(f"condition {condition!r} not in cache (have {tuple(rhat)})")
        return np.asarray(rhat[condition])
    if condition != "intact":
        raise ValueError(
            f"condition={condition!r} needs the 3-condition twin cache "
            f"({_TWIN_CONDITIONS_CACHE.name}); loaded cache is intact-only.")
    return np.asarray(rhat)


def compute_femfraction_data(condition="intact", refresh=False, count_bins=None):
    """Per-cell 1-alpha (unclipped) for the neurons (B_obs) and the twin
    ``condition`` (B_model), matched per cell on the fig2-frame intersection.

    ``count_bins`` is the Figure 2 counting window the estimate is built on;
    it defaults to Figure 2's reported window so panel E's f_FEM is the same
    quantity Figure 2 panel C reports. ``count_bins=0`` selects the legacy
    instantaneous single-bin estimator.

    Returns {'B_obs_uncl', 'B_model_uncl', 'B_obs', 'B_model', 'subj',
    'session', 'count_bins'} as flat (n_cells,) arrays. Cached per condition
    and window.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if count_bins is None:
        count_bins = FIG2_REPORTED_WINDOW_BINS

    cache = _cache_path(condition, count_bins)
    if cache.exists() and not refresh:
        with open(cache, "rb") as f:
            return dill.load(f)

    # The legacy supp caches hold the INSTANTANEOUS single-bin estimator, which
    # is not this computation at any counting window (it is biased low against
    # Figure 2 by 0.071 at 25 ms). Migrate it only for the legacy window, never
    # into a windowed cache.
    legacy = _LEGACY_CACHE[condition]
    if not count_bins and legacy.exists() and not refresh:
        print(f"[fig3.femfrac] migrating {legacy.name} -> {cache.name} (identical)")
        with open(legacy, "rb") as f:
            out = dill.load(f)
        with open(cache, "wb") as f:
            dill.dump(out, f)
        return out

    twin = _load_twin_source(context=f"[{condition}] ")
    aligned_by = {a["session"]: a for a in load_aligned_cache()}

    B_obs, B_model, B_obs_uncl, B_model_uncl, subj, sess_list = [], [], [], [], [], []
    for sr in twin:
        sess = sr["session"]
        if sess not in aligned_by:
            continue
        included = _fig2_included(aligned_by[sess])
        tnm = np.asarray(sr["neuron_mask"])
        cols = np.array([j for j, o in enumerate(tnm) if int(o) in included], dtype=int)
        if cols.size < 3:
            continue
        rhat = _select_rhat(sr, condition)[:, :, cols]
        robs = np.asarray(sr["robs_used"])[:, :, cols]
        dfs = np.asarray(sr["dfs_used"])[:, :, cols]
        eyepos = np.asarray(sr["eyepos_used"], np.float64)
        base_valid = np.asarray(sr["valid_mask"], bool)
        r_eye = np.hypot(eyepos[..., 0], eyepos[..., 1])
        valid_mask = base_valid & np.isfinite(r_eye) & (r_eye < FIG2_FIXATION_RADIUS)

        comp = decompose_model_session(rhat, robs, eyepos, valid_mask, dfs,
                                       count_bins=count_bins)
        B_obs.extend(comp["B_obs"])
        B_model.extend(comp["B_model"])
        B_obs_uncl.extend(comp["B_obs_uncl"])
        B_model_uncl.extend(comp["B_model_uncl"])
        subj.extend([sr["subject"]] * cols.size)
        sess_list.extend([sess] * cols.size)
        print(f"[fig3.femfrac]   {sess}: {cols.size} cells")

    out = {
        "B_obs": np.asarray(B_obs, float),
        "B_model": np.asarray(B_model, float),
        "B_obs_uncl": np.asarray(B_obs_uncl, float),
        "B_model_uncl": np.asarray(B_model_uncl, float),
        "subj": np.asarray(subj, dtype=object).astype(str),
        "session": np.asarray(sess_list, dtype=object).astype(str),
        "count_bins": int(count_bins),
    }
    with open(cache, "wb") as f:
        dill.dump(out, f)
    print(f"[fig3.femfrac]   cached [{condition}] ({out['B_obs'].size} cells) -> {cache}")
    return out


def in01(v):
    """fig2's 1-alpha inclusion: finite and within [0, 1] (unclipped values)."""
    v = np.asarray(v, float)
    return np.isfinite(v) & (v >= 0.0) & (v <= 1.0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build fig3 panel-E f_FEM caches.")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    for c in CONDITIONS:
        d = compute_femfraction_data(condition=c, refresh=args.refresh)
        v = d["B_model_uncl"][in01(d["B_model_uncl"])]
        print(f"  [{c}] model median 1-a = {np.median(v):.3f} (n={v.size})")
