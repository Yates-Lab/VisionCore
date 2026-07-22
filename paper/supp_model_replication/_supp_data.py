"""Shared data layer for the twin -> fig2 replication supplement.

The claim: the fig3 digital twin reproduces the fig2 covariance-decomposition
results (panels C, E, F, G, I). To test it on equal footing we push the twin's
predictions through the *identical* estimator fig2 uses -- ``decompose_session``
(paper/covariance_decomposition/decompose.py) followed by the stage-2/3 derived
metrics (derive.py) -- so any difference is the twin, not the pipeline.

Population (see TWIN_IMPROVEMENTS.md item 5): the twin and fig2 include cells by
different rules, so we analyze their per-session INTERSECTION -- cells that are
both fig2-analyzed (rate > 2 Hz & split-half PSTH R^2 > 0.05) and twin-covered
(> 200 spikes on the fixrsvp inference trials). That is 1279 / 1355 = 94.4% of
fig2's neurons across 23 shared sessions. Inclusion is locked to fig2's
real-data rate/PSTH-R^2 (passed as the record's ``rate_hz``/``psth_r2``), so the
population is identical regardless of which counts feed the covariance.

Three ``counts_mode`` inputs, all in the twin frame on the same intersection:
  empirical : real observed spikes (robs_used)      -- the in-frame fig2 anchor.
  raw       : deterministic twin rates (rhat_used)  -- rate structure only.
  poisson   : Poisson(rhat_used)                    -- twin rates realized as
              conditionally-independent counts. Fano=1 and noise-corr=0 are then
              the built-in null, so any Fano>1 / rho>0 the pipeline recovers
              comes purely from FEM-driven rate modulation.

Usage:
    from _supp_data import compute_supp_bundle, load_fig2_reference
    twin = compute_supp_bundle("poisson")     # fig2-shaped bundle
    emp  = compute_supp_bundle("empirical")
"""
from __future__ import annotations

import sys

import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR, FIGURES_DIR, STATS_DIR

# --- pipeline imports (shared covariance-decomposition + fig3 twin cache) ------
_COVDECOMP = str(VISIONCORE_ROOT / "paper" / "covariance_decomposition")
_FIG3 = str(VISIONCORE_ROOT / "paper" / "fig3")
for _p in (_COVDECOMP, _FIG3):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import derive  # noqa: E402
from decompose import decompose_session, WINDOW_BINS_DEFAULT  # noqa: E402
from data_loading import (  # noqa: E402
    load_cache as load_aligned_cache, FIXATION_RADIUS,
)

# --- constants ----------------------------------------------------------------
SUBJECTS = derive.SUBJECTS
SUBJECT_COLORS = derive.SUBJECT_COLORS
MIN_RATE_HZ = derive.MIN_RATE_HZ            # 2.0
MIN_PSTH_R2 = derive.MIN_PSTH_R2            # 0.05
# The twin inference cache (fig3) kept fixation < 1.0 deg, but fig2's aligned
# cache uses FIXATION_RADIUS (0.5 deg). To compare on the SAME fixation window we
# retighten the analyzed samples to |eye| < FIXATION_RADIUS here -- the model's
# rates at the retained bins are unchanged, so no re-inference is needed.
SUPP_FIXATION_RADIUS = FIXATION_RADIUS      # 0.5 (match fig2)

TWIN_CACHE = CACHE_DIR / "fig3_digitaltwin.pkl"
# Twin inference re-run on fig2's exact frame (fixation < 0.5 deg); built by
# _supp_inference.py. Preferred over TWIN_CACHE (1.0 deg) so 'empirical' matches
# published fig2. Falls back to TWIN_CACHE if not yet built.
#
# Two on-disk formats are supported:
#   supp_twin_fig2frame_conditions.pkl : rhat_used is {cond: array} for the three
#       within-model conditions (intact/zeroed/stabilized). Preferred.
#   supp_twin_fig2frame.pkl            : rhat_used is a single array (intact only,
#       legacy). Usable only for condition='intact'.
SUPP_INFERENCE_CONDITIONS_CACHE = CACHE_DIR / "supp_twin_fig2frame_conditions.pkl"
SUPP_INFERENCE_CACHE = CACHE_DIR / "supp_twin_fig2frame.pkl"
# Within-model conditions (fig3 naming). 'intact' == the full twin.
SUPP_CONDITIONS = ("intact", "zeroed", "stabilized")
PANEL_C_CACHE = CACHE_DIR / "supp_panel_c.pkl"
FIG2_DERIVED_CACHE = CACHE_DIR / "covdecomp_derived.pkl"
FIG2_REF_CACHE = CACHE_DIR / "supp_fig2_reference.pkl"

FIG_DIR = FIGURES_DIR / "supp_model_replication"
STAT_DIR = STATS_DIR / "supp_model_replication"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

# Exploration defaults (kept below fig2's production 1000 shuffles). Bump for a
# final figure. Windows must span (1,2,3,6) bins so derive's SUBSPACE_WINDOW_IDX
# (=2, the 25 ms window) stays valid.
DEFAULT_N_SHUFFLES = 200
DEFAULT_WINDOWS_BINS = tuple(WINDOW_BINS_DEFAULT)

COUNTS_MODES = ("empirical", "raw", "poisson")

_MEM_CACHE: dict = {}


def configure_matplotlib():
    import matplotlib as mpl
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


# ---------------------------------------------------------------------------
# Intersection-population record construction (twin frame)
# ---------------------------------------------------------------------------

def _fig2_included(aligned_rec):
    """orig-neuron-index -> (rate_hz, psth_r2) for cells passing fig2 inclusion."""
    nm = np.asarray(aligned_rec["neuron_mask"])
    rate = np.asarray(aligned_rec["rate_hz"], float)
    psth = np.asarray(aligned_rec["psth_r2"], float)
    keep = np.isfinite(rate) & (rate > MIN_RATE_HZ) & np.isfinite(psth) & (psth > MIN_PSTH_R2)
    return {int(o): (float(r), float(p))
            for o, r, p, k in zip(nm, rate, psth, keep) if k}


def _draw_counts(rhat, counts_mode, rng):
    """Turn deterministic twin rates into the record's ``robs`` for one mode."""
    if counts_mode == "raw":
        return np.asarray(rhat, np.float64)
    if counts_mode == "poisson":
        finite = np.isfinite(rhat)
        lam = np.where(finite, np.clip(rhat, 0.0, None), 0.0)
        counts = rng.poisson(lam).astype(np.float64)
        counts[~finite] = np.nan          # preserve the missing-sample pattern
        return counts
    raise ValueError(f"_draw_counts does not handle {counts_mode!r}")


def _twin_source_path():
    """Preferred twin inference cache: the 3-condition build if present, else the
    legacy intact-only fig2-frame cache, else the fig3 1.0-deg cache."""
    if SUPP_INFERENCE_CONDITIONS_CACHE.exists():
        return SUPP_INFERENCE_CONDITIONS_CACHE
    if SUPP_INFERENCE_CACHE.exists():
        return SUPP_INFERENCE_CACHE
    return TWIN_CACHE


def _load_twin_source(context=""):
    src = _twin_source_path()
    note = ""
    if src is TWIN_CACHE:
        note = "  (fig2-frame cache missing; run _supp_inference.py to match fig2)"
    elif src is SUPP_INFERENCE_CACHE:
        note = "  (intact-only legacy cache; only condition='intact' available)"
    print(f"[supp] {context}inference source: {src.name}{note}")
    with open(src, "rb") as f:
        return dill.load(f)


def _select_rhat(sr, condition):
    """Per-session model rates for one within-model condition.

    Handles both cache formats: rhat_used is either a {cond: array} dict
    (3-condition build) or a single array (legacy intact-only build)."""
    rhat = sr["rhat_used"]
    if isinstance(rhat, dict):
        if condition not in rhat:
            raise KeyError(f"condition {condition!r} not in cache "
                           f"(have {tuple(rhat)})")
        return np.asarray(rhat[condition])
    if condition != "intact":
        raise ValueError(
            f"condition={condition!r} requires the 3-condition inference cache "
            f"({SUPP_INFERENCE_CONDITIONS_CACHE.name}); the loaded cache is "
            f"intact-only. Run _supp_inference.py.")
    return np.asarray(rhat)


def build_records(counts_mode, condition="intact", seed=0):
    """Twin-frame aligned-session records on the fig2 x twin intersection.

    ``condition`` selects the within-model route (intact/zeroed/stabilized); it
    matters only when ``counts_mode`` derives from the model rates (raw/poisson).
    For counts_mode='empirical' (real spikes) the condition is irrelevant.

    Returns a list of dicts matching ``decompose_session``'s ``aligned`` input
    (robs, eyepos, valid_mask, neuron_mask, rate_hz, psth_r2, meta ...).
    """
    if counts_mode not in COUNTS_MODES:
        raise ValueError(f"counts_mode must be one of {COUNTS_MODES}")
    if condition not in SUPP_CONDITIONS:
        raise ValueError(f"condition must be one of {SUPP_CONDITIONS}")
    twin = _load_twin_source()
    aligned_by = {a["session"]: a for a in load_aligned_cache()}
    rng = np.random.default_rng(seed)

    records = []
    for sr in twin:
        sess = sr["session"]
        if sess not in aligned_by:
            continue
        included = _fig2_included(aligned_by[sess])
        tnm = np.asarray(sr["neuron_mask"])
        cols = np.array([j for j, o in enumerate(tnm) if int(o) in included], dtype=int)
        if cols.size < 3:
            continue
        orig = tnm[cols]
        rate_hz = np.array([included[int(o)][0] for o in orig], float)
        psth_r2 = np.array([included[int(o)][1] for o in orig], float)

        rhat = _select_rhat(sr, condition)[:, :, cols]
        robs = np.asarray(sr["robs_used"])[:, :, cols]
        if counts_mode == "empirical":
            counts = np.asarray(robs, np.float64)
        else:
            counts = _draw_counts(rhat, counts_mode, rng)

        # Match fig2's fixation window: restrict analyzed samples to |eye| < 0.5
        # deg (the twin cache was built at < 1.0 deg). decompose_session builds
        # its close-pair segments from valid_mask, so tightening it here puts the
        # FEM estimator on the same eye-position window fig2 uses.
        eyepos = np.asarray(sr["eyepos_used"], np.float64)
        base_valid = np.asarray(sr["valid_mask"], bool)
        r_eye = np.hypot(eyepos[..., 0], eyepos[..., 1])
        fix_ok = np.isfinite(r_eye) & (r_eye < SUPP_FIXATION_RADIUS)
        valid_mask = base_valid & fix_ok

        records.append({
            "session": sess,
            "subject": sr["subject"],
            "robs": counts,
            "eyepos": eyepos,
            "valid_mask": valid_mask,
            "neuron_mask": orig,
            "rate_hz": rate_hz,
            "psth_r2": psth_r2,
            "contam_rate": None,
            "n_trials_total": counts.shape[0],
            "n_trials_good": counts.shape[0],
            "n_neurons_total": counts.shape[2],
            "n_neurons_used": counts.shape[2],
        })
    return records


# ---------------------------------------------------------------------------
# Decompose + derive (identical estimator to fig2)
# ---------------------------------------------------------------------------

def _decompose_all(records, windows_bins, n_shuffles, n_jobs=-1):
    from joblib import Parallel, delayed

    def _one(rec):
        try:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(1):
                return decompose_session(rec, windows_bins=windows_bins,
                                         n_shuffles=n_shuffles)
        except ImportError:
            return decompose_session(rec, windows_bins=windows_bins,
                                     n_shuffles=n_shuffles)

    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(_one)(r) for r in records)


def _derive_bundle(raw_session_results):
    """Run derive's stage-2/3 on raw decompose output -> fig2-shaped bundle."""
    sr = derive._to_mats_schema(raw_session_results, target=derive.TARGET)
    sr = derive._apply_session_unit_floor(sr)
    windows_ms = [r["window_ms"] for r in sr[0]["results"]]
    windows_bins = [r["window_bins"] for r in sr[0]["results"]]
    session_names = [x["session"] for x in sr]
    subjects = [x["subject"] for x in sr]

    metrics = derive._compute_metrics(sr, windows_ms, windows_bins)
    m_by_window, subj_pn_by_window, alpha_stats = derive._compute_alpha_stats(
        metrics, windows_ms)
    fano_stats = derive._compute_fano_stats(metrics, windows_ms)
    nc_stats = derive._compute_nc_stats(metrics, windows_ms)
    subspace = derive._compute_subspace(sr, session_names, subjects)

    bundle = dict(
        metrics=metrics,
        m_by_window=m_by_window,
        subject_per_neuron_by_window=subj_pn_by_window,
        alpha_stats=alpha_stats,
        fano_stats=fano_stats,
        nc_stats=nc_stats,
        WINDOWS_MS=windows_ms,
        WINDOWS_BINS=windows_bins,
        SUBJECTS=SUBJECTS,
        SUBJECT_COLORS=SUBJECT_COLORS,
        session_names=session_names,
        subjects=subjects,
        n_sessions=len(sr),
        SUBSPACE_WINDOW_IDX=derive.SUBSPACE_WINDOW_IDX,
        SUBSPACE_K=derive.SUBSPACE_K,
        **subspace,
    )
    return bundle


def _cache_path(counts_mode, condition, n_shuffles, seed):
    # 'empirical' (real spikes) is condition-independent, so it keeps a single
    # condition-free cache; model-derived modes are cached per condition.
    cond = "" if counts_mode == "empirical" else f"_{condition}"
    return CACHE_DIR / f"supp_repl_{counts_mode}{cond}_s{n_shuffles}_seed{seed}.pkl"


def compute_supp_bundle(counts_mode, condition="intact",
                        n_shuffles=DEFAULT_N_SHUFFLES,
                        windows_bins=DEFAULT_WINDOWS_BINS, seed=0,
                        refresh=False):
    """fig2-shaped bundle for one (counts_mode, condition), cached on disk + mem.

    ``condition`` (intact/zeroed/stabilized) selects the within-model route and
    is ignored for counts_mode='empirical' (real spikes)."""
    cond = "intact" if counts_mode == "empirical" else condition
    key = (counts_mode, cond, n_shuffles, tuple(windows_bins), seed)
    if not refresh and key in _MEM_CACHE:
        return _MEM_CACHE[key]

    path = _cache_path(counts_mode, cond, n_shuffles, seed)
    if path.exists() and not refresh:
        print(f"[supp] loading {counts_mode}/{cond} bundle from {path}")
        with open(path, "rb") as f:
            bundle = dill.load(f)
        _MEM_CACHE[key] = bundle
        return bundle

    print(f"[supp] computing {counts_mode}/{cond} bundle "
          f"(n_shuffles={n_shuffles}, windows={tuple(windows_bins)}, seed={seed})")
    records = build_records(counts_mode, condition=cond, seed=seed)
    print(f"[supp]   {len(records)} intersection sessions, "
          f"{sum(r['n_neurons_used'] for r in records)} cells")
    raw = _decompose_all(records, windows_bins, n_shuffles)
    bundle = _derive_bundle(raw)
    with open(path, "wb") as f:
        dill.dump(bundle, f)
    print(f"[supp]   cached -> {path}")
    _MEM_CACHE[key] = bundle
    return bundle


def compute_poisson_draws(n_draws=5, condition="intact",
                          n_shuffles=DEFAULT_N_SHUFFLES,
                          windows_bins=DEFAULT_WINDOWS_BINS, base_seed=100,
                          refresh=False):
    """List of Poisson-mode bundles over independent draws (seeds base_seed+i)."""
    return [
        compute_supp_bundle("poisson", condition=condition, n_shuffles=n_shuffles,
                            windows_bins=windows_bins, seed=base_seed + i,
                            refresh=refresh)
        for i in range(n_draws)
    ]


# ---------------------------------------------------------------------------
# Panel C: per-cell 1-alpha (raw twin rates vs neurons) on the fig2 frame
# ---------------------------------------------------------------------------

def _panel_c_cache_path(condition):
    return CACHE_DIR / f"supp_panel_c_{condition}.pkl"


def compute_panel_c_data(condition="intact", refresh=False):
    """Per-cell 1-alpha (estimator B, target='full') on the fig2-frame intersection.

    B_obs   : neurons (real spikes) through the close-pair estimator
              (condition-independent).
    B_model : twin RAW rates for ``condition`` (intact/zeroed/stabilized) through
              the identical estimator.
    Matched per cell, on the same intersection population (fig2-analyzed AND
    twin-covered) the E/F/G/I panels use -- so panel C describes the same cells
    and the same frame. Replaces the fig3 per-cell bundle (covdecomp_model.pkl,
    the 1.0 deg frame); no ccmax filter, since fig2 inclusion already gates
    reliability.
    """
    if condition not in SUPP_CONDITIONS:
        raise ValueError(f"condition must be one of {SUPP_CONDITIONS}")
    cache_path = _panel_c_cache_path(condition)
    if cache_path.exists() and not refresh:
        with open(cache_path, "rb") as f:
            return dill.load(f)

    from model_decompose import decompose_model_session  # noqa: E402

    twin = _load_twin_source(context=f"panel C [{condition}] ")
    aligned_by = {a["session"]: a for a in load_aligned_cache()}

    B_obs, B_model, subj, sess_list = [], [], [], []
    B_obs_uncl, B_model_uncl = [], []
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
        valid_mask = base_valid & np.isfinite(r_eye) & (r_eye < SUPP_FIXATION_RADIUS)

        comp = decompose_model_session(rhat, robs, eyepos, valid_mask, dfs)
        B_obs.extend(comp["B_obs"])
        B_model.extend(comp["B_model"])
        B_obs_uncl.extend(comp["B_obs_uncl"])
        B_model_uncl.extend(comp["B_model_uncl"])
        subj.extend([sr["subject"]] * cols.size)
        sess_list.extend([sess] * cols.size)
        print(f"[supp]   panel C {sess}: {cols.size} cells")

    out = {
        # Clipped [0,1] values (legacy). Prefer the *_uncl arrays + a [0,1]
        # exclusion to match fig2 (avoids the clip pile-up at 0 / 1).
        "B_obs": np.asarray(B_obs, float),
        "B_model": np.asarray(B_model, float),
        "B_obs_uncl": np.asarray(B_obs_uncl, float),
        "B_model_uncl": np.asarray(B_model_uncl, float),
        "subj": np.asarray(subj, dtype=object).astype(str),
        "session": np.asarray(sess_list, dtype=object).astype(str),
    }
    with open(cache_path, "wb") as f:
        dill.dump(out, f)
    print(f"[supp]   cached panel C [{condition}] "
          f"({out['B_obs'].size} cells) -> {cache_path}")
    return out


# ---------------------------------------------------------------------------
# Published fig2 reference (small extract of the 8 GB derived bundle)
# ---------------------------------------------------------------------------

_REF_KEYS = (
    "metrics", "m_by_window", "subject_per_neuron_by_window", "alpha_stats",
    "fano_stats", "nc_stats", "WINDOWS_MS", "WINDOWS_BINS", "SUBJECTS",
    "SUBJECT_COLORS", "session_names", "subjects", "n_sessions",
    "SUBSPACE_WINDOW_IDX", "SUBSPACE_K",
    "pr_fem_list", "pr_psth_list", "pr_resid_list", "sub_names", "sub_subjects",
    "overlap_k1_list", "overlap_k_list", "var_p_given_f", "var_f_given_p",
    "spectra_psth", "spectra_fem", "null_var_p_given_f", "null_var_f_given_p",
    "null_overlap_k", "null_overlap_k1", "null_session_idx", "null_subjects",
)


def load_fig2_reference(refresh=False):
    """Published fig2 derived stats (the real paper result, fig2 frame + all
    1355 cells), extracted once from covdecomp_derived.pkl into a small cache.
    The outer reference; the in-frame anchor is compute_supp_bundle('empirical').
    """
    if FIG2_REF_CACHE.exists() and not refresh:
        with open(FIG2_REF_CACHE, "rb") as f:
            return dill.load(f)
    print(f"[supp] extracting fig2 reference from {FIG2_DERIVED_CACHE} ...")
    with open(FIG2_DERIVED_CACHE, "rb") as f:
        full = dill.load(f)
    ref = {k: full[k] for k in _REF_KEYS if k in full}
    with open(FIG2_REF_CACHE, "wb") as f:
        dill.dump(ref, f)
    print(f"[supp]   cached fig2 reference -> {FIG2_REF_CACHE}")
    return ref


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build supplement twin bundles.")
    ap.add_argument("--modes", nargs="+", default=list(COUNTS_MODES))
    ap.add_argument("--conditions", nargs="+", default=["intact"],
                    help="Within-model conditions for model-derived modes.")
    ap.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--fig2-ref", action="store_true", help="Also build fig2 reference.")
    args = ap.parse_args()
    for mode in args.modes:
        # 'empirical' is condition-independent; build it once.
        conds = ["intact"] if mode == "empirical" else args.conditions
        for cond in conds:
            b = compute_supp_bundle(mode, condition=cond, n_shuffles=args.n_shuffles,
                                    seed=args.seed, refresh=args.refresh)
            w = b["WINDOWS_MS"][b["SUBSPACE_WINDOW_IDX"]]
            fs = b["fano_stats"][w]
            nc = b["nc_stats"][w]
            print(f"  [{mode}/{cond}] @25ms: Fano unc={fs['slope_unc']:.3f} "
                  f"cor={fs['slope_cor']:.3f} | NC z_u={nc['z_u_mean']:.3f} "
                  f"z_c={nc['z_c_mean']:.3f} | {b['n_sessions']} sessions")
    if args.fig2_ref:
        load_fig2_reference(refresh=args.refresh)
