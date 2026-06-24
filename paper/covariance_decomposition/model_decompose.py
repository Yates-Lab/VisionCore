"""Model digital-twin covariance decomposition (the model side of fig 3 panel D).

Centralizes the per-cell model 1-alpha computation that was scattered across the
fig3 scripts. Consumes the digital-twin inference cache produced by
``paper/fig3/_fig3_data.py`` (the heavy GPU model forward pass stays there) and
computes, per cell, 1-alpha three ways in the SAME fig3 frame:

  A_model : rate_variance_components(rhat)         -- all-samples one-way ANOVA.
  B_model : pipeline_one_minus_alpha(rhat, eye)    -- close-pair estimator B,
            target='full' (the matched production estimator) on model rates.
  B_obs   : pipeline_one_minus_alpha(robs, eye)    -- estimator B on the neurons,
            in the fig3 frame (the neuron-side companion to B_model).

Cached as ``covdecomp_model.pkl``. ``A_model`` is the estimator currently drawn
on the panel-D model axis (``generate_fig3d.compute_model_one_minus_alpha``);
``B_model`` / ``B_obs`` place the model and the neurons on the identical matched
estimator (the equal-footing comparison). The estimator B path uses the same
``decompose(target='full', cpsth_method='mcfarland', closepair_density='direct')``
as the empirical pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import dill

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR
from VisionCore.covariance import rate_variance_components, pipeline_one_minus_alpha

THRESHOLD = 0.05
MIN_TRIALS_PER_PHASE = 10
MODEL_CACHE = CACHE_DIR / "covdecomp_model.pkl"


def decompose_model_session(rhat, robs, eye, valid_mask, dfs,
                            threshold=THRESHOLD,
                            min_trials_per_phase=MIN_TRIALS_PER_PHASE):
    """Per-cell model 1-alpha (A_model, B_model, B_obs) for one session.

    Runs PER CELL: the digital-twin rates carry per-cell data-filter (dfs) NaNs,
    so each cell is decomposed on its own ``valid_mask & dfs[:, :, ni]!=0`` sample
    set (the multi-cell estimator's all-cells-finite requirement would discard
    nearly every sample). Estimator B (close-pair, target='full') is therefore
    applied to single-cell slices, matching how estimator A is evaluated.

    Parameters
    ----------
    rhat : ndarray (n_trials, n_time, n_cells)  -- deterministic model rates.
    robs : ndarray (n_trials, n_time, n_cells)  -- observed spike counts.
    eye  : ndarray (n_trials, n_time, 2)        -- eye position (deg).
    valid_mask : ndarray (n_trials, n_time)     -- eye-finite sample mask.
    dfs  : ndarray (n_trials, n_time, n_cells)  -- per-cell data filter.

    Returns
    -------
    dict of (n_cells,) arrays: A_model, B_model, B_obs.
    """
    n_cells = rhat.shape[2]
    valid_mask = np.asarray(valid_mask, bool)
    A_model = np.full(n_cells, np.nan)
    B_model = np.full(n_cells, np.nan)
    B_obs = np.full(n_cells, np.nan)

    for ni in range(n_cells):
        valid_ni = valid_mask & (dfs[:, :, ni] != 0)

        A_model[ni] = rate_variance_components(
            rhat[:, :, ni], valid=valid_ni,
            min_trials_per_phase=min_trials_per_phase,
        )["one_minus_alpha"]

        B_model[ni] = pipeline_one_minus_alpha(
            rhat[:, :, ni:ni + 1], eye, valid=valid_ni, threshold=threshold,
            min_trials_per_phase=min_trials_per_phase,
        )["one_minus_alpha"][0]

        B_obs[ni] = pipeline_one_minus_alpha(
            robs[:, :, ni:ni + 1], eye, valid=valid_ni, threshold=threshold,
            min_trials_per_phase=min_trials_per_phase,
        )["one_minus_alpha"][0]

    return {"A_model": A_model, "B_model": B_model, "B_obs": B_obs}


def load_model_data(refresh=False):
    """Canonical per-cell model decomposition bundle, cached.

    Returns a dict with per-session results and flattened per-cell arrays:
        session_results : list of {session, subject, A_model, B_model, B_obs,
                                    alpha (empirical fig2 1-alpha), ccmax}
        subj, A_model, B_model, B_obs, fig2, ccmax : flattened (n_cells_total,)
    """
    if MODEL_CACHE.exists() and not refresh:
        print(f"Loading cached model decomposition from {MODEL_CACHE}")
        with open(MODEL_CACHE, "rb") as f:
            return dill.load(f)

    # The digital-twin inference cache lives in the fig3 package.
    fig3_dir = str(VISIONCORE_ROOT / "paper" / "fig3")
    if fig3_dir not in sys.path:
        sys.path.insert(0, fig3_dir)
    from _fig3_data import load_fig3_data

    data = load_fig3_data()
    per_session = []
    flat = {k: [] for k in ("subj", "A_model", "B_model", "B_obs", "fig2", "ccmax")}
    for si, sr in enumerate(data["session_results"]):
        rhat = sr["rhat_used"]; robs = sr["robs_used"]; eye = sr["eyepos_used"]
        vmask = sr["valid_mask"]; dfs = sr["dfs_used"]
        alpha = np.asarray(sr["alpha"], float)
        ccmax = np.asarray(sr["ccmax"], float)
        print(f"[{si+1}/{len(data['session_results'])}] {sr['session']} "
              f"({sr['subject']}): {rhat.shape[0]} trials, {rhat.shape[2]} cells")
        comp = decompose_model_session(rhat, robs, eye, vmask, dfs)
        rec = {"session": sr["session"], "subject": sr["subject"],
               "A_model": comp["A_model"], "B_model": comp["B_model"],
               "B_obs": comp["B_obs"], "alpha": alpha, "ccmax": ccmax}
        per_session.append(rec)
        flat["subj"].extend([sr["subject"]] * len(alpha))
        flat["A_model"].extend(comp["A_model"])
        flat["B_model"].extend(comp["B_model"])
        flat["B_obs"].extend(comp["B_obs"])
        flat["fig2"].extend(1.0 - alpha)
        flat["ccmax"].extend(ccmax)

    bundle = {"session_results": per_session,
              "subj": np.asarray(flat["subj"], dtype=object).astype(str)}
    for k in ("A_model", "B_model", "B_obs", "fig2", "ccmax"):
        bundle[k] = np.asarray(flat[k], dtype=float)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_CACHE, "wb") as f:
        dill.dump(bundle, f)
    print(f"\nCached model decomposition to {MODEL_CACHE}")
    return bundle


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute model digital-twin decomposition.")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    load_model_data(refresh=args.refresh)
