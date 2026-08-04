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
from VisionCore.covariance import (
    rate_variance_components, pipeline_one_minus_alpha,
    decompose_trajectory, extract_valid_segments, extract_windows,
)

THRESHOLD = 0.05
MIN_TRIALS_PER_PHASE = 10
MODEL_CACHE = CACHE_DIR / "covdecomp_model.pkl"


# ---------------------------------------------------------------------------
# Figure 2's estimator, applied per cell
# ---------------------------------------------------------------------------

def one_minus_alpha_windowed(rate, eyepos, valid, t_count=None, t_hist=None,
                             threshold=THRESHOLD,
                             min_trials_per_time_bin=MIN_TRIALS_PER_PHASE,
                             min_seg_len=None):
    """Per-cell unclipped 1-alpha under Figure 2's production estimator.

    ``pipeline_one_minus_alpha`` treats each (trial, bin) as one sample and
    matches close pairs on INSTANTANEOUS eye position, which is not the quantity
    Figure 2 reports: Figure 2 sums counts over a ``t_count``-bin window and
    matches pairs on RMS distance over the ``t_hist + t_count``-bin trajectory.
    This runs Figure 2's path -- the same `decompose_trajectory` plus uncentred
    close-pair `Crate` that `decompose.decompose_session` uses -- on a single
    cell, so a model/neuron f_FEM computed here is numerically the same
    quantity Figure 2 panel C reports.

    ``rate`` may be spike counts or deterministic model rates; `extract_windows`
    sums it over the counting window either way.

    Returns ``{one_minus_alpha, crate_diag, cpsth_diag, n_close_pairs,
    n_samples, n_segments}``; ``one_minus_alpha`` is NaN when the estimator is
    undefined (too few samples, or a non-positive rate variance).
    """
    # Imported lazily: `decompose` pulls in the data-loading stack, and this
    # module is imported by callers that never touch the Figure 2 estimator.
    from decompose import (
        _uncentred_crate, MIN_SEG_LEN_DEFAULT, T_HIST_MS_DEFAULT, DT,
        WEIGHT_CLIP_DEFAULT, TIME_BIN_WEIGHTING_DEFAULT, CPSTH_METHOD_DEFAULT,
        CLOSEPAIR_DENSITY_DEFAULT, N_BOOT_DEFAULT,
    )
    from fig3_windows import FIG2_REPORTED_WINDOW_BINS

    if t_count is None:
        t_count = FIG2_REPORTED_WINDOW_BINS
    if t_hist is None:
        t_hist = int(round(T_HIST_MS_DEFAULT / (DT * 1000)))
    if min_seg_len is None:
        min_seg_len = MIN_SEG_LEN_DEFAULT

    rate = np.asarray(rate, dtype=np.float64)
    if rate.ndim == 2:
        rate = rate[:, :, None]
    if rate.shape[2] != 1:
        raise ValueError(f"expected a single cell, got {rate.shape[2]}")

    undefined = {"one_minus_alpha": np.nan, "crate_diag": np.nan,
                 "cpsth_diag": np.nan, "n_close_pairs": 0, "n_samples": 0,
                 "n_segments": 0}

    segments = extract_valid_segments(np.asarray(valid, bool),
                                      min_len_bins=min_seg_len)
    if not len(segments):
        return undefined
    counts, trajectories, T_idx = extract_windows(
        np.nan_to_num(rate, nan=0.0), np.nan_to_num(
            np.asarray(eyepos, np.float64), nan=0.0),
        segments, t_count, t_hist,
    )
    if counts is None or counts.shape[0] < 2:
        return dict(undefined, n_segments=len(segments))

    real = decompose_trajectory(
        counts, trajectories, T_idx, target="full", threshold=threshold,
        weight_clip=WEIGHT_CLIP_DEFAULT,
        time_bin_weighting=TIME_BIN_WEIGHTING_DEFAULT,
        cpsth_method=CPSTH_METHOD_DEFAULT, n_boot=N_BOOT_DEFAULT, seed=42,
        min_trials_per_time_bin=min_trials_per_time_bin,
        closepair_density=CLOSEPAIR_DENSITY_DEFAULT,
    )
    # Figure 2 overrides decompose_trajectory's Crate with the uncentred
    # close-pair form; match that or the ratio is not Figure 2's 1-alpha.
    Crate, n_close, _p, _r, _pp, _ok = _uncentred_crate(
        counts, trajectories, T_idx, "full", threshold,
        Erate=real["Erate"], time_bin_weighting=TIME_BIN_WEIGHTING_DEFAULT,
        weight_clip=WEIGHT_CLIP_DEFAULT,
        closepair_density=CLOSEPAIR_DENSITY_DEFAULT,
    )
    crate = float(np.diag(Crate)[0])
    cpsth = float(np.diag(real["Cpsth"])[0])
    oma = 1.0 - cpsth / crate if crate > 0 else np.nan
    return {"one_minus_alpha": oma, "crate_diag": crate, "cpsth_diag": cpsth,
            "n_close_pairs": int(n_close), "n_samples": int(counts.shape[0]),
            "n_segments": len(segments)}


def decompose_model_session(rhat, robs, eye, valid_mask, dfs,
                            count_bins=None,
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
    count_bins : int, optional
        Figure 2 counting window for B_model/B_obs. Defaults to
        ``fig3_windows.FIG2_REPORTED_WINDOW_BINS`` (3 bins, 25 ms), which puts
        them on Figure 2's reported estimator -- window AND RMS-trajectory
        close-pair matching -- so panel E's f_FEM is numerically the quantity
        Figure 2 panel C reports. Pass ``count_bins=0`` for the legacy
        instantaneous single-bin estimator, which is biased low against Figure 2
        (median -0.071 at 25 ms, -0.031 even against its own 8.33 ms window).

    Returns
    -------
    dict of (n_cells,) arrays: A_model, B_model, B_obs.
    """
    from fig3_windows import FIG2_REPORTED_WINDOW_BINS
    if count_bins is None:
        count_bins = FIG2_REPORTED_WINDOW_BINS

    n_cells = rhat.shape[2]
    valid_mask = np.asarray(valid_mask, bool)
    A_model = np.full(n_cells, np.nan)
    B_model = np.full(n_cells, np.nan)
    B_obs = np.full(n_cells, np.nan)
    # Unclipped companions: 1 - diag(Cpsth)/diag(Crate) WITHOUT the [0,1] clip
    # the pipeline applies (covariance.py). fig2 excludes out-of-[0,1] cells
    # rather than clipping them onto the boundaries, so callers that mirror fig2
    # (e.g. the twin-replication supplement) need the raw value + an exclusion.
    B_model_uncl = np.full(n_cells, np.nan)
    B_obs_uncl = np.full(n_cells, np.nan)

    def _uncl(d):
        # pipeline_one_minus_alpha returns per-cell diagonals; single-cell here.
        crate = float(np.asarray(d["crate_diag"])[0])
        if not (crate > 0):
            return np.nan
        return 1.0 - float(np.asarray(d["cpsth_diag"])[0]) / crate

    for ni in range(n_cells):
        valid_ni = valid_mask & (dfs[:, :, ni] != 0)

        A_model[ni] = rate_variance_components(
            rhat[:, :, ni], valid=valid_ni,
            min_trials_per_phase=min_trials_per_phase,
        )["one_minus_alpha"]

        if count_bins:
            for arr, clipped, uncl in ((rhat, B_model, B_model_uncl),
                                       (robs, B_obs, B_obs_uncl)):
                d = one_minus_alpha_windowed(
                    arr[:, :, ni:ni + 1], eye, valid_ni, t_count=count_bins,
                    threshold=threshold,
                    min_trials_per_time_bin=min_trials_per_phase,
                )
                uncl[ni] = d["one_minus_alpha"]
                clipped[ni] = (np.clip(d["one_minus_alpha"], 0.0, 1.0)
                               if np.isfinite(d["one_minus_alpha"]) else np.nan)
        else:
            mdl = pipeline_one_minus_alpha(
                rhat[:, :, ni:ni + 1], eye, valid=valid_ni, threshold=threshold,
                min_trials_per_phase=min_trials_per_phase,
            )
            B_model[ni] = mdl["one_minus_alpha"][0]
            B_model_uncl[ni] = _uncl(mdl)

            obs = pipeline_one_minus_alpha(
                robs[:, :, ni:ni + 1], eye, valid=valid_ni, threshold=threshold,
                min_trials_per_phase=min_trials_per_phase,
            )
            B_obs[ni] = obs["one_minus_alpha"][0]
            B_obs_uncl[ni] = _uncl(obs)

    return {"A_model": A_model, "B_model": B_model, "B_obs": B_obs,
            "B_model_uncl": B_model_uncl, "B_obs_uncl": B_obs_uncl}


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
