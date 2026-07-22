"""Data layer for the twin saccade-modulation supplement.

Question: the twin's extraretinal contribution --- ``residual = twin(intact) -
twin(zeroed)``, where ``zeroed`` removes only the behavior (eye-velocity /
eye-position) input --- looked sparse and biphasic in fig4 panel H. Is it a
stereotyped, saccade-locked modulation, and is it consistent across units?

This module reuses the existing conditions cache
``supp_twin_fig2frame_conditions.pkl`` (per session: robs / rhat[intact,zeroed,
stabilized] / dfs / eyepos, all per trial x bin, affine-rescaled, fig2's 0.5 deg
frame) rather than re-running the twin. The one thing that cache does NOT carry
is the mapping from session time to (trial, bin), which we need to place the
session's detected saccades (``saccades/saccades.json``) onto the cached trials.

`build_saccade_alignment` reconstructs that mapping by re-loading each session's
fixRSVP dset (CPU only, no model forward) and replaying the exact trial iteration
the inference used, then **gates on a bit-wise match** between the reconstructed
per-trial eyepos and the cache's ``eyepos_used`` before trusting the alignment.
Saccade ``start_time``s are mapped to dataset samples with
``get_inds_from_times`` and reduced to (cached-trial-position, bin) pairs.

`compute_sta_bundle` then builds the saccade-triggered averages (residual /
observed / intact / zeroed / eye-speed) over a -100/+200 ms window, per neuron
and pooled, plus the example-neuron raster payload.

Usage:
    uv run python paper/supp_twin_saccade_modulation/_supp_saccade_data.py [--force]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "fig3"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "covariance_decomposition"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_model_replication"))

from _fig3_data import (  # noqa: E402
    CHECKPOINT_PATH, VALID_TIME_BINS, MIN_FIX_DUR, subject_from_session,
)
from data_loading import FIXATION_RADIUS  # noqa: E402  (0.5, fig2 frame)
from _supp_inference import SUPP_INFERENCE_CONDITIONS_CACHE  # noqa: E402

# --- constants -------------------------------------------------------------
DT = 1 / 120.0                       # s per bin
STA_PRE_MS = 100.0                   # window before saccade onset
STA_POST_MS = 200.0                  # window after saccade onset
STA_PRE = int(round(STA_PRE_MS / 1000.0 / DT))    # 12 bins
STA_POST = int(round(STA_POST_MS / 1000.0 / DT))  # 24 bins
STA_LAGS = np.arange(-STA_PRE, STA_POST + 1)       # bins, len 37
STA_T = STA_LAGS * DT * 1000.0                     # ms axis
BASELINE_MASK = STA_T < -25.0        # pre-saccade window for baseline subtraction

# Example neuron for the raster row (fig4 panel B/H pin).
EXAMPLE_SESSION = "Allen_2022-04-08"
EXAMPLE_NEURON_ID = 62

# Reliability: split-half correlation of the observed PSTH.
RELIABLE_THRESHOLD = 0.5

ALIGNMENT_CACHE = CACHE_DIR / "supp_saccade_alignment.pkl"
STA_CACHE = CACHE_DIR / "supp_saccade_sta.pkl"


# --- cache loaders ---------------------------------------------------------
def load_conditions_cache():
    """Load the reused twin conditions cache (list of per-session dicts)."""
    if not SUPP_INFERENCE_CONDITIONS_CACHE.exists():
        raise FileNotFoundError(
            f"Conditions cache not found: {SUPP_INFERENCE_CONDITIONS_CACHE}. "
            "Run paper/supp_model_replication/_supp_inference.py first."
        )
    with open(SUPP_INFERENCE_CONDITIONS_CACHE, "rb") as f:
        return dill.load(f)


# --- saccade -> (trial, bin) alignment -------------------------------------
def _reconstruct_trial_order(dset, eyepos_used):
    """Replay the inference's per-trial iteration on `dset` and return
    (orig_trial_ids, eyepos_recon) for the cached (good) trials.

    orig_trial_ids[i] is the raw fixRSVP trial id that maps to cached trial
    position i. eyepos_recon is compared against `eyepos_used` by the caller as
    an alignment gate.
    """
    trial_inds = np.asarray(dset.covariates["trial_inds"]).ravel()
    psth_inds = np.asarray(dset.covariates["psth_inds"]).ravel()
    eyepos_flat = np.asarray(dset["eyepos"])

    fixation = np.hypot(eyepos_flat[:, 0], eyepos_flat[:, 1]) < FIXATION_RADIUS
    trials = np.unique(trial_inds)
    NT = len(trials)
    T = int(psth_inds.max()) + 1

    eyepos = np.full((NT, T, 2), np.nan)
    fix_dur = np.full(NT, np.nan)
    for itrial in range(NT):
        ix = (trial_inds == trials[itrial]) & fixation
        if not np.any(ix):
            continue
        t_inds = psth_inds[ix].astype(int)
        fix_dur[itrial] = len(t_inds)
        eyepos[itrial, t_inds] = eyepos_flat[ix]

    good = fix_dur > MIN_FIX_DUR
    iix = np.arange(min(VALID_TIME_BINS, T))
    return trials[good], eyepos[good][:, iix], trial_inds, psth_inds


def build_saccade_alignment(force=False):
    """Per session: map detected saccades to (cached-trial-position, bin).

    Returns {session: {"sacc_trial": int[], "sacc_bin": int[], "n_raw": int,
    "eyepos_gate": float}}. Cached to ALIGNMENT_CACHE. Loads the model on CPU and
    each fixRSVP dset once; runs no forward pass.
    """
    if ALIGNMENT_CACHE.exists() and not force:
        print(f"Loading saccade alignment from {ALIGNMENT_CACHE}")
        with open(ALIGNMENT_CACHE, "rb") as f:
            return dill.load(f)

    import torch
    from DataYatesV1.utils.io import YatesV1Session
    from eval.eval_stack_multidataset import load_model
    from eval.eval_stack_utils import load_single_dataset

    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))

    cond = load_conditions_cache()
    cache_by_session = {r["session"]: r for r in cond}

    print(f"Loading model (CPU, no forward) from: {CHECKPOINT_PATH}")
    model, _ = load_model(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    name_to_idx = {n: i for i, n in enumerate(model.names)}

    out = {}
    for session_name, rec in cache_by_session.items():
        if session_name not in name_to_idx:
            print(f"  {session_name}: not in model.names, skipping")
            continue
        dataset_idx = name_to_idx[session_name]
        print(f"\n--- {session_name} [{dataset_idx}] ---")
        try:
            train_data, val_data, _ = load_single_dataset(model, dataset_idx)
            fixrsvp_inds = torch.cat([
                train_data.get_dataset_inds("fixrsvp"),
                val_data.get_dataset_inds("fixrsvp"),
            ], dim=0)
        except Exception as e:  # noqa: BLE001
            print(f"  Skipping: {e}")
            continue
        dset_idx_local = fixrsvp_inds[:, 0].unique().item()
        dset = train_data.dsets[dset_idx_local]

        orig_ids, eyepos_recon, trial_inds, psth_inds = _reconstruct_trial_order(
            dset, rec["eyepos_used"])

        # Alignment gate: reconstructed eyepos must match the cache bit-wise.
        cache_eye = rec["eyepos_used"]
        if eyepos_recon.shape != cache_eye.shape:
            print(f"  GATE FAIL: shape {eyepos_recon.shape} != {cache_eye.shape}")
            continue
        both_nan = np.isnan(eyepos_recon) & np.isnan(cache_eye)
        diff = np.abs(np.nan_to_num(eyepos_recon) - np.nan_to_num(cache_eye))
        diff[both_nan] = 0.0
        gate = float(np.nanmax(diff))
        if gate > 1e-9:
            print(f"  GATE FAIL: max|eyepos_recon - cache| = {gate}")
            continue

        id_to_pos = {int(t): i for i, t in enumerate(orig_ids)}

        # Load + map saccades.
        sess = YatesV1Session(session_name)
        sacc_path = sess.sess_dir / "saccades" / "saccades.json"
        if not sacc_path.exists():
            print(f"  No saccades.json at {sacc_path}, skipping")
            continue
        saccades = json.load(open(sacc_path))
        sacc_times = torch.tensor([s["start_time"] for s in saccades],
                                  dtype=torch.float64)
        inds = train_data.get_inds_from_times(sacc_times)
        inds = inds[inds[:, 0] == dset_idx_local]
        time_inds = inds[:, 1].cpu().numpy().astype(int)

        sacc_trial, sacc_bin = [], []
        for ti in time_inds:
            otrial = int(trial_inds[ti])
            pbin = int(psth_inds[ti])
            pos = id_to_pos.get(otrial)
            if pos is None or not (0 <= pbin < VALID_TIME_BINS):
                continue
            sacc_trial.append(pos)
            sacc_bin.append(pbin)

        out[session_name] = {
            "sacc_trial": np.asarray(sacc_trial, dtype=int),
            "sacc_bin": np.asarray(sacc_bin, dtype=int),
            "n_raw": len(saccades),
            "n_mapped": len(sacc_trial),
            "eyepos_gate": gate,
        }
        print(f"  gate OK ({gate:.1e}); {len(saccades)} saccades, "
              f"{len(sacc_trial)} mapped into cached fixation trials")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(ALIGNMENT_CACHE, "wb") as f:
        dill.dump(out, f)
    print(f"\nCached saccade alignment for {len(out)} sessions to {ALIGNMENT_CACHE}")
    return out


# --- saccade-triggered averages --------------------------------------------
def _sta_over_saccades(arr_tb, sacc_trial, sacc_bin):
    """Saccade-triggered average of a (trials, bins) array.

    Returns (sta[len(STA_LAGS)], n_valid[len(STA_LAGS)]). NaNs (invalid/masked
    bins, out-of-range lags) are ignored per lag.
    """
    n_trials, n_bins = arr_tb.shape
    acc = np.zeros(len(STA_LAGS))
    cnt = np.zeros(len(STA_LAGS))
    for tr, b0 in zip(sacc_trial, sacc_bin):
        for li, lag in enumerate(STA_LAGS):
            b = b0 + lag
            if 0 <= b < n_bins:
                v = arr_tb[tr, b]
                if np.isfinite(v):
                    acc[li] += v
                    cnt[li] += 1
    with np.errstate(invalid="ignore"):
        sta = acc / cnt
    return sta, cnt


def _eye_speed(eyepos_tb):
    """Eye speed (deg/s) per (trial, bin) from per-bin eyepos, central diff."""
    n_trials, n_bins, _ = eyepos_tb.shape
    vel = np.full((n_trials, n_bins), np.nan)
    d = (eyepos_tb[:, 2:, :] - eyepos_tb[:, :-2, :]) / (2 * DT)
    vel[:, 1:-1] = np.hypot(d[:, :, 0], d[:, :, 1])
    return vel


def _split_half_reliability(robs_tb):
    """Split-half correlation of the observed PSTH for one neuron.
    robs_tb: (trials, bins) counts with NaN where masked."""
    n_trials = robs_tb.shape[0]
    if n_trials < 4:
        return np.nan
    idx = np.arange(n_trials)
    a = robs_tb[idx % 2 == 0]
    b = robs_tb[idx % 2 == 1]
    psth_a = np.nanmean(a, axis=0)
    psth_b = np.nanmean(b, axis=0)
    m = np.isfinite(psth_a) & np.isfinite(psth_b)
    if m.sum() < 3 or np.nanstd(psth_a[m]) < 1e-9 or np.nanstd(psth_b[m]) < 1e-9:
        return np.nan
    return float(np.corrcoef(psth_a[m], psth_b[m])[0, 1])


def compute_sta_bundle(force=False):
    """Build per-neuron and pooled saccade-triggered averages + example payload.

    Returns a dict cached to STA_CACHE.
    """
    if STA_CACHE.exists() and not force:
        print(f"Loading STA bundle from {STA_CACHE}")
        with open(STA_CACHE, "rb") as f:
            return dill.load(f)

    cond = load_conditions_cache()
    align = build_saccade_alignment()

    per_neuron = {k: [] for k in
                  ["residual", "observed", "intact", "zeroed"]}
    n_sacc_neuron = []
    reliability = []
    subjects = []
    sessions = []
    speed_sta_sessions = []
    example = None

    for rec in cond:
        session = rec["session"]
        if session not in align:
            continue
        a = align[session]
        st, sb = a["sacc_trial"], a["sacc_bin"]
        if len(st) == 0:
            continue

        robs = rec["robs_used"] / DT               # (trials, bins, neurons) rate
        intact = rec["rhat_used"]["intact"] / DT
        zeroed = rec["rhat_used"]["zeroed"] / DT
        resid = intact - zeroed
        dfs = rec["dfs_used"]
        robs = np.where(dfs > 0, robs, np.nan)
        n_trials, n_bins, n_neurons = robs.shape

        speed = _eye_speed(rec["eyepos_used"])
        speed_sta, _ = _sta_over_saccades(speed, st, sb)
        speed_sta_sessions.append(speed_sta)

        arrs = {"residual": resid, "observed": robs,
                "intact": intact, "zeroed": zeroed}
        for ni in range(n_neurons):
            for key, arr in arrs.items():
                sta, cnt = _sta_over_saccades(arr[:, :, ni], st, sb)
                per_neuron[key].append(sta)
            n_sacc_neuron.append(int(np.nanmin(cnt)))
            reliability.append(_split_half_reliability(robs[:, :, ni]))
            subjects.append(rec["subject"])
            sessions.append(session)

        if session == EXAMPLE_SESSION:
            loc = np.where(rec["neuron_mask"] == EXAMPLE_NEURON_ID)[0]
            if len(loc):
                ni = int(loc[0])
                example = _build_example(rec, ni, st, sb, speed)

    bundle = {
        "sta": {k: np.asarray(v) for k, v in per_neuron.items()},
        "n_sacc": np.asarray(n_sacc_neuron),
        "reliability": np.asarray(reliability),
        "subjects": np.asarray(subjects),
        "sessions": np.asarray(sessions),
        "speed_sta_sessions": np.asarray(speed_sta_sessions),
        "lags_ms": STA_T,
        "baseline_mask": BASELINE_MASK,
        "example": example,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STA_CACHE, "wb") as f:
        dill.dump(bundle, f)
    print(f"Cached STA bundle to {STA_CACHE}")
    return bundle


def _build_example(rec, ni, sacc_trial, sacc_bin, speed):
    """Example-neuron raster payload: seriated trials, all five aligned strips,
    per-trial saccade markers, and the neuron's STA set."""
    from _fig3_helpers import order_single_neuron_by_seriation, N_BINS_B

    robs = rec["robs_used"][:, :, ni] / DT
    intact = rec["rhat_used"]["intact"][:, :, ni] / DT
    zeroed = rec["rhat_used"]["zeroed"][:, :, ni] / DT
    dfs = rec["dfs_used"][:, :, ni]
    resid = intact - zeroed

    _, _, order, first_bin = order_single_neuron_by_seriation(
        np.where(dfs > 0, robs, np.nan), intact, dfs)
    end = first_bin + N_BINS_B

    def strip(arr, mask=True):
        s = arr[:, first_bin:end].astype(float).copy()
        if mask:
            s[dfs[:, first_bin:end] == 0] = np.nan
        return s[order]

    # map saccade (trial, bin) -> raster row (order position) and windowed bin
    pos_of_trial = {int(t): i for i, t in enumerate(order)}
    ex_sacc = []
    for tr, b in zip(sacc_trial, sacc_bin):
        row = pos_of_trial.get(int(tr))
        if row is not None and first_bin <= b < end:
            ex_sacc.append((row, b - first_bin))

    # STA set for this neuron
    sta = {k: _sta_over_saccades(a, sacc_trial, sacc_bin)[0]
           for k, a in {"residual": resid, "observed":
                        np.where(dfs > 0, robs, np.nan),
                        "intact": intact, "zeroed": zeroed}.items()}
    sta["speed"], _ = _sta_over_saccades(speed, sacc_trial, sacc_bin)

    return {
        "session": rec["session"], "neuron_id": EXAMPLE_NEURON_ID,
        "observed": strip(robs), "intact": strip(intact),
        "zeroed": strip(zeroed), "residual": strip(resid, mask=False),
        "speed": strip(speed, mask=False),
        "sacc_markers": np.asarray(ex_sacc),
        "window_s": N_BINS_B * DT,
        "sta": sta,
        "n_sacc": len(sacc_trial),
    }


def baseline_subtract(sta):
    """Subtract the pre-saccade baseline (mean over BASELINE_MASK lags)."""
    base = np.nanmean(sta[..., BASELINE_MASK], axis=-1, keepdims=True)
    return sta - base


def print_stats(bundle=None):
    if bundle is None:
        bundle = compute_sta_bundle()
    rel = bundle["reliability"]
    good = np.isfinite(rel) & (rel > RELIABLE_THRESHOLD)
    resid = baseline_subtract(bundle["sta"]["residual"])
    t = bundle["lags_ms"]
    post = (t >= 0) & (t <= 150)

    print("\n=== Saccade-modulation summary ===")
    print(f"neurons total: {len(rel)}  reliable (split-half>{RELIABLE_THRESHOLD}): "
          f"{int(good.sum())}")
    print(f"saccades/session mapped: median "
          f"{np.median([a['n_mapped'] for a in build_saccade_alignment().values()]):.0f}")
    peak = np.nanmax(np.abs(resid[:, post]), axis=1)
    print(f"|residual STA| peak (0-150ms), reliable units: "
          f"median {np.nanmedian(peak[good]):.3f} sp/s, "
          f"90th pct {np.nanpercentile(peak[good], 90):.3f}")
    # population mean trace peak vs pre-saccade noise
    pop = np.nanmean(resid[good], axis=0)
    pop_sem = np.nanstd(resid[good], axis=0) / np.sqrt(good.sum())
    imax = np.nanargmax(np.abs(pop))
    print(f"population-mean residual STA extremum: {pop[imax]:+.4f} sp/s at "
          f"{t[imax]:+.0f} ms (SEM {pop_sem[imax]:.4f})")
    print(f"eye-speed STA peaks at lag "
          f"{t[np.nanargmax(np.nanmean(bundle['speed_sta_sessions'], axis=0))]:+.0f} ms "
          "(should be ~0)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-align", action="store_true")
    args = ap.parse_args()
    if args.force_align:
        build_saccade_alignment(force=True)
    bundle = compute_sta_bundle(force=args.force)
    print_stats(bundle)
