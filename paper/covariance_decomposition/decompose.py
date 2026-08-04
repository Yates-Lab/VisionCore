"""Per-session empirical LOTC decomposition driver (stage 1).

Pure-numpy port of the validated methods pipeline
(``ryan/methods_eyepos_matching/pipeline.py``), specialized to the production
configuration: the eye-position-distribution-matched ``target='full'``
estimator, ``cpsth_method='mcfarland'``, pair-count time-bin weighting, the
directly-estimated close-pair density, and the uncentred close-pair
``C_rate = MM - Erate (x) Erate`` form. Each session runs one target with its
own KDE-reweighted eye-shuffle null.

Windowing semantics match the legacy ``extract_windows``: count window of
``t_count`` bins after a FIXED ``t_hist`` history (``T_HIST_MS_DEFAULT`` = 25 ms
= 3 bins at 120 Hz) that does not depend on the counting window, stride
``t_count``, ``T_idx`` = the count-window start bin. Close pairs are therefore
matched over ``t_hist + t_count`` bins of trajectory at every counting window,
so an across-window comparison varies only the counting window. ``C_total`` is the
pair-count-weighted covariance returned by ``decompose_trajectory``, centred on
the same weighted mean ``Erate`` as ``C_psth``/``C_rate`` and computed over the
same retained time bins, so the law of total covariance holds term by term and
the Fano numerator ``C_noise = C_total - C_rate`` is weighting-consistent.
(Previously this was the unweighted ``np.cov(counts, ddof=1)``, which weighted
each bin by its trial count ``n_t`` and used the unweighted sample mean.)

Output schema (per session): session, subject, rate_hz, psth_r2, neuron_mask,
qc, meta, windows[]. Each ``windows[w]`` carries window_bins/window_ms,
n_samples, n_close_pairs, Ctotal, and ``targets[target] = {Crate, Cpsth, Erate,
one_minus_alpha, Shuffled_Crates, n_shuffles_requested/kept/dropped,
pair_density_degenerate}``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from VisionCore.paths import CACHE_DIR
from VisionCore.covariance import (
    decompose_trajectory, _rms_traj_close_pairs, _geometric_median, _density_fn,
    extract_valid_segments, extract_windows,
)
from data_loading import load_cache

DT = 1 / 120
WINDOW_BINS_DEFAULT = (1, 2, 3, 6)
TARGETS_DEFAULT = ("full",)
THRESHOLD_DEFAULT = 0.05
T_HIST_MS_DEFAULT = 25.0       # fixed matching history; see decompose_session
MIN_SEG_LEN_DEFAULT = 36
MIN_TRIALS_PER_TIME_BIN_DEFAULT = 10
N_BOOT_DEFAULT = 20            # only used if cpsth_method='split_half'
N_SHUFFLES_DEFAULT = 1000
CPSTH_METHOD_DEFAULT = "mcfarland"
TIME_BIN_WEIGHTING_DEFAULT = "pair_count"
CLOSEPAIR_DENSITY_DEFAULT = "direct"
WEIGHT_CLIP_DEFAULT = 1e6

DECOMP_CACHE = CACHE_DIR / "covdecomp_empirical.pkl"


# ---------------------------------------------------------------------------
# Ctotal (legacy-compatible)
# ---------------------------------------------------------------------------

def _ctotal_unweighted(counts):
    """Sample covariance of `counts` rows with an isfinite-sum filter (legacy)."""
    ok = np.isfinite(counts.sum(axis=1))
    X = counts[ok]
    if X.shape[0] < 2:
        n_cells = counts.shape[1]
        return np.full((n_cells, n_cells), np.nan)
    return np.cov(X.T, ddof=1)


# ---------------------------------------------------------------------------
# Uncentred close-pair Crate (legacy form) + eye-shuffle null
# ---------------------------------------------------------------------------

MIN_PAIR_DENSITY_POINTS = 3


def _fit_pair_density(E):
    """KDE on the close-pair midpoint set, or ``None`` if it is degenerate.

    ``gaussian_kde`` needs a full-rank 2-D sample covariance, so at least three
    midpoints spanning both dimensions. With exactly two close pairs the two
    midpoints are always collinear, the covariance is rank 1, and the internal
    Cholesky raises ``LinAlgError``. Rather than substitute a regularized
    parametric density -- which would manufacture a plausible-looking number out
    of a two-point fit -- report the failure and let the caller drop the draw.
    """
    if len(E) < MIN_PAIR_DENSITY_POINTS:
        return None
    try:
        return _density_fn(E, "kde")
    except np.linalg.LinAlgError:
        return None


def _uncentred_crate(counts, trajectories, T_idx, target, threshold,
                     Erate, time_bin_weighting, weight_clip,
                     phat=None, rho=None, reduction="geometric_median",
                     closepair_density="direct", phat_pair=None):
    """Close-pair Crate as ``MM - Erate (x) Erate`` with the single-point-
    reduction importance weights (matches the §4.5 cell-side reference and the
    legacy uncentred estimator). Returns (Crate, n_pairs, phat, rho, phat_pair,
    pair_density_ok); representative points ``rho``, the KDE ``phat`` and (for
    direct density) the close-pair KDE ``phat_pair`` are computed on demand and
    returned for reuse. ``pair_density_ok`` is False when the close-pair midpoint
    density this target needs could not be estimated, which makes the returned
    Crate unusable -- see ``_fit_pair_density``.
    """
    n_cells = counts.shape[1]
    gi, gj, tpair, _mid = _rms_traj_close_pairs(trajectories, T_idx, threshold)
    n_pairs = len(gi)
    if n_pairs == 0:
        return np.full((n_cells, n_cells), np.nan), 0, phat, rho, phat_pair, False

    if rho is None:
        rho = (_geometric_median(trajectories) if reduction == "geometric_median"
               else trajectories.mean(axis=1))
    if phat is None:
        phat = _density_fn(rho, "kde")
    # The close-pair midpoint density is consumed only by the target='full'
    # importance weights below, so don't demand it (or drop draws over it)
    # for the other targets.
    pair_density_ok = True
    if closepair_density == "direct" and target == "full" and phat_pair is None:
        phat_pair = _fit_pair_density(0.5 * (rho[gi] + rho[gj]))
        pair_density_ok = phat_pair is not None

    if target == "full":
        rho_mid = 0.5 * (rho[gi] + rho[gj])
        if phat_pair is not None:
            pw_q = (np.clip(phat(rho_mid), 1e-12, None)
                    / np.clip(phat_pair(rho_mid), 1e-12, None))
        else:
            pw_q = 1.0 / np.clip(phat(rho_mid), 1e-12, None)
        pw_q = np.clip(pw_q, None, weight_clip * np.median(pw_q))
    else:
        pw_q = np.ones(n_pairs)

    if time_bin_weighting == "pair_count":
        pw_tt = np.ones(n_pairs)
    elif time_bin_weighting == "uniform":
        _, inv = np.unique(tpair, return_inverse=True)
        nP_t = np.bincount(inv).astype(float)
        pw_tt = 1.0 / nP_t[inv]
    elif time_bin_weighting == "trial_count":
        _, inv = np.unique(tpair, return_inverse=True)
        nP_t = np.bincount(inv).astype(float)
        nt_by_T = {int(u): int((T_idx == u).sum()) for u in np.unique(T_idx)}
        nt_per_pair = np.array([nt_by_T[int(u)] for u in tpair], dtype=float)
        pw_tt = nt_per_pair / nP_t[inv]
    else:
        raise ValueError(f"unknown time_bin_weighting: {time_bin_weighting!r}")

    pw = pw_q * pw_tt
    pw = pw / pw.sum()
    prod = (counts[gi].T * pw) @ counts[gj]
    MM = 0.5 * (prod + prod.T)
    Crate = MM - np.outer(Erate, Erate)
    return Crate, n_pairs, phat, rho, phat_pair, pair_density_ok


def _run_corrected_shuffles(counts, trajectories, T_idx, threshold, n_shuffles,
                            time_bin_weighting, seed, Erate, target, phat, rho,
                            weight_clip, closepair_density):
    """Eye-shuffle null: permute the (sample -> trajectory) row map, re-enumerate
    close pairs, recompute the same target-reweighted Crate. ``phat`` (KDE on
    the representative-point set) is permutation-invariant and reused; ``rho`` is
    permuted; ``phat_pair`` is re-fit per shuffle (passed None). ``Erate`` is
    held at the real per-target value.

    A shuffle whose surviving close-pair set is too small to support the midpoint
    density is dropped rather than estimated some other way: its Crate would be a
    second moment over one or two pairs, which is not a usable null draw under
    any density. Returns ``(crates, n_dropped)``; the permutation is drawn before
    the drop test, so dropping does not perturb the later shuffles.
    """
    rng = np.random.default_rng(seed)
    N = counts.shape[0]
    out = []
    n_dropped = 0
    for _ in range(n_shuffles):
        perm = rng.permutation(N)
        Crate_shuf, _n, _p, _r, _pp, ok = _uncentred_crate(
            counts, trajectories[perm], T_idx, target, threshold,
            Erate=Erate, time_bin_weighting=time_bin_weighting,
            weight_clip=weight_clip, phat=phat, rho=rho[perm],
            closepair_density=closepair_density, phat_pair=None,
        )
        if not ok:
            n_dropped += 1
            continue
        out.append(Crate_shuf)
    return out, n_dropped


# ---------------------------------------------------------------------------
# Per-session driver
# ---------------------------------------------------------------------------

def decompose_session(aligned,
                      windows_bins: Sequence[int] = WINDOW_BINS_DEFAULT,
                      targets: Sequence[str] = TARGETS_DEFAULT,
                      threshold: float = THRESHOLD_DEFAULT,
                      t_hist_ms: float = T_HIST_MS_DEFAULT,
                      dt: float = DT,
                      min_seg_len: int = MIN_SEG_LEN_DEFAULT,
                      time_bin_weighting: str = TIME_BIN_WEIGHTING_DEFAULT,
                      cpsth_method: str = CPSTH_METHOD_DEFAULT,
                      n_boot: int = N_BOOT_DEFAULT,
                      n_shuffles: int = N_SHUFFLES_DEFAULT,
                      min_trials_per_time_bin: int = MIN_TRIALS_PER_TIME_BIN_DEFAULT,
                      seed: int = 42,
                      closepair_density: str = CLOSEPAIR_DENSITY_DEFAULT,
                      weight_clip: float = WEIGHT_CLIP_DEFAULT,
                      verbose: bool = False):
    """Run the LOTC decomposition on one aligned-session record (all targets)."""
    robs = np.nan_to_num(np.asarray(aligned["robs"], dtype=np.float64), nan=0.0)
    eyepos = np.nan_to_num(np.asarray(aligned["eyepos"], dtype=np.float64), nan=0.0)
    valid_mask = np.asarray(aligned["valid_mask"], dtype=bool)

    segments = extract_valid_segments(valid_mask, min_len_bins=min_seg_len)
    if verbose:
        print(f"  [{aligned['session']}] {len(segments)} valid segments")
    # FIXED matching history, independent of the counting window. Previously
    # `t_hist = max(t_hist_bins, t_count)`, which made the history scale with
    # the counting window (L = 2W) and so confounded counting-window duration
    # with matching-history duration in any across-window comparison.
    t_hist_bins = int(round(t_hist_ms / (dt * 1000)))

    per_window = []
    for t_count in windows_bins:
        t_hist = t_hist_bins
        counts, trajectories, T_idx = extract_windows(
            robs, eyepos, segments, t_count, t_hist
        )
        if counts is None or counts.shape[0] < 100:
            continue

        # Ctotal is taken from decompose_trajectory (below), which computes it
        # under the SAME pair-count time-bin weighting and the SAME weighted
        # mean Erate as Cpsth/Crate, over the same retained time bins. This is
        # what makes the factorization exact term by term; the previous
        # unweighted np.cov weighted each bin by its trial count n_t and was
        # centred on the unweighted sample mean, so C_noise = C_total - C_rate
        # mixed two weightings and two means.
        Ctotal = None
        per_target = {}
        n_close_pairs = None
        phat = rho = phat_pair = None
        for tgt in targets:
            real = decompose_trajectory(
                counts, trajectories, T_idx, target=tgt,
                threshold=threshold, weight_clip=weight_clip,
                time_bin_weighting=time_bin_weighting,
                cpsth_method=cpsth_method, n_boot=n_boot, seed=seed,
                min_trials_per_time_bin=min_trials_per_time_bin,
                closepair_density=closepair_density,
            )
            if Ctotal is None:
                Ctotal = real["Ctotal"]
            # Override Crate with the uncentred close-pair form (legacy /
            # §4.5-reference consistent; see note_pipeline.md §7.2 item 6).
            Crate, n_close, phat, rho, phat_pair, real_pair_density_ok = \
                _uncentred_crate(
                    counts, trajectories, T_idx, tgt, threshold,
                    Erate=real["Erate"], time_bin_weighting=time_bin_weighting,
                    weight_clip=weight_clip, phat=phat, rho=rho,
                    closepair_density=closepair_density, phat_pair=phat_pair,
                )
            if n_close_pairs is None:
                n_close_pairs = n_close

            with np.errstate(divide="ignore", invalid="ignore"):
                alpha = np.clip(np.diag(real["Cpsth"]) / np.diag(Crate), 0.0, 1.0)
            one_minus_alpha = 1.0 - alpha
            one_minus_alpha[~(np.diag(Crate) > 0)] = np.nan

            shuffled_crates, n_shuffles_dropped = [], 0
            if n_shuffles > 0:
                shuffled_crates, n_shuffles_dropped = _run_corrected_shuffles(
                    counts, trajectories, T_idx, threshold, n_shuffles,
                    time_bin_weighting, seed=seed, Erate=real["Erate"],
                    target=tgt, phat=phat, rho=rho, weight_clip=weight_clip,
                    closepair_density=closepair_density,
                )

            per_target[tgt] = {
                "Crate": Crate,
                "Cpsth": real["Cpsth"],
                "Erate": real["Erate"],
                "one_minus_alpha": one_minus_alpha,
                "Shuffled_Crates": shuffled_crates,
                # Null-draw bookkeeping: shuffles whose close-pair set could not
                # support the midpoint density (see _fit_pair_density).
                "n_shuffles_requested": int(n_shuffles),
                "n_shuffles_kept": len(shuffled_crates),
                "n_shuffles_dropped": int(n_shuffles_dropped),
                # Never observed on real (unshuffled) data; recorded so it can
                # not become silent if it ever does occur.
                "pair_density_degenerate": bool(not real_pair_density_ok),
            }

        per_window.append({
            "window_bins": int(t_count),
            "window_ms": float(t_count * dt * 1000),
            "n_samples": int(counts.shape[0]),
            "n_close_pairs": int(n_close_pairs),
            "Ctotal": Ctotal,
            "targets": per_target,
        })

    return {
        "session": aligned["session"],
        "subject": aligned["subject"],
        "rate_hz": aligned["rate_hz"],
        "psth_r2": aligned["psth_r2"],
        "neuron_mask": aligned["neuron_mask"],
        "qc": {"contam_rate": aligned["contam_rate"]},
        "meta": {
            "n_trials_total": aligned["n_trials_total"],
            "n_trials_good": aligned["n_trials_good"],
            "n_neurons_total": aligned["n_neurons_total"],
            "n_neurons_used": aligned["n_neurons_used"],
        },
        "windows": per_window,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _decompose_one(aligned, **kw):
    """Pickle-friendly worker: pin BLAS to one thread, then decompose."""
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            return decompose_session(aligned, **kw)
    except ImportError:
        return decompose_session(aligned, **kw)


def compute_decomposition(refresh=False, n_jobs=-1, **kw):
    """Run the per-session decomposition over all aligned sessions and cache it."""
    if DECOMP_CACHE.exists() and not refresh:
        print(f"Loading cached decomposition from {DECOMP_CACHE}")
        import dill
        with open(DECOMP_CACHE, "rb") as f:
            return dill.load(f)

    import dill
    from joblib import Parallel, delayed

    aligned_sessions = load_cache()
    print(f"Decomposing {len(aligned_sessions)} sessions "
          f"(targets={kw.get('targets', TARGETS_DEFAULT)}, "
          f"n_shuffles={kw.get('n_shuffles', N_SHUFFLES_DEFAULT)})")
    session_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_decompose_one)(a, **kw) for a in aligned_sessions
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DECOMP_CACHE, "wb") as f:
        dill.dump(session_results, f)
    print(f"\nCached {len(session_results)} session decompositions to {DECOMP_CACHE}")
    return session_results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute per-session LOTC decomposition.")
    ap.add_argument("--refresh", action="store_true", help="Force recompute.")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    compute_decomposition(refresh=args.refresh, n_jobs=args.n_jobs)
