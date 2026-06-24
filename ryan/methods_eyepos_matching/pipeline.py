"""Methods-side per-session LOTC decomposition driver.

Replaces ``legacy.compute_fig2_data._compute_one_session`` with a pure-numpy
implementation built on ``estimators.decompose_trajectory``. Same windowing
semantics as ``legacy.covariance.extract_windows`` (count window of t_count,
trajectory of max(t_hist_bins, t_count) + t_count = 2*t_count for t_count>=1
at the default t_hist_ms=10), same segmentation, same Ctotal definition.

For each (session, window) we run all three targets (naive / full / central) in
one pass. Shuffle nulls are run for target='naive' only -- that is enough to
reproduce the legacy alpha/Fano/NC empirical-p values for the equivalence
check; full/central nulls are deferred (writeup §7.5).

Shuffle semantic: permute the (sample -> trajectory) row map, re-enumerate
close pairs, recompute the close-pair second moment. ``Erate`` (under
target='naive', sw=1) is invariant under row permutation, so the shuffle
output is Crate_shuf = MM_shuf - Erate Erate^T -- close-pair-only, matching
legacy `Shuffled_Intercepts`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from estimators import (                                          # noqa: E402
    decompose_trajectory, _rms_traj_close_pairs, _geometric_median, _density_fn,
)
from legacy.covariance import extract_valid_segments              # noqa: E402

DT = 1 / 120
WINDOW_BINS_DEFAULT = (1, 2, 3, 6)
TARGETS_DEFAULT = ("naive", "full", "central")
THRESHOLD_DEFAULT = 0.05
T_HIST_MS_DEFAULT = 10.0
MIN_SEG_LEN_DEFAULT = 36
MIN_TRIALS_PER_TIME_BIN_DEFAULT = 10
N_BOOT_DEFAULT = 20
N_SHUFFLES_DEFAULT = 100


# ---------------------------------------------------------------------------
# Numpy port of legacy.covariance.extract_windows
# ---------------------------------------------------------------------------

def _extract_windows_numpy(robs, eyepos, segments, t_count, t_hist):
    """Stride windows within each segment; return (counts, trajectories, T_idx).

    Mirrors ``legacy.covariance.extract_windows`` exactly:
      * count window covers [t0+t_hist : t0+total_len) (the t_count bins after
        the history),
      * trajectory covers [t0 : t0+total_len),
      * stride = t_count within each segment,
      * T_idx = t0 + t_hist (the start bin of the count window).

    total_len = t_hist + t_count.
    """
    total_len = t_hist + t_count
    trial_indices, time_indices = [], []

    for (tr, start, stop) in segments:
        if (stop - start) < total_len:
            continue
        t_starts = np.arange(start, stop - total_len + 1, t_count)
        trial_indices.extend([tr] * len(t_starts))
        time_indices.extend(t_starts.tolist())

    if len(trial_indices) == 0:
        return None, None, None

    idx_tr = np.asarray(trial_indices)
    idx_t0 = np.asarray(time_indices)

    # gather (N, total_len, 2) eye trajectory
    offsets = np.arange(total_len)[None, :]
    gather_t = idx_t0[:, None] + offsets                       # (N, total_len)
    gather_tr = np.broadcast_to(idx_tr[:, None], gather_t.shape)
    trajectories = eyepos[gather_tr, gather_t, :]              # (N, total_len, 2)

    # gather (N, n_cells) summed spike counts over the t_count window
    spike_offsets = np.arange(t_hist, total_len)[None, :]
    gather_t_spk = idx_t0[:, None] + spike_offsets             # (N, t_count)
    gather_tr_spk = np.broadcast_to(idx_tr[:, None], gather_t_spk.shape)
    S_raw = robs[gather_tr_spk, gather_t_spk, :]               # (N, t_count, NC)
    counts = S_raw.sum(axis=1)                                 # (N, NC)

    T_idx = idx_t0 + t_hist
    return counts, trajectories, T_idx


# ---------------------------------------------------------------------------
# Ctotal: match legacy torch.cov(correction=1) on isfinite rows
# ---------------------------------------------------------------------------

def _ctotal_unweighted(counts):
    """Sample covariance of `counts` rows with isfinite-sum filter.

    Matches ``legacy.run_covariance_decomposition``'s Ctotal computation:
        ix = np.isfinite(SpikeCounts.sum(1))
        Ctotal = torch.cov(SpikeCounts[ix].T, correction=1)
    """
    ok = np.isfinite(counts.sum(axis=1))
    X = counts[ok]
    if X.shape[0] < 2:
        n_cells = counts.shape[1]
        return np.full((n_cells, n_cells), np.nan)
    return np.cov(X.T, ddof=1)


# ---------------------------------------------------------------------------
# Naive close-pair shuffle (Crate only)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Legacy-compatible Crate override
# ---------------------------------------------------------------------------
#
# ``estimators.decompose_trajectory`` computes its close-pair Crate from
# centred S (``S_c = S - Erate``) for numerical precision in the §4.4
# synthetic validation. The legacy ``estimate_rate_covariance`` with
# ``intercept_mode='below_threshold'`` computes Crate from raw S as
# ``MM - Erate ⊗ Erate``, which differs by cross terms
# ``Erate (E_close[S]-Erate)^T + (E_close[S]-Erate) Erate^T`` whenever the
# close-pair-set average of S is biased away from Erate (true for real
# eccentric cells). The two are different estimators of the close-pair
# rate covariance.
#
# For the equivalence comparison and for the methods pipeline's main
# Stage-1 output (consumed by Fig. 7 + Fig. 8), we override Crate with the
# legacy uncentred form. The §4.4 centred estimator remains the
# theoretically-validated default in ``estimators.decompose_trajectory``
# and is what fig_trajectory.py uses. The importance weights here match the
# §4.4 single-point reduction: each trajectory is reduced to its geometric
# median rho_i and one KDE p_hat is fit on {rho_i}; Direction 1 weights close
# pairs by 1/p_hat(rho_mid) at the midpoint of the two representative points
# (p^2 implied by p_hat, exactly as in §4.2 -- no separate close-pair KDE).
# ---------------------------------------------------------------------------

def _legacy_compat_crate(counts, trajectories, T_idx, target, threshold,
                         Erate, time_bin_weighting, weight_clip,
                         phat=None, rho=None, reduction="geometric_median",
                         closepair_density="direct", phat_pair=None):
    """Recompute the close-pair Crate as ``MM - Erate ⊗ Erate`` (legacy form).

    Re-enumerates close pairs and applies the §4.4 single-point-reduction
    importance weights. Returns (Crate, n_pairs, phat, rho, phat_pair) where the
    representative points ``rho``, the KDE ``phat`` and (for
    ``closepair_density='direct'``) the close-pair KDE ``phat_pair`` are computed
    on-demand if not supplied (so callers can share them across targets for
    free). ``closepair_density='direct'`` only affects the Direction-1 (full)
    close-pair weight; central's close-pair weight is 1 regardless.
    """
    n_cells = counts.shape[1]
    gi, gj, tpair, _mid_traj = _rms_traj_close_pairs(
        trajectories, T_idx, threshold
    )
    n_pairs = len(gi)
    if n_pairs == 0:
        return np.full((n_cells, n_cells), np.nan), 0, phat, rho, phat_pair

    if rho is None:
        rho = (_geometric_median(trajectories) if reduction == "geometric_median"
               else trajectories.mean(axis=1))
    if phat is None:
        phat = _density_fn(rho, "kde")
    if closepair_density == "direct" and phat_pair is None:
        phat_pair = _density_fn(0.5 * (rho[gi] + rho[gj]), "kde")

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
        unique_t = np.unique(T_idx)
        nt_by_T = {int(u): int((T_idx == u).sum()) for u in unique_t}
        nt_per_pair = np.array([nt_by_T[int(u)] for u in tpair], dtype=float)
        pw_tt = nt_per_pair / nP_t[inv]
    else:
        raise ValueError(f"unknown time_bin_weighting: {time_bin_weighting!r}")

    pw = pw_q * pw_tt
    pw = pw / pw.sum()
    prod = (counts[gi].T * pw) @ counts[gj]
    MM = 0.5 * (prod + prod.T)
    Crate = MM - np.outer(Erate, Erate)
    return Crate, n_pairs, phat, rho, phat_pair


def _enumerate_close_pairs(trajectories, T_idx, threshold):
    """Same close-pair logic as ``estimators._rms_traj_close_pairs`` but only
    returns the (gi, gj, tpair) tuple; no midpoint trajectory (the naive
    shuffle does not need it).
    """
    N, T, _ = trajectories.shape
    inv_sqrt_T = 1.0 / float(np.sqrt(float(T)))
    flat = trajectories.reshape(N, -1)
    gi_acc, gj_acc, t_acc = [], [], []
    for t in np.unique(T_idx):
        ix = np.where(T_idx == t)[0]
        if len(ix) < 2:
            continue
        F = flat[ix]
        D = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=-1) * inv_sqrt_T
        i, j = np.triu_indices(len(ix), k=1)
        d = D[i, j]
        close = d < threshold
        if not close.any():
            continue
        gi = ix[i[close]]
        gj = ix[j[close]]
        gi_acc.append(gi)
        gj_acc.append(gj)
        t_acc.append(np.full(int(close.sum()), t))
    if not gi_acc:
        return (np.zeros(0, int), np.zeros(0, int), np.zeros(0, int))
    return (np.concatenate(gi_acc),
            np.concatenate(gj_acc),
            np.concatenate(t_acc))


def _naive_close_pair_crate(counts, gi, gj, tpair, T_idx, Erate,
                            time_bin_weighting):
    """Naive close-pair Crate = MM - Erate Erate^T."""
    n_cells = counts.shape[1]
    if len(gi) == 0:
        return np.full((n_cells, n_cells), np.nan)

    if time_bin_weighting == "pair_count":
        pw_t = np.ones(len(gi))
    elif time_bin_weighting == "uniform":
        _, inv = np.unique(tpair, return_inverse=True)
        nP_t = np.bincount(inv).astype(float)
        pw_t = 1.0 / nP_t[inv]
    elif time_bin_weighting == "trial_count":
        _, inv = np.unique(tpair, return_inverse=True)
        nP_t = np.bincount(inv).astype(float)
        unique_t = np.unique(T_idx)
        nt_by_T = {int(u): int((T_idx == u).sum()) for u in unique_t}
        nt_per_pair = np.array([nt_by_T[int(u)] for u in tpair], dtype=float)
        pw_t = nt_per_pair / nP_t[inv]
    else:
        raise ValueError(f"unknown time_bin_weighting: {time_bin_weighting!r}")

    pw = pw_t / pw_t.sum()
    prod = (counts[gi].T * pw) @ counts[gj]
    MM = 0.5 * (prod + prod.T)
    return MM - np.outer(Erate, Erate)


def _run_naive_shuffles(counts, trajectories, T_idx, threshold, n_shuffles,
                        time_bin_weighting, seed, Erate):
    """Permute (sample -> trajectory) rows, re-enumerate close pairs, recompute
    Crate. Returns list of (n_cells, n_cells) ndarrays.

    Note on Erate invariance: under target='naive' the per-sample weight is 1
    and Erate is the per-time-bin mean of counts (subject to the across-bin
    pair-count weighting). Both are functions of (counts, T_idx) only -- not
    of trajectories -- so Erate is invariant under the trajectory shuffle and
    we can use the real-fit Erate without recomputing it.
    """
    rng = np.random.default_rng(seed)
    N = counts.shape[0]
    out = []
    for _ in range(n_shuffles):
        perm = rng.permutation(N)
        traj_shuf = trajectories[perm]
        gi, gj, tpair = _enumerate_close_pairs(traj_shuf, T_idx, threshold)
        out.append(_naive_close_pair_crate(
            counts, gi, gj, tpair, T_idx, Erate, time_bin_weighting
        ))
    return out


# ---------------------------------------------------------------------------
# Per-session driver
# ---------------------------------------------------------------------------

def _format_window_record(t_count, dt, counts_shape, per_target,
                          Ctotal, n_close_pairs):
    """Compose one window's result block; schema chosen to feed straight into
    metrics.py (analog of legacy ``mats[w]`` + ``results[w]``)."""
    return {
        "window_bins": int(t_count),
        "window_ms": float(t_count * dt * 1000),
        "n_samples": int(counts_shape[0]),
        "n_close_pairs": int(n_close_pairs),
        "Ctotal": Ctotal,
        "targets": per_target,
    }


def decompose_session(aligned,
                      windows_bins: Sequence[int] = WINDOW_BINS_DEFAULT,
                      targets: Sequence[str] = TARGETS_DEFAULT,
                      threshold: float = THRESHOLD_DEFAULT,
                      t_hist_ms: float = T_HIST_MS_DEFAULT,
                      dt: float = DT,
                      min_seg_len: int = MIN_SEG_LEN_DEFAULT,
                      time_bin_weighting: str = "pair_count",
                      cpsth_method: str = "split_half",
                      n_boot: int = N_BOOT_DEFAULT,
                      n_shuffles: int = N_SHUFFLES_DEFAULT,
                      min_trials_per_time_bin: int = MIN_TRIALS_PER_TIME_BIN_DEFAULT,
                      seed: int = 42,
                      closepair_density: str = "direct",
                      verbose: bool = False):
    """Run the methods LOTC decomposition on one aligned-session record.

    Parameters mirror ``legacy.compute_fig2_data`` constants where applicable.
    Returns a dict with the per-session metadata and a ``windows`` list, one
    entry per ``windows_bins``. Each window entry has its own per-target dict
    of ``{Crate, Cpsth, Erate, one_minus_alpha, n_close_pairs, Shuffled_Crates}``;
    ``Ctotal`` lives at the window level since it does not depend on target.
    """
    robs = np.asarray(aligned["robs"], dtype=np.float64)
    eyepos = np.asarray(aligned["eyepos"], dtype=np.float64)
    valid_mask = np.asarray(aligned["valid_mask"], dtype=bool)

    # legacy sanitizes via nan_to_num inside run_covariance_decomposition --
    # match it so close-pair pools and Erate are comparable.
    robs = np.nan_to_num(robs, nan=0.0)
    eyepos = np.nan_to_num(eyepos, nan=0.0)

    segments = extract_valid_segments(valid_mask, min_len_bins=min_seg_len)
    if verbose:
        print(f"  [{aligned['session']}] {len(segments)} valid segments")
    t_hist_bins = int(t_hist_ms / (dt * 1000))

    per_window = []
    for t_count in windows_bins:
        t_hist = max(t_hist_bins, t_count)
        counts, trajectories, T_idx = _extract_windows_numpy(
            robs, eyepos, segments, t_count, t_hist
        )
        if counts is None or counts.shape[0] < 100:
            if verbose:
                print(f"  [{aligned['session']}] window={t_count} skipped "
                      f"(n_samples={None if counts is None else counts.shape[0]})")
            continue

        Ctotal = _ctotal_unweighted(counts)

        per_target = {}
        n_close_pairs = None
        # representative points + KDE(s) are eye-only and target-independent;
        # compute once per window and share across targets
        phat = rho = phat_pair = None
        for tgt in targets:
            real = decompose_trajectory(
                counts, trajectories, T_idx, target=tgt,
                threshold=threshold, weight_clip=1e6,
                time_bin_weighting=time_bin_weighting,
                cpsth_method=cpsth_method, n_boot=n_boot,
                seed=seed,
                min_trials_per_time_bin=min_trials_per_time_bin,
                closepair_density=closepair_density,
            )

            # Override Crate with the uncentred close-pair form
            # `MM - Erate ⊗ Erate` for ALL targets. The centred form
            # in `decompose_trajectory` and the uncentred form here are
            # both consistent for C_rate^q under the corresponding target,
            # but they are DIFFERENT estimators in finite samples (see
            # note_pipeline.md §7.3, item 6 — centred vs uncentred).
            # We use uncentred for two reasons:
            #
            #   (1) the §4.5 cell-side analysis (`generate_realdata.py`,
            #       Fig. 5) used the single-bin `decompose`, which is
            #       uncentred -- the methods §7 pipeline numbers must
            #       align with those §4.5 reference numbers;
            #   (2) the legacy `compute_conditional_second_moments` /
            #       `estimate_rate_covariance` is uncentred, so for
            #       `target='naive'` this also enables the §7.2 equivalence
            #       audit.
            #
            # The centred form remains the default in
            # `estimators.decompose_trajectory` because its numerical-
            # precision argument holds on the §4.6 synthetic validation.
            Crate, n_close, phat, rho, phat_pair = _legacy_compat_crate(
                counts, trajectories, T_idx, tgt, threshold,
                Erate=real["Erate"],
                time_bin_weighting=time_bin_weighting,
                weight_clip=1e6, phat=phat, rho=rho,
                closepair_density=closepair_density, phat_pair=phat_pair,
            )
            if n_close_pairs is None:
                n_close_pairs = n_close

            with np.errstate(divide="ignore", invalid="ignore"):
                alpha = np.clip(np.diag(real["Cpsth"]) / np.diag(Crate),
                                0.0, 1.0)
            one_minus_alpha = 1.0 - alpha
            one_minus_alpha[~(np.diag(Crate) > 0)] = np.nan

            shuffled_crates = []
            if tgt == "naive" and n_shuffles > 0:
                shuffled_crates = _run_naive_shuffles(
                    counts, trajectories, T_idx, threshold, n_shuffles,
                    time_bin_weighting, seed=seed,
                    Erate=real["Erate"],
                )

            per_target[tgt] = {
                "Crate": Crate,
                "Cpsth": real["Cpsth"],
                "Erate": real["Erate"],
                "one_minus_alpha": one_minus_alpha,
                "Shuffled_Crates": shuffled_crates,
            }

        per_window.append(_format_window_record(
            t_count, dt, counts.shape, per_target, Ctotal, n_close_pairs
        ))

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
# Legacy adapter: run the SNAPSHOT's _compute_one_session against the aligned
# cache (no prepare_data, CPU only). Used by compute_methods_data.py --legacy
# so the comparator pickle is regeneratable from the snapshot alone.
# ---------------------------------------------------------------------------

def decompose_session_legacy(aligned, device: str = "cpu",
                             windows_bins: Sequence[int] = WINDOW_BINS_DEFAULT,
                             n_shuffles: int = N_SHUFFLES_DEFAULT,
                             dt: float = DT):
    """Run the snapshot's per-session decomposition against an aligned record.

    The snapshot's ``_compute_one_session`` runs ``prepare_data`` + aligns +
    decomposes. We need only the decomposition part: the aligned arrays already
    come from the cache. So we replicate the post-align portion verbatim and
    call ``legacy.run_covariance_decomposition`` directly.
    """
    import torch
    from legacy.covariance import run_covariance_decomposition
    from legacy.compute_fig2_data import (
        INTERCEPT_MODE, INTERCEPT_KWARGS
    )

    robs = np.asarray(aligned["robs"])
    eyepos = np.asarray(aligned["eyepos"])
    valid_mask = np.asarray(aligned["valid_mask"], dtype=bool)

    # legacy._compute_one_session computes psth from the ORIGINAL robs (with
    # NaNs preserved); run_covariance_decomposition does its own nan_to_num
    # internally on a local copy. Match that ordering exactly -- including the
    # NaN-propagating .mean (not nanmean), since legacy uses .mean.
    psth = robs.mean(axis=0)

    results, mats = run_covariance_decomposition(
        robs, eyepos, valid_mask,
        window_sizes_bins=list(windows_bins),
        dt=dt,
        n_shuffles=n_shuffles,
        intercept_mode=INTERCEPT_MODE,
        intercept_kwargs=INTERCEPT_KWARGS,
        seed=42,
        device=device,
    )

    return {
        "session": aligned["session"],
        "subject": aligned["subject"],
        "results": results,
        "mats": mats,
        "neuron_mask": aligned["neuron_mask"],
        "meta": {
            "n_trials_total": aligned["n_trials_total"],
            "n_trials_good": aligned["n_trials_good"],
            "n_neurons_total": aligned["n_neurons_total"],
            "n_neurons_used": aligned["n_neurons_used"],
        },
        "psth": psth,
        "rate_hz": aligned["rate_hz"],
        "psth_r2": aligned["psth_r2"],
        "qc": {"contam_rate": aligned["contam_rate"]},
    }
