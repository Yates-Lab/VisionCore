"""Extended within-model perturbation ladder for the fig3 single-trial panel.

Forks figure 3's panel D (`paper/fig3/_fig3_ablation_data.py`, conditions
intact / zeroed / stabilized) into a six-condition ladder that separates *where*
reafference is removed from *whether* the extraretinal input is available, and
scores every condition on two metrics instead of one.

Conditions (all are inference-time perturbations of ONE trained twin; the
weights never change):

  full                 retinal stimulus + behavior, both intact (reference)
  ablated              behavior zeroed; the separate extraretinal route removed
  stab_window          retinal input stabilized WITHIN the model's own 33-frame
                       window: each lag keeps the image that was really on
                       screen, but the whole window is cropped at the last
                       frame's gaze. Gaze-contingent position shifts across
                       prediction times survive; only retinal image motion
                       inside the window is removed. The strictest reafference
                       control available -- it changes nothing else.
  stab_trial           retinal input frozen at each trial's own centroid gaze
  stab_global          retinal input frozen at ONE session-global centroid gaze
                       (reproduces fig3 panel D's `stabilized`)
  stab_global_ablated  session-global freeze AND behavior zeroed: both FEM
                       routes removed at once

Metrics, per neuron (all in `_ext_metrics.py`):
  ve        single-trial r^2 against the observed spike counts
  bps       single-trial Poisson bits per spike against each unit's own
            mean-rate null (`models.losses.calc_poisson_bits_per_spike`)
  selfcons  Poisson self-consistency fraction: the same likelihood gain divided
            by the gain expected if the condition's own predicted rates
            generated independent Poisson counts

plus a `psth` baseline (the leave-one-out PSTH used as the predictor) scored on
every metric, so the trial-average reference appears as a real box on each
predictive row rather than only as a line.

Self-contained: owns `outputs/cache/fig3e_extended_ablation.pkl` and imports
nothing from `paper/fig3`. The population masks are rebuilt here from the shared
covariance-decomposition (fig2) caches so this figure describes the same cells
fig2 and fig3 report.
"""
import sys

import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR

# `eval.*` / `models.*` are top-level packages under the repo root, not the
# installed VisionCore package, so the root has to be importable regardless of
# how this module is entered (script, -m, or import from a sibling folder).
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

from _ext_stim import FixRsvpRenderer
from _ext_metrics import bits_per_spike, poisson_self_consistency


CACHE_PATH = CACHE_DIR / "fig3e_extended_ablation.pkl"

# --- analysis parameters (mirrored from fig3 so the populations coincide) ---
DT = 1 / 120
VALID_TIME_BINS = 120
MIN_FIX_DUR = 20
MIN_TOTAL_SPIKES = 200
SUBJECTS = ["Allen", "Logan"]

CHECKPOINT_DIR = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120"
CHECKPOINT_SUBDIR = "2026-03-31_11-33-32_learned_resnet_concat_convgru_gaussian"
EXPERIMENT_SUBDIR = "learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4"
BEST_CKPT = "epoch=374-val_bps_overall=0.6395.ckpt"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{CHECKPOINT_SUBDIR}/{EXPERIMENT_SUBDIR}/{BEST_CKPT}"

# Forward passes are chunked over the trial's prediction times. Batch elements
# are independent (the recurrence runs over the lag axis inside each sample's
# cube, not across samples), so this is exactly equivalent to one big batch and
# keeps peak activation memory small enough to share the GPU with other jobs.
CHUNK = 32

# Model conditions, in the figure's left-to-right order (`psth` is not a model
# condition -- it is the leave-one-out trial average).
CONDS = ["full", "ablated", "stab_window", "stab_trial", "stab_global",
         "stab_global_ablated"]
BOX_ORDER = ["psth"] + CONDS

STAB_CONDS = ["stab_window", "stab_trial", "stab_global", "stab_global_ablated"]
ZERO_BEHAVIOR_CONDS = ["ablated", "stab_global_ablated"]

# Conditions that have a residual against the full twin (`full` is the
# reference, and the PSTH is not a model, so neither has one).
RESID_CONDS = [c for c in CONDS if c != "full"]

COND_LABEL = {
    "psth": "Trial\naverage\n(PSTH)",
    "full": "Retinal +\nbehavioral\n(full)",
    "ablated": "Retinal\nonly\n(ablated)",
    "stab_window": "Stabilized\nwithin\nwindow",
    "stab_trial": "Stabilized\nwithin\ntrial",
    "stab_global": "Stabilized\nacross\ntrials",
    "stab_global_ablated": "Stabilized\n+ ablated",
}


def subject_from_session(session_name):
    return session_name.split("_")[0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _var_explained(pred, true, axis=None):
    return 1 - np.nanvar(pred - true, axis=axis) / np.nanvar(true, axis=axis)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _run_model_chunked(model, stim, behavior, dataset_idx, chunk=CHUNK):
    """`run_model` over sub-batches of prediction times, concatenated."""
    from eval.eval_stack_utils import run_model

    outs = []
    for s in range(0, stim.shape[0], chunk):
        batch = {'stim': stim[s:s + chunk], 'behavior': behavior[s:s + chunk]}
        out = run_model(model, batch, dataset_idx=dataset_idx)
        outs.append(out['rhat'].detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _run_inference(session_filter=None, cache_path=CACHE_PATH):
    """Run the twin over every Allen/Logan fixRSVP session under all six
    conditions. Writes `cache_path` and returns the per-session results."""
    import torch
    from tqdm import tqdm
    from DataYatesV1 import get_free_device
    from eval.eval_stack_multidataset import load_model
    from eval.eval_stack_utils import load_single_dataset, rescale_rhat

    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))

    device = get_free_device()
    print(f"Loading model from: {CHECKPOINT_PATH}")
    model, model_info = load_model(checkpoint_path=CHECKPOINT_PATH, device=str(device))
    model.model.eval()
    model.model.convnet.use_checkpointing = False
    print(f"Model loaded: {model_info['experiment']}, epoch {model_info['epoch']}")

    results = []
    for dataset_idx, session_name in enumerate(model.names):
        subject = subject_from_session(session_name)
        if subject not in SUBJECTS:
            continue
        if session_filter is not None and session_name not in session_filter:
            continue
        print(f"\n--- {session_name} ({subject}) ---")

        try:
            train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
            fixrsvp_inds = torch.cat([
                train_data.get_dataset_inds('fixrsvp'),
                val_data.get_dataset_inds('fixrsvp'),
            ], dim=0)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        dset = train_data.dsets[fixrsvp_inds[:, 0].unique().item()]
        trial_inds = np.asarray(dset.covariates['trial_inds']).ravel()
        psth_inds_flat = np.asarray(dset.covariates['psth_inds']).ravel()
        robs_flat = np.asarray(dset['robs'])
        eyepos_flat = np.asarray(dset['eyepos'])
        fixation = np.hypot(eyepos_flat[:, 0], eyepos_flat[:, 1]) < 1.0

        trials = np.unique(trial_inds)
        NT, NC = len(trials), robs_flat.shape[1]
        T = int(psth_inds_flat.max()) + 1
        stim_lags = np.array(dataset_config['keys_lags']['stim'])
        samp = dataset_config.get('sampling', {})
        factor = (int(samp['source_rate']) // int(samp['target_rate'])) if samp else 1

        # -- stimulus machinery + the two hard gates -------------------------
        rend = FixRsvpRenderer(session_name)
        align = rend.alignment_maxabs(dset['stim'].numpy(), factor)
        if align != 0:
            print(f"  Skipping: raw<->model frame alignment gate failed ({align})")
            continue
        # Gate exactly the raw samples this session will crop at: prediction
        # times are chosen on the model's average-pooled eye position, so the raw
        # frame they map to is what must re-render bit-exactly.
        gate_idx = np.where(fixation)[0] * factor
        gate_idx = gate_idx[gate_idx < rend.n_raw]
        native = rend.verify_native_render(gate_idx)
        if native != 0:
            print(f"  Skipping: canvas re-render gate failed ({native})")
            continue

        n_emb = dset['stim'].shape[0]
        glob_np, n_tr_glob = rend.build_frozen_stim("global", factor, n_emb)
        trial_np, n_tr_trial = rend.build_frozen_stim("trial", factor, n_emb)
        stab_stim = {
            "stab_global": torch.from_numpy(glob_np),
            "stab_trial": torch.from_numpy(trial_np),
            "stab_global_ablated": torch.from_numpy(glob_np),
        }
        print(f"  gates OK (align={align}, native={native}); frozen renders: "
              f"global {n_tr_glob} trials, trial {n_tr_trial} trials; "
              f"{rend.n_contents} canvases")

        robs = np.full((NT, T, NC), np.nan)
        dfs = np.full((NT, T, NC), np.nan)
        fix_dur = np.full(NT, np.nan)
        rhat = {c: np.full((NT, T, NC), np.nan) for c in CONDS}

        n_clamped_tot = 0
        for itrial in tqdm(range(NT), desc=f"  {session_name}"):
            ix = (trial_inds == trials[itrial]) & fixation
            if not np.any(ix):
                continue
            stim_indices = np.where(ix)[0]
            stim_lag_indices = stim_indices[:, None] - stim_lags[None, :]
            behavior0 = dset['behavior'][ix]
            behavior_zero = torch.zeros_like(behavior0)
            t_inds = psth_inds_flat[ix].astype(int)
            fix_dur[itrial] = len(t_inds)
            robs[itrial, t_inds] = robs_flat[ix]
            dfs[itrial, t_inds] = np.asarray(dset['dfs'][ix])

            window_cube = None
            for c in CONDS:
                if c == "stab_window":
                    if window_cube is None:
                        cube, n_cl = rend.build_window_stabilized_cube(
                            stim_indices, stim_lags, factor)
                        window_cube = torch.from_numpy(cube)
                        n_clamped_tot += n_cl
                    stim = window_cube
                elif c in stab_stim:
                    stim = stab_stim[c][stim_lag_indices]
                else:
                    stim = dset['stim'][stim_lag_indices]
                rhat[c][itrial, t_inds] = _run_model_chunked(
                    model, stim.permute(0, 2, 1, 3, 4),
                    behavior_zero if c in ZERO_BEHAVIOR_CONDS else behavior0,
                    dataset_idx,
                )

        good_trials = fix_dur > MIN_FIX_DUR
        if good_trials.sum() < 10:
            print(f"  Skipping: only {good_trials.sum()} good trials")
            continue
        iix = np.arange(min(VALID_TIME_BINS, T))
        robs = robs[good_trials][:, iix]
        dfs = dfs[good_trials][:, iix]
        rhat = {c: r[good_trials][:, iix] for c, r in rhat.items()}

        neuron_mask = np.where(np.nansum(robs, axis=(0, 1)) > MIN_TOTAL_SPIKES)[0]
        if len(neuron_mask) < 3:
            print(f"  Skipping: only {len(neuron_mask)} neurons pass spike threshold")
            continue
        robs = robs[:, :, neuron_mask]
        dfs = dfs[:, :, neuron_mask]
        rhat = {c: r[:, :, neuron_mask] for c, r in rhat.items()}
        n_trials, n_time, n_neurons = robs.shape
        print(f"  {n_trials} trials, {n_time} bins, {n_neurons} neurons")

        def rescale(r):
            rr, _ = rescale_rhat(
                torch.from_numpy(robs.reshape(-1, n_neurons)),
                torch.from_numpy(r.reshape(-1, n_neurons)),
                torch.from_numpy(dfs.reshape(-1, n_neurons)),
                mode='affine',
            )
            return rr.reshape(n_trials, n_time, n_neurons).detach().cpu().numpy()

        rhat_rs = {c: rescale(r) for c, r in rhat.items()}

        robs_m = robs.copy()
        robs_m[dfs == 0] = np.nan
        rhat_m = {c: r.copy() for c, r in rhat_rs.items()}
        for c in CONDS:
            rhat_m[c][dfs == 0] = np.nan

        # Leave-one-out PSTH: the trial-average baseline, scored like a model.
        #
        # It goes through the SAME affine rescaling as every model condition.
        # That is not cosmetic: the raw empirical PSTH has structural zeros
        # (bins where no other trial happened to spike), and a Poisson
        # likelihood charges log(0) whenever the held-out trial spikes there, so
        # an unrescaled PSTH scores a large negative bits/spike that reflects the
        # estimator's zeros rather than the trial average's predictive power.
        # `rescale_rhat`'s affine form is exponentiated (a*rhat + b with a, b > 0),
        # so it is strictly positive by construction and its additive term is a
        # data-fit floor instead of an arbitrary one. Every box on the figure
        # then receives identical treatment.
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rbar = np.zeros_like(robs_m)
                for i in range(n_trials):
                    other = np.setdiff1d(np.arange(n_trials), i)
                    rbar[i] = np.nanmean(robs_m[other], axis=0)
        rbar_rs = rescale(np.nan_to_num(rbar, nan=0.0))
        rbar_rs_m = rbar_rs.copy()
        rbar_rs_m[dfs == 0] = np.nan
        rbar_rs_m[~np.isfinite(rbar)] = np.nan

        ve = {c: _var_explained(rhat_m[c], robs_m, axis=(0, 1)) for c in CONDS}
        ve["psth"] = _var_explained(rbar_rs_m, robs_m, axis=(0, 1))
        ve_psth_unrescaled = _var_explained(rbar, robs_m, axis=(0, 1))

        bps, bps_raw, floored = {}, {}, {}
        for c in CONDS:
            bps[c], floored[c] = bits_per_spike(rhat_rs[c], robs, dfs)
            bps_raw[c], _ = bits_per_spike(rhat[c], robs, dfs)
        bps["psth"], floored["psth"] = bits_per_spike(rbar_rs_m, robs, dfs)
        bps_raw["psth"], floored["psth_unrescaled"] = bits_per_spike(rbar, robs, dfs)

        # --- Poisson self-consistency: observed gain / its own Poisson reference ---
        # Same rescaled predictions, same valid-bin mask, same positive floor and
        # same mean-rate null as the bits/spike row above, so `gain_obs` is that
        # row's likelihood gain before the per-spike normalization and the ratio
        # only swaps the denominator. `G_self` is the gain expected if each
        # condition's own predicted rates generated independent Poisson counts.
        selfcons, gain_obs, gain_self = {}, {}, {}
        for c in CONDS:
            selfcons[c], gain_obs[c], gain_self[c], _ = poisson_self_consistency(
                rhat_rs[c], robs, dfs)
        (selfcons["psth"], gain_obs["psth"], gain_self["psth"],
         _) = poisson_self_consistency(rbar_rs_m, robs, dfs)

        # --- how much of the twin's rate modulation each perturbation moves ---
        # Model vs model: the residual between the full twin's predicted rate and
        # each perturbed twin's, as a fraction of the full twin's own rate
        # variance. The observed spikes play no part.
        #
        # Computed on the RAW model output, not the affine-rescaled rates. The
        # rescaling is fit per condition against the observed counts, so it would
        # partly absorb the very change being measured; the raw output is what
        # the twin actually predicts. Note a pure DC shift between conditions
        # cancels inside a variance, so this measures changed *modulation*, not a
        # changed mean rate. The rescaled version is stored alongside for
        # comparison.
        valid_bins = np.isfinite(dfs) & (dfs > 0)

        def _bin_var(x):
            return np.nanvar(np.where(valid_bins, x, np.nan), axis=(0, 1))

        rate_var = {"raw": _bin_var(rhat["full"]), "rescaled": _bin_var(rhat_rs["full"])}
        resid_frac, resid_frac_rs = {}, {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for c in RESID_CONDS:
                resid_frac[c] = _bin_var(rhat["full"] - rhat[c]) / rate_var["raw"]
                resid_frac_rs[c] = (_bin_var(rhat_rs["full"] - rhat_rs[c])
                                    / rate_var["rescaled"])
        for c in RESID_CONDS:
            resid_frac[c][~(rate_var["raw"] > 0)] = np.nan
            resid_frac_rs[c][~(rate_var["rescaled"] > 0)] = np.nan
        print("  residual/rate variance  " + "  ".join(
            f"{c}: {np.nanmedian(resid_frac[c]):.3f}" for c in RESID_CONDS))

        # One median over every prediction time in the session (eye-tracking
        # validity masked inside `window_displacement_px`).
        med_disp = rend.window_displacement_px(
            np.where(fixation)[0], stim_lags, factor)
        print(f"  window stabilization removed {med_disp:.2f} px "
              f"({med_disp / 37.5:.3f} deg) median in-window gaze displacement; "
              f"{n_clamped_tot} lag(s) clamped at session start")
        print("  medians  " + "  ".join(
            f"{c}: r2={np.nanmedian(ve[c]):+.4f} bps={np.nanmedian(bps[c]):+.4f} "
            f"S={np.nanmedian(selfcons[c]):+.3f}"
            for c in BOX_ORDER))
        print(f"  PSTH rescaling: r2 {np.nanmedian(ve_psth_unrescaled):+.4f} -> "
              f"{np.nanmedian(ve['psth']):+.4f}, bps "
              f"{np.nanmedian(bps_raw['psth']):+.4f} -> {np.nanmedian(bps['psth']):+.4f} "
              f"({floored['psth_unrescaled']} floored bin(s) removed)")
        n_fl_model = sum(v for c, v in floored.items() if c in CONDS)
        if n_fl_model:
            print(f"  WARNING: BPS floor hit {n_fl_model} model bin(s) "
                  f"(of {robs.size} per condition) -- expected 0, since the "
                  f"affine rescaling is strictly positive")
        if floored["psth"]:
            print(f"  BPS floor hit {floored['psth']} rescaled-PSTH bin(s) "
                  f"({100 * floored['psth'] / robs.size:.3f}% of bins; residual "
                  f"structural zeros the rescaling could not lift)")

        results.append({
            "session": session_name, "subject": subject,
            "neuron_mask": neuron_mask, "n_neurons": n_neurons,
            "n_trials": n_trials,
            "ve": ve, "bps": bps, "bps_unrescaled": bps_raw,
            "selfcons": selfcons, "gain_obs": gain_obs, "gain_self": gain_self,
            "ve_psth_unrescaled": ve_psth_unrescaled,
            "resid_frac": resid_frac, "resid_frac_rescaled": resid_frac_rs,
            "rate_var_full": rate_var["raw"],
            "diagnostics": {
                "align_maxabs": align, "native_maxabs": native,
                "n_trials_frozen_global": n_tr_glob,
                "n_trials_frozen_trial": n_tr_trial,
                "median_window_disp_px": med_disp,
                "n_window_lags_clamped": n_clamped_tot,
                "n_bps_floored": floored,
            },
        })

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        dill.dump(results, f)
    print(f"\nCached {len(results)} sessions to {cache_path}")
    return results


# ---------------------------------------------------------------------------
# Population masks (rebuilt from the shared fig2 caches)
# ---------------------------------------------------------------------------
def _fig2_inclusion():
    """Return `(per_cell, floored_sessions)` from the aligned covariance cache:
    `per_cell[(session, neuron_id)]` is fig2's window-independent inclusion
    (rate > 2 Hz & split-half PSTH R^2 > 0.10), and `floored_sessions` is the set
    of sessions with at least `MIN_SESSION_UNITS` such units -- the same two
    rules `paper/covariance_decomposition/derive.py` applies, recomputed from the
    72 MB aligned cache so this loader never has to open the multi-GB derived
    bundle."""
    covdecomp = str(VISIONCORE_ROOT / "paper" / "covariance_decomposition")
    if covdecomp not in sys.path:
        sys.path.insert(0, covdecomp)
    import derive
    from data_loading import load_cache as load_aligned_cache

    per_cell, floored = {}, set()
    for a in load_aligned_cache():
        rate = np.asarray(a["rate_hz"], float)
        psth = np.asarray(a["psth_r2"], float)
        keep = (np.isfinite(rate) & (rate > derive.MIN_RATE_HZ)
                & np.isfinite(psth) & (psth > derive.MIN_PSTH_R2))
        for o, k in zip(np.asarray(a["neuron_mask"]), keep):
            per_cell[(a["session"], int(o))] = bool(k)
        if int(keep.sum()) >= derive.MIN_SESSION_UNITS:
            floored.add(a["session"])
    return per_cell, floored


# The eye-shuffle null is NOT a zero-rate-variance null. `_run_corrected_shuffles`
# permutes the trajectories but leaves `T_idx` alone, and close pairs are still
# enumerated within a time-in-trial group, so a shuffled `Crate` still carries
# the PSTH variance -- which is why fig2 uses this null for `1 - alpha`. So
# `p_rate` tests whether a unit has resolvable *FEM* modulation, not whether it
# has usable rate variance. This row is a fraction of the *total* explainable
# variance, so it must not be gated on that: the only exclusion is a denominator
# that is not a usable variance (handled in `ceiling.py`). `p_rate` is carried
# through for anyone who wants the FEM-resolvable subset.
P_RATE_MAX = 0.05


def _attach_ceiling(agg, keys):
    """Join fig2's per-unit single-trial ceiling and normalize each r^2 by it.

    `R2_max = Var(lambda)/Var(y) = diag(Crate)/diag(Ctotal)` bounds the r^2 of
    any predictor that is a function of stimulus and gaze, so `ve / R2_max` is
    the fraction of the explainable variance a condition actually captured.
    Taken at `PRODUCTION_WINDOW_BINS` (one 120 Hz bin), which is both the twin's
    resolution and the bin the `ve` row is computed on -- the ceiling is
    strongly window-dependent, so the two must match.
    """
    covdecomp = str(VISIONCORE_ROOT / "paper" / "covariance_decomposition")
    if covdecomp not in sys.path:
        sys.path.insert(0, covdecomp)
    from ceiling import load_ceiling, PRODUCTION_WINDOW_BINS

    table = load_ceiling()
    w = PRODUCTION_WINDOW_BINS

    def field(key, name):
        return table.get(key, {}).get(w, {}).get(name, np.nan)

    r2_max = np.array([field(k, "r2_max") for k in keys], dtype=float)
    p_rate = np.array([field(k, "p_rate") for k in keys], dtype=float)

    pop = np.asarray(agg["population"], dtype=bool)
    missing = int((pop & ~np.isfinite(r2_max)).sum())
    unresolved = int((pop & np.isfinite(r2_max) & (p_rate > P_RATE_MAX)).sum())

    agg["r2_max"] = r2_max
    agg["p_rate"] = p_rate
    with np.errstate(divide="ignore", invalid="ignore"):
        agg["r2_norm"] = {c: np.asarray(agg["ve"][c], dtype=float) / r2_max
                          for c in BOX_ORDER}

    n_used = int((pop & np.isfinite(r2_max)).sum())
    print(f"Ceiling (window_bins={w}): {n_used}/{int(pop.sum())} population "
          f"cells normalized; dropped {missing} without a usable rate variance. "
          f"({unresolved} of the retained cells have no FEM modulation "
          f"resolvable against the eye-shuffle null, p > {P_RATE_MAX}; they are "
          f"kept -- this row is a fraction of total explainable variance.)")


def aggregate(results):
    """Flatten per-cell arrays across sessions, in results-order x
    neuron_mask-order."""
    agg = {
        "ve": {c: np.concatenate([r["ve"][c] for r in results]) for c in BOX_ORDER},
        "bps": {c: np.concatenate([r["bps"][c] for r in results]) for c in BOX_ORDER},
        "bps_unrescaled": {
            c: np.concatenate([r["bps_unrescaled"][c] for r in results])
            for c in BOX_ORDER
        },
        "subjects": np.array(
            [r["subject"] for r in results for _ in range(r["n_neurons"])]),
        "sessions": np.array(
            [r["session"] for r in results for _ in range(r["n_neurons"])]),
    }
    if all("resid_frac" in r for r in results):
        for key in ("resid_frac", "resid_frac_rescaled"):
            agg[key] = {c: np.concatenate([r[key][c] for r in results])
                        for c in RESID_CONDS}
    else:
        print("NOTE: cache predates the residual-variance row; rerun with "
              "--recompute to populate it.")
    # The cache holds per-neuron summaries, not the per-bin rate traces, so a
    # metric added after a sweep cannot be reconstructed from it.
    if all("selfcons" in r for r in results):
        for key in ("selfcons", "gain_obs", "gain_self"):
            agg[key] = {c: np.concatenate([r[key][c] for r in results])
                        for c in BOX_ORDER}
    else:
        print("NOTE: cache predates the Poisson self-consistency row; rerun "
              "with --recompute to populate it.")
    return agg


_cached_data = None


def load_extended_data(recompute=False, cache_path=CACHE_PATH):
    """Return flattened per-cell arrays plus the fig2 `population` mask."""
    global _cached_data
    if _cached_data is not None and not recompute:
        return _cached_data

    if cache_path.exists() and not recompute:
        print(f"Loading cached extended-ablation results from {cache_path}")
        with open(cache_path, "rb") as f:
            results = dill.load(f)
    else:
        results = _run_inference(cache_path=cache_path)

    per_cell, floored = _fig2_inclusion()
    kept = [r for r in results if r["session"] in floored]
    dropped = [r["session"] for r in results if r["session"] not in floored]
    if dropped:
        print(f"Session floor (fig2 population): dropping {len(dropped)} "
              f"sub-floor session(s): {dropped}")
    results = kept

    agg = aggregate(results)
    # One key list drives both the population mask and the ceiling join, so the
    # two cannot drift out of alignment with the flattened per-cell arrays.
    keys = [(r["session"], int(nid))
            for r in results for nid in r["neuron_mask"]]
    agg["population"] = np.asarray(
        [per_cell.get(k, False) for k in keys], dtype=bool)
    _attach_ceiling(agg, keys)
    print(f"Extended ablation: {len(results)} sessions, "
          f"{len(agg['population'])} cells "
          f"({int(agg['population'].sum())} in the fig2 population)")

    _cached_data = {**agg, "results": results}
    return _cached_data


def main():
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--sessions", type=str, default=None,
                   help="Comma-separated session names (smoke test).")
    p.add_argument("--cache", type=str, default=None,
                   help="Alternate cache path (keeps a smoke test off the real cache).")
    args = p.parse_args()

    cache_path = Path(args.cache) if args.cache else CACHE_PATH
    if args.sessions:
        _run_inference(session_filter=set(args.sessions.split(",")),
                       cache_path=cache_path)
    else:
        load_extended_data(recompute=args.recompute, cache_path=cache_path)


if __name__ == "__main__":
    main()
