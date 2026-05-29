"""
Talk-ready full-vs-behavior-zeroed model comparison on fixRSVP.

This is a focused extraction of the intact/zeroed behavior comparison from
ryan/behavior-vs-vision/within_model_perturbation.py. It does not import or
modify Ryan's code. All generated files are written under:

    outputs/behavior-vs-vision/model_comparison/

Typical usage:

    python jake/behavior-vs-vision/model_comparison.py --recompute

For a fast rebuild of the population plots from Ryan's existing summary cache:

    python jake/behavior-vs-vision/model_comparison.py --from-ryan-cache

For the talk workflow, keep Ryan's full-population summary and recompute only a
few sessions for example-cell PSTH panels:

    python jake/behavior-vs-vision/model_comparison.py --from-ryan-cache \
        --recompute-examples --sessions Allen_2022-02-16 Logan_2020-03-04

Plain --from-ryan-cache cannot make example-cell PSTH panels because Ryan's
cache intentionally does not store trial-aligned predictions; add
--recompute-examples when those panels are needed.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Running this file by path puts jake/behavior-vs-vision on sys.path, not the
# repo root. Add the repo root before importing project modules.
VISIONCORE_ROOT = Path(__file__).resolve().parents[2]
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

for _workspace_pkg in ("DataYatesV1", "DataRowleyV1V2"):
    _pkg_root = VISIONCORE_ROOT.parent / _workspace_pkg
    if (_pkg_root / _workspace_pkg).exists() and str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))

# Keep Matplotlib's generated font/config cache inside the requested output
# tree instead of ~/.config/matplotlib or a random /tmp directory.
_mpl_config_dir = (
    VISIONCORE_ROOT
    / "outputs"
    / "behavior-vs-vision"
    / "model_comparison"
    / ".matplotlib"
)
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BEHAVIOR_DIR = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120/"
    "2026-03-31_11-33-32_learned_resnet_concat_convgru_gaussian/"
    "learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4"
)

RYAN_WITHIN_MODEL_CACHE = (
    VISIONCORE_ROOT / "outputs" / "cache" / "behavior_vs_vision_within_model.pkl"
)

OUTPUT_ROOT = VISIONCORE_ROOT / "outputs" / "behavior-vs-vision"
RUN_ROOT = OUTPUT_ROOT / "model_comparison"
FIG_DIR = RUN_ROOT / "figures"
STAT_DIR = RUN_ROOT / "stats"
LOCAL_CACHE_DIR = RUN_ROOT / "cache"
LOCAL_CACHE_PATH = LOCAL_CACHE_DIR / "full_vs_zeroed_fixrsvp.pkl"

for _path in (FIG_DIR, STAT_DIR, LOCAL_CACHE_DIR):
    _path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

FULL_COLOR = "#147A8C"
ZEROED_COLOR = "#E07A2F"
DATA_COLOR = "#222222"
SUBJECT_COLORS = {"Allen": "#3B6FB6", "Logan": "#2F8F5B"}
SAMPLE_RATE_HZ = 120.0
BIN_MS = 1000.0 / SAMPLE_RATE_HZ


def set_plot_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def clean_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def save_figure(fig: mpl.figure.Figure, stem: str, formats: Iterable[str], dpi: int) -> None:
    formats = list(dict.fromkeys(["pdf", *formats]))
    for fmt in formats:
        out = FIG_DIR / f"{stem}.{fmt}"
        fig.savefig(out, bbox_inches="tight", dpi=dpi if fmt != "svg" else None)
        print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisConfig:
    valid_time_bins: int
    min_fix_dur: int
    min_total_spikes: int
    fixation_radius: float
    ccnorm_splits: int
    seed: int


def subject_from_session(session_name: str) -> str:
    return session_name.split("_", maxsplit=1)[0]


def find_best_ckpt(ckpt_dir: Path) -> Path:
    ckpts = list(ckpt_dir.glob("epoch=*-val_bps_overall=*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No val_bps_overall checkpoints in {ckpt_dir}")
    return max(ckpts, key=lambda p: float(p.stem.split("val_bps_overall=")[1]))


def safe_wilcoxon(x: np.ndarray, alternative: str = "two-sided") -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or np.allclose(x, 0):
        return math.nan, math.nan
    try:
        from scipy.stats import wilcoxon

        stat, pval = wilcoxon(x, alternative=alternative)
        return float(stat), float(pval)
    except Exception:
        return math.nan, math.nan


def format_pvalue(pval: float) -> str:
    if not np.isfinite(pval):
        return "n/a"
    if pval < 1e-4:
        return f"{pval:.1e}"
    return f"{pval:.4f}"


def finite_pair(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y)


def robust_lims(values: np.ndarray, pad_frac: float = 0.06) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(values, [0.5, 99.5])
    lo = min(lo, 0.0)
    hi = max(hi, 0.05)
    pad = (hi - lo) * pad_frac
    return float(lo - pad), float(hi + pad)


def pearson_1d(x: np.ndarray, y: np.ndarray, min_n: int = 20) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n:
        return math.nan
    x = x[ok]
    y = y[ok]
    if x.std() < 1e-8 or y.std() < 1e-8:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Model/data inference path
# ---------------------------------------------------------------------------


_EVAL_FUNCS: dict | None = None


def import_eval_stack():
    global _EVAL_FUNCS
    if _EVAL_FUNCS is not None:
        return _EVAL_FUNCS

    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))
    from eval.eval_stack_multidataset import load_model
    from eval.eval_stack_utils import (
        bits_per_spike,
        ccnorm_split_half_variable_trials,
        load_single_dataset,
        rescale_rhat,
        run_model,
    )

    _EVAL_FUNCS = {
        "load_model": load_model,
        "load_single_dataset": load_single_dataset,
        "run_model": run_model,
        "rescale_rhat": rescale_rhat,
        "bits_per_spike": bits_per_spike,
        "ccnorm_split_half_variable_trials": ccnorm_split_half_variable_trials,
    }
    return _EVAL_FUNCS


def load_behavior_model(ckpt_dir: Path, device: str | None):
    from DataYatesV1 import get_free_device

    funcs = import_eval_stack()
    ckpt = find_best_ckpt(ckpt_dir)
    if device is None:
        device = str(get_free_device())

    print(f"Behavior checkpoint: {ckpt}")
    print(f"Device: {device}")
    model, info = funcs["load_model"](checkpoint_path=str(ckpt), device=device)
    model.model.eval()
    if hasattr(model.model, "convnet"):
        model.model.convnet.use_checkpointing = False
    return model, info


def gather_fixrsvp(model, dataset_idx: int, cfg: AnalysisConfig):
    funcs = import_eval_stack()
    train_data, val_data, dataset_config = funcs["load_single_dataset"](model, dataset_idx)

    fixrsvp_inds = torch.cat(
        [
            train_data.get_dataset_inds("fixrsvp"),
            val_data.get_dataset_inds("fixrsvp"),
        ],
        dim=0,
    )
    dset_idx_local = fixrsvp_inds[:, 0].unique().item()
    dset = train_data.dsets[dset_idx_local]

    trial_inds = np.asarray(dset.covariates["trial_inds"]).ravel()
    psth_inds = np.asarray(dset.covariates["psth_inds"]).ravel()
    eyepos = np.asarray(dset["eyepos"])
    fixation = np.hypot(eyepos[:, 0], eyepos[:, 1]) < cfg.fixation_radius

    trials = np.unique(trial_inds)
    n_trials = len(trials)
    n_cells = np.asarray(dset["robs"]).shape[1]
    n_time = int(psth_inds.max()) + 1

    cids = np.asarray(dataset_config.get("cids", np.arange(n_cells)))
    if cids.shape[0] != n_cells:
        cids = np.arange(n_cells)

    return {
        "dset": dset,
        "dataset_config": dataset_config,
        "trial_inds": trial_inds,
        "psth_inds": psth_inds,
        "fixation": fixation,
        "trials": trials,
        "n_trials": n_trials,
        "n_time": n_time,
        "n_cells": n_cells,
        "cids": cids,
    }


def run_condition(
    model,
    dataset_idx: int,
    info: dict,
    *,
    zero_behavior: bool,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    funcs = import_eval_stack()
    run_model = funcs["run_model"]

    dset = info["dset"]
    trial_inds = info["trial_inds"]
    psth_inds = info["psth_inds"]
    fixation = info["fixation"]
    trials = info["trials"]
    n_trials = info["n_trials"]
    n_time = info["n_time"]
    n_cells = info["n_cells"]

    rhat = np.full((n_trials, n_time, n_cells), np.nan, dtype=np.float32)
    robs = np.full((n_trials, n_time, n_cells), np.nan, dtype=np.float32)
    dfs = np.full((n_trials, n_time, n_cells), np.nan, dtype=np.float32)
    fix_dur = np.full(n_trials, np.nan, dtype=np.float32)
    eyepos = np.full((n_trials, n_time, 2), np.nan, dtype=np.float32)

    robs_flat = np.asarray(dset["robs"])
    eyepos_flat = np.asarray(dset["eyepos"])
    stim_lags = np.asarray(info["dataset_config"]["keys_lags"]["stim"])

    for itrial in tqdm(range(n_trials), leave=False, desc=desc):
        ix = (trial_inds == trials[itrial]) & fixation
        if not np.any(ix):
            continue

        stim_indices = np.where(ix)[0]
        stim_lag_indices = stim_indices[:, None] - stim_lags[None, :]
        stim = dset["stim"][stim_lag_indices].permute(0, 2, 1, 3, 4)
        behavior = dset["behavior"][ix]
        if zero_behavior:
            behavior = torch.zeros_like(behavior)

        out = run_model(
            model,
            {"stim": stim, "behavior": behavior},
            dataset_idx=dataset_idx,
        )

        t_inds = psth_inds[ix].astype(int)
        fix_dur[itrial] = len(t_inds)
        rhat[itrial, t_inds] = out["rhat"].detach().cpu().numpy()
        robs[itrial, t_inds] = robs_flat[ix]
        dfs[itrial, t_inds] = np.asarray(dset["dfs"][ix])
        eyepos[itrial, t_inds] = eyepos_flat[ix]

        del out, stim, behavior

    return rhat, robs, dfs, fix_dur, eyepos


def affine_rescale(robs: np.ndarray, rhat: np.ndarray, dfs: np.ndarray) -> np.ndarray:
    funcs = import_eval_stack()
    rescale_rhat = funcs["rescale_rhat"]

    n_trials, n_time, n_cells = robs.shape
    rhat_rs, _ = rescale_rhat(
        torch.from_numpy(robs.reshape(-1, n_cells)).float(),
        torch.from_numpy(rhat.reshape(-1, n_cells)).float(),
        torch.from_numpy(dfs.reshape(-1, n_cells)).float(),
        mode="affine",
    )
    return rhat_rs.reshape(n_trials, n_time, n_cells).detach().cpu().numpy()


def compute_bps(rhat_rs: np.ndarray, robs: np.ndarray, dfs: np.ndarray) -> np.ndarray:
    funcs = import_eval_stack()
    bits_per_spike = funcs["bits_per_spike"]

    n_trials, n_time, n_cells = robs.shape
    dfs_mask = (np.nan_to_num(dfs.reshape(-1, n_cells), nan=0.0) > 0.5).astype(
        np.float32
    )
    return (
        bits_per_spike(
            torch.from_numpy(rhat_rs.reshape(-1, n_cells)).float(),
            torch.from_numpy(robs.reshape(-1, n_cells)).float(),
            torch.from_numpy(dfs_mask),
        )
        .detach()
        .cpu()
        .numpy()
    )


def compute_ccnorm(
    robs: np.ndarray,
    rhat_rs: np.ndarray,
    dfs: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    funcs = import_eval_stack()
    ccnorm_split_half_variable_trials = funcs["ccnorm_split_half_variable_trials"]

    dfs_mask = np.nan_to_num(dfs, nan=0.0) > 0.5
    cc1, abs1, max1, _, _ = ccnorm_split_half_variable_trials(
        robs.copy(),
        rhat_rs.copy(),
        dfs_mask.copy(),
        n_splits=n_splits,
        rng=seed,
        return_components=True,
    )
    cc2, abs2, max2, _, _ = ccnorm_split_half_variable_trials(
        robs.copy(),
        rhat_rs.copy(),
        dfs_mask.copy(),
        n_splits=n_splits,
        rng=seed + 1,
        return_components=True,
    )
    unstable = (cc1 - cc2) ** 2 > 0.01
    ccnorm = 0.5 * (cc1 + cc2)
    ccabs = 0.5 * (abs1 + abs2)
    ccmax = 0.5 * (max1 + max2)
    ccnorm[unstable] = np.nan
    return ccnorm, ccabs, ccmax


def masked_trial_mean(arr: np.ndarray, dfs: np.ndarray) -> np.ndarray:
    valid = (np.nan_to_num(dfs, nan=0.0) > 0.5) & np.isfinite(arr)
    counts = valid.sum(axis=0)
    sums = np.where(valid, arr, 0.0).sum(axis=0)
    out = sums / np.maximum(counts, 1)
    out[counts == 0] = np.nan
    return out


def masked_trial_sem(arr: np.ndarray, dfs: np.ndarray) -> np.ndarray:
    valid = (np.nan_to_num(dfs, nan=0.0) > 0.5) & np.isfinite(arr)
    n = valid.sum(axis=0)
    mean = masked_trial_mean(arr, dfs)
    resid = np.where(valid, arr - mean[None, :, :], 0.0)
    var = np.sum(resid * resid, axis=0) / np.maximum(n - 1, 1)
    sem = np.sqrt(var) / np.sqrt(np.maximum(n, 1))
    sem[n <= 1] = np.nan
    return sem


def nanpercentile_by_time(mat: np.ndarray, q: float) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2:
        return np.array([], dtype=float)
    out = np.full(mat.shape[1], np.nan, dtype=float)
    for i_time in range(mat.shape[1]):
        vals = mat[:, i_time]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[i_time] = np.percentile(vals, q)
    return out


def psth_corr_per_cell(
    robs: np.ndarray,
    rhat: np.ndarray,
    dfs: np.ndarray,
    *,
    min_time: int = 20,
) -> np.ndarray:
    y = masked_trial_mean(robs, dfs)
    p = masked_trial_mean(rhat, dfs)
    out = np.full(y.shape[1], np.nan)
    for i_cell in range(y.shape[1]):
        out[i_cell] = pearson_1d(y[:, i_cell], p[:, i_cell], min_n=min_time)
    return out


def gaze_sort_scores(eyepos: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return one scalar gaze score per trial for row sorting.

    The score is the first principal component of each trial's median gaze
    position in the displayed fixRSVP window. If the 2-D gaze cloud is
    degenerate, it falls back to horizontal eye position.
    """
    trial_pos = np.full((eyepos.shape[0], 2), np.nan, dtype=float)
    finite_eye = np.isfinite(eyepos).all(axis=-1)
    for i_trial in range(eyepos.shape[0]):
        mask = valid[i_trial] & finite_eye[i_trial]
        if mask.any():
            trial_pos[i_trial] = np.nanmedian(eyepos[i_trial, mask], axis=0)

    scores = np.full(eyepos.shape[0], np.nan, dtype=float)
    ok = np.isfinite(trial_pos).all(axis=1)
    if ok.sum() < 2:
        return scores

    centered = trial_pos[ok] - np.nanmean(trial_pos[ok], axis=0, keepdims=True)
    if np.nanmax(np.abs(centered)) < 1e-8:
        scores[ok] = trial_pos[ok, 0]
        return scores

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    scores[ok] = centered @ vh[0]
    return scores


def build_example_candidates(
    session: str,
    subject: str,
    neuron_mask: np.ndarray,
    cids: np.ndarray,
    robs: np.ndarray,
    rhat_full: np.ndarray,
    rhat_zeroed: np.ndarray,
    dfs: np.ndarray,
    eyepos: np.ndarray,
    metrics: dict,
    *,
    max_per_session: int,
    min_example_bps: float,
) -> list[dict]:
    psth_data = masked_trial_mean(robs, dfs)
    psth_full = masked_trial_mean(rhat_full, dfs)
    psth_zeroed = masked_trial_mean(rhat_zeroed, dfs)

    corr_full = psth_corr_per_cell(robs, rhat_full, dfs)
    corr_zeroed = psth_corr_per_cell(robs, rhat_zeroed, dfs)
    delta_bps = metrics["bps_zeroed"] - metrics["bps_full"]
    ccnorm_full = metrics["ccnorm_full"]

    score = (
        0.8 * np.nan_to_num(corr_full, nan=-np.inf)
        + np.nan_to_num(metrics["bps_full"], nan=-np.inf)
        - 1.5 * np.abs(np.nan_to_num(delta_bps, nan=np.inf))
    )
    if np.isfinite(ccnorm_full).any():
        score += 0.25 * np.nan_to_num(ccnorm_full, nan=0.0)

    order = np.argsort(score)[::-1]

    def make_example(local_idx: int, selection: str) -> dict:
        cell_valid = np.nan_to_num(dfs[:, :, local_idx], nan=0.0) > 0.5
        trial_gaze = gaze_sort_scores(eyepos, cell_valid)
        return {
            "session": session,
            "subject": subject,
            "cell_index": int(neuron_mask[local_idx]),
            "cid": int(cids[local_idx]),
            "score": float(score[local_idx]),
            "selection": selection,
            "bps_full": float(metrics["bps_full"][local_idx]),
            "bps_zeroed": float(metrics["bps_zeroed"][local_idx]),
            "delta_bps": float(delta_bps[local_idx]),
            "ccnorm_full": float(metrics["ccnorm_full"][local_idx]),
            "ccnorm_zeroed": float(metrics["ccnorm_zeroed"][local_idx]),
            "psth_corr_full": float(corr_full[local_idx]),
            "psth_corr_zeroed": float(corr_zeroed[local_idx]),
            "data": psth_data[:, local_idx].astype(np.float32),
            "full": psth_full[:, local_idx].astype(np.float32),
            "zeroed": psth_zeroed[:, local_idx].astype(np.float32),
            "trial_data": np.where(
                cell_valid, robs[:, :, local_idx], np.nan
            ).astype(np.float32),
            "trial_full": np.where(
                cell_valid, rhat_full[:, :, local_idx], np.nan
            ).astype(np.float32),
            "trial_zeroed": np.where(
                cell_valid, rhat_zeroed[:, :, local_idx], np.nan
            ).astype(np.float32),
            "trial_gaze": trial_gaze.astype(np.float32),
        }

    strict: list[int] = []
    fallback: list[int] = []
    lower_corr_fallback: list[int] = []
    for local_idx in order:
        if not np.isfinite(score[local_idx]):
            continue
        if not np.isfinite(metrics["bps_full"][local_idx]):
            continue
        if not np.isfinite(corr_full[local_idx]) or corr_full[local_idx] < 0.55:
            lower_corr_fallback.append(int(local_idx))
            continue
        if metrics["bps_full"][local_idx] < min_example_bps:
            fallback.append(int(local_idx))
            continue
        strict.append(int(local_idx))

    selected = [(idx, "higher_bps") for idx in strict[:max_per_session]]
    if len(selected) < max_per_session:
        remaining = max_per_session - len(selected)
        selected.extend((idx, "lower_bps_fallback") for idx in fallback[:remaining])
    if len(selected) < max_per_session:
        remaining = max_per_session - len(selected)
        selected.extend(
            (idx, "lower_corr_fallback")
            for idx in lower_corr_fallback[:remaining]
        )
    return [make_example(local_idx, selection) for local_idx, selection in selected]


def process_session(
    model,
    dataset_idx: int,
    session_name: str,
    cfg: AnalysisConfig,
    *,
    examples_per_session: int,
    min_example_bps: float,
) -> dict | None:
    subject = subject_from_session(session_name)
    print(f"\n--- {session_name} ({subject}) [{dataset_idx}] ---")

    try:
        info = gather_fixrsvp(model, dataset_idx, cfg)
    except Exception as exc:
        print(f"Skipping {session_name}: could not load fixRSVP data: {exc}")
        return None

    print(
        f"Trials={info['n_trials']} time={info['n_time']} cells={info['n_cells']}"
    )

    rhat_full, robs, dfs, fix_dur, eyepos = run_condition(
        model,
        dataset_idx,
        info,
        zero_behavior=False,
        desc=f"{session_name} full",
    )
    rhat_zeroed, _, _, _, _ = run_condition(
        model,
        dataset_idx,
        info,
        zero_behavior=True,
        desc=f"{session_name} zeroed",
    )

    good_trials = fix_dur > cfg.min_fix_dur
    if good_trials.sum() < 10:
        print(f"Skipping {session_name}: only {good_trials.sum()} good trials")
        return None

    iix = np.arange(min(cfg.valid_time_bins, robs.shape[1]))
    robs = robs[good_trials][:, iix]
    dfs = dfs[good_trials][:, iix]
    rhat_full = rhat_full[good_trials][:, iix]
    rhat_zeroed = rhat_zeroed[good_trials][:, iix]
    eyepos = eyepos[good_trials][:, iix]

    neuron_mask = np.where(np.nansum(robs, axis=(0, 1)) > cfg.min_total_spikes)[0]
    if neuron_mask.size < 3:
        print(f"Skipping {session_name}: only {neuron_mask.size} cells pass threshold")
        return None

    cids = info["cids"][neuron_mask]
    robs = robs[:, :, neuron_mask]
    dfs = dfs[:, :, neuron_mask]
    rhat_full = rhat_full[:, :, neuron_mask]
    rhat_zeroed = rhat_zeroed[:, :, neuron_mask]

    print(
        f"Using {robs.shape[0]} trials, {robs.shape[1]} bins, "
        f"{robs.shape[2]} cells"
    )
    print("Affine rescale: full behavior")
    rhat_full_rs = affine_rescale(robs, rhat_full, dfs)
    print("Affine rescale: zeroed behavior")
    rhat_zeroed_rs = affine_rescale(robs, rhat_zeroed, dfs)

    print("Metrics: BPS")
    bps_full = compute_bps(rhat_full_rs, robs, dfs)
    bps_zeroed = compute_bps(rhat_zeroed_rs, robs, dfs)
    print("Metrics: CCnorm")
    ccnorm_full, ccabs_full, ccmax_full = compute_ccnorm(
        robs,
        rhat_full_rs,
        dfs,
        n_splits=cfg.ccnorm_splits,
        seed=cfg.seed + dataset_idx * 100,
    )
    ccnorm_zeroed, ccabs_zeroed, ccmax_zeroed = compute_ccnorm(
        robs,
        rhat_zeroed_rs,
        dfs,
        n_splits=cfg.ccnorm_splits,
        seed=cfg.seed + dataset_idx * 100 + 10,
    )

    metrics = {
        "bps_full": bps_full,
        "bps_zeroed": bps_zeroed,
        "ccnorm_full": ccnorm_full,
        "ccnorm_zeroed": ccnorm_zeroed,
        "ccabs_full": ccabs_full,
        "ccabs_zeroed": ccabs_zeroed,
        "ccmax_full": ccmax_full,
        "ccmax_zeroed": ccmax_zeroed,
    }

    examples = build_example_candidates(
        session_name,
        subject,
        neuron_mask,
        cids,
        robs,
        rhat_full_rs,
        rhat_zeroed_rs,
        dfs,
        eyepos,
        metrics,
        max_per_session=examples_per_session,
        min_example_bps=min_example_bps,
    )

    print(
        "Session medians: "
        f"BPS full={np.nanmedian(bps_full):.3f}, "
        f"zeroed={np.nanmedian(bps_zeroed):.3f}, "
        f"delta={np.nanmedian(bps_zeroed - bps_full):+.4f}"
    )

    return {
        "session": session_name,
        "subject": subject,
        "n_trials": int(robs.shape[0]),
        "n_time": int(robs.shape[1]),
        "n_cells": int(robs.shape[2]),
        "neuron_mask": neuron_mask.astype(int),
        "cids": cids.astype(int),
        "metrics": metrics,
        "examples": examples,
    }


def recompute_results(args: argparse.Namespace, *, save_cache: bool = True) -> dict:
    cfg = AnalysisConfig(
        valid_time_bins=args.valid_time_bins,
        min_fix_dur=args.min_fix_dur,
        min_total_spikes=args.min_total_spikes,
        fixation_radius=args.fixation_radius,
        ccnorm_splits=args.ccnorm_splits,
        seed=args.seed,
    )

    model, model_info = load_behavior_model(args.behavior_dir, args.device)
    all_sessions = list(model.names)

    if args.sessions:
        requested = set(args.sessions)
        selected = [s for s in all_sessions if s in requested]
        missing = sorted(requested.difference(selected))
        if missing:
            print(f"Requested sessions not found in model: {missing}")
    else:
        selected = [
            s for s in all_sessions if subject_from_session(s) in set(args.subjects)
        ]

    if args.max_sessions is not None:
        selected = selected[: args.max_sessions]

    print(f"\nProcessing {len(selected)} sessions")

    session_to_idx = {name: idx for idx, name in enumerate(all_sessions)}
    results = []
    for session_name in selected:
        r = process_session(
            model,
            session_to_idx[session_name],
            session_name,
            cfg,
            examples_per_session=args.example_candidates_per_session,
            min_example_bps=args.min_example_bps,
        )
        if r is not None:
            results.append(r)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "source": "recompute",
        "model_info": {
            k: str(v) if isinstance(v, Path) else v for k, v in model_info.items()
        },
        "behavior_dir": str(args.behavior_dir),
        "config": vars(args),
        "sessions": results,
    }
    if save_cache:
        with open(LOCAL_CACHE_PATH, "wb") as f:
            dill.dump(payload, f)
        print(f"Cached recomputed results to {LOCAL_CACHE_PATH}")
    return payload


# ---------------------------------------------------------------------------
# Ryan summary cache path
# ---------------------------------------------------------------------------


def load_ryan_summary_cache(path: Path) -> dict:
    print(f"Loading Ryan summary cache: {path}")
    with open(path, "rb") as f:
        data = dill.load(f)

    sessions = []
    for r in data:
        bps_full = np.asarray(r["bps"]["beh_intact"])
        bps_zeroed = np.asarray(r["bps"]["beh_zeroed"])
        ccnorm_full = np.asarray(r["cc_norm"]["beh_intact"])
        ccnorm_zeroed = np.asarray(r["cc_norm"]["beh_zeroed"])
        n_cells = len(bps_full)
        neuron_mask = np.asarray(r.get("fixrsvp_mask", np.arange(n_cells))).astype(int)

        sessions.append(
            {
                "session": r["session"],
                "subject": r.get("subject", subject_from_session(r["session"])),
                "n_trials": math.nan,
                "n_time": math.nan,
                "n_cells": int(n_cells),
                "neuron_mask": neuron_mask,
                "cids": neuron_mask,
                "metrics": {
                    "bps_full": bps_full,
                    "bps_zeroed": bps_zeroed,
                    "ccnorm_full": ccnorm_full,
                    "ccnorm_zeroed": ccnorm_zeroed,
                    "ccabs_full": np.asarray(r["cc_abs"]["beh_intact"]),
                    "ccabs_zeroed": np.asarray(r["cc_abs"]["beh_zeroed"]),
                    "ccmax_full": np.asarray(r["cc_max"]["beh_intact"]),
                    "ccmax_zeroed": np.asarray(r["cc_max"]["beh_zeroed"]),
                },
                "examples": [],
            }
        )

    return {
        "source": f"ryan-summary-cache:{path}",
        "sessions": sessions,
    }


def load_or_compute(args: argparse.Namespace) -> dict:
    if args.recompute:
        return recompute_results(args)

    if args.from_ryan_cache:
        if not args.ryan_cache.exists():
            raise FileNotFoundError(args.ryan_cache)
        payload = load_ryan_summary_cache(args.ryan_cache)
        if args.recompute_examples:
            example_args = args
            if args.sessions is None:
                example_args = argparse.Namespace(**vars(args))
                example_args.sessions = [r["session"] for r in payload["sessions"]]
            example_payload = recompute_results(example_args, save_cache=False)
            examples_by_session = {
                r["session"]: r.get("examples", [])
                for r in example_payload.get("sessions", [])
            }
            n_examples = 0
            for r in payload["sessions"]:
                r["examples"] = examples_by_session.get(r["session"], [])
                n_examples += len(r["examples"])
            payload["example_source"] = example_payload.get("source", "recompute")
            print(f"Attached {n_examples} recomputed example candidates")
        return payload

    if LOCAL_CACHE_PATH.exists():
        print(f"Loading local cache: {LOCAL_CACHE_PATH}")
        with open(LOCAL_CACHE_PATH, "rb") as f:
            return dill.load(f)

    raise FileNotFoundError(
        f"No local cache at {LOCAL_CACHE_PATH}. Run with --recompute or "
        "--from-ryan-cache."
    )


# ---------------------------------------------------------------------------
# Flattening, stats, and exports
# ---------------------------------------------------------------------------


def flatten_cells(payload: dict) -> dict[str, np.ndarray]:
    rows = []
    for session_idx, r in enumerate(payload["sessions"]):
        m = r["metrics"]
        n = len(m["bps_full"])
        for i in range(n):
            rows.append(
                {
                    "session_idx": session_idx,
                    "session": r["session"],
                    "subject": r["subject"],
                    "cell_index": int(r["neuron_mask"][i]),
                    "cid": int(r["cids"][i]),
                    "bps_full": float(m["bps_full"][i]),
                    "bps_zeroed": float(m["bps_zeroed"][i]),
                    "delta_bps": float(m["bps_zeroed"][i] - m["bps_full"][i]),
                    "ccnorm_full": float(m["ccnorm_full"][i]),
                    "ccnorm_zeroed": float(m["ccnorm_zeroed"][i]),
                    "delta_ccnorm": float(m["ccnorm_zeroed"][i] - m["ccnorm_full"][i]),
                    "ccmax_full": float(m["ccmax_full"][i]),
                    "ccmax_zeroed": float(m["ccmax_zeroed"][i]),
                }
            )

    keys = rows[0].keys() if rows else []
    flat = {k: np.asarray([row[k] for row in rows]) for k in keys}
    flat["_rows"] = rows
    return flat


def session_summary(payload: dict) -> list[dict]:
    out = []
    for r in payload["sessions"]:
        m = r["metrics"]
        delta_bps = np.asarray(m["bps_zeroed"]) - np.asarray(m["bps_full"])
        delta_cc = np.asarray(m["ccnorm_zeroed"]) - np.asarray(m["ccnorm_full"])
        out.append(
            {
                "session": r["session"],
                "subject": r["subject"],
                "n_cells": r["n_cells"],
                "median_bps_full": float(np.nanmedian(m["bps_full"])),
                "median_bps_zeroed": float(np.nanmedian(m["bps_zeroed"])),
                "median_delta_bps": float(np.nanmedian(delta_bps)),
                "median_ccnorm_full": float(np.nanmedian(m["ccnorm_full"])),
                "median_ccnorm_zeroed": float(np.nanmedian(m["ccnorm_zeroed"])),
                "median_delta_ccnorm": float(np.nanmedian(delta_cc)),
            }
        )
    return out


def write_csvs(payload: dict, flat: dict[str, np.ndarray]) -> None:
    cell_csv = STAT_DIR / "cell_metrics_full_vs_zeroed.csv"
    rows = flat.get("_rows", [])
    if rows:
        with open(cell_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {cell_csv}")

    summary_rows = session_summary(payload)
    summary_csv = STAT_DIR / "session_summary_full_vs_zeroed.csv"
    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved {summary_csv}")

    stats_csv = STAT_DIR / "population_stats_full_vs_zeroed.csv"
    if rows:
        stats_rows = []
        for subject in ["All"] + sorted(set(flat["subject"])):
            mask = np.ones(len(flat["delta_bps"]), dtype=bool)
            if subject != "All":
                mask = flat["subject"] == subject
            mask &= np.isfinite(flat["delta_bps"])
            d_bps = flat["delta_bps"][mask].astype(float)
            stat, pval = safe_wilcoxon(d_bps, alternative="two-sided")
            stats_rows.append(
                {
                    "subject": subject,
                    "metric": "delta_bps_zeroed_minus_full",
                    "n_cells": int(d_bps.size),
                    "median": float(np.nanmedian(d_bps)) if d_bps.size else math.nan,
                    "mean": float(np.nanmean(d_bps)) if d_bps.size else math.nan,
                    "wilcoxon_stat": stat,
                    "wilcoxon_p": pval,
                }
            )

            mask = np.ones(len(flat["delta_ccnorm"]), dtype=bool)
            if subject != "All":
                mask = flat["subject"] == subject
            mask &= np.isfinite(flat["delta_ccnorm"])
            d_cc = flat["delta_ccnorm"][mask].astype(float)
            stat, pval = safe_wilcoxon(d_cc, alternative="two-sided")
            stats_rows.append(
                {
                    "subject": subject,
                    "metric": "delta_ccnorm_zeroed_minus_full",
                    "n_cells": int(d_cc.size),
                    "median": float(np.nanmedian(d_cc)) if d_cc.size else math.nan,
                    "mean": float(np.nanmean(d_cc)) if d_cc.size else math.nan,
                    "wilcoxon_stat": stat,
                    "wilcoxon_p": pval,
                }
            )

        with open(stats_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stats_rows)
        print(f"Saved {stats_csv}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_session_summary(payload: dict, formats: Iterable[str], dpi: int) -> None:
    rows = session_summary(payload)
    if not rows:
        return

    subjects = np.asarray([r["subject"] for r in rows])
    delta_bps = np.asarray([r["median_delta_bps"] for r in rows], dtype=float)
    cc_full = np.asarray([r["median_ccnorm_full"] for r in rows], dtype=float)
    cc_zeroed = np.asarray([r["median_ccnorm_zeroed"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1))

    ax = axes[0]
    vp = ax.violinplot(
        [delta_bps[np.isfinite(delta_bps)]],
        positions=[0],
        widths=0.65,
        showextrema=False,
        showmedians=False,
    )
    for body in vp["bodies"]:
        body.set_facecolor("#C7CCD1")
        body.set_edgecolor("#5E6670")
        body.set_alpha(0.8)

    for subj in sorted(set(subjects)):
        mask = (subjects == subj) & np.isfinite(delta_bps)
        rng = np.random.default_rng(sum(ord(c) for c in subj))
        x = rng.normal(0, 0.035, size=mask.sum())
        ax.scatter(
            x,
            delta_bps[mask],
            s=28,
            color=SUBJECT_COLORS.get(subj, "#666666"),
            alpha=0.8,
            edgecolor="white",
            linewidth=0.4,
            label=subj,
        )
    med = np.nanmedian(delta_bps)
    ax.plot([-0.28, 0.28], [med, med], color=DATA_COLOR, lw=1.6)
    ax.axhline(0, color="#666666", lw=0.9, ls="--")
    ax.set_xticks([0])
    ax.set_xticklabels(["zeroed - full"])
    ax.set_ylabel("Median delta BPS per session")
    ax.set_title(f"Behavior contribution (n={len(rows)} sessions)")
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)

    ax = axes[1]
    x_full, x_zero = 0, 1
    rng = np.random.default_rng(2026)
    jitter = rng.normal(0, 0.035, size=len(subjects))
    for i, subj in enumerate(subjects):
        color = SUBJECT_COLORS.get(subj, "#666666")
        x_pair = [x_full + jitter[i], x_zero + jitter[i]]
        ax.plot(x_pair, [cc_full[i], cc_zeroed[i]], color=color, alpha=0.35, lw=1)
        ax.scatter(
            x_pair,
            [cc_full[i], cc_zeroed[i]],
            color=color,
            s=22,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

    box = ax.boxplot(
        [cc_full[np.isfinite(cc_full)], cc_zeroed[np.isfinite(cc_zeroed)]],
        positions=[x_full, x_zero],
        widths=0.45,
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(box["boxes"], [FULL_COLOR, ZEROED_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.24)
        patch.set_edgecolor(color)
    for part in ("whiskers", "caps", "medians"):
        for artist in box[part]:
            artist.set_color("#333333")
            artist.set_linewidth(1.0)

    _, p_cc = safe_wilcoxon(cc_zeroed - cc_full, alternative="two-sided")
    ax.set_xticks([x_full, x_zero])
    ax.set_xticklabels(["full", "behavior zeroed"])
    ax.set_ylabel("Median CCnorm per session")
    ax.set_title(f"PSTH prediction unchanged (p={format_pvalue(p_cc)})")
    ax.set_ylim(bottom=0)
    clean_axis(ax)

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "session_summary_delta_bps_ccnorm", formats, dpi)
    plt.close(fig)


def plot_population_scatter(flat: dict[str, np.ndarray], formats: Iterable[str], dpi: int) -> None:
    if len(flat.get("bps_full", [])) == 0:
        return

    bps_full = flat["bps_full"].astype(float)
    bps_zeroed = flat["bps_zeroed"].astype(float)
    subjects = flat["subject"]
    ok = finite_pair(bps_full, bps_zeroed)
    if not ok.any():
        return

    delta = bps_zeroed[ok] - bps_full[ok]
    med = np.nanmedian(delta)
    within = 100.0 * np.mean(np.abs(delta) <= 0.01)

    fig, ax = plt.subplots(figsize=(4.1, 4.1))
    for subj in sorted(set(subjects)):
        mask = ok & (subjects == subj)
        if not mask.any():
            continue
        ax.scatter(
            bps_full[mask],
            bps_zeroed[mask],
            s=13,
            alpha=0.42,
            color=SUBJECT_COLORS.get(subj, "#666666"),
            edgecolor="none",
            rasterized=True,
            label=f"{subj} (n={mask.sum()})",
        )

    lims = robust_lims(np.concatenate([bps_full[ok], bps_zeroed[ok]]))
    ax.plot(lims, lims, color="#222222", ls="--", lw=1.0, alpha=0.75)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("BPS, full behavior")
    ax.set_ylabel("BPS, behavior zeroed")
    ax.set_title(
        "Per-cell model performance\n"
        f"median delta={med:+.4f} BPS; {within:.0f}% within +/-0.01"
    )
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)

    fig.tight_layout()
    save_figure(fig, "population_bps_scatter_full_vs_zeroed", formats, dpi)
    plt.close(fig)


def plot_population_delta_violin(
    flat: dict[str, np.ndarray],
    formats: Iterable[str],
    dpi: int,
) -> None:
    if len(flat.get("delta_bps", [])) == 0:
        return

    subjects = flat["subject"]
    categories = ["All"] + sorted(set(subjects))
    data = []
    for cat in categories:
        mask = np.isfinite(flat["delta_bps"])
        if cat != "All":
            mask &= subjects == cat
        data.append(flat["delta_bps"][mask].astype(float))

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    positions = np.arange(len(categories))
    vp = ax.violinplot(
        data,
        positions=positions,
        widths=0.65,
        showextrema=False,
        showmedians=False,
    )
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor("#BFC6CE" if i == 0 else SUBJECT_COLORS.get(categories[i], "#888888"))
        body.set_edgecolor("#333333")
        body.set_alpha(0.42 if i == 0 else 0.34)

    rng = np.random.default_rng(123)
    for i, vals in enumerate(data):
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if vals.size > 650:
            idx = rng.choice(vals.size, size=650, replace=False)
            vals_plot = vals[idx]
        else:
            vals_plot = vals
        x = rng.normal(positions[i], 0.055, size=vals_plot.size)
        color = "#30343B" if i == 0 else SUBJECT_COLORS.get(categories[i], "#666666")
        ax.scatter(
            x,
            vals_plot,
            s=7,
            alpha=0.22,
            color=color,
            edgecolor="none",
            rasterized=True,
        )
        med = np.nanmedian(vals)
        ax.plot([positions[i] - 0.25, positions[i] + 0.25], [med, med], color="#111111", lw=1.4)

    ax.axhline(0, color="#555555", lw=0.9, ls="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{cat}\n(n={len(vals)})" for cat, vals in zip(categories, data)])
    ax.set_ylabel("Delta BPS (zeroed - full)")
    ax.set_title("Per-cell BPS changes cluster near zero")
    clean_axis(ax)

    fig.tight_layout()
    save_figure(fig, "population_delta_bps_violin", formats, dpi)
    plt.close(fig)


def plot_population_ccnorm_scatter(
    flat: dict[str, np.ndarray],
    formats: Iterable[str],
    dpi: int,
) -> None:
    if len(flat.get("ccnorm_full", [])) == 0:
        return

    full = flat["ccnorm_full"].astype(float)
    zeroed = flat["ccnorm_zeroed"].astype(float)
    subjects = flat["subject"]
    ok = finite_pair(full, zeroed)
    if not ok.any():
        return

    delta = zeroed[ok] - full[ok]
    med = np.nanmedian(delta)
    within = 100.0 * np.mean(np.abs(delta) <= 0.02)

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    for subj in sorted(set(subjects)):
        mask = ok & (subjects == subj)
        if not mask.any():
            continue
        ax.scatter(
            full[mask],
            zeroed[mask],
            s=13,
            alpha=0.42,
            color=SUBJECT_COLORS.get(subj, "#666666"),
            edgecolor="none",
            rasterized=True,
            label=f"{subj} (n={mask.sum()})",
        )

    lo, hi = robust_lims(np.concatenate([full[ok], zeroed[ok]]))
    lo = max(0.0, lo)
    hi = min(1.05, hi)
    ax.plot([lo, hi], [lo, hi], color="#222222", ls="--", lw=1.0, alpha=0.75)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("CCnorm, full behavior")
    ax.set_ylabel("CCnorm, behavior zeroed")
    ax.set_title(
        "PSTH prediction\n"
        f"median delta={med:+.4f}; {within:.0f}% within +/-0.02"
    )
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)

    fig.tight_layout()
    save_figure(fig, "population_ccnorm_scatter_full_vs_zeroed", formats, dpi)
    plt.close(fig)


def collect_examples(
    payload: dict,
    n_examples: int,
    *,
    balance_sessions: bool = True,
    examples_per_session: int | None = None,
) -> list[dict]:
    by_session = []
    for session in payload["sessions"]:
        examples = [
            ex
            for ex in session.get("examples", [])
            if np.isfinite(ex.get("score", np.nan))
        ]
        examples.sort(key=lambda ex: ex["score"], reverse=True)
        if examples:
            by_session.append(examples)

    if examples_per_session is not None:
        selected = []
        for session_examples in by_session:
            selected.extend(session_examples[:examples_per_session])
        return selected[:n_examples] if n_examples else selected

    if not balance_sessions:
        examples = [ex for session_examples in by_session for ex in session_examples]
        examples.sort(key=lambda ex: ex["score"], reverse=True)
        return examples[:n_examples]

    selected = []
    rank = 0
    while len(selected) < n_examples:
        added = False
        for session_examples in by_session:
            if rank >= len(session_examples):
                continue
            selected.append(session_examples[rank])
            added = True
            if len(selected) >= n_examples:
                break
        if not added:
            break
        rank += 1
    return selected


def write_displayed_examples_csv(
    payload: dict,
    n_examples: int,
    examples_per_session: int | None,
) -> None:
    examples = collect_examples(
        payload,
        n_examples,
        examples_per_session=examples_per_session,
    )
    rows = []
    for i, ex in enumerate(examples, start=1):
        rows.append(
            {
                "display_rank": i,
                "session": ex["session"],
                "subject": ex["subject"],
                "cell_index": ex["cell_index"],
                "cid": ex["cid"],
                "selection": ex.get("selection", ""),
                "score": ex["score"],
                "bps_full": ex["bps_full"],
                "bps_zeroed": ex["bps_zeroed"],
                "delta_bps": ex["delta_bps"],
                "psth_corr_full": ex["psth_corr_full"],
                "psth_corr_zeroed": ex["psth_corr_zeroed"],
                "ccnorm_full": ex["ccnorm_full"],
                "ccnorm_zeroed": ex["ccnorm_zeroed"],
            }
        )
    if not rows:
        return

    out = STAT_DIR / "displayed_example_cells_full_vs_zeroed.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")


def plot_example_cells(
    payload: dict,
    n_examples: int,
    formats: Iterable[str],
    dpi: int,
    examples_per_session: int | None,
) -> None:
    examples = collect_examples(
        payload,
        n_examples,
        examples_per_session=examples_per_session,
    )
    if not examples:
        print(
            "No example PSTH data available. Use --recompute, or "
            "--from-ryan-cache --recompute-examples, to generate examples."
        )
        return

    n = len(examples)
    n_cols = 2 if n > 1 else 1
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 2.35 * n_rows),
        squeeze=False,
        sharex=True,
    )
    time_ms_raw = np.arange(len(examples[0]["data"])) * BIN_MS
    finite_time = np.zeros_like(time_ms_raw, dtype=bool)
    for ex in examples:
        finite_time |= (
            np.isfinite(ex["data"])
            | np.isfinite(ex["full"])
            | np.isfinite(ex["zeroed"])
        )
    t0 = time_ms_raw[finite_time][0] if finite_time.any() else 0.0
    time_ms = time_ms_raw - t0

    for ax, ex in zip(axes.ravel(), examples):
        y = np.asarray(ex["data"], dtype=float) * SAMPLE_RATE_HZ
        full = np.asarray(ex["full"], dtype=float) * SAMPLE_RATE_HZ
        zeroed = np.asarray(ex["zeroed"], dtype=float) * SAMPLE_RATE_HZ
        full_trials = np.asarray(ex.get("trial_full", []), dtype=float)
        zeroed_trials = np.asarray(ex.get("trial_zeroed", []), dtype=float)
        full_lo = nanpercentile_by_time(full_trials, 12.5) * SAMPLE_RATE_HZ
        full_hi = nanpercentile_by_time(full_trials, 87.5) * SAMPLE_RATE_HZ
        zeroed_lo = nanpercentile_by_time(zeroed_trials, 12.5) * SAMPLE_RATE_HZ
        zeroed_hi = nanpercentile_by_time(zeroed_trials, 87.5) * SAMPLE_RATE_HZ
        if full_lo.size != time_ms.size:
            full_lo = np.full_like(time_ms, np.nan, dtype=float)
            full_hi = np.full_like(time_ms, np.nan, dtype=float)
        if zeroed_lo.size != time_ms.size:
            zeroed_lo = np.full_like(time_ms, np.nan, dtype=float)
            zeroed_hi = np.full_like(time_ms, np.nan, dtype=float)

        ax.fill_between(
            time_ms,
            full_lo,
            full_hi,
            color=FULL_COLOR,
            alpha=0.13,
            linewidth=0,
            label="full central 75%",
        )
        ax.fill_between(
            time_ms,
            zeroed_lo,
            zeroed_hi,
            color=ZEROED_COLOR,
            alpha=0.13,
            linewidth=0,
            label="zeroed central 75%",
        )
        ax.plot(time_ms, y, color=DATA_COLOR, lw=1.9, label="data PSTH")
        ax.plot(time_ms, full, color=FULL_COLOR, lw=1.8, label="full behavior")
        ax.plot(
            time_ms,
            zeroed,
            color=ZEROED_COLOR,
            lw=1.8,
            ls=(0, (4, 2)),
            label="behavior zeroed",
        )
        ax.set_title(
            f"{ex['session']} cell {ex['cell_index']}\n"
            f"BPS {ex['bps_full']:.3f} vs {ex['bps_zeroed']:.3f}, "
            f"r={ex['psth_corr_full']:.2f}"
        )
        ax.set_ylabel("spikes/s")
        ax.set_ylim(bottom=0)
        clean_axis(ax)

    for ax in axes.ravel()[len(examples) :]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Time in PSTH window (ms)")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "example_cells_psth_full_vs_zeroed", formats, dpi)
    plt.close(fig)


def _trial_heatmap_arrays(ex: dict, order_mode: str) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Prepare observed/full/zeroed trial matrices for Ryan-style heatmaps."""
    mats = [
        np.asarray(ex["trial_data"], dtype=float),
        np.asarray(ex["trial_full"], dtype=float),
        np.asarray(ex["trial_zeroed"], dtype=float),
    ]
    finite_time = np.zeros(mats[0].shape[1], dtype=bool)
    finite_trial = np.ones(mats[0].shape[0], dtype=bool)
    for mat in mats:
        finite_time |= np.isfinite(mat).any(axis=0)
        finite_trial &= np.isfinite(mat).sum(axis=1) >= 10

    if not finite_time.any() or not finite_trial.any():
        return [np.empty((0, 0)) for _ in mats], np.array([]), 0

    first = int(np.argmax(finite_time))
    last = int(len(finite_time) - np.argmax(finite_time[::-1]))
    trial_idx = np.where(finite_trial)[0]

    if order_mode == "gaze_sorted":
        gaze = np.asarray(ex.get("trial_gaze", np.full(mats[0].shape[0], np.nan)), dtype=float)
        scores = gaze[trial_idx]
        if np.isfinite(scores).sum() >= 2:
            fill = np.nanmedian(scores[np.isfinite(scores)])
            scores = np.where(np.isfinite(scores), scores, fill)
            trial_idx = trial_idx[np.argsort(scores)]
    elif order_mode != "original":
        raise ValueError(f"Unknown order_mode={order_mode!r}")

    # Convert spikes/bin to sp/s to match Ryan's Figure 3 visual convention.
    out = [mat[trial_idx, first:last] * SAMPLE_RATE_HZ for mat in mats]
    time_ms = (np.arange(first, last) - first) * BIN_MS
    return out, time_ms, len(trial_idx)


def _add_heatmap_scale_bar(ax: mpl.axes.Axes, n_trials: int, duration_ms: float) -> None:
    if n_trials <= 0 or duration_ms <= 0:
        return
    x0 = duration_ms * 0.05
    x1 = min(x0 + 100.0, duration_ms * 0.45)
    y0 = n_trials + max(1.0, n_trials * 0.045)
    ax.plot([x0, x1], [y0, y0], color="black", lw=1.7, clip_on=False)
    ax.text(
        0.5 * (x0 + x1),
        y0 + max(1.0, n_trials * 0.035),
        "100 ms",
        ha="center",
        va="top",
        fontsize=7,
        clip_on=False,
    )

    y_top = max(0.0, n_trials - 10.0)
    y_bot = n_trials
    xv = -duration_ms * 0.035
    ax.plot([xv, xv], [y_top, y_bot], color="black", lw=1.7, clip_on=False)
    ax.text(
        xv - duration_ms * 0.035,
        0.5 * (y_top + y_bot),
        "10 trials",
        ha="right",
        va="center",
        rotation=90,
        fontsize=7,
        clip_on=False,
    )


def plot_trial_heatmaps(
    payload: dict,
    n_examples: int,
    order_mode: str,
    formats: Iterable[str],
    dpi: int,
    examples_per_session: int | None,
) -> None:
    examples = [
        ex
        for ex in collect_examples(
            payload,
            n_examples,
            examples_per_session=examples_per_session,
        )
        if all(k in ex for k in ("trial_data", "trial_full", "trial_zeroed"))
    ]
    if not examples:
        print(
            f"No trial heatmap data available for {order_mode}. Use "
            "--recompute or --from-ryan-cache --recompute-examples."
        )
        return

    prepared = []
    vmax_pool = []
    for ex in examples:
        mats, time_ms, n_trials = _trial_heatmap_arrays(ex, order_mode)
        if n_trials == 0 or time_ms.size == 0:
            continue
        prepared.append((ex, mats, time_ms, n_trials))
        for mat in mats:
            vmax_pool.append(mat[np.isfinite(mat)])

    if not prepared:
        print(f"No finite trial heatmap data available for {order_mode}.")
        return

    values = np.concatenate([v for v in vmax_pool if v.size])
    vmax = float(np.nanpercentile(values, 99.2)) if values.size else 1.0
    vmax = max(vmax, 1.0)

    n_rows = len(prepared)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(6.9, max(1.35 * n_rows, 2.2)),
        squeeze=False,
        constrained_layout=False,
    )
    cmap = mpl.colormaps["gray_r"].copy()
    cmap.set_bad("#F3F4F5")
    column_titles = ["Observed", "Full model", "Behavior zeroed"]
    order_title = "gaze-sorted trials" if order_mode == "gaze_sorted" else "original trial order"
    last_im = None

    for row, (ex, mats, time_ms, n_trials) in enumerate(prepared):
        duration_ms = float(time_ms[-1] + BIN_MS)
        for col, (ax, mat, title) in enumerate(zip(axes[row], mats, column_titles)):
            last_im = ax.imshow(
                mat,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=0,
                vmax=vmax,
                origin="upper",
                extent=[0, duration_ms, n_trials, 0],
            )
            if row == 0:
                ax.set_title(title, pad=5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(
                    f"{ex['session']}\ncell {ex['cell_index']}\n"
                    f"BPS {ex['bps_full']:.3f}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=48,
                    fontsize=8,
                )
        if row == n_rows - 1:
            _add_heatmap_scale_bar(axes[row, 0], n_trials, duration_ms)

    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes,
            fraction=0.018,
            pad=0.012,
            shrink=0.88,
        )
        cbar.set_label("sp/s")
        cbar.outline.set_linewidth(0.6)

    fig.suptitle(f"Single-trial fixRSVP predictions, {order_title}", y=0.995, fontsize=11)
    fig.subplots_adjust(left=0.18, right=0.91, top=0.92, bottom=0.08, wspace=0.03, hspace=0.18)
    suffix = "gaze_sorted" if order_mode == "gaze_sorted" else "original_order"
    save_figure(fig, f"example_trial_heatmaps_{suffix}_full_vs_zeroed", formats, dpi)
    plt.close(fig)


def make_all_figures(payload: dict, args: argparse.Namespace) -> None:
    set_plot_style()
    flat = flatten_cells(payload)
    write_csvs(payload, flat)
    formats = args.formats

    plot_session_summary(payload, formats, args.dpi)
    plot_population_scatter(flat, formats, args.dpi)
    plot_population_delta_violin(flat, formats, args.dpi)
    plot_population_ccnorm_scatter(flat, formats, args.dpi)
    write_displayed_examples_csv(payload, args.n_examples, args.examples_per_session)
    plot_example_cells(
        payload,
        args.n_examples,
        formats,
        args.dpi,
        args.examples_per_session,
    )
    plot_trial_heatmaps(
        payload,
        args.n_examples,
        "gaze_sorted",
        formats,
        args.dpi,
        args.examples_per_session,
    )
    plot_trial_heatmaps(
        payload,
        args.n_examples,
        "original",
        formats,
        args.dpi,
        args.examples_per_session,
    )

    n_cells = len(flat.get("delta_bps", []))
    n_sessions = len(payload.get("sessions", []))
    print(f"\nDone. Source: {payload.get('source', 'unknown')}")
    print(f"Sessions: {n_sessions}; cells: {n_cells}")
    print(f"Outputs: {RUN_ROOT}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create full-vs-behavior-zeroed fixRSVP talk figures."
    )
    parser.add_argument("--recompute", action="store_true", help="Run model inference.")
    parser.add_argument(
        "--from-ryan-cache",
        action="store_true",
        help="Use Ryan's summary cache for population plots only.",
    )
    parser.add_argument(
        "--recompute-examples",
        action="store_true",
        help=(
            "With --from-ryan-cache, rerun selected sessions to add PSTH "
            "example-cell panels while keeping Ryan's 24-session summary."
        ),
    )
    parser.add_argument("--ryan-cache", type=Path, default=RYAN_WITHIN_MODEL_CACHE)
    parser.add_argument("--behavior-dir", type=Path, default=BEHAVIOR_DIR)
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["Allen", "Logan"],
        help="Subjects to include when recomputing.",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Exact session names to recompute. Overrides --subjects.",
    )
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--valid-time-bins", type=int, default=120)
    parser.add_argument("--min-fix-dur", type=int, default=20)
    parser.add_argument("--min-total-spikes", type=int, default=200)
    parser.add_argument("--fixation-radius", type=float, default=1.0)
    parser.add_argument("--ccnorm-splits", type=int, default=200)
    parser.add_argument(
        "--n-examples",
        type=int,
        default=48,
        help="Maximum number of example cells to display across the contact sheets.",
    )
    parser.add_argument(
        "--examples-per-session",
        type=int,
        default=2,
        help="Number of displayed example cells to take from each session.",
    )
    parser.add_argument(
        "--example-candidates-per-session",
        type=int,
        default=8,
        help="Number of ranked example candidates to retain from each recomputed session.",
    )
    parser.add_argument(
        "--min-example-bps",
        type=float,
        default=0.12,
        help="Minimum full-model BPS for automatically selected example cells.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        choices=["png", "pdf", "svg"],
        help="Figure formats to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_or_compute(args)
    make_all_figures(payload, args)


if __name__ == "__main__":
    main()
