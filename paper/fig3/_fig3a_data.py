"""Asset loader for figure 3 panel A schematic.

Renders full-screen example stimuli (backimage, gaborium, gratings, fixrsvp)
for the pinned panel-B session, extracts a representative sampling ROI from
each prepared `.dset`, and pulls a lag-stack of `dset['stim']` for the
"reconstructed video" cube. Results are cached so subsequent runs of the
panel are fast.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import dill
import numpy as np

from _fig3_data import CACHE_DIR, DT
from _fig3_helpers import PANEL_B_SESSION


PANEL_A_CACHE_PATH = CACHE_DIR / "fig3a_assets.pkl"

# Preferred fixRSVP image IDs, tried in order. The first one with frames in
# the session is used. Edit if the chosen image is unappealing (some IDs in
# the RSVP set are aversive — e.g. the larval/maggot image).
PREFERRED_FIXRSVP_IMAGE_IDS = [21, 12, 7, 18, 3, 9, 14, 25, 5, 16]

# Pinned free-viewing trace segment, chosen via browse_freeview_segments.py.
# (trial_idx, start_global_sample); 5 s window at 120 Hz follows. The image
# overlaid behind the trace stays the one `_render_backimage` selects — only
# the trace's gaze samples are pinned.
PINNED_FREEVIEW = (327, 91923)

# Current/predicted frame for panel A's natural-image model input, as a local
# index into the pinned free-viewing window. Hand-picked so a high-contrast
# saccade sits early in the displayed window (its post-saccadic transient then
# spans the prediction-target raster). Advancing it moves the model input
# forward in time (saccade lands earlier); ~6 bins ≈ 50 ms at 120 Hz.
TRAIN_CUR_FRAME_LOCAL = 312


@dataclass
class PanelAAssets:
    """All the raw arrays panel A needs to draw itself."""
    session: str
    screen_shape: tuple                 # (H, W) full screen in pixels
    pix_per_deg: float                  # screen pixels per visual degree
    screens: dict                       # type -> (H, W) uint8 full-screen image
    rois: dict                          # type -> (2, 2) int  [[r0, r1], [c0, c1]]
    lag_cube: np.ndarray                # (n_lags, h, w) float
    lag_indices: list                   # which lag indices were sampled
    arch: dict                          # introspected architecture summary
    frontend_weights: np.ndarray        # (num_channels, kernel_size) learned temporal kernels
    readout_mean: np.ndarray            # (2,) Gaussian center in normalized [-1,1]
    readout_std: np.ndarray             # (2,) Gaussian std
    readout_features: np.ndarray        # (n_feat,) per-feature weight vector
    behavior_t: np.ndarray              # (T,) seconds (fixrsvp FEM trace)
    behavior_eyepos: np.ndarray         # (T, 2) deg
    behavior_speed: np.ndarray          # (T,) deg/s
    behavior_roi_seq_px: np.ndarray     # (T, 2, 2) fixrsvp dset rois aligned to trace
    freeview_trace_px: np.ndarray       # (N, 2) pixel coords on backimage screen
    freeview_roi_seq_px: np.ndarray     # (N, 2, 2) backimage dset rois for pinned window
    example_neurons: list = None        # list of 3 dicts: per-neuron readout + trace snippet
    psth_neurons: list = None            # list of dicts: best-ccnorm units' observed/predicted PSTHs
    stab_lag_cube: Optional[np.ndarray] = None  # (n_lags, h, w) reafferent-ablated
                                        # ("stabilized") copy of lag_cube: same RSVP
                                        # frames, retinal ROI frozen at the trial
                                        # medoid gaze (flashes kept, motion removed)
    # ── Natural-image (training) model input + prediction target ────────────
    # Panel A's top row shows the training objective: a natural-image model
    # input (space×space×time crop + behavior) and the observed spike-count
    # raster it is trained to predict. All sourced from the pinned backimage
    # free-viewing window (PINNED_FREEVIEW), so they match the gaze trace drawn
    # on the natural-image training screen.
    train_lag_cube: Optional[np.ndarray] = None      # (n_lags, h, w) oldest→newest
    train_cur_roi_px: Optional[np.ndarray] = None    # (2,2) backimage-screen ROI at
                                                     # the cube's current (newest) frame
    train_behavior_t: Optional[np.ndarray] = None    # (T,) seconds
    train_behavior_eyepos: Optional[np.ndarray] = None  # (T, 2) deg
    train_behavior_speed: Optional[np.ndarray] = None   # (T,) deg/s
    train_raster: Optional[np.ndarray] = None        # (n_units, T) observed spikes/bin


# ----------------------------------------------------------------------------
# Architecture introspection (from the YAML; cheap, no GPU)
# ----------------------------------------------------------------------------
def _load_arch_info():
    """Return a compact summary of the model architecture from its YAML."""
    import yaml
    from VisionCore.paths import VISIONCORE_ROOT

    cfg_path = (VISIONCORE_ROOT / "experiments" / "model_configs"
                / "learned_resnet_concat_convgru_gaussian.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    block_kernels = []
    for blk in cfg["convnet"]["params"]["block_configs"]:
        k = blk["conv_params"]["kernel_size"]
        block_kernels.append(tuple(k))

    return {
        "adapter_grid": cfg["adapter"]["params"]["grid_size"],
        "frontend_k": cfg["frontend"]["params"]["kernel_size"],
        "frontend_channels": cfg["frontend"]["params"]["num_channels"],
        "convnet_channels": cfg["convnet"]["params"]["channels"],
        "convnet_kernels": block_kernels,
        "behavior_dim": cfg["modulator"]["params"]["behavior_dim"],
        "feature_dim": cfg["modulator"]["params"]["feature_dim"],
        "gru_hidden": cfg["recurrent"]["params"]["hidden_dim"],
        "gru_kernel": cfg["recurrent"]["params"]["kernel_size"],
    }


# ----------------------------------------------------------------------------
# Frontend weight extraction (from trained checkpoint)
# ----------------------------------------------------------------------------
def _load_n_trained_units():
    """Total trained units across all per-dataset readout heads, summed from
    the checkpoint's `model.readouts.*.mean` row counts. Cheap — loads only
    the state_dict (CPU), no session data."""
    import re
    import torch
    from _fig3_data import CHECKPOINT_PATH
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    total = 0
    for k, v in sd.items():
        if re.match(r"model\.readouts\.\d+\.mean$", k):
            total += int(v.shape[0])
    return total


def _load_frontend_weights():
    """Pull learned (num_channels, kernel_size) temporal kernels from the
    pinned digital-twin checkpoint.

    The frontend is a depthwise temporal conv with weight shape
    (num_channels, 1, kernel_size, 1, 1). We squeeze to (num_channels,
    kernel_size). Loaded from the parametrization's `.original` tensor;
    weight-norm scales differ per channel but the *shape* of each kernel
    is what we want to plot, and downstream rendering normalises each
    trace independently.
    """
    import torch
    from _fig3_data import CHECKPOINT_PATH
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    key = "model.frontend.temporal_conv.conv.parametrizations.weight.original"
    w = sd[key].detach().cpu().numpy()
    return np.ascontiguousarray(w.squeeze(axis=(1, 3, 4)))   # (C, K)


def _load_example_neurons(n=3, window_s=0.5, min_rho=0.85):
    """Pick `n` neurons that are well fit by the model and span the baseline
    firing-rate distribution, then return per-neuron assets needed by the
    panel-A schematic.

    Selection strategy:
      * pool every neuron across all cached sessions whose PSTH correlation
        (`rho`) between observed and predicted is >= min_rho (loosens if
        too few qualify);
      * partition by baseline firing rate (terciles low/mid/high);
      * in each tercile, keep the highest-rho neuron.

    For each chosen neuron we additionally extract:
      * Gaussian readout (mean, std) and depthwise feature weights from the
        trained checkpoint;
      * trial-averaged observed and predicted PSTHs (0.5 s, 120 Hz), so the
        schematic shows cross-condition generalisation rather than the noise
        floor of a single trial.
    """
    import torch
    from _fig3_data import (
        CHECKPOINT_PATH, CACHE_PATH, DT, DATASET_CONFIGS_PATH,
    )

    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"fig3 inference cache missing at {CACHE_PATH} — "
            "run load_fig3_data() first so example neurons can be picked."
        )
    print(f"  loading fig3 inference cache from {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        session_results = dill.load(f)

    # Map session name → dataset_idx via the dataset-configs YAML.
    from models.config_loader import load_dataset_configs
    cfgs = load_dataset_configs(str(DATASET_CONFIGS_PATH))
    name2idx = {c["session"]: i for i, c in enumerate(cfgs)}

    print(f"  loading checkpoint state_dict from {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    n_bins = int(round(window_s / DT))

    # Flatten candidates across all sessions.
    cand = []
    for si, sr in enumerate(session_results):
        rhos = np.asarray(sr["rhos"])           # observed/predicted PSTH corr
        ccn = np.asarray(sr["ccnorm"])
        rmean = np.asarray(sr["robs_mean"])     # (T, n_neurons), spikes/bin
        with np.errstate(invalid="ignore"):
            baseline = np.nanmean(rmean, axis=0) / DT
        for ni in range(sr["n_neurons"]):
            if not np.isfinite(rhos[ni]):
                continue
            cand.append({
                "si": si,
                "ni": ni,
                "rho": float(rhos[ni]),
                "ccnorm": float(ccn[ni]) if np.isfinite(ccn[ni]) else float("nan"),
                "baseline": float(baseline[ni]),
                "session": sr["session"],
            })

    if not cand:
        raise RuntimeError("No valid neurons found in fig3 inference cache.")

    # Loosen threshold until we have at least 4*n eligible neurons so the
    # tercile partition has room to pick well-spaced baselines.
    thr = min_rho
    while True:
        elig = [c for c in cand if c["rho"] >= thr]
        if len(elig) >= 4 * n or thr <= 0.0:
            break
        thr -= 0.02
    print(f"  example-neuron pool: {len(elig)} neurons with PSTH-rho ≥ {thr:.2f}")

    # Partition eligible neurons into `n` baseline-rate terciles; pick the
    # highest-rho neuron in each.
    baselines = np.array([c["baseline"] for c in elig])
    quantiles = np.linspace(0, 1, n + 1)
    edges = np.quantile(baselines, quantiles)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    chosen = []
    for k in range(n):
        lo, hi = edges[k], edges[k + 1]
        bucket = [c for c in elig if lo <= c["baseline"] <= hi
                  and not any(c["si"] == cc["si"] and c["ni"] == cc["ni"]
                              for cc in chosen)]
        if not bucket:
            bucket = [c for c in elig
                      if not any(c["si"] == cc["si"] and c["ni"] == cc["ni"]
                                 for cc in chosen)]
        bucket.sort(key=lambda c: c["rho"], reverse=True)
        chosen.append(bucket[0])

    # Sort low → high baseline so the visual stack reads bottom-up.
    chosen.sort(key=lambda c: c["baseline"])

    # Light Gaussian smoothing (σ = 15 ms) applied to the PSTH for display
    # so the lines read as continuous tuning curves rather than jagged bins.
    from scipy.ndimage import gaussian_filter1d
    sigma_bins = max(0.015 / DT, 1e-6)

    out = []
    for c in chosen:
        si, ni = c["si"], c["ni"]
        sr = session_results[si]
        di = name2idx[sr["session"]]
        nmask = np.asarray(sr["neuron_mask"])
        neuron_id = int(nmask[ni])

        mean = sd[f"model.readouts.{di}.mean"][neuron_id].detach().cpu().numpy().astype(np.float32)
        std  = sd[f"model.readouts.{di}.std"][neuron_id].detach().cpu().numpy().astype(np.float32)
        feats = (sd[f"model.readouts.{di}.features.weight"][neuron_id]
                 .detach().cpu().numpy().squeeze().astype(np.float32))

        # Trial-averaged PSTH (already computed during inference); spikes/bin.
        robs_psth = np.asarray(sr["robs_mean"][:, ni], dtype=float)
        rhat_psth = np.asarray(sr["rhat_mean"][:, ni], dtype=float)
        dfs = np.asarray(sr["dfs_used"][:, :, ni], dtype=float)

        # Restrict to bins with substantial trial coverage so the PSTH
        # average isn't dominated by 1–2 short trials at the tails.
        n_per_bin = (dfs > 0).sum(axis=0)
        min_trials = max(5, int(0.25 * dfs.shape[0]))
        valid = n_per_bin >= min_trials
        if not valid.any():
            continue
        first_bin = int(np.argmax(valid))
        end_bin = first_bin + n_bins
        if end_bin > robs_psth.shape[0]:
            end_bin = robs_psth.shape[0]
            first_bin = end_bin - n_bins
            if first_bin < 0:
                continue

        robs_snip = robs_psth[first_bin:end_bin] / DT
        rhat_snip = rhat_psth[first_bin:end_bin] / DT
        # Drop residual nans (no valid trials at this bin) by carrying the
        # last finite value forward — keeps the line connected.
        for arr in (robs_snip, rhat_snip):
            nan_mask = ~np.isfinite(arr)
            if nan_mask.any():
                last = 0.0
                for i in range(len(arr)):
                    if nan_mask[i]:
                        arr[i] = last
                    else:
                        last = arr[i]
        robs_snip = gaussian_filter1d(robs_snip, sigma=sigma_bins, mode="nearest")
        rhat_snip = gaussian_filter1d(rhat_snip, sigma=sigma_bins, mode="nearest")

        t_axis = np.arange(n_bins) * DT
        psth_corr = float(np.corrcoef(robs_snip, rhat_snip)[0, 1])

        print(f"    neuron {neuron_id} @ {sr['session']}: "
              f"baseline={c['baseline']:.1f} sp/s, rho={c['rho']:.2f}, "
              f"ccnorm={c['ccnorm']:.2f}, window PSTH corr={psth_corr:.2f}")

        out.append({
            "session": sr["session"],
            "neuron_id": neuron_id,
            "rho": float(c["rho"]),
            "ccnorm": float(c["ccnorm"]),
            "baseline_rate": float(c["baseline"]),
            "mean": mean,
            "std": std,
            "features": feats,
            "t": t_axis.astype(np.float32),
            "robs_rate": robs_snip.astype(np.float32),
            "rhat_rate": rhat_snip.astype(np.float32),
        })

    if not out:
        raise RuntimeError("Failed to extract any example-neuron snippets.")
    return out


def _extract_psth_snippet(sr, ni, n_bins, sigma_bins=None):
    """Trial-averaged observed/predicted PSTH (sp/s) for neuron `ni` in
    session-result `sr`, restricted to a `n_bins` window starting at the first
    well-covered bin. Returns (t_axis, robs_rate, rhat_rate) or None when no
    bin has adequate trial coverage. With `sigma_bins` falsy the raw per-bin
    rates are returned unsmoothed (an honest representation of the model
    output); pass a bin sigma only for optional display smoothing."""
    robs_psth = np.asarray(sr["robs_mean"][:, ni], dtype=float)
    rhat_psth = np.asarray(sr["rhat_mean"][:, ni], dtype=float)
    dfs = np.asarray(sr["dfs_used"][:, :, ni], dtype=float)

    n_per_bin = (dfs > 0).sum(axis=0)
    min_trials = max(5, int(0.25 * dfs.shape[0]))
    valid = n_per_bin >= min_trials
    if not valid.any():
        return None
    first_bin = int(np.argmax(valid))
    end_bin = first_bin + n_bins
    if end_bin > robs_psth.shape[0]:
        end_bin = robs_psth.shape[0]
        first_bin = end_bin - n_bins
        if first_bin < 0:
            return None

    robs_snip = robs_psth[first_bin:end_bin] / DT
    rhat_snip = rhat_psth[first_bin:end_bin] / DT
    # Carry the last finite value forward across residual nan bins so the
    # displayed line stays connected.
    for arr in (robs_snip, rhat_snip):
        nan_mask = ~np.isfinite(arr)
        if nan_mask.any():
            last = 0.0
            for i in range(len(arr)):
                if nan_mask[i]:
                    arr[i] = last
                else:
                    last = arr[i]
    if sigma_bins:
        from scipy.ndimage import gaussian_filter1d
        robs_snip = gaussian_filter1d(robs_snip, sigma=sigma_bins, mode="nearest")
        rhat_snip = gaussian_filter1d(rhat_snip, sigma=sigma_bins, mode="nearest")
    t_axis = np.arange(n_bins) * DT
    return t_axis, robs_snip, rhat_snip


# Manual curation of the panel-B prediction units. Each entry is
# (session, neuron_id) chosen from the select_psth_units.py contact sheet. When
# None, the top-`n` reliable units by ccnorm are used automatically.
MANUAL_PSTH_UNITS = [
    ("Allen_2022-04-06", 76),   # ccnorm 0.97, r 0.82
    ("Logan_2020-01-07", 16),   # ccnorm 0.94, r 0.89
    ("Logan_2020-01-09", 10),   # ccnorm 0.89, r 0.82
]


def _load_best_ccnorm_psths(n=3, window_s=0.5):
    """Return observed/predicted PSTHs for the panel-B prediction column, for
    display beside the readouts. Reads only the fig3 inference cache (no
    checkpoint or raw-session load), so it can be backfilled into an existing
    asset cache cheaply. Units are the `n` highest-ccnorm reliable cells, or
    the hand-picked `MANUAL_PSTH_UNITS` when set. Predictions are drawn
    unsmoothed (raw per-bin model output)."""
    from _fig3_data import CACHE_PATH

    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"fig3 inference cache missing at {CACHE_PATH} — "
            "run load_fig3_data() first so best-ccnorm PSTHs can be picked."
        )
    print(f"  loading fig3 inference cache from {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        session_results = dill.load(f)

    n_bins = int(round(window_s / DT))
    sigma_bins = None  # predictions shown unsmoothed (raw per-bin model output)

    cand = []
    from _fig3_data import CCMAX_THRESHOLD

    for si, sr in enumerate(session_results):
        ccn = np.asarray(sr["ccnorm"])
        ccm = np.asarray(sr["ccmax"])
        nmask = np.asarray(sr["neuron_mask"])
        for ni in range(sr["n_neurons"]):
            # Gate on the split-half reliability ceiling (ccmax) and drop
            # ccnorm > 1 — those are normalization artifacts on low-reliability
            # units, not genuinely best-fit cells. Rank the survivors by ccnorm.
            if not (np.isfinite(ccn[ni]) and np.isfinite(ccm[ni])):
                continue
            if ccm[ni] < CCMAX_THRESHOLD or ccn[ni] > 1.0:
                continue
            cand.append({
                "si": si, "ni": ni,
                "ccnorm": float(ccn[ni]),
                "session": sr["session"],
                "neuron_id": int(nmask[ni]),
            })
    cand.sort(key=lambda c: c["ccnorm"], reverse=True)

    if MANUAL_PSTH_UNITS:
        by_key = {(c["session"], c["neuron_id"]): c for c in cand}
        missing = [k for k in MANUAL_PSTH_UNITS if k not in by_key]
        if missing:
            print(f"  WARNING: manual PSTH units not in candidate pool: {missing}")
        cand = [by_key[k] for k in MANUAL_PSTH_UNITS if k in by_key]
        n = len(cand)

    out = []
    for c in cand:
        snip = _extract_psth_snippet(session_results[c["si"]], c["ni"],
                                     n_bins, sigma_bins)
        if snip is None:
            continue
        t_axis, robs_snip, rhat_snip = snip
        out.append({
            "session": c["session"],
            "neuron_id": c["neuron_id"],
            "ccnorm": c["ccnorm"],
            "t": t_axis.astype(np.float32),
            "robs_rate": robs_snip.astype(np.float32),
            "rhat_rate": rhat_snip.astype(np.float32),
        })
        print(f"    psth[{len(out)}] {c['session']} n{c['neuron_id']}: "
              f"ccnorm={c['ccnorm']:.2f}")
        if len(out) >= n:
            break

    if not out:
        raise RuntimeError("No best-ccnorm PSTH snippets could be extracted.")
    return out


def _load_readout_example():
    """Pick a representative neuron from readout 0 and return its
    (mean, std, features) weights. Selection prefers a neuron with
    well-localised feature-weight energy (so the heat-strip reads as
    structured, not flat noise).
    """
    import torch
    from _fig3_data import CHECKPOINT_PATH
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    means = sd["model.readouts.0.mean"].detach().cpu().numpy()
    stds  = sd["model.readouts.0.std"].detach().cpu().numpy()
    feats = sd["model.readouts.0.features.weight"].detach().cpu().numpy()
    feats = feats.squeeze(axis=(2, 3))   # (N_neurons, n_feat)
    # Score by L4/L2 ratio (high → energy concentrated in few features)
    norm2 = np.sqrt((feats ** 2).sum(axis=1) + 1e-12)
    norm4 = np.power((feats ** 4).sum(axis=1) + 1e-12, 0.25)
    score = norm4 / norm2
    idx = int(np.argmax(score))
    return means[idx].astype(np.float32), stds[idx].astype(np.float32), feats[idx].astype(np.float32)


# ----------------------------------------------------------------------------
# Full-screen stimulus rendering
# ----------------------------------------------------------------------------
def _full_screen_rect(settings):
    """Return ROI = full screen as (2,2) array, rows then columns (pixel coords)."""
    sr = settings["screenRect"].astype(int)  # [x0, y0, x1, y1] or similar
    # exp/general.py and gratings/fix_rsvp/gaborium use (rows, cols) = (y, x)
    # screenRect convention: [x0, y0, x1, y1]
    return np.array([[sr[1], sr[3]], [sr[0], sr[2]]], dtype=int)


def _paste_on_screen(image, dest_rect, bkgnd, screen_shape):
    """Paste a destRect-sized image onto a full screen canvas filled with bkgnd."""
    H, W = screen_shape
    canvas = np.full((H, W), int(bkgnd), dtype=np.uint8)
    x0, y0, x1, y1 = [int(v) for v in dest_rect]
    h = y1 - y0
    w = x1 - x0
    if image.shape[:2] != (h, w):
        from PIL import Image as PILImage
        image = np.array(PILImage.fromarray(image).resize((w, h), resample=2))
    if image.ndim == 3:
        image = image.mean(axis=2).astype(np.uint8)
    # Clip to screen bounds
    y0c, y1c = max(0, y0), min(H, y1)
    x0c, x1c = max(0, x0), min(W, x1)
    sy0, sy1 = y0c - y0, h - (y1 - y1c)
    sx0, sx1 = x0c - x0, w - (x1 - x1c)
    canvas[y0c:y1c, x0c:x1c] = image[sy0:sy1, sx0:sx1]
    return canvas


def _render_backimage(exp, protocols):
    from DataYatesV1.exp.backimage import BackImageTrial
    idxs = [i for i, p in enumerate(protocols) if p == "BackImage"]
    if not idxs:
        return None
    # Pick a trial whose image is interesting (not all flat). Just take the
    # middle one.
    iT = idxs[len(idxs) // 2]
    trial = BackImageTrial(exp["D"][iT], exp["S"])
    img = trial.get_image()
    if img.ndim == 3:
        img = img.mean(axis=2).astype(np.uint8)
    screen_shape = _screen_shape_from_settings(exp["S"])
    canvas = _paste_on_screen(img, trial.dest_rect, trial.bkgnd, screen_shape)
    return canvas


def _render_gaborium(exp, protocols):
    from DataYatesV1.exp.gaborium import GaboriumTrial
    idxs = [i for i, p in enumerate(protocols) if p == "ForageGabor"]
    if not idxs:
        return None
    iT = idxs[len(idxs) // 2]
    trial = GaboriumTrial(exp["D"][iT], exp["S"], method="gpu")
    roi = _full_screen_rect(exp["S"])
    # Many gaborium trials' first frame is just background — find a frame
    # with actual probes by scanning the probe history.
    p_index = trial.p_index
    valid = np.where(np.isfinite(p_index) & (p_index != 0))[0]
    target_frame = int(valid[len(valid) // 2]) if len(valid) else 0
    frames = trial.get_frames([target_frame], roi=roi)
    return frames[0]


def _render_gratings(exp, protocols):
    """Render a full-screen low-SF, high-contrast grating.

    Scans every grating trial for the lowest non-zero spatial frequency and
    renders that one frame at contrast=1.0 so the bars are unambiguous in
    the schematic.
    """
    from DataYatesV1.exp.gratings import GratingsTrial
    idxs = [i for i, p in enumerate(protocols)
            if p in ("ForageGrating", "ForageDriftingGrating")]
    if not idxs:
        return None

    best = None    # (sf, trial_idx, frame_idx, ori)
    for iT in idxs:
        try:
            trial = GratingsTrial(exp["D"][iT], exp["S"])
        except Exception:
            continue
        sfs = np.asarray(trial.spatial_frequencies, dtype=float)
        mask = np.isfinite(sfs) & (sfs > 0)
        if not np.any(mask):
            continue
        local_min = sfs[mask].min()
        if best is None or local_min < best[0]:
            frame_idx = int(np.where((sfs == local_min) & mask)[0][0])
            best = (local_min, iT, frame_idx, float(trial.orientations[frame_idx]))

    if best is None:
        # Fallback to old midpoint behaviour
        iT = idxs[len(idxs) // 2]
        trial = GratingsTrial(exp["D"][iT], exp["S"])
        roi = _full_screen_rect(exp["S"])
        return trial.get_frames([trial.n_frames // 2], roi=roi)[0]

    sf, iT, frame_idx, ori = best
    print(f"    gratings: trial {iT}, frame {frame_idx}, "
          f"sf={sf:.3f} cpd, ori={ori:.1f}°")
    trial = GratingsTrial(exp["D"][iT], exp["S"])
    roi = _full_screen_rect(exp["S"])
    # Render directly via gen_grating so we can override the contrast.
    grating = trial.gen_grating(sf, ori, contrast=1.0, roi=roi)
    img = np.clip(grating + trial.bkgnd + 0.5, 0, 255).astype(np.uint8)
    return img


def _render_fixrsvp(exp, protocols):
    """Render a fixRSVP frame, picking a non-aversive image whose gaze
    position is as close to screen centre as possible.

    Returns
    -------
    image : np.ndarray
        Full-screen rendered frame.
    position_deg : np.ndarray
        (2,) gaze position in degrees relative to screen centre — used
        downstream to centre the ROI marker on the chosen face location.
    """
    from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
    idxs = [i for i, p in enumerate(protocols)
            if p == "FixRsvpStim" and FixRsvpTrial.is_valid(exp["D"][i])]
    if not idxs:
        return None, None

    roi = _full_screen_rect(exp["S"])
    # For each preferred image_id, scan every trial and rank candidates by
    # how close the gaze position is to screen centre. Use the closest.
    for desired_id in PREFERRED_FIXRSVP_IMAGE_IDS:
        candidates = []
        for iT in idxs:
            trial = FixRsvpTrial(exp["D"][iT], exp["S"])
            ids = np.asarray(trial.image_ids)
            positions = np.asarray(trial.positions)
            hits = np.where(ids == desired_id)[0]
            for h in hits:
                pos = positions[h]
                if not np.all(np.isfinite(pos)):
                    continue
                dist = float(np.hypot(pos[0], pos[1]))
                candidates.append((dist, iT, int(h), pos))
        if not candidates:
            continue
        candidates.sort(key=lambda c: c[0])
        dist, iT, target, pos = candidates[0]
        print(f"    fixrsvp: trial {iT}, frame {target}, image_id={desired_id}, "
              f"position=({pos[0]:+.2f},{pos[1]:+.2f})° (dist={dist:.2f}°)")
        trial = FixRsvpTrial(exp["D"][iT], exp["S"])
        image = np.asarray(trial.get_rois(target, roi=roi)).squeeze()
        return image, pos

    # Fallback
    iT = idxs[len(idxs) // 2]
    trial = FixRsvpTrial(exp["D"][iT], exp["S"])
    valid = np.where(trial.image_ids > 0)[0]
    target = int(valid[len(valid) // 2]) if len(valid) else 0
    pos = np.asarray(trial.positions[target])
    print(f"    fixrsvp: fallback trial {iT}, frame {target}")
    return np.asarray(trial.get_rois(target, roi=roi)).squeeze(), pos


def _screen_shape_from_settings(settings):
    sr = settings["screenRect"].astype(int)
    # (height, width)
    return (int(sr[3] - sr[1]), int(sr[2] - sr[0]))


# ----------------------------------------------------------------------------
# Representative ROI per stimulus type (in screen pixel coords)
# ----------------------------------------------------------------------------
def _representative_roi(dset_path: Path):
    """Return a (2, 2) int ROI = [[r0, r1], [c0, c1]] from a prepared .dset.

    Picks the most common ROI across frames (mode of the corner).
    """
    from models.data.datasets import DictDataset
    dset = DictDataset.load(str(dset_path))
    roi = np.asarray(dset["roi"])  # may be (N, 2, 2)
    if roi.ndim == 4:
        roi = roi.reshape(-1, 2, 2)
    elif roi.ndim != 3:
        raise ValueError(f"Unexpected ROI shape {roi.shape}")
    # Median corner — robust to outliers.
    med = np.median(roi, axis=0).astype(int)
    return med


# ----------------------------------------------------------------------------
# Lag cube
# ----------------------------------------------------------------------------
def _pick_lag_window(stim, trial_inds, n_lags):
    """Pick the highest-contrast contiguous `n_lags` window inside one trial.

    Deterministic (seeded). Returns `(chosen_trial, win_idxs)` where `win_idxs`
    are ascending (chronological) frame indices into `stim`. Shared by the
    moving lag cube and its stabilized counterpart so both describe the exact
    same frames.
    """
    unique_trials, counts = np.unique(trial_inds, return_counts=True)
    long_trials = unique_trials[counts >= n_lags]
    rng = np.random.default_rng(0)
    best = None
    best_var = -np.inf
    for t in long_trials:
        idxs = np.where(trial_inds == t)[0]
        if len(idxs) < n_lags:
            continue
        start = (len(idxs) - n_lags) // 2
        window = idxs[start:start + n_lags]
        var = stim[window].var()
        if var > best_var:
            best_var = var + rng.uniform(0, 1e-3)
            best = (int(t), window)
    if best is None:
        raise RuntimeError(f"No fixrsvp trial long enough for a {n_lags}-lag window")
    return best


def _lag_cube_from_fixrsvp(session_dir: Path, n_lags: int = 33):
    """Pull the exact stimulus context the model sees at one inference time.

    The training config uses 33 lags (`keys_lags.stim: [0..32]`) at 120 Hz —
    a 275 ms window. We pick a fixrsvp time bin `t` with ≥ n_lags-1 prior
    frames in the same trial and return frames[t-n_lags+1 : t+1].

    Returns (cube, lag_indices). cube[0] is the OLDEST frame (longest lag);
    cube[-1] is the most recent (lag 0).
    """
    from models.data.datasets import DictDataset
    dset = DictDataset.load(str(session_dir / "datasets" / "fixrsvp.dset"))
    stim = np.asarray(dset["stim"])  # (N, H, W) — one ROI snapshot per time bin

    if stim.ndim != 3:
        raise ValueError(f"Unexpected stim shape {stim.shape}")

    trial_inds = np.asarray(dset.covariates["trial_inds"]).ravel().astype(int)
    chosen_trial, win_idxs = _pick_lag_window(stim, trial_inds, n_lags)
    cube = stim[win_idxs]
    lag_indices = list(range(n_lags))
    print(f"    lag cube: trial {chosen_trial}, {n_lags} frames "
          f"({n_lags / 120 * 1000:.0f} ms at 120 Hz)")
    return cube.astype(np.float32), lag_indices


def _stabilized_lag_cube_from_fixrsvp(sess, n_lags: int = 33):
    """Reafferent-ablated ("stabilized") counterpart of the moving lag cube.

    Re-renders the SAME window of RSVP frames with the retinal ROI held fixed, so
    image flashes still update mid-window but the gaze-induced retinal motion is
    removed. This is a single-trial illustration of the "extraretinal only
    (stabilized)" manipulation quantified in panels C/D/E; for legibility the ROI
    is frozen at THIS example trial's own medoid gaze, which keeps the flashed
    image centred in the crop. The quantified ablation instead freezes every trial
    at the session-global centroid (`build_stabilized_stim`); anchoring the single
    panel trial there would crop away from its own stimulus and read as mostly
    blank, so the picture uses the trial-local medoid.

    Returns a (n_lags, H, W) float32 cube ordered oldest → newest, matching
    `_lag_cube_from_fixrsvp`.
    """
    from models.data.datasets import DictDataset
    from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
    from DataYatesV1.utils.general import get_clock_functions
    from _fig3_ablation_data import FIX_RADIUS

    exp = sess.exp
    dset = DictDataset.load(str(sess.sess_dir / "datasets" / "fixrsvp.dset"))
    raw_stim = np.asarray(dset["stim"])
    trial_inds = np.asarray(dset.covariates["trial_inds"]).ravel().astype(int)
    t_bins = np.asarray(dset.covariates["t_bins"]).ravel()
    roi_all = np.asarray(dset["roi"])
    dpi_pix = np.asarray(dset.covariates["dpi_pix"])
    dpi_valid = np.asarray(dset.covariates["dpi_valid"]).ravel() > 0
    eyepos = np.asarray(dset["eyepos"])
    fixation = np.hypot(eyepos[:, 0], eyepos[:, 1]) < FIX_RADIUS

    chosen_trial, win_idxs = _pick_lag_window(raw_stim, trial_inds, n_lags)
    m = trial_inds == chosen_trial
    valid = m & fixation & dpi_valid
    if valid.sum() == 0:
        raise RuntimeError(
            f"stabilized cube: no valid fixation frames in trial {chosen_trial}")

    # Illustration ROI: the example trial's own medoid gaze (sample minimizing
    # summed distance to the trial's other valid gazes), which keeps the flashed
    # image centred in the frozen crop. Schematic-only; the quantified ablation
    # (C/D/E) freezes at the session-global centroid instead.
    vidx = np.where(valid)[0]
    d = np.sqrt(((dpi_pix[vidx][:, None] - dpi_pix[vidx][None]) ** 2).sum(-1)).sum(1)
    med = int(vidx[np.argmin(d)])

    trial = FixRsvpTrial(exp["D"][chosen_trial], exp["S"])
    ptb2ephys, _ = get_clock_functions(exp)
    start_idx = np.where(trial.image_ids == 2)[0][0]
    flip_times = ptb2ephys(trial.flip_times[start_idx:])
    hist_idx = np.searchsorted(flip_times, t_bins[m], side="right") - 1 + start_idx
    roi_const = np.repeat(roi_all[med][None], m.sum(), axis=0)
    stab_trial = np.asarray(trial.get_rois(hist_idx, roi=roi_const))

    pos = np.searchsorted(np.where(m)[0], win_idxs)
    stab_cube = stab_trial[pos].astype(np.float32)
    print(f"    stabilized lag cube: trial {chosen_trial}, ROI frozen at this "
          f"trial's medoid gaze (flashes preserved, reafferent motion removed)")
    return stab_cube


def _train_model_input_and_raster(session_dir: Path, *, n_lags: int = 33,
                                  n_future: int = 30, fs: float = 120.0):
    """Build panel A's natural-image training objective from the pinned
    backimage free-viewing window.

    Returns the model input the twin actually consumes at one inference time
    (a space × space × time crop of the gaze-contingent stimulus history plus
    the eye-position/velocity covariates) together with the observed
    spike-count raster (units × time) it is trained to predict. The highlighted
    raster column is the cube's current (newest) frame — the single timepoint
    the shown input window predicts.

    The window is centred on the largest saccade in the pinned segment (placed
    at the middle of the lag cube) so the post-saccadic response transient it
    drives is visible in the prediction-target raster. Everything is sourced
    from the same `PINNED_FREEVIEW` window drawn on the natural-image training
    screen, so the gaze trace, the magnified ROI crop, and the predicted spikes
    all describe one moment in one free-viewing trial.
    """
    from models.data.datasets import DictDataset
    dset = DictDataset.load(str(session_dir / "datasets" / "backimage.dset"))
    stim = np.asarray(dset["stim"])                      # (N, H, W)
    robs = np.asarray(dset["robs"])                      # (N, n_units)
    eyepos = np.asarray(dset["eyepos"])                  # (N, 2) deg
    roi_all = np.asarray(dset["roi"])                    # (N, 2, 2)
    trial_inds = np.asarray(dset.covariates["trial_inds"]).ravel().astype(int)

    if PINNED_FREEVIEW is None:
        raise RuntimeError("train model input needs PINNED_FREEVIEW set")
    pin_trial, pin_start = PINNED_FREEVIEW
    win_n = int(5.0 * fs)                                # match the displayed 5 s trace
    if pin_start + win_n > len(eyepos):
        raise RuntimeError(
            f"PINNED_FREEVIEW window {pin_start}:{pin_start + win_n} exceeds "
            f"backimage.dset (len={len(eyepos)})")
    seg_trials = np.unique(trial_inds[pin_start:pin_start + win_n])
    if not (len(seg_trials) == 1 and seg_trials[0] == pin_trial):
        raise RuntimeError(
            f"pinned window spans trials {seg_trials}; expected only {pin_trial}")

    # Hard-coded current/predicted frame (local index within the pinned window),
    # chosen so a high-contrast saccade sits early in the displayed window and
    # its post-saccadic transient is visible across the raster. The lag cube is
    # the 33-frame history ending here; the raster spans the same history plus a
    # few future bins.
    t = TRAIN_CUR_FRAME_LOCAL

    g0 = pin_start + t - n_lags + 1                      # global start of cube
    cube = stim[g0:g0 + n_lags].astype(np.float32)       # (n_lags, H, W) oldest→newest
    cur_roi = roi_all[pin_start + t].astype(int)         # (2, 2)

    r0 = g0                                              # raster window start (global)
    r1 = pin_start + t + n_future + 1                    # exclusive
    raster = np.asarray(robs[r0:r1], dtype=np.float32).T  # (n_units, n_bins), natural order

    # Behavior covariates over the raster window (eye position x/y + speed).
    ep_win = eyepos[r0:r1].astype(np.float32)
    dxy2 = np.diff(ep_win, axis=0, prepend=ep_win[:1])
    speed_win = (np.linalg.norm(dxy2, axis=1) * fs).astype(np.float32)
    t_axis = (np.arange(r1 - r0) / fs).astype(np.float32)

    print(f"    train model input: pinned trial {pin_trial}, current frame "
          f"t={t} ({n_lags} lags + {n_future} future bins, raster "
          f"{raster.shape[0]}×{raster.shape[1]})")
    return {
        "train_lag_cube": cube,
        "train_cur_roi_px": cur_roi,
        "train_behavior_t": t_axis,
        "train_behavior_eyepos": ep_win,
        "train_behavior_speed": speed_win,
        "train_raster": raster,
    }


def _align_cube_last_frame(cube, screen, roi):
    """Replace `cube[-1]` (the most-recent frame = cube front face) with the
    resized test-screen ROI patch so the cube's current frame is visually
    identical to what's on the test screen. In place; returns `cube`."""
    from PIL import Image as PILImage
    r0, r1 = int(roi[0, 0]), int(roi[0, 1])
    c0, c1 = int(roi[1, 0]), int(roi[1, 1])
    r0, r1 = max(0, r0), min(screen.shape[0], r1)
    c0, c1 = max(0, c0), min(screen.shape[1], c1)
    roi_patch = screen[r0:r1, c0:c1]
    th, tw = cube.shape[1], cube.shape[2]
    if roi_patch.size > 0:
        resized = np.asarray(PILImage.fromarray(roi_patch).resize(
            (tw, th), PILImage.BILINEAR), dtype=cube.dtype)
        cube[-1] = resized
    return cube


def _real_freeview_trace(session_dir, pix_per_deg, screen_shape, *,
                         fs=120.0, target_s=5.0, max_abs_deg=7.0):
    """Pull a free-viewing eye-position segment from backimage.dset.

    If ``PINNED_FREEVIEW`` is set, returns the pinned (trial, start_global)
    window directly without scoring. Otherwise scans all trials and picks
    the highest-scoring fixation-heavy window (legacy auto-pick).
    """
    from models.data.datasets import DictDataset
    dset_path = session_dir / "datasets" / "backimage.dset"
    if not dset_path.exists():
        return None
    dset = DictDataset.load(str(dset_path))
    eyepos = np.asarray(dset["eyepos"])
    roi_seq_full = np.asarray(dset["roi"])      # (N, 2, 2)
    trial_inds = np.asarray(dset.covariates["trial_inds"]).ravel().astype(int)
    if "dpi_valid" in dset.covariates:
        valid = np.asarray(dset.covariates["dpi_valid"]).ravel().astype(bool)
    else:
        valid = np.ones(len(eyepos), dtype=bool)

    target_n = int(target_s * fs)
    H, W = screen_shape
    cx_pix, cy_pix = W / 2.0, H / 2.0

    def _to_px(ep_seg):
        return np.column_stack([
            ep_seg[:, 0] * pix_per_deg + cx_pix,
            -ep_seg[:, 1] * pix_per_deg + cy_pix,
        ]).astype(np.float32)

    if PINNED_FREEVIEW is not None:
        pin_trial, pin_start = PINNED_FREEVIEW
        end = pin_start + target_n
        if end > len(eyepos):
            raise RuntimeError(
                f"PINNED_FREEVIEW ({pin_trial}, {pin_start}) extends past "
                f"backimage.dset (len={len(eyepos)})")
        seg_trials = np.unique(trial_inds[pin_start:end])
        if not (len(seg_trials) == 1 and seg_trials[0] == pin_trial):
            raise RuntimeError(
                f"PINNED_FREEVIEW trial mismatch: pin says trial={pin_trial} "
                f"but samples {pin_start}:{end} cover trials {seg_trials}")
        ep_seg = eyepos[pin_start:end]
        roi_seg = roi_seq_full[pin_start:end].astype(np.int32)
        print(f"    real trace: pinned trial={pin_trial} start={pin_start} "
              f"({target_n} samples, {target_n/fs:.1f} s)")
        return _to_px(ep_seg), roi_seg

    # Velocity threshold separating fixation from saccade (deg/s).
    fixation_vel_thresh = 20.0  # samples below this count as fixations

    best = None     # (score, ep_seg)
    for t in np.unique(trial_inds):
        mask = trial_inds == int(t)
        idxs = np.where(mask)[0]
        if len(idxs) < target_n + 20:
            continue
        ep = eyepos[idxs]
        vld = valid[idxs]
        # Slide a window with stride 30 samples (0.25 s) across the trial.
        stride = 30
        for start in range(0, len(idxs) - target_n + 1, stride):
            sl = slice(start, start + target_n)
            if not vld[sl].all():
                continue
            window = ep[sl]
            if (np.abs(window) > max_abs_deg).any():
                continue
            # Instantaneous gaze velocity (deg/s).
            dxy = np.diff(window, axis=0, prepend=window[:1])
            speed = np.linalg.norm(dxy, axis=1) * fs
            fix_frac = float(np.mean(speed < fixation_vel_thresh))
            if fix_frac < 0.85:    # require ≥85% of samples in fixation
                continue
            spread = float(window.std(axis=0).mean())
            if spread < 0.8 or spread > 4.5:
                continue
            # Prefer high fixation fraction (clean look) with moderate spread.
            score = fix_frac * 10.0 + spread * 0.3
            if best is None or score > best[0]:
                best = (score, window.copy(),
                        roi_seq_full[idxs[sl]].copy())

    if best is None:
        return None
    score, ep_seg, roi_seg = best
    print(f"    real trace: {len(ep_seg)} samples ({len(ep_seg)/fs:.1f} s), "
          f"score={score:.2f}")
    return _to_px(ep_seg), roi_seg.astype(np.int32)


def _synth_freeview_trace(screen_shape, pix_per_deg, *, fs=120.0,
                          n_fixations=8, seed=2):
    """Synthesise a plausible free-viewing scan-path in screen-pixel coords.

    A real free-viewing trace from BackImage data would be ideal, but the
    backimage dataset stores residual FEM relative to the gaze, not absolute
    gaze position. This synthesises a scan: a sequence of fixation targets
    spread across the central portion of the screen, with each fixation
    carrying low-amplitude FEM drift (~0.15°), connected by short ballistic
    saccades.
    """
    H, W = screen_shape
    rng = np.random.default_rng(seed)
    cx, cy = W / 2.0, H / 2.0
    # Place fixation targets in a roughly ±6° box around the screen centre,
    # distributed loosely so the scan covers the image.
    # Spread targets across most of the central image area (±8°) so the
    # scan visibly covers a wide swath of the natural image.
    radius_px = 8.0 * pix_per_deg
    targets = []
    for k in range(n_fixations):
        # Stratify angles around the circle so the path traverses the
        # image rather than clustering.
        base_ang = 2 * np.pi * (k + rng.uniform(-0.15, 0.15)) / n_fixations
        rad = rng.uniform(0.45, 1.0) * radius_px
        targets.append((cx + rad * np.cos(base_ang),
                        cy + rad * np.sin(base_ang)))

    fem_amp_px = 0.18 * pix_per_deg
    n_fix_pts = int(0.35 * fs)
    n_sac_pts = max(4, int(0.05 * fs))
    pts = []
    for i, (tx, ty) in enumerate(targets):
        # FEM drift (low-pass random walk) anchored at the target
        wx = rng.standard_normal(n_fix_pts)
        wy = rng.standard_normal(n_fix_pts)
        kernel = np.exp(-np.arange(15) / 4.0)
        kernel /= kernel.sum()
        dx = np.convolve(wx, kernel, mode="same") * fem_amp_px
        dy = np.convolve(wy, kernel, mode="same") * fem_amp_px
        for x, y in zip(tx + dx, ty + dy):
            pts.append((x, y))
        if i < len(targets) - 1:
            ntx, nty = targets[i + 1]
            for k in range(n_sac_pts):
                a = (k + 1) / n_sac_pts
                pts.append((tx * (1 - a) + ntx * a,
                            ty * (1 - a) + nty * a))
    return np.asarray(pts, dtype=np.float32)


def _behavior_segment(session_dir: Path, fs: float = 120.0,
                      window_s: float = 0.5):
    """Extract a representative fixation segment: eye position (x,y) and speed."""
    from models.data.datasets import DictDataset
    dset = DictDataset.load(str(session_dir / "datasets" / "fixrsvp.dset"))
    eyepos = np.asarray(dset["eyepos"])
    roi_seq_full = np.asarray(dset["roi"])      # (N, 2, 2)
    trial_inds = np.asarray(dset.covariates["trial_inds"]).ravel().astype(int)

    # Pick a long, well-fixated trial: small eye-position range, ≥ window_s of data.
    target_n = int(window_s * fs)
    unique = np.unique(trial_inds)
    best = None
    best_score = -np.inf
    rng = np.random.default_rng(1)
    for t in unique:
        mask = trial_inds == t
        n = mask.sum()
        if n < target_n:
            continue
        ep = eyepos[mask]
        # Prefer trials with realistic FEM scale (not pure noise): std in [0.05, 0.5]°
        sd = ep.std(axis=0).mean()
        if not (0.05 < sd < 0.5):
            continue
        # Score: combination of length and small but nonzero motion
        score = n - 100 * abs(sd - 0.15) + rng.uniform(0, 1)
        if score > best_score:
            best_score = score
            best = t
    if best is None:
        best = unique[0]
    mask = trial_inds == int(best)
    sample_idxs = np.where(mask)[0][:target_n]
    ep = eyepos[sample_idxs]
    roi_seg = roi_seq_full[sample_idxs].astype(np.int32)
    t = np.arange(len(ep)) / fs
    # Speed (deg/s) — finite difference
    dxy = np.diff(ep, axis=0, prepend=ep[:1])
    speed = np.linalg.norm(dxy, axis=1) * fs
    return t, ep.astype(np.float32), speed.astype(np.float32), roi_seg


# ----------------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------------
def load_panel_a_assets(recompute: bool = False) -> PanelAAssets:
    if PANEL_A_CACHE_PATH.exists() and not recompute:
        print(f"Loading fig3a assets from {PANEL_A_CACHE_PATH}")
        with open(PANEL_A_CACHE_PATH, "rb") as f:
            assets = dill.load(f)
        # One-time backfills so existing caches gain fields added after they
        # were written, without a full raw-session recompute.
        dirty = False
        if "n_trained_units" not in assets.arch:
            assets.arch["n_trained_units"] = _load_n_trained_units()
            dirty = True
            print(f"  backfilled n_trained_units="
                  f"{assets.arch['n_trained_units']} into cache")
        if getattr(assets, "psth_neurons", None) is None:
            print("  backfilling best-ccnorm PSTHs into cache...")
            assets.psth_neurons = _load_best_ccnorm_psths(n=3)
            dirty = True
        if getattr(assets, "stab_lag_cube", None) is None:
            print("  backfilling stabilized (reafferent-ablated) lag cube...")
            from DataYatesV1.utils.io import get_session
            subj, dt = assets.session.split("_")
            _sess = get_session(subj, dt)
            stab = _stabilized_lag_cube_from_fixrsvp(_sess)
            if "fixrsvp" in assets.screens and "fixrsvp" in assets.rois:
                stab = _align_cube_last_frame(
                    stab, assets.screens["fixrsvp"], assets.rois["fixrsvp"])
            assets.stab_lag_cube = stab
            dirty = True
        if getattr(assets, "train_raster", None) is None:
            print("  backfilling natural-image model input + prediction-target "
                  "raster...")
            from DataYatesV1.utils.io import get_session
            subj, dt = assets.session.split("_")
            _sess = get_session(subj, dt)
            for k, v in _train_model_input_and_raster(_sess.sess_dir).items():
                setattr(assets, k, v)
            dirty = True
        if dirty:
            with open(PANEL_A_CACHE_PATH, "wb") as f:
                dill.dump(assets, f)
        return assets

    from DataYatesV1.utils.io import get_session
    from DataYatesV1.exp.general import get_trial_protocols

    print(f"Loading raw session {PANEL_B_SESSION} for fig3a assets...")
    subject, date = PANEL_B_SESSION.split("_")
    sess = get_session(subject, date)
    if sess is None:
        raise RuntimeError(f"Could not load session {PANEL_B_SESSION}")
    exp = sess.exp
    protocols = get_trial_protocols(exp)
    screen_shape = _screen_shape_from_settings(exp["S"])
    pix_per_deg = float(exp["S"]["pixPerDeg"])

    print("  rendering backimage...")
    screens = {"backimage": _render_backimage(exp, protocols)}
    print("  rendering gaborium...")
    screens["gaborium"] = _render_gaborium(exp, protocols)
    print("  rendering gratings...")
    screens["gratings"] = _render_gratings(exp, protocols)
    print("  rendering fixrsvp...")
    fixrsvp_render = _render_fixrsvp(exp, protocols)
    fixrsvp_position_deg = None
    if fixrsvp_render is not None:
        fixrsvp_img, fixrsvp_position_deg = fixrsvp_render
        if fixrsvp_img is not None:
            screens["fixrsvp"] = fixrsvp_img

    # Drop None entries if any stimulus type is missing in this session.
    screens = {k: v for k, v in screens.items() if v is not None}

    print("  extracting representative ROIs...")
    sess_dir = sess.sess_dir
    rois = {}
    for stim_type in screens.keys():
        dset_path = sess_dir / "datasets" / f"{stim_type}.dset"
        if dset_path.exists():
            rois[stim_type] = _representative_roi(dset_path)

    # For fixrsvp: keep the dset ROI's *size* (it's what the model sees) but
    # re-centre it on the chosen frame's gaze position so the cyan marker
    # sits over the rendered face.
    if "fixrsvp" in rois and fixrsvp_position_deg is not None:
        r0, r1 = rois["fixrsvp"][0]
        c0, c1 = rois["fixrsvp"][1]
        h_px = int(r1 - r0)
        w_px = int(c1 - c0)
        center_pix = np.asarray(exp["S"]["centerPix"], dtype=float).ravel()
        # screen y axis: row index increases downward; +y deg = up → subtract.
        face_col = int(round(center_pix[0] + fixrsvp_position_deg[0] * pix_per_deg))
        face_row = int(round(center_pix[1] - fixrsvp_position_deg[1] * pix_per_deg))
        new_roi = np.array([
            [face_row - h_px // 2, face_row - h_px // 2 + h_px],
            [face_col - w_px // 2, face_col - w_px // 2 + w_px],
        ], dtype=int)
        rois["fixrsvp"] = new_roi
        print(f"    fixrsvp ROI recentered on chosen frame: rows {new_roi[0]}, cols {new_roi[1]}")

    print("  loading lag-cube from fixrsvp.dset...")
    cube, lag_idx = _lag_cube_from_fixrsvp(sess_dir, n_lags=33)

    print("  rendering stabilized (reafferent-ablated) lag cube...")
    stab_cube = _stabilized_lag_cube_from_fixrsvp(sess)

    # Align both cubes' most-recent frame (last in the lag stack) with the actual
    # test-screen ROI content, so their shared "current frame" (the cube front
    # face) is visually identical to what's on the test screen and to each other.
    if "fixrsvp" in screens and "fixrsvp" in rois:
        cube = _align_cube_last_frame(cube, screens["fixrsvp"], rois["fixrsvp"])
        stab_cube = _align_cube_last_frame(stab_cube, screens["fixrsvp"],
                                           rois["fixrsvp"])
        print(f"    cube[-1] aligned to test-screen ROI ({cube.shape[1]}×"
              f"{cube.shape[2]}) for moving + stabilized cubes")

    print("  extracting behavior segment...")
    beh_t, beh_eye, beh_speed, beh_roi_seq = _behavior_segment(sess_dir)

    print("  building natural-image model input + prediction-target raster...")
    train_objective = _train_model_input_and_raster(sess_dir)

    print("  extracting free-viewing eye trace...")
    freeview_result = _real_freeview_trace(sess_dir, pix_per_deg, screen_shape)
    if freeview_result is None:
        print("    (real trace unavailable — falling back to synthesis)")
        freeview_trace_px = _synth_freeview_trace(screen_shape, pix_per_deg)
        freeview_roi_seq = np.zeros((len(freeview_trace_px), 2, 2), dtype=np.int32)
    else:
        freeview_trace_px, freeview_roi_seq = freeview_result

    arch = _load_arch_info()
    arch["n_trained_units"] = _load_n_trained_units()

    print("  loading frontend weights from checkpoint...")
    frontend_weights = _load_frontend_weights()
    print("  loading readout example neuron...")
    rd_mean, rd_std, rd_feats = _load_readout_example()
    print("  loading example neurons (readouts + trace snippets)...")
    example_neurons = _load_example_neurons(n=3)
    print("  loading best-ccnorm PSTH neurons...")
    psth_neurons = _load_best_ccnorm_psths(n=3)

    assets = PanelAAssets(
        session=PANEL_B_SESSION,
        screen_shape=screen_shape,
        pix_per_deg=pix_per_deg,
        screens=screens,
        rois=rois,
        lag_cube=cube,
        lag_indices=lag_idx,
        stab_lag_cube=stab_cube,
        arch=arch,
        frontend_weights=frontend_weights,
        readout_mean=rd_mean,
        readout_std=rd_std,
        readout_features=rd_feats,
        behavior_t=beh_t,
        behavior_eyepos=beh_eye,
        behavior_speed=beh_speed,
        behavior_roi_seq_px=beh_roi_seq,
        freeview_trace_px=freeview_trace_px,
        freeview_roi_seq_px=freeview_roi_seq,
        example_neurons=example_neurons,
        psth_neurons=psth_neurons,
        **train_objective,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(PANEL_A_CACHE_PATH, "wb") as f:
        dill.dump(assets, f)
    print(f"Cached fig3a assets to {PANEL_A_CACHE_PATH}")
    return assets


if __name__ == "__main__":
    import sys
    recompute = "--recompute" in sys.argv
    a = load_panel_a_assets(recompute=recompute)
    print(f"\nSession: {a.session}")
    print(f"Screen shape: {a.screen_shape}")
    print(f"Screens: {list(a.screens.keys())}")
    for k, v in a.rois.items():
        print(f"  ROI[{k}]: {v.tolist()}")
    print(f"Lag cube shape: {a.lag_cube.shape}")
    print(f"Arch: {a.arch}")
