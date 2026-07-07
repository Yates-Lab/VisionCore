"""Shared primitives for the population-geometry analysis.

Builds a co-centered in-silico foveal population from the fig4 digital twin:
every selected real unit is instantiated with its learned feature selectivity
and Gaussian width but re-centered at the ROI center (mean -> 0), so the whole
population reads out from one common retinotopic location.

Reuses the driving substrate from ryan/digital-twin-fem/_common.py (stimulus
shifting, PopulationReadout, rate-map compute) but pins the fig4 checkpoint and
provides selection + centered-population construction.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

# Make the digital-twin-fem primitives importable (hyphenated dir).
_DTFEM = VISIONCORE_ROOT / "ryan" / "digital-twin-fem"
if str(_DTFEM) not in sys.path:
    sys.path.insert(0, str(_DTFEM))

from _common import (  # noqa: E402  (reused substrate)
    PopulationReadout, compute_rate_map_batched, make_counterfactual_stim,
    extract_fixrsvp_eye_traces, _zero_behavior,
    PPD, N_LAGS, DT, IMAGE_SHAPE, OUT_SIZE,
)

# fig4 checkpoint (epoch=374, val_bps=0.6395) — the canonical digital-twin model.
CHECKPOINT_DIR = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120"
CHECKPOINT_SUBDIR = "2026-03-31_11-33-32_learned_resnet_concat_convgru_gaussian"
EXPERIMENT_SUBDIR = "learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4"
BEST_CKPT = "epoch=374-val_bps_overall=0.6395.ckpt"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{CHECKPOINT_SUBDIR}/{EXPERIMENT_SUBDIR}/{BEST_CKPT}"

OUT_DIR = FIGURES_DIR / "population_geometry"
OUT_DIR.mkdir(parents=True, exist_ok=True)
INVENTORY_CSV = OUT_DIR / "unit_inventory.csv"

# Population selection thresholds (chosen from Step 1.1 distributions).
CONTAM_MAX = 10.0     # contamination % (min rejectable RPV)
CCNORM_MIN = 0.5      # twin fidelity
RATE_MIN = 1.0        # Hz


def load_twin(device=None):
    """Load the fig4 digital twin. Returns (model, info, device)."""
    from DataYatesV1 import get_free_device
    from eval.eval_stack_multidataset import load_model
    if device is None:
        device = get_free_device()
    model, info = load_model(checkpoint_path=CHECKPOINT_PATH, device=str(device))
    model.model.eval()
    model.model.convnet.use_checkpointing = False
    return model, info, device


def select_units(contam_max=CONTAM_MAX, ccnorm_min=CCNORM_MIN, rate_min=RATE_MIN,
                 inventory_csv=INVENTORY_CSV) -> pd.DataFrame:
    """Return the selected-unit table (one row per synthetic neuron)."""
    df = pd.read_csv(inventory_csv)
    m = ((df.contam_pct < contam_max) & (df.ccnorm > ccnorm_min)
         & (df.firing_rate > rate_min))
    sel = df[m].reset_index(drop=True)
    print(f"Selected {len(sel)} units / {sel.session.nunique()} sessions "
          f"(contam<{contam_max}, ccnorm>{ccnorm_min}, rate>{rate_min})")
    return sel


@dataclass
class CenteredPopulation:
    readout: torch.nn.Module          # PopulationReadout -> (T, N, 1, 1)
    session: np.ndarray               # (N,) originating session per neuron
    readout_idx: np.ndarray           # (N,) readout-unit index within that session
    feat_size: tuple[int, int]        # (H_feat, W_feat) of the core output
    displacement: np.ndarray          # (N,) |native mean| in grid units (pre-centering)
    N: int


def _probe_feat_size(model):
    device = model.device
    dtype = next(model.model.parameters()).dtype
    with torch.no_grad():
        dummy = torch.zeros(1, 1, N_LAGS, OUT_SIZE[0], OUT_SIZE[1],
                            device=device, dtype=dtype)
        beh = _zero_behavior(model, 1, device, dtype)
        core_out = model.model.core_forward(dummy, beh)
    return int(core_out.shape[-2]), int(core_out.shape[-1])


def build_centered_population(model, units: pd.DataFrame,
                              center: bool = True) -> CenteredPopulation:
    """Assemble a population readout from selected units.

    Each unit keeps its learned feature weights + Gaussian (std, theta). With
    center=True the readout is re-centered (mean -> 0) at full feature-map
    resolution so PopulationReadout's valid conv collapses to a single ROI-center
    readout per unit; with center=False the unit's native position is kept (for
    quantifying the co-centering perturbation).
    """
    H_feat, W_feat = _probe_feat_size(model)
    feat_w, biases, space_w = [], [], []
    sessions, readout_idxs, disp = [], [], []

    for sess_name, grp in units.groupby("session", sort=False):
        if sess_name not in model.names:
            print(f"  skip {sess_name}: not in model.names")
            continue
        ridx = model.names.index(sess_name)
        ro = model.model.readouts[ridx]
        cids = grp.readout_idx.to_numpy()

        feat_w.append(ro.features.weight.detach().cpu()[cids])
        biases.append(ro.bias.detach().cpu()[cids])

        native_mean = ro.mean.data.clone()
        disp.append(np.linalg.norm(native_mean.detach().cpu().numpy()[cids], axis=1))
        if center:
            ro.mean.data.zero_()
        mask = ro.compute_gaussian_mask(H_feat, W_feat, model.device).detach().cpu()[cids]
        ro.mean.data.copy_(native_mean)          # restore
        space_w.append(mask)

        sessions.extend([sess_name] * len(cids))
        readout_idxs.extend(int(c) for c in cids)

    feat_w = torch.cat(feat_w, dim=0)
    biases = torch.cat(biases, dim=0)
    space_w = torch.cat(space_w, dim=0)          # (N, H_feat, W_feat) -> full kernel
    readout = PopulationReadout(feat_w, biases, space_w)
    N = feat_w.shape[0]
    return CenteredPopulation(
        readout=readout,
        session=np.array(sessions),
        readout_idx=np.array(readout_idxs, dtype=int),
        feat_size=(H_feat, W_feat),
        displacement=np.concatenate(disp),
        N=N,
    )


def population_rates(model, pop: CenteredPopulation, image: np.ndarray,
                     eye_trace: np.ndarray) -> np.ndarray:
    """Return (T+1, N) predicted rates (pre-Poisson) for a static patch + eye trace.

    image: (H, W) full-field patch (0..255). eye_trace: (T, 2) deg (drift). The
    patch is broadcast over time and gaze-shifted by the trace, mirroring
    generate_trials but returning rates instead of Poisson samples.
    """
    T = eye_trace.shape[0]
    full_stack = np.broadcast_to(image[None, :, :],
                                 (T + N_LAGS + 1, *image.shape)).copy()
    trace = torch.from_numpy(eye_trace).float()
    eye_stim = make_counterfactual_stim(full_stack, trace, ppd=PPD, scale_factor=1.0,
                                        n_lags=N_LAGS, out_size=OUT_SIZE)
    stim_norm = (eye_stim - 127.0) / 255.0
    readout_dev = pop.readout.to(model.device)
    rate_map = compute_rate_map_batched(model, readout_dev, stim_norm)  # (T+1, N, 1, 1)
    return rate_map[:, :, 0, 0].detach().cpu().numpy()


def static_response(model, pop: CenteredPopulation, image: np.ndarray,
                    settle: int | None = None) -> np.ndarray:
    """Steady-state (N,) rate to a static full-field image (no eye movement).

    The eye trace must be at least N_LAGS long for the lag embedding; a static
    (zero-motion) trace of that length lets the ConvGRU settle. Returns the mean
    of the last few output frames.
    """
    if settle is None:
        settle = N_LAGS
    settle = max(settle, N_LAGS)
    rates = population_rates(model, pop, image, np.zeros((settle, 2), dtype=np.float32))
    return rates[-3:].mean(axis=0)


# ---------------------------------------------------------------------------
# Behavior reconstruction for synthetic drift (see note_synthetic_behavior.md)
# Replays the eye_vel op-chain from multi_basic_120_long.yaml, substituting the
# per-array maxnorm with a scale imputed from pooled real fixRSVP velocities.
# ---------------------------------------------------------------------------
BASIS_CFG = dict(num_delta_funcs=0, num_cosine_funcs=10, history_bins=50,
                 causal=False, log_spacing=False, peak_range_ms=[30, 200],
                 normalize=True)
SPLITRELU_CFG = dict(split_dim=1, trainable_gain=False)
_basis_fn = None
_sr_fn = None
_VEL_SCALE_CACHE = OUT_DIR / "vel_scale.npy"


def _ensure_transform_fns():
    global _basis_fn, _sr_fn
    if _basis_fn is None:
        from models.data.transforms import _make_basis, _make_splitrelu
        _basis_fn = _make_basis(BASIS_CFG)
        _sr_fn = _make_splitrelu(SPLITRELU_CFG)


def estimate_vel_scale(model, quantile: float = 0.999, cache: bool = True) -> float:
    """Impute the maxnorm velocity scale from pooled real fixRSVP drift traces.

    maxnorm divides velocity by max|component|; we use a high quantile of pooled
    |component| velocities across real fixations as a robust stand-in.
    """
    if cache and _VEL_SCALE_CACHE.exists():
        return float(np.load(_VEL_SCALE_CACHE))
    traces, _ = extract_fixrsvp_eye_traces(model)
    comps = []
    for tr in traces:
        n = int(np.isfinite(tr[:, 0]).sum())
        if n > 2:
            comps.append(np.abs(np.diff(tr[:n], axis=0)).ravel())
    scale = float(np.quantile(np.concatenate(comps), quantile))
    if cache:
        np.save(_VEL_SCALE_CACHE, np.array(scale))
    print(f"imputed velocity maxnorm scale (q{quantile}): {scale:.5f} deg/frame")
    return scale


_KAPPA_CACHE = OUT_DIR / "kappa.npy"


def get_drift_params(model, fixation_radius_deg: float = 0.5,
                     cache: bool = True) -> tuple[float, float]:
    """Return (kappa deg^2/s, vel_scale) measured from real fixRSVP drift, cached."""
    if cache and _KAPPA_CACHE.exists() and _VEL_SCALE_CACHE.exists():
        return float(np.load(_KAPPA_CACHE)), float(np.load(_VEL_SCALE_CACHE))
    from _drift import estimate_kappa
    traces, _ = extract_fixrsvp_eye_traces(model, fixation_radius_deg=fixation_radius_deg)
    kappa, _, _ = estimate_kappa(traces, dt=DT, fit_lags=10)
    comps = [np.abs(np.diff(t[:int(np.isfinite(t[:, 0]).sum())], axis=0)).ravel()
             for t in traces if np.isfinite(t[:, 0]).sum() > 2]
    vel_scale = float(np.quantile(np.concatenate(comps), 0.999))
    if cache:
        np.save(_KAPPA_CACHE, np.array(kappa))
        np.save(_VEL_SCALE_CACHE, np.array(vel_scale))
    return kappa, vel_scale


def build_behavior(eyepos_deg: np.ndarray, vel_scale: float) -> torch.Tensor:
    """Reconstruct the (L, behavior_dim) behavior tensor from an eye trace (deg).

    eye_vel = splitrelu(temporal_basis(symlog(diff(eyepos)/vel_scale)));
    behavior = concat([eye_vel, eye_pos]).
    """
    _ensure_transform_fns()
    x = torch.from_numpy(np.asarray(eyepos_deg, dtype=np.float32))
    vel = torch.diff(x, dim=0, prepend=x[:1])
    vel = vel / (vel_scale + 1e-8)
    vel = torch.sign(vel) * torch.log1p(torch.abs(vel))
    with torch.no_grad():
        ev = _sr_fn(_basis_fn(vel))          # (L, C'')
    return torch.cat([ev, x], dim=1)         # eye_vel ++ eye_pos


def _behavior_for_frames(eye_trace: np.ndarray, vel_scale: float) -> torch.Tensor:
    """Behavior aligned to the T+1 stim frames produced by make_counterfactual_stim.

    Mirrors the stim's front padding (duplicate first N_LAGS samples), builds the
    behavior over the padded trace, and slices the frames whose 'current' sample
    the lag embedding uses (index N_LAGS-1 onward).
    """
    padded = np.concatenate([eye_trace[:N_LAGS], eye_trace], axis=0)
    beh = build_behavior(padded, vel_scale)      # (N_LAGS + T, behavior_dim)
    return beh[N_LAGS - 1:]                       # (T + 1, behavior_dim)


def compute_rate_map_beh(model, readout, stim, behavior, batch_size: int = 32):
    """Like compute_rate_map_batched but feeds a per-frame behavior tensor.

    behavior: (T, behavior_dim) aligned to stim frames, or None to zero it.
    """
    device = next(model.model.parameters()).device
    dtype = next(model.model.parameters()).dtype
    T = stim.shape[0]
    model.model.eval()
    readout.eval()
    chunks = []
    with torch.no_grad():
        for t0 in range(0, T, batch_size):
            t1 = min(t0 + batch_size, T)
            x = stim[t0:t1].to(device)
            if behavior is None:
                beh = _zero_behavior(model, x.shape[0], device, dtype)
            else:
                beh = behavior[t0:t1].to(device=device, dtype=dtype)
            core_out = model.model.core_forward(x, beh)
            y = model.model.activation(readout(core_out[:, :, -1]))
            chunks.append(y.cpu())
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def drift_rates(model, pop: CenteredPopulation, image: np.ndarray,
                eye_trace: np.ndarray, vel_scale: float | None = None) -> np.ndarray:
    """(T+1, N) predicted rates for a static patch drifted by eye_trace.

    vel_scale=None -> zero behavior (retinal-only); else reconstruct
    drift-consistent behavior with that maxnorm scale.
    """
    T = eye_trace.shape[0]
    full_stack = np.broadcast_to(image[None, :, :],
                                 (T + N_LAGS + 1, *image.shape)).copy()
    trace = torch.from_numpy(eye_trace.astype(np.float32))
    eye_stim = make_counterfactual_stim(full_stack, trace, ppd=PPD,
                                        scale_factor=1.0, n_lags=N_LAGS,
                                        out_size=OUT_SIZE)
    stim_norm = (eye_stim - 127.0) / 255.0
    behavior = None if vel_scale is None else _behavior_for_frames(eye_trace, vel_scale)
    readout_dev = pop.readout.to(model.device)
    rate_map = compute_rate_map_beh(model, readout_dev, stim_norm, behavior)
    return rate_map[:, :, 0, 0].detach().cpu().numpy()


def grating(orientation_deg: float, sf_cyc_deg: float, phase: float = 0.0,
            contrast: float = 1.0, ppd: float = PPD,
            image_shape=IMAGE_SHAPE) -> np.ndarray:
    """Render a full-field sinusoidal grating (0..255 float)."""
    H, W = image_shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xx = (xx - W / 2) / ppd
    yy = (yy - H / 2) / ppd
    theta = np.deg2rad(orientation_deg)
    proj = xx * np.cos(theta) + yy * np.sin(theta)
    g = np.cos(2 * np.pi * sf_cyc_deg * proj + phase)
    return (127.0 + 127.0 * contrast * g).astype(np.float32)
