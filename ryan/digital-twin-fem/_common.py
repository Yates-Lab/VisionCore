"""Shared primitives for the digital-twin FEM analyses.

See VisionCore/ryan/digital-twin-fem/README.md for the research question and
plans/2026-04-13-digital-twin-fem-design.md for the design rationale.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR, FIGURES_DIR

# Ensure VisionCore root is importable for eval modules.
if str(VISIONCORE_ROOT) not in sys.path:
    sys.path.insert(0, str(VISIONCORE_ROOT))

# Model checkpoint (same as fig3).
CHECKPOINT_DIR = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120"
CHECKPOINT_SUBDIR = "2026-03-31_12-03-23_learned_resnet_concat_convgru_gaussian"
EXPERIMENT_SUBDIR = "learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga1"
BEST_CKPT = "epoch=193-val_bps_overall=0.6000.ckpt"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{CHECKPOINT_SUBDIR}/{EXPERIMENT_SUBDIR}/{BEST_CKPT}"

# Stimulus / temporal constants.
PPD = 37.50476617
N_LAGS = 32
DT = 1.0 / 120.0
IMAGE_SHAPE = (540, 540)
OUT_SIZE = (151, 151)

# Output directory for figure-4 diagnostics.
FIG_DIR = FIGURES_DIR / "fig4"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# FEM and decoder enums.
FEM_CONDITIONS = ("real", "none", "scaled_0.5", "scaled_2.0")
DECODER_KINDS = ("instantaneous", "time_averaged", "flattened_temporal")


# ---------------------------------------------------------------------------
# Population readout + stimulus/rate helpers
# (Inlined from the legacy scripts.spatial_info / mcfarland_sim modules so
# this analysis has no dependency on the deprecated scripts/ layout.)
# ---------------------------------------------------------------------------
class PopulationReadout(nn.Module):
    """Per-unit (feature Conv1x1 + Gaussian spatial mask) readout over a feature map.

    Output spatial shape is `(feature_H - kernel_H + 1, feature_W - kernel_W + 1)`
    with valid padding; the rate-map shape is `(T, n_units, H_grid, W_grid)`.
    """

    def __init__(self, feat_weights: torch.Tensor, biases: torch.Tensor,
                 space_weights: torch.Tensor):
        super().__init__()
        n_units, in_ch = feat_weights.shape[0], feat_weights.shape[1]
        self.features = nn.Conv2d(in_ch, n_units, kernel_size=1, bias=False)
        self.features.weight = nn.Parameter(feat_weights, requires_grad=False)
        self.bias = nn.Parameter(biases, requires_grad=False)
        self.space_weights = nn.Parameter(space_weights[:, None, :, :], requires_grad=False)
        self.n_units = n_units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        space = F.conv2d(feat, self.space_weights, groups=self.n_units, padding="valid")
        return space + self.bias[None, :, None, None]


def _eye_deg_to_norm(eye_deg: torch.Tensor, ppd: float,
                     img_size: tuple[int, int]) -> torch.Tensor:
    """Convert (T, 2) eye position in degrees to grid_sample [-1, 1] coords."""
    H, W = img_size
    eye_deg = eye_deg.to(dtype=torch.float32)
    x_pix = eye_deg[:, 0] * ppd
    y_pix = eye_deg[:, 1] * ppd
    x_norm = 2.0 * x_pix / (W - 1)
    y_norm = -2.0 * y_pix / (H - 1)  # grid_sample y-axis points down
    return torch.stack((x_norm, y_norm), dim=-1)


def _shift_movie_with_eye(
    movie: torch.Tensor, eye_xy: torch.Tensor,
    out_size: tuple[int, int] = (100, 100),
    center: tuple[float, float] = (0.0, 0.0),
    mode: str = "bilinear", padding_mode: str = "zeros",
    scale_factor: float = 1.0, align_corners: bool = True,
) -> torch.Tensor:
    """Sample an eye-shifted crop from `movie` via `grid_sample`.

    movie: (T, H, W) or (T, C, H, W). eye_xy: (T, 2) in normalized [-1, 1].
    """
    if movie.dim() == 3:
        movie = movie.unsqueeze(1)
        squeeze_C = True
    elif movie.dim() == 4:
        squeeze_C = False
    else:
        raise ValueError("movie must have shape (T, H, W) or (T, C, H, W)")

    T, _, H, W = movie.shape
    device, dtype = movie.device, movie.dtype
    eye_xy = eye_xy.to(device=device, dtype=dtype)
    outH, outW = out_size
    cx, cy = center

    x_extent = (outW / W) * scale_factor
    y_extent = (outH / H) * scale_factor
    ys = torch.linspace(-y_extent, y_extent, outH, device=device, dtype=dtype)
    xs = torch.linspace(-x_extent, x_extent, outW, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack((grid_x + cx, grid_y + cy), dim=-1).unsqueeze(0)

    grid = base_grid - eye_xy.view(T, 1, 1, 2)
    out = F.grid_sample(movie, grid, mode=mode, padding_mode=padding_mode,
                        align_corners=align_corners)
    if squeeze_C:
        out = out[:, 0]
    return out


def _embed_time_lags(movie: torch.Tensor, n_lags: int = 32) -> torch.Tensor:
    """Embed time lags into a movie tensor.

    movie: (T, H, W) or (T, 1, H, W) -> (T - n_lags + 1, 1, n_lags, H, W).
    """
    if movie.dim() == 3:
        movie = movie.unsqueeze(1)
    T, C, H, W = movie.shape
    out_frames = T - n_lags + 1
    lagged = torch.zeros(out_frames, C, n_lags, H, W,
                         dtype=movie.dtype, device=movie.device)
    for lag in range(n_lags):
        lagged[:, :, lag] = movie[n_lags - 1 - lag: T - lag]
    return lagged


def make_counterfactual_stim(
    full_stack: np.ndarray, eyepos: torch.Tensor,
    ppd: float = PPD, scale_factor: float = 1.0,
    n_lags: int = N_LAGS, out_size: tuple[int, int] = (101, 101),
) -> torch.Tensor:
    """Reconstruct a gaze-contingent stimulus tensor from an eye trace.

    Returns a (T, 1, n_lags, H, W) tensor ready to feed to the model after
    pixel normalization.
    """
    eye_norm = _eye_deg_to_norm(torch.fliplr(eyepos), ppd, full_stack.shape[1:3])
    eye_movie = _shift_movie_with_eye(
        torch.from_numpy(full_stack[:eyepos.shape[0] + n_lags]).float(),
        torch.cat([eye_norm[:n_lags], eye_norm], dim=0),
        out_size=out_size, center=(0.0, 0.0),
        scale_factor=scale_factor, mode="bilinear",
    )
    return _embed_time_lags(eye_movie, n_lags=n_lags)


def _zero_behavior(model, batch_size: int, device, dtype) -> torch.Tensor | None:
    """Return a zero-valued behavior tensor matching the modulator's behavior_dim.

    If the model has no modulator, returns None (core_forward will skip gracefully).
    """
    mod = getattr(model.model, "modulator", None)
    if mod is None:
        return None
    behavior_dim = getattr(mod, "behavior_dim", None)
    if behavior_dim is None:
        return None
    return torch.zeros(batch_size, behavior_dim, device=device, dtype=dtype)


def compute_rate_map_batched(
    model, readout: nn.Module, stim: torch.Tensor, batch_size: int = 32,
) -> torch.Tensor:
    """Run core + readout on a (T, 1, n_lags, H, W) stim in time-batches.

    Returns a (T, n_units, H_grid, W_grid) rate tensor on CPU.
    """
    device = next(model.model.parameters()).device
    dtype = next(model.model.parameters()).dtype
    T = stim.shape[0]
    model.model.eval()
    readout.eval()
    y_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for t_start in range(0, T, batch_size):
            t_end = min(t_start + batch_size, T)
            x = stim[t_start:t_end].to(device)
            beh = _zero_behavior(model, x.shape[0], device, dtype)
            core_out = model.model.core_forward(x, beh)
            y_batch = model.model.activation(readout(core_out[:, :, -1]))
            y_chunks.append(y_batch.cpu())
            del y_batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return torch.cat(y_chunks, dim=0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_digital_twin(device: str | torch.device | None = None):
    """Load the figure-3 digital-twin model. Returns (model, model_info, device)."""
    from DataYatesV1 import get_free_device
    from eval.eval_stack_multidataset import load_model

    if device is None:
        device = get_free_device()
    print(f"Loading digital twin from {CHECKPOINT_PATH}")
    model, model_info = load_model(checkpoint_path=CHECKPOINT_PATH, device=str(device))
    model.model.eval()
    model.model.convnet.use_checkpointing = False
    return model, model_info, device


# ---------------------------------------------------------------------------
# CCmax-reliable-unit filter (from fig3 cache — fig2's neuron_mask is a
# spike-threshold, not a CCmax reliability mask).
# ---------------------------------------------------------------------------
def load_ccmax_reliable_units(ccmax_threshold: float = 0.80) -> dict[str, np.ndarray]:
    """Return {session_name: neuron_indices} where each neuron passes fig3's ccmax filter.

    Uses the fig3 cache (`fig3_digitaltwin.pkl`), which contains per-session
    `neuron_mask` (spike-count filter, indexing into the original neurons) and
    `ccmax` (noise-ceiling estimate, same length as neuron_mask). Reliable units
    are those with `ccmax > ccmax_threshold`; the returned indices reference the
    original (pre-mask) neuron space used by `model.model.readouts[idx]`.
    """
    fig3_cache_path = CACHE_DIR / "fig3_digitaltwin.pkl"
    if not fig3_cache_path.exists():
        raise FileNotFoundError(
            f"Figure 3 cache not found at {fig3_cache_path}. "
            "Run generate_figure3.py first to cache the CCmax estimates."
        )
    with open(fig3_cache_path, "rb") as f:
        session_results = dill.load(f)

    reliable: dict[str, np.ndarray] = {}
    for sr in session_results:
        sess = sr["session"]
        mask = np.asarray(sr["neuron_mask"])
        ccmax = np.asarray(sr["ccmax"])
        keep = ccmax > ccmax_threshold
        reliable[sess] = mask[keep].astype(int)
    return reliable


# ---------------------------------------------------------------------------
# Simulated population: reliable units × foveal grid positions.
# ---------------------------------------------------------------------------
@dataclass
class SimulatedPopulation:
    """A random subsample of (unit_index, grid_row, grid_col) simulated neurons."""

    readout: torch.nn.Module
    unit_ids: np.ndarray           # (N, 3): (global_unit_idx, grid_row, grid_col)
    session_names: list[str]       # per global_unit_idx, the originating session
    grid_shape: tuple[int, int]    # (H, W) of the rate-map grid
    N: int


def build_population(
    model,
    N: int,
    rng: np.random.Generator,
    ccmax_threshold: float = 0.80,
    grid_extent_deg: float = 0.5,  # noqa: ARG001 (metadata; not enforced in index space)
    feature_grid: tuple[int, int] = (14, 14),
) -> SimulatedPopulation:
    """Build a pooled simulated V1 population.

    Population = (ccmax-reliable real units across sessions) × (foveal grid positions),
    randomly subsampled to N. The grid shape is probed on a dummy forward because
    PopulationReadout output size = feature_map - kernel + 1 and depends on the
    convnet's reduction from OUT_SIZE.
    """
    reliable = load_ccmax_reliable_units(ccmax_threshold)
    feat_weights, biases, space_weights, session_names = [], [], [], []
    for sess_name, cids in reliable.items():
        if sess_name not in model.names or len(cids) == 0:
            continue
        ridx = model.names.index(sess_name)
        ro = model.model.readouts[ridx]
        fw = ro.features.weight.detach().cpu()[cids]
        b = ro.bias.detach().cpu()[cids]
        sw = ro.compute_gaussian_mask(feature_grid[0], feature_grid[1],
                                      model.device).detach().cpu()[cids]
        feat_weights.append(fw)
        biases.append(b)
        space_weights.append(sw)
        session_names.extend([sess_name] * len(cids))

    if not feat_weights:
        raise RuntimeError("No reliable units found across sessions; check fig3 cache.")

    feat_weights = torch.cat(feat_weights, dim=0)
    biases = torch.cat(biases, dim=0)
    space_weights = torch.cat(space_weights, dim=0)
    readout = PopulationReadout(feat_weights, biases, space_weights)
    n_units = feat_weights.shape[0]

    # Probe the rate-map grid shape with a one-frame dummy forward through the
    # core + readout. The core output has spatial dims given by the model's
    # convnet reduction of OUT_SIZE; PopulationReadout then does a valid-padding
    # conv with the feature_grid kernel.
    device = model.device
    dtype = next(model.model.parameters()).dtype
    with torch.no_grad():
        dummy = torch.zeros(1, 1, N_LAGS, OUT_SIZE[0], OUT_SIZE[1],
                            device=device, dtype=dtype)
        beh = _zero_behavior(model, dummy.shape[0], device, dtype)
        core_out = model.model.core_forward(dummy, beh)
        y = readout.to(device)(core_out[:, :, -1])
    H_grid, W_grid = int(y.shape[-2]), int(y.shape[-1])
    readout.cpu()

    # Enumerate (unit, row, col) candidates and subsample.
    uu, rr, cc = np.meshgrid(np.arange(n_units), np.arange(H_grid), np.arange(W_grid),
                             indexing="ij")
    candidates = np.stack([uu.ravel(), rr.ravel(), cc.ravel()], axis=1)
    total = candidates.shape[0]
    if N > total:
        raise ValueError(f"Requested N={N} exceeds candidate pool size {total}.")
    idx = rng.choice(total, size=N, replace=False)
    unit_ids = candidates[idx]

    return SimulatedPopulation(
        readout=readout,
        unit_ids=unit_ids,
        session_names=session_names,
        grid_shape=(H_grid, W_grid),
        N=N,
    )


# ---------------------------------------------------------------------------
# Tumbling-E renderer
# ---------------------------------------------------------------------------
def render_tumbling_e(
    orientation_deg: float,
    size_deg: float,
    ppd: float = PPD,
    image_shape: tuple[int, int] = IMAGE_SHAPE,
    fg_value: float = 255.0,
    bg_value: float = 0.0,
) -> np.ndarray:
    """Render a Snellen-style tumbling E, rotated by `orientation_deg`.

    orientation_deg in {0, 90, 180, 270} yields the four canonical orientations
    (0 = arms to the right). Size is the total height/width of the E in degrees;
    standard E is 5x5 stroke-widths so stroke = size/5.
    """
    from scipy.ndimage import zoom, rotate

    H, W = image_shape
    size_px = max(5, int(round(size_deg * ppd)))
    base = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ], dtype=np.float32)
    zoom_factor = size_px / 5.0
    e_img = zoom(base, zoom=zoom_factor, order=0)
    e_rot = rotate(e_img, angle=orientation_deg, reshape=False, order=0,
                   mode="constant", cval=0.0)
    img = np.full((H, W), bg_value, dtype=np.float32)
    h_e, w_e = e_rot.shape
    r0 = (H - h_e) // 2
    c0 = (W - w_e) // 2
    mask = e_rot > 0.5
    img[r0:r0 + h_e, c0:c0 + w_e][mask] = fg_value
    return img


# ---------------------------------------------------------------------------
# Eye-trace counterfactuals
# ---------------------------------------------------------------------------
def make_eye_trace(condition: str, real_trace: np.ndarray) -> np.ndarray:
    """Return an eye-position trace [T, 2] in degrees for a given FEM condition.

    `real_trace` is a source [T, 2] real fixRSVP trace. The 'none' condition
    holds position at the trace mean; the scaled conditions preserve mean and
    scale the deviations. 'simulated_brownian' is reserved (TODO).
    """
    if condition == "real":
        return real_trace.copy()
    mean = np.nanmean(real_trace, axis=0, keepdims=True)
    if condition == "none":
        return np.tile(mean, (real_trace.shape[0], 1)).astype(np.float32)
    if condition == "scaled_0.5":
        return (mean + 0.5 * (real_trace - mean)).astype(np.float32)
    if condition == "scaled_2.0":
        return (mean + 2.0 * (real_trace - mean)).astype(np.float32)
    if condition == "simulated_brownian":
        raise NotImplementedError("Simulated Brownian drift: TODO (see README §TODOs).")
    raise ValueError(f"Unknown FEM condition: {condition!r}")


# ---------------------------------------------------------------------------
# fixRSVP eye-trace pool extractor
# ---------------------------------------------------------------------------
def _load_fixrsvp_only(model, dataset_idx: int):
    """Load only the fixRSVP portion of a dataset.

    Mirrors `eval.eval_stack_utils.load_single_dataset` but sets types to
    exactly ['fixrsvp'] so the other stimulus types (backimage, gaborium,
    gratings) are not materialized. Returns (train_dset, val_dset, config).
    """
    import copy
    import contextlib
    import os

    from models.data import prepare_data

    if hasattr(model, "dataset_configs"):
        cfg = copy.deepcopy(model.dataset_configs[dataset_idx])
    elif hasattr(model.model, "dataset_configs"):
        cfg = copy.deepcopy(model.model.dataset_configs[dataset_idx])
    elif hasattr(model, "cfgs"):
        cfg = copy.deepcopy(model.cfgs[dataset_idx])
    else:
        raise RuntimeError("Cannot locate dataset configs on model.")

    cfg["types"] = ["fixrsvp"]
    cfg.setdefault("keys_lags", {})["eyepos"] = 0

    with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), \
            contextlib.redirect_stderr(devnull):
        train_dset, val_dset, cfg = prepare_data(cfg, strict=False)
    return train_dset, val_dset, cfg


def extract_fixrsvp_eye_traces(
    model,
    max_T: int = 540,
    fixation_radius_deg: float = 1.0,
    min_fix_dur: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull real fixRSVP eye traces, trimming to in-fixation samples.

    Returns (eye_traces [K, max_T, 2] float with NaN padding, durations [K] int).
    Loads only fixRSVP per session, not the full multi-stimulus dataset.
    """
    traces: list[np.ndarray] = []
    for dataset_idx, name in enumerate(model.names):
        try:
            train_data, val_data, _ = _load_fixrsvp_only(model, dataset_idx)
            inds = torch.cat([
                train_data.get_dataset_inds("fixrsvp"),
                val_data.get_dataset_inds("fixrsvp"),
            ], dim=0)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        if inds.shape[0] == 0:
            continue
        ds_local = inds[:, 0].unique().item()
        dset = train_data.dsets[ds_local]
        trial_inds = np.asarray(dset.covariates["trial_inds"]).ravel()
        eyepos = np.asarray(dset["eyepos"])
        fixation = np.hypot(eyepos[:, 0], eyepos[:, 1]) < fixation_radius_deg
        for ti in np.unique(trial_inds):
            ix = (trial_inds == ti) & fixation
            ep = eyepos[ix]
            if ep.shape[0] < min_fix_dur:
                continue
            padded = np.full((max_T, 2), np.nan, dtype=np.float32)
            n_keep = min(ep.shape[0], max_T)
            padded[:n_keep] = ep[:n_keep]
            traces.append(padded)
    if not traces:
        raise RuntimeError("No fixRSVP eye traces extracted; check dataset configs.")
    traces_arr = np.stack(traces, axis=0)
    dur = np.array([
        np.where(np.isnan(t[:, 0]))[0][0] if np.isnan(t[:, 0]).any() else t.shape[0]
        for t in traces_arr
    ])
    return traces_arr, dur


# ---------------------------------------------------------------------------
# Trial generator
# ---------------------------------------------------------------------------
def generate_trials(
    model,
    population: SimulatedPopulation,
    stim_image: np.ndarray,
    eye_trace: np.ndarray,
    n_trials: int,
    rng: np.random.Generator,
    out_size: tuple[int, int] = OUT_SIZE,
    n_lags: int = N_LAGS,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Run the model on a counterfactual stimulus, gather subsampled units, Poisson-sample.

    Returns spike-count trials of shape [n_trials, T, N_pop] where T = eye-trace
    length after NaN trimming.
    """
    if device is None:
        device = next(model.model.parameters()).device

    T_fix = int(np.sum(~np.isnan(eye_trace[:, 0])))
    if T_fix < 1:
        raise ValueError("Eye trace has no valid (non-NaN) samples.")
    trace = torch.from_numpy(eye_trace[:T_fix]).float()
    full_stack = np.broadcast_to(
        stim_image[None, :, :],
        (T_fix + n_lags + 1, *stim_image.shape),
    ).copy()

    eye_stim = make_counterfactual_stim(
        full_stack, trace,
        ppd=PPD, scale_factor=1.0,
        n_lags=n_lags, out_size=out_size,
    )
    stim_norm = (eye_stim - 127.0) / 255.0

    readout_dev = population.readout.to(device)
    rate_map = compute_rate_map_batched(model, readout_dev, stim_norm)
    # rate_map: (T, N_units, H, W). Gather at sampled (unit, row, col).
    u = population.unit_ids[:, 0]
    r = population.unit_ids[:, 1]
    c = population.unit_ids[:, 2]
    rates = rate_map[:, u, r, c].detach().cpu().numpy()  # (T, N_pop)

    # Poisson-sample trials. Model outputs rates per frame; multiply by DT for
    # per-frame expected counts.
    lam = np.clip(rates * DT, 0.0, None)[None, :, :]  # (1, T, N_pop)
    lam = np.broadcast_to(lam, (n_trials, *lam.shape[1:]))
    y = rng.poisson(lam).astype(np.int32)
    return y


# ---------------------------------------------------------------------------
# Decoder fit/eval
# ---------------------------------------------------------------------------
def _reshape_for_decoder(y: np.ndarray, kind: str, t_window: int,
                        t_bin: int | None = None) -> np.ndarray:
    """y: (n_trials, T, N). Returns (n_trials, D) for the requested decoder."""
    if kind == "instantaneous":
        if t_bin is None:
            raise ValueError("instantaneous decoder requires t_bin.")
        return y[:, t_bin, :].astype(np.float32)
    yw = y[:, :t_window, :]
    if kind == "time_averaged":
        return yw.mean(axis=1).astype(np.float32)
    if kind == "flattened_temporal":
        return yw.reshape(yw.shape[0], -1).astype(np.float32)
    raise ValueError(f"Unknown decoder kind: {kind!r}")


def fit_eval_decoder(
    y_train: np.ndarray, labels_train: np.ndarray,
    y_test: np.ndarray, labels_test: np.ndarray,
    kind: str, t_window: int, t_bin: int | None = None,
    l2: float = 1.0,
) -> tuple[float, LogisticRegression, StandardScaler]:
    """Fit multinomial logistic regression with L2 and return (acc, clf, scaler)."""
    X_tr = _reshape_for_decoder(y_train, kind, t_window, t_bin)
    X_te = _reshape_for_decoder(y_test, kind, t_window, t_bin)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)
    clf = LogisticRegression(C=1.0 / l2, solver="lbfgs", max_iter=1000)
    clf.fit(X_tr, labels_train)
    acc = float(clf.score(X_te, labels_test))
    return acc, clf, scaler


# ---------------------------------------------------------------------------
# GRU memory-off context manager
# ---------------------------------------------------------------------------
@contextmanager
def gru_memory_off(model):
    """Temporarily force the ConvGRU to zero its hidden state at every timestep.

    The inner cell loop normally carries `h` across timesteps; here we pass
    `None` to each cell at every step so the cell re-initializes to zero.
    Same weights, same architecture, memory removed. Restored on exit.
    """
    rec = model.model.recurrent
    original_forward = rec.forward

    def memoryless_forward(x, hidden=None):  # noqa: ARG001
        x2 = rearrange(x, "b c t h w -> b t c h w")
        _, T, _, _, _ = x2.shape
        outputs = []
        for t in range(T):
            inp = x2[:, t]
            for cell in rec.cells:
                inp = cell(inp, None)
            outputs.append(inp)
        y_seq = torch.stack(outputs, dim=1)
        return rearrange(y_seq, "b t c h w -> b c t h w")

    rec.forward = memoryless_forward
    try:
        yield
    finally:
        rec.forward = original_forward
