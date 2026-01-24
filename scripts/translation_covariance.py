"""
Translation-Induced Covariance Analysis (dedicated script)

This script computes fixational eye movement (FEM) induced covariance across stimuli.
It loads backimage_results, reconstructs null vs. real eye-trace responses per stimulus,
computes the FEM covariance, top-2 PCA subspace, translation-gradient subspace, principal
angles, and across-stimulus capture matrix.

Outputs:
- Per-stimulus products (*.npz) with Sigma, U_pca2, U_grad2, angles, capture2
- Aggregated all_cov_results.pkl
- capture_matrix.npy

Notes:
- Requires the model and spatial readout to compute rate maps.
- Attempts to load outputs (for readout construction) from scripts/mcfarland_outputs_mono.pkl.
- If readout cannot be constructed, the script will exit with an informative message.
"""
#%% 
import os
import sys
import pickle
from typing import Tuple, Optional

import numpy as np
import torch
from numpy.linalg import eigh
from scipy.linalg import subspace_angles
import importlib
from numpy.random import default_rng

# Ensure repository root is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

# Prefer real DataYatesV1 package if present to discover backgrounds dir
DATA_YATES_ROOT = '/home/declan/DataYatesV1'
if os.path.isdir(DATA_YATES_ROOT) and DATA_YATES_ROOT not in sys.path:
    sys.path.append(DATA_YATES_ROOT)

# Project imports
from spatial_info import make_counterfactual_stim, get_spatial_readout, compute_rate_map_batched
from utils import get_model_and_dataset_configs
#%% 
# Interactive config
DEBUG = False
NUM_STIMULI = 5 if DEBUG else 20
NUM_TRACES = 5 if DEBUG else 50
OUTPUT_DIR = os.path.join(ROOT, 'declan', 'translation_covariance')
# Optional: directories to search if cached image is missing
IMAGE_SEARCH_DIRS = [
    os.path.join(ROOT, 'declan'),
    os.path.join(ROOT, 'data'),
    os.path.join(ROOT, 'datasets'),
]
# Try to add DataYatesV1 backgrounds directory via import (preferred)
BACKIMAGE_DIR = ''
try:
    mod = importlib.import_module('DataYatesV1.exp.support')
    get_backimage_directory = getattr(mod, 'get_backimage_directory', None)
    if callable(get_backimage_directory):
        try:
            _dir_obj = get_backimage_directory()
            BACKIMAGE_DIR = str(_dir_obj)
        except Exception:
            BACKIMAGE_DIR = ''
    else:
        BACKIMAGE_DIR = ''
except Exception:
    BACKIMAGE_DIR = ''

# Fallback to known absolute path if import fails
if not BACKIMAGE_DIR:
    _fallback = '/home/declan/DataYatesV1/DataYatesV1/exp/SupportData/Backgrounds'
    if os.path.isdir(_fallback):
        BACKIMAGE_DIR = _fallback

if BACKIMAGE_DIR and BACKIMAGE_DIR not in IMAGE_SEARCH_DIRS:
    IMAGE_SEARCH_DIRS.insert(0, BACKIMAGE_DIR)
# Allow runtime override via environment variable (colon-separated paths)
_ENV_DIRS = os.environ.get('VC_IMAGE_DIRS', '')
if _ENV_DIRS:
    for _d in _ENV_DIRS.split(':'):
        _d = _d.strip()
        if _d and _d not in IMAGE_SEARCH_DIRS:
            IMAGE_SEARCH_DIRS.append(_d)
# Set True to run full save pipeline; otherwise use smoke test below
RUN_FULL = True
SEED = 41
GEN_SHUFFLE_BASELINE = True
GEN_RANDOM_BASELINE = True
TAUS = [1.0, 2.0, 4.0]
RANDOM_BASELINE_REPEATS = 20 if DEBUG else 200

def enable_autoreload():
    pass


def get_free_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_backimage_results(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_model_and_readout() -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load model and construct spatial readout using cached outputs.

    Returns (model, readout). Raises RuntimeError if outputs/readout cannot be constructed.
    """
    model, _ = get_model_and_dataset_configs()
    device = get_free_device()
    model = model.to(device)

    # Try to load cached outputs for readout construction
    outputs_path_candidates = [
        os.path.join(ROOT, 'scripts', 'mcfarland_outputs_mono.pkl'),
        os.path.join(ROOT, 'scripts', 'mcfarland_outputs.pkl'),
    ]
    outputs = None
    for p in outputs_path_candidates:
        if os.path.exists(p):
            import dill
            with open(p, 'rb') as f:
                outputs = dill.load(f)
            break
    if outputs is None:
        raise RuntimeError(
            'Could not find cached outputs for readout construction. '
            'Expected scripts/mcfarland_outputs_mono.pkl. Please generate or provide this file.'
        )
    readout = get_spatial_readout(model, outputs).to(device)
    return model, readout


def get_trial_stim_and_rates(eyepos: np.ndarray, full_stack: np.ndarray,
                             model: torch.nn.Module,
                             readout: torch.nn.Module,
                             ppd: float = 37.5,
                             out_size: Tuple[int, int] = (151, 151),
                             n_lags: int = 32,
                             scale: float = 1.0,
                             plot: bool = False):
    """Simulate null vs real eye-trace rate maps for a single trial.
    Returns (y_real, y_null, stim_real, stim_null)."""
    # Normalize eyepos to shape [T,2] and trim to valid length
    eyepos_arr = np.asarray(eyepos, dtype=np.float32)
    if eyepos_arr.ndim == 1:
        # Attempt reshape to [T,2]
        if eyepos_arr.size % 2 == 0:
            eyepos_arr = eyepos_arr.reshape(-1, 2)
        else:
            eyepos_arr = eyepos_arr[:-1].reshape(-1, 2) if eyepos_arr.size > 1 else np.zeros((0, 2), dtype=np.float32)
    elif eyepos_arr.ndim == 2:
        if eyepos_arr.shape[1] == 2:
            pass
        elif eyepos_arr.shape[0] == 2:
            eyepos_arr = eyepos_arr.T
        else:
            eyepos_arr = eyepos_arr[:, :2]
    else:
        flat = eyepos_arr.reshape(-1)
        eyepos_arr = flat[:-1].reshape(-1, 2) if flat.size % 2 == 1 else flat.reshape(-1, 2)

    nan_rows = np.where(np.isnan(eyepos_arr).any(axis=1))[0]
    T_valid = nan_rows[0] if len(nan_rows) > 0 else len(eyepos_arr)
    eyepos_arr = eyepos_arr[:T_valid]
    eyepos_t = torch.from_numpy(eyepos_arr).float()
    null_eyepos = torch.zeros_like(eyepos_t) + eyepos_t.mean(0)

    # Reconstruct stimuli
    eye_stim = make_counterfactual_stim(full_stack, eyepos_t, out_size=out_size, n_lags=n_lags, scale_factor=scale)
    eye_stim_null = make_counterfactual_stim(full_stack, null_eyepos, out_size=out_size, n_lags=n_lags, scale_factor=scale)

    # Compute rates on normalized stimulus
    y = compute_rate_map_batched(model, readout, (eye_stim - 127.0) / 255.0)
    y_null = compute_rate_map_batched(model, readout, (eye_stim_null - 127.0) / 255.0)
    return y, y_null, eye_stim, eye_stim_null


# Helpers: compress [T,N,H,W] -> [T,N]
def compress_to_population_vec(y_tensor):
    if isinstance(y_tensor, torch.Tensor):
        y_np = y_tensor.detach().cpu().numpy()
    else:
        y_np = np.asarray(y_tensor)
    return y_np.mean(axis=(2, 3)) if y_np.ndim == 4 else y_np


def compute_delta_y(y, y_null):
    y_tn = compress_to_population_vec(y)
    mu_tn = compress_to_population_vec(y_null)
    return y_tn - mu_tn, y_tn, mu_tn


def covariance_from_samples(X):
    X = np.asarray(X, dtype=np.float32)
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    mask = np.isfinite(Xc).all(axis=1)
    Xc = Xc[mask]
    if Xc.shape[0] < 3:
        return np.full((Xc.shape[1], Xc.shape[1]), np.nan, dtype=np.float32)
    return np.cov(Xc, rowvar=False)


def top2_pca_subspace(Sigma):
    w, V = eigh(Sigma)
    order = np.argsort(w)[::-1]
    return w[order][:2], V[:, order][:, :2]


def fit_dxdy_subspace(deltaY_list, dx_list, dy_list):
    dY = np.concatenate(deltaY_list, axis=0)  # [S,N]
    Dpos = np.stack([np.concatenate(dx_list), np.concatenate(dy_list)], axis=1)  # [S,2]
    mask = np.isfinite(dY).all(axis=1) & np.isfinite(Dpos).all(axis=1)
    dY = dY[mask]
    Dpos = Dpos[mask]
    G, *_ = np.linalg.lstsq(Dpos, dY, rcond=None)  # [2,N]
    U = G.T
    Q, _ = np.linalg.qr(U)
    return Q[:, :2], G


def fit_2d_subspace_from_regressors(deltaY_list, reg1_list, reg2_list):
    """Generic 2D subspace fit from arbitrary regressors (reg1, reg2)."""
    dY = np.concatenate(deltaY_list, axis=0)
    R = np.stack([np.concatenate(reg1_list), np.concatenate(reg2_list)], axis=1)
    mask = np.isfinite(dY).all(axis=1) & np.isfinite(R).all(axis=1)
    dY = dY[mask]
    R = R[mask]
    if dY.shape[0] < 3:
        # Not enough samples, return NaNs to signal skip
        return np.full((dY.shape[1], 2), np.nan, dtype=np.float32), np.full((2, dY.shape[1]), np.nan, dtype=np.float32)
    G, *_ = np.linalg.lstsq(R, dY, rcond=None)  # [2,N]
    U = G.T
    Q, _ = np.linalg.qr(U)
    return Q[:, :2], G


def principal_angles_2d(U1, U2):
    return subspace_angles(U1, U2)


def capture_fraction(Ui, Sigma_j):
    num = np.trace(Ui.T @ Sigma_j @ Ui)
    den = np.trace(Sigma_j)
    return float(num / den) if den > 0 and np.isfinite(den) else np.nan


def alignment_score(U1, U2):
    """Mean cos^2 of principal angles between 2D subspaces."""
    th = principal_angles_2d(U1, U2)
    return float(np.mean(np.cos(th) ** 2))


def _to_gray_float32(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        # Convert RGB/RGBA to grayscale using luminance weights
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        arr = 0.2989 * r + 0.5870 * g + 0.1140 * b
    arr = arr.astype(np.float32)
    # Ensure a reasonable 0-255 scale
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0.0, 255.0)
    return arr


def ensure_min_trace_length(trace_arr: np.ndarray, min_len: int) -> np.ndarray:
    """Pad trace to at least min_len by repeating the first sample.
    Returns a copy with shape [T_pad, 2]."""
    trace_arr = np.asarray(trace_arr, dtype=np.float32)
    if trace_arr.ndim == 1:
        trace_arr = trace_arr.reshape(-1, 2) if trace_arr.size % 2 == 0 else trace_arr[:-1].reshape(-1, 2)
    elif trace_arr.ndim == 2:
        if trace_arr.shape[1] != 2:
            trace_arr = trace_arr[:, :2] if trace_arr.shape[1] > 2 else trace_arr.T if trace_arr.shape[0] == 2 else trace_arr
    else:
        flat = trace_arr.reshape(-1)
        trace_arr = flat.reshape(-1, 2) if flat.size % 2 == 0 else flat[:-1].reshape(-1, 2)
    # Trim NaNs
    nan_rows = np.where(np.isnan(trace_arr).any(axis=1))[0]
    T_valid = nan_rows[0] if len(nan_rows) > 0 else len(trace_arr)
    trace_arr = trace_arr[:T_valid]
    if trace_arr.shape[0] >= min_len:
        return trace_arr
    if trace_arr.shape[0] == 0:
        base = np.zeros((1, 2), dtype=np.float32)
    else:
        base = trace_arr[:1]
    pad_rows = min_len - trace_arr.shape[0]
    pad = np.repeat(base, pad_rows, axis=0)
    return np.vstack([trace_arr, pad])


def find_image_on_disk(filename: str) -> str:
    """Search for the given filename under IMAGE_SEARCH_DIRS; return first match or ''."""
    for base in IMAGE_SEARCH_DIRS:
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            if filename in files:
                return os.path.join(root, filename)
    return ''


def load_image_for_entry(entry: dict, image_key: str) -> Optional[np.ndarray]:
    """Return an HxW float32 image in [0,255], or None if unavailable."""
    # 1) Direct cached array
    if 'image' in entry and entry['image'] is not None:
        try:
            return _to_gray_float32(entry['image'])
        except Exception:
            pass
    # 2) Known path fields
    for k in ('image_path', 'path', 'filepath', 'file_path', 'img_path'):
        if k in entry and entry[k]:
            p = entry[k]
            if isinstance(p, (list, tuple)):
                p = p[0]
            if isinstance(p, bytes):
                p = p.decode('utf-8', errors='ignore')
            if isinstance(p, str) and os.path.exists(p):
                try:
                    try:
                        from PIL import Image
                        with Image.open(p) as im:
                            return _to_gray_float32(np.array(im))
                    except Exception:
                        import imageio.v2 as iio
                        im = iio.imread(p)
                        return _to_gray_float32(im)
                except Exception:
                    pass
    # 3) Heuristic search by filename
    if isinstance(image_key, str):
        candidate = find_image_on_disk(os.path.basename(image_key))
        if candidate:
            try:
                try:
                    from PIL import Image
                    with Image.open(candidate) as im:
                        return _to_gray_float32(np.array(im))
                except Exception:
                    import imageio.v2 as iio
                    im = iio.imread(candidate)
                    return _to_gray_float32(im)
            except Exception:
                pass
    return None

#%% 

enable_autoreload()
try:
    model, readout = load_model_and_readout()
except Exception as e:
    print(f"[ERROR] {e}")
    print("Aborting: Model/readout required to compute rate maps.")
    sys.exit(1)

# Load image+eyepos metadata
results_path = os.path.join(ROOT, 'declan', 'backimage_fixation_results.pkl')
backimage_results = load_backimage_results(results_path)

# Parameters (from config)
num_stimuli = NUM_STIMULI
num_traces = NUM_TRACES
output_dir = OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)

# Select stimuli by trial count
sorted_images = sorted(backimage_results.items(), key=lambda x: -x[1]['n_trials'])
stimuli = [img for img, _ in sorted_images[:num_stimuli]]

# Pre-build trace pools for shuffle baseline
stim_to_traces = {}
for img in stimuli:
    entry = backimage_results[img]
    traces = entry.get('eyepos', [])
    if len(traces) > 0:
        stim_to_traces[img] = traces[:num_traces]
traces_pool_all = [(img, tr) for img, trs in stim_to_traces.items() for tr in trs]

cov_results = {}

#%%
# Smoke test cell (runs when RUN_FULL=False)
if not RUN_FULL and len(stimuli) > 0:
    s_idx = 0
    image_file = stimuli[s_idx]
    print(f"\n[Smoke Test] Stimulus {s_idx+1}/{len(stimuli)}: {image_file}")
    search_dirs = [d for d in IMAGE_SEARCH_DIRS if os.path.isdir(d)]
    if BACKIMAGE_DIR:
        print(f"[Smoke Test] Using DataYates backgrounds: {BACKIMAGE_DIR}")
    print(f"[Smoke Test] Searching in: {', '.join(search_dirs)}")
    entry = backimage_results[image_file]
    eyepos_all = entry.get('eyepos', [])
    eyepos_traces = eyepos_all[:num_traces]
    backimage_image = load_image_for_entry(entry, image_file)
    if backimage_image is not None:
        n_lags = 32
        out_size = (151, 151)
        scale = 1.0
        T_null = 540
        null_trace = np.zeros((T_null, 2), dtype=np.float32)
        full_stack_null = np.repeat(backimage_image[np.newaxis, ...], T_null + n_lags, axis=0)
        y_null, _, _, _ = get_trial_stim_and_rates(null_trace, full_stack_null, model, readout,
                                                   out_size=out_size, n_lags=n_lags, scale=scale)
        deltaY_list, dx_list, dy_list = [], [], []
        for trace in eyepos_traces:
            trace_arr = ensure_min_trace_length(trace, n_lags)
            T_stim = trace_arr.shape[0]
            Trep = T_stim + n_lags
            stack_rep = np.repeat(backimage_image[np.newaxis, ...], Trep, axis=0)
            y, _, _, _ = get_trial_stim_and_rates(trace_arr, stack_rep, model, readout,
                                                  out_size=out_size, n_lags=n_lags, scale=scale)
            dy_tn, _, _ = compute_delta_y(y, y_null[:y.shape[0]])
            deltaY_list.append(dy_tn)
            # Build dx/dy series aligned with embedded time lags: prepad n_lags with first sample
            pos_series = np.vstack([np.repeat(trace_arr[:1], n_lags, axis=0), trace_arr])
            dx_list.append(pos_series[:dy_tn.shape[0], 0])
            dy_list.append(pos_series[:dy_tn.shape[0], 1])
        X = np.concatenate(deltaY_list, axis=0)
        Sigma = covariance_from_samples(X)
        evals2, U_pca2 = top2_pca_subspace(Sigma)
        U_grad2, G = fit_dxdy_subspace(deltaY_list, dx_list, dy_list)
        theta = principal_angles_2d(U_pca2, U_grad2)
        frac2 = capture_fraction(U_pca2, Sigma)
        print(f"[Smoke Test] evals2[:2] = {np.round(evals2, 4)}")
        print(f"[Smoke Test] angles (deg) = {np.degrees(theta)}")
        print(f"[Smoke Test] capture2 = {frac2:.3f}")
    else:
        print("[Smoke Test] Missing image (cached or on disk); skipping.")
        if BACKIMAGE_DIR:
            print(f"[Hint] Confirm filename exists under {BACKIMAGE_DIR}.")

#%% Full run cell (gated by RUN_FULL)
if RUN_FULL:
    for s_idx, image_file in enumerate(stimuli):
        print(f"\nProcessing stimulus {s_idx+1}/{len(stimuli)}: {image_file}")
        entry = backimage_results[image_file]
        eyepos_all = entry.get('eyepos', [])
        if len(eyepos_all) == 0:
            print("  - No eyepos; skipping.")
            continue
        eyepos_traces = eyepos_all[:num_traces]

        # Robust image fetch
        backimage_image = load_image_for_entry(entry, image_file)
        if backimage_image is None:
            print("  - Missing image (cached or on disk); skipping.")
            if BACKIMAGE_DIR:
                print(f"  - Checked DataYates backgrounds: {BACKIMAGE_DIR}")
            continue

        n_lags = 32
        out_size = (151, 151)
        scale = 1.0

        # Build a repeated stack for temporal context per trace
        deltaY_list, dx_list, dy_list = [], [], []
        vx_list, vy_list = [], []
        fdx_lists_by_tau = {tau: [] for tau in TAUS}
        fdy_lists_by_tau = {tau: [] for tau in TAUS}
        # For split-half reliability
        deltaY_list_A, deltaY_list_B = [], []

        # Null response per stimulus using mean-position trace
        T_null = 540
        null_trace = np.zeros((T_null, 2), dtype=np.float32)
        full_stack_null = np.repeat(backimage_image[np.newaxis, ...], T_null + n_lags + 1, axis=0)
        y_null, _, _, _ = get_trial_stim_and_rates(null_trace, full_stack_null, model, readout,
                                                   out_size=out_size, n_lags=n_lags, scale=scale)
        # Helper to compute velocity and filtered position given pos_series
        def _compute_velocity(series_xy):
            v = np.zeros_like(series_xy)
            v[1:] = series_xy[1:] - series_xy[:-1]
            return v
        def _compute_filtered(series_1d, alpha):
            out = np.zeros_like(series_1d)
            for t in range(1, len(series_1d)):
                out[t] = alpha * out[t-1] + (1.0 - alpha) * series_1d[t]
            return out

        for t_idx, trace in enumerate(eyepos_traces):
            trace_arr = ensure_min_trace_length(trace, n_lags)
            T_stim = trace_arr.shape[0]
            Trep = T_stim + n_lags
            stack_rep = np.repeat(backimage_image[np.newaxis, ...], Trep, axis=0)
            y, _, _, _ = get_trial_stim_and_rates(trace_arr, stack_rep, model, readout,
                                                  out_size=out_size, n_lags=n_lags, scale=scale)
            dy_tn, _, _ = compute_delta_y(y, y_null[:y.shape[0]])
            deltaY_list.append(dy_tn)
            # Split-half assign
            if (t_idx % 2) == 0:
                deltaY_list_A.append(dy_tn)
            else:
                deltaY_list_B.append(dy_tn)
            # Build regressors aligned with embedded time lags
            pos_series = np.vstack([np.repeat(trace_arr[:1], n_lags, axis=0), trace_arr])
            pos_series = pos_series[:dy_tn.shape[0]]
            dx_list.append(pos_series[:, 0])
            dy_list.append(pos_series[:, 1])
            vel = _compute_velocity(pos_series)
            vx_list.append(vel[:, 0])
            vy_list.append(vel[:, 1])
            for tau in TAUS:
                alpha = float(np.exp(-1.0 / tau))
                fx = _compute_filtered(pos_series[:, 0], alpha)
                fy = _compute_filtered(pos_series[:, 1], alpha)
                fdx_lists_by_tau[tau].append(fx)
                fdy_lists_by_tau[tau].append(fy)

        if len(deltaY_list) == 0:
            print("  - No valid δy; skipping.")
            continue

        # Covariance and subspaces
        X = np.concatenate(deltaY_list, axis=0)
        Sigma = covariance_from_samples(X)
        evals2, U_pca2 = top2_pca_subspace(Sigma)
        # Position-based gradient subspace (preserve original outputs)
        U_grad2, G_pos = fit_dxdy_subspace(deltaY_list, dx_list, dy_list)
        theta = principal_angles_2d(U_pca2, U_grad2)
        frac2 = capture_fraction(U_pca2, Sigma)

        # Within-stimulus split-half reliability (A vs B)
        principal_angles_within = np.array([np.nan, np.nan], dtype=np.float32)
        within_score = np.nan
        try:
            if len(deltaY_list_A) >= 1 and len(deltaY_list_B) >= 1:
                XA = np.concatenate(deltaY_list_A, axis=0)
                XB = np.concatenate(deltaY_list_B, axis=0)
                SA = covariance_from_samples(XA)
                SB = covariance_from_samples(XB)
                if np.isfinite(SA).all() and np.isfinite(SB).all():
                    _evalA, U_A = top2_pca_subspace(SA)
                    _evalB, U_B = top2_pca_subspace(SB)
                    principal_angles_within = principal_angles_2d(U_A, U_B)
                    within_score = float(np.mean(np.cos(principal_angles_within) ** 2))
        except Exception:
            pass

        # Additional regressors: velocity and filtered position
        U_grad2_vel, G_vel = fit_2d_subspace_from_regressors(deltaY_list, vx_list, vy_list)
        angles_pca_vs_pos = principal_angles_2d(U_pca2, U_grad2)
        score_pca_vs_pos = float(np.mean(np.cos(angles_pca_vs_pos) ** 2))
        angles_pca_vs_vel = principal_angles_2d(U_pca2, U_grad2_vel) if np.isfinite(U_grad2_vel).all() else np.array([np.nan, np.nan])
        score_pca_vs_vel = float(np.mean(np.cos(angles_pca_vs_vel) ** 2)) if np.isfinite(U_grad2_vel).all() else np.nan
        angles_pca_vs_fpos = {}
        score_pca_vs_fpos = {}
        for tau in TAUS:
            U_fpos, G_fpos = fit_2d_subspace_from_regressors(deltaY_list, fdx_lists_by_tau[tau], fdy_lists_by_tau[tau])
            if np.isfinite(U_fpos).all():
                ang = principal_angles_2d(U_pca2, U_fpos)
                sc = float(np.mean(np.cos(ang) ** 2))
            else:
                ang = np.array([np.nan, np.nan])
                sc = np.nan
            angles_pca_vs_fpos[tau] = ang
            score_pca_vs_fpos[tau] = sc

        # Save per-stimulus products
        out_npz = os.path.join(output_dir, f"{s_idx:03d}_cov_products.npz")
        np.savez(
            out_npz,
            stimulus=image_file,
            Sigma=Sigma,
            evals2=evals2,
            U_pca2=U_pca2,
            U_grad2=U_grad2,
            principal_angles=theta,
            capture2=frac2,
            principal_angles_within_pca2=principal_angles_within,
            within_alignment_score=within_score,
            angles_pca_vs_pos=angles_pca_vs_pos,
            score_pca_vs_pos=score_pca_vs_pos,
            angles_pca_vs_vel=angles_pca_vs_vel,
            score_pca_vs_vel=score_pca_vs_vel,
            **{f"angles_pca_vs_fpos_tau{int(tau)}": angles_pca_vs_fpos[tau] for tau in TAUS},
            **{f"score_pca_vs_fpos_tau{int(tau)}": score_pca_vs_fpos[tau] for tau in TAUS},
        )
        cov_results[image_file] = {
            'Sigma': Sigma, 'evals2': evals2, 'U_pca2': U_pca2,
            'U_grad2': U_grad2, 'principal_angles': theta, 'capture2': frac2,
            'principal_angles_within_pca2': principal_angles_within,
            'within_alignment_score': within_score,
            'angles_pca_vs_pos': angles_pca_vs_pos,
            'score_pca_vs_pos': score_pca_vs_pos,
            'angles_pca_vs_vel': angles_pca_vs_vel,
            'score_pca_vs_vel': score_pca_vs_vel,
            'angles_pca_vs_fpos': angles_pca_vs_fpos,
            'score_pca_vs_fpos': score_pca_vs_fpos,
        }
        print(f"  - Saved: {out_npz} | angles (deg) = {np.degrees(theta)} | capture2 = {frac2:.3f} | within={within_score:.3f}")

    # Save all results and capture matrix
    with open(os.path.join(output_dir, 'all_cov_results.pkl'), 'wb') as f:
        pickle.dump(cov_results, f)

    keys = list(cov_results.keys())
    S = len(keys)
    capture_mat = np.full((S, S), np.nan, dtype=np.float32)
    for i in range(S):
        Ui = cov_results[keys[i]]['U_pca2']
        for j in range(S):
            Sj = cov_results[keys[j]]['Sigma']
            capture_mat[i, j] = capture_fraction(Ui, Sj)
    np.save(os.path.join(output_dir, 'capture_matrix.npy'), capture_mat)
    print(f"\n✓ Saved all covariance results and capture_matrix to {output_dir}")

    # Random subspace baseline (alignment)
    if GEN_RANDOM_BASELINE and S > 1:
        N = cov_results[keys[0]]['U_pca2'].shape[0]
        rng = default_rng(SEED)
        offdiag_values = []
        for r in range(RANDOM_BASELINE_REPEATS):
            U_rand = []
            for _ in range(S):
                A = rng.standard_normal((N, 2))
                Q, _ = np.linalg.qr(A)
                U_rand.append(Q[:, :2])
            # Compute off-diagonal alignment scores
            for i in range(S):
                for j in range(S):
                    if i == j:
                        continue
                    th = principal_angles_2d(U_rand[i], U_rand[j])
                    offdiag_values.append(np.mean(np.cos(th) ** 2))
        offdiag_values = np.asarray(offdiag_values, dtype=np.float32)
        np.save(os.path.join(output_dir, 'random_baseline_alignment.npy'), offdiag_values)
        print(f"✓ Saved random baseline alignment distribution ({offdiag_values.size} values)")

    # Trace–stimulus mismatch shuffle baseline
    if GEN_SHUFFLE_BASELINE and S > 0:
        cov_results_shuf = {}
        for s_idx, image_file in enumerate(keys):
            print(f"\n[Shuffle] Stimulus {s_idx+1}/{S}: {image_file}")
            entry = backimage_results[image_file]
            backimage_image = load_image_for_entry(entry, image_file)
            if backimage_image is None:
                print("  - Missing image; skipping shuf.")
                continue
            # Null response per stimulus
            T_null = 540
            null_trace = np.zeros((T_null, 2), dtype=np.float32)
            full_stack_null = np.repeat(backimage_image[np.newaxis, ...], T_null + n_lags + 1, axis=0)
            y_null, _, _, _ = get_trial_stim_and_rates(null_trace, full_stack_null, model, readout,
                                                       out_size=out_size, n_lags=n_lags, scale=scale)
            # Draw shuffled traces from pool excluding this stimulus
            pool = [tr for (img, tr) in traces_pool_all if img != image_file]
            if len(pool) == 0:
                print("  - Empty trace pool; skipping shuf.")
                continue
            rng = default_rng(SEED + s_idx)
            chosen = [pool[i] for i in rng.choice(len(pool), size=min(num_traces, len(pool)), replace=False)]
            deltaY_list_shuf = []
            for trace in chosen:
                trace_arr = ensure_min_trace_length(trace, n_lags)
                T_stim = trace_arr.shape[0]
                Trep = T_stim + n_lags
                stack_rep = np.repeat(backimage_image[np.newaxis, ...], Trep, axis=0)
                y, _, _, _ = get_trial_stim_and_rates(trace_arr, stack_rep, model, readout,
                                                      out_size=out_size, n_lags=n_lags, scale=scale)
                dy_tn, _, _ = compute_delta_y(y, y_null[:y.shape[0]])
                deltaY_list_shuf.append(dy_tn)
            if len(deltaY_list_shuf) == 0:
                print("  - No δy_shuf; skipping.")
                continue
            Xs = np.concatenate(deltaY_list_shuf, axis=0)
            Sigma_shuf = covariance_from_samples(Xs)
            evals2_shuf, U_pca2_shuf = top2_pca_subspace(Sigma_shuf)
            frac2_shuf = capture_fraction(U_pca2_shuf, Sigma_shuf)
            # Save per-stimulus shuffled products
            out_npz_shuf = os.path.join(output_dir, f"{s_idx:03d}_cov_products_shuf.npz")
            np.savez(out_npz_shuf, stimulus=image_file, Sigma=Sigma_shuf, evals2=evals2_shuf,
                     U_pca2=U_pca2_shuf, capture2=frac2_shuf)
            cov_results_shuf[image_file] = {
                'Sigma': Sigma_shuf, 'evals2': evals2_shuf, 'U_pca2': U_pca2_shuf, 'capture2': frac2_shuf
            }
            print(f"  - Shuf saved: {out_npz_shuf} | capture2_shuf = {frac2_shuf:.3f}")
        # Save aggregated shuf results and capture matrix
        with open(os.path.join(output_dir, 'all_cov_results_shuf.pkl'), 'wb') as f:
            pickle.dump(cov_results_shuf, f)
        keys_shuf = list(cov_results_shuf.keys())
        S_shuf = len(keys_shuf)
        if S_shuf > 0:
            capture_mat_shuf = np.full((S_shuf, S_shuf), np.nan, dtype=np.float32)
            for i in range(S_shuf):
                Ui = cov_results_shuf[keys_shuf[i]]['U_pca2']
                for j in range(S_shuf):
                    Sj = cov_results_shuf[keys_shuf[j]]['Sigma']
                    capture_mat_shuf[i, j] = capture_fraction(Ui, Sj)
            np.save(os.path.join(output_dir, 'capture_matrix_shuf.npy'), capture_mat_shuf)
            print(f"\n✓ Saved shuffle results and capture_matrix_shuf to {output_dir}")

#%%

