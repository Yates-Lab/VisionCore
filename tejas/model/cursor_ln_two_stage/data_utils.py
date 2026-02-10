"""Load VisionCore data and center stim on RF for a given cell (STA/STE-based)."""
import os
import sys
import numpy as np
import torch
from copy import deepcopy

# VisionCore root
_VISIONCORE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _VISIONCORE_ROOT not in sys.path:
    sys.path.insert(0, _VISIONCORE_ROOT)

from DataYatesV1 import calc_sta
from DataYatesV1.utils.rf import Gaussian2D


def get_dataset_from_config(subject, date, dataset_configs_path):
    """Same as two_stage: load train/val and config."""
    from tejas.model.util import get_dataset_from_config as _get
    return _get(subject, date, dataset_configs_path)


def center_rf_from_ste(stim_np, robs_np, dfs_np, cell_idx, peak_lag=None):
    """
    Get RF center (x0, y0) by fitting Gaussian to STE at peak lag for one cell.
    stim_np: (N, lags, H, W)
    robs_np, dfs_np: (N, C)
    Returns: (x0, y0) in pixel coords, and peak_lag used.
    """
    n_lags = stim_np.shape[1]
    # STE = spike-triggered second moment (squared stim)
    stes = []
    for lag in range(n_lags):
        ste = calc_sta(
            stim_np[:, lag],
            robs_np,
            [0],
            dfs=dfs_np.squeeze(),
            progress=False,
            stim_modifier=lambda x: x**2,
        )
        stes.append(ste)
    # calc_sta returns (n_units, n_lags, n_c, n_y, n_x); we call with lags=[0] so (n_units, 1, 1, H, W) per lag
    ste_all = np.stack(stes, axis=0)  # (n_lags, n_units, 1, H, W)
    n_units = ste_all.shape[1]
    # take cell_idx unit, squeeze: (n_lags, H, W)
    ste_cell = ste_all[:, cell_idx].squeeze()
    if ste_cell.ndim == 2:
        ste_cell = ste_cell[None, ...]  # (1, H, W) -> one lag
    if peak_lag is None:
        peak_vals = ste_cell.reshape(ste_cell.shape[0], -1).max(axis=1)
        peak_lag = int(np.argmax(peak_vals))
    peak_lag = min(max(0, peak_lag), ste_cell.shape[0] - 1)
    cell_ste = ste_cell[peak_lag] if ste_cell.ndim == 3 else ste_cell
    try:
        rf = Gaussian2D(*Gaussian2D.est_p0(cell_ste))
        rf.fit(cell_ste)
        x0, y0 = rf.x0, rf.y0
    except Exception:
        x0 = cell_ste.shape[1] // 2
        y0 = cell_ste.shape[0] // 2
    return x0, y0, peak_lag


def center_stim(dset_dict, x0, y0, dim_for_centering=30):
    """
    Crop stim to a patch centered at (x0, y0). Handles stim shape (N, 1, lags, H, W).
    Returns new dict with stim (N, lags, dim, dim); robs and dfs unchanged.
    """
    stim = dset_dict["stim"]
    if stim.ndim == 5:
        stim = stim.squeeze(1)  # (N, lags, H, W)
    N, L, H, W = stim.shape
    x_min = int(x0 - dim_for_centering / 2)
    x_max = int(x0 + dim_for_centering / 2)
    y_min = int(y0 - dim_for_centering / 2)
    y_max = int(y0 + dim_for_centering / 2)
    # Clamp to image bounds
    x_min = max(0, min(x_min, W - dim_for_centering))
    x_max = x_min + dim_for_centering
    y_min = max(0, min(y_min, H - dim_for_centering))
    y_max = y_min + dim_for_centering
    out = deepcopy(dset_dict)
    out["stim"] = stim[:, :, y_min:y_max, x_min:x_max].clone()
    return out


def load_and_center_for_cell(cell_idx=14, dim_for_centering=30, subject="Allen", date="2022-04-13", config_path=None):
    """Load train/val, compute RF for cell_idx, return centered train/val dicts and STA for that cell."""
    if config_path is None:
        config_path = os.path.join(_VISIONCORE_ROOT, "experiments", "dataset_configs", "multi_basic_240_gaborium_20lags.yaml")
    train_dset, val_dset, dataset_config = get_dataset_from_config(subject, date, config_path)
    train_full = train_dset[:]
    val_full = val_dset[:]
    # Shapes: stim (N, 1, lags, H, W), robs (N, C), dfs (N, C)
    stim = train_full["stim"]
    if stim.ndim == 5:
        stim = stim.squeeze(1)  # (N, lags, H, W)
    robs = train_full["robs"].numpy() if torch.is_tensor(train_full["robs"]) else train_full["robs"]
    dfs = train_full["dfs"].numpy() if torch.is_tensor(train_full["dfs"]) else train_full["dfs"]
    x0, y0, _ = center_rf_from_ste(stim, robs, dfs, cell_idx)
    train_centered = center_stim(train_full, x0, y0, dim_for_centering)
    val_centered = center_stim(val_full, x0, y0, dim_for_centering)
    # STA for this cell only (for LN init)
    n_lags = train_centered["stim"].shape[1]
    robs_c = train_full["robs"][:, [cell_idx]].cpu()
    dfs_c = train_full["dfs"][:, [cell_idx]].cpu().squeeze()
    stas = []
    for lag in range(n_lags):
        s = train_centered["stim"][:, lag].detach().cpu()
        sta = calc_sta(s, robs_c, [0], dfs=dfs_c, progress=False)
        stas.append(sta.squeeze().cpu().numpy())
    sta_cell = np.stack(stas, axis=0)  # (n_lags, dim, dim)
    return train_centered, val_centered, sta_cell, dataset_config
