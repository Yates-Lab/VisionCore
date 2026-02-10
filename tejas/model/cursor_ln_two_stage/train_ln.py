"""
Train LN model for cell 14: same data loading as two_stage.py, centered patch, STA init, LBFGS.
Run from VisionCore root: uv run python tejas/model/cursor_ln_two_stage/train_ln.py
"""
import os
import sys
from copy import deepcopy

_VISIONCORE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _VISIONCORE_ROOT not in sys.path:
    sys.path.insert(0, _VISIONCORE_ROOT)

import torch
import torch.nn as nn
from tqdm import tqdm

from tejas.model.cursor_ln_two_stage.data_utils import load_and_center_for_cell
from tejas.model.cursor_ln_two_stage.models_ln import LinearNonLinearModel, AffineSoftplus
from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator

CELL_IDX = 14
DIM_CENTER = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_ITER = 500


def main():
    print("Loading and centering data for cell", CELL_IDX, "...")
    train_centered, val_centered, sta_cell, dataset_config = load_and_center_for_cell(
        cell_idx=CELL_IDX, dim_for_centering=DIM_CENTER
    )
    # Shapes: stim (N, lags, dim, dim), robs (N, C), dfs (N, C)
    stim_tr = train_centered["stim"].to(DEVICE)
    stim_val = val_centered["stim"].to(DEVICE)
    robs_tr = train_centered["robs"][:, [CELL_IDX]].to(DEVICE)
    dfs_tr = train_centered["dfs"][:, [CELL_IDX]].to(DEVICE)
    robs_val = val_centered["robs"][:, [CELL_IDX]].to(DEVICE)
    dfs_val = val_centered["dfs"][:, [CELL_IDX]].to(DEVICE)

    kernel_shape = (stim_tr.shape[1], stim_tr.shape[2], stim_tr.shape[3])
    model = LinearNonLinearModel(kernel_shape).to(DEVICE)
    sta_t = torch.from_numpy(sta_cell).float().to(DEVICE)
    sta_t = sta_t / (sta_t.norm() + 1e-8)
    model.kernel_.data.copy_(sta_t)
    model.kernel_.requires_grad = False
    # Initial scale/bias fit (like get_initial_weight_and_bias_ln)
    gen_tr = (stim_tr.view(stim_tr.shape[0], -1) @ sta_t.view(-1)).detach()
    gen_val = (stim_val.view(stim_val.shape[0], -1) @ sta_t.view(-1)).detach()
    aff = AffineSoftplus(learn_bias=True).to(DEVICE)
    loss_fn = MaskedPoissonNLLLoss(pred_key="rhat", target_key="robs", mask_key="dfs")
    opt_aff = torch.optim.LBFGS(aff.parameters(), lr=1.0, max_iter=100, line_search_fn="strong_wolfe")
    for _ in range(3):
        def cl():
            opt_aff.zero_grad()
            b = {"generator": gen_tr.unsqueeze(1), "robs": robs_tr, "dfs": dfs_tr}
            out = aff(b)
            L = loss_fn(out)
            L.backward()
            return L
        opt_aff.step(cl)
    with torch.no_grad():
        model.scale.data.copy_(aff.weight)
        model.bias.data.copy_(aff.bias)
    model.kernel_.requires_grad = True

    loss_fn = MaskedPoissonNLLLoss(pred_key="rhat", target_key="robs", mask_key="dfs")
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=MAX_ITER, line_search_fn="strong_wolfe")

    batch_tr = {"stim": stim_tr, "robs": robs_tr, "dfs": dfs_tr}
    batch_val = {"stim": stim_val, "robs": robs_val, "dfs": dfs_val}

    best_val_bps = -float("inf")
    best_state = None

    def closure():
        nonlocal best_val_bps, best_state
        optimizer.zero_grad()
        out_tr = model(batch_tr)
        train_loss = loss_fn(out_tr)
        train_agg = PoissonBPSAggregator(device=DEVICE)
        train_agg(out_tr)
        tbps = train_agg.closure().item()
        train_loss.backward()
        with torch.no_grad():
            out_val = model(batch_val)
            val_agg = PoissonBPSAggregator(device=DEVICE)
            val_agg(out_val)
            vbps = val_agg.closure().item()
        if vbps > best_val_bps:
            best_val_bps = vbps
            best_state = deepcopy(model.state_dict())
        print(f"  train_loss={train_loss.item():.4f} train_bps={tbps:.4f} val_bps={vbps:.4f} best_val_bps={best_val_bps:.4f}")
        return train_loss

    print("Running LBFGS ...")
    optimizer.step(closure)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model with val_bps={best_val_bps:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
