"""
Quick diagnostic: compare two-stage vs LN outputs, check if two-stage can overfit a tiny subset.
Run from VisionCore root: uv run python tejas/model/cursor_ln_two_stage/diagnose_two_stage.py
"""
import os
import sys
_VISIONCORE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _VISIONCORE_ROOT not in sys.path:
    sys.path.insert(0, _VISIONCORE_ROOT)

import torch
from torch.utils.data import DataLoader, TensorDataset

from tejas.model.cursor_ln_two_stage.data_utils import load_and_center_for_cell
from tejas.model.cursor_ln_two_stage.models_two_stage import TwoStage
from tejas.model.cursor_ln_two_stage.models_ln import LinearNonLinearModel
from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator

CELL_IDX = 14
DIM = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64


def main():
    print("Loading data ...")
    train_c, val_c, sta_cell, cfg = load_and_center_for_cell(cell_idx=CELL_IDX, dim_for_centering=DIM)
    n_lags = len(cfg["keys_lags"]["stim"])
    stim_tr = train_c["stim"]
    if stim_tr.ndim == 4:
        stim_tr = stim_tr.unsqueeze(1)
    robs_tr = train_c["robs"][:, [CELL_IDX]]
    dfs_tr = train_c["dfs"][:, [CELL_IDX]]
    stim_val = val_c["stim"]
    if stim_val.ndim == 4:
        stim_val = stim_val.unsqueeze(1)
    robs_val = val_c["robs"][:, [CELL_IDX]]
    dfs_val = val_c["dfs"][:, [CELL_IDX]]

    mean_rate = (robs_tr * dfs_tr).sum().item() / (dfs_tr.sum().clamp(min=1e-8).item())
    print(f"Mean train rate (masked): {mean_rate:.6f}")
    print(f"robs range: [{robs_tr.min().item():.4f}, {robs_tr.max().item():.4f}]")

    # --- Two-stage: one forward, inspect outputs ---
    print("\n--- Two-stage model ---")
    model_ts = TwoStage((DIM, DIM), 1, n_lags=n_lags, height=2, order=5).to(DEVICE)
    with torch.no_grad():
        batch0 = {"stim": stim_tr[:BATCH_SIZE].to(DEVICE), "robs": robs_tr[:BATCH_SIZE].to(DEVICE), "dfs": dfs_tr[:BATCH_SIZE].to(DEVICE)}
        pos_f, neg_f, _ = model_ts.get_pyr_feats(batch0)
        sc = (pos_f.std().item() + neg_f.std().item()) / 2.0 + 1e-6
        model_ts.feat_scale.fill_(sc)
        model_ts.beta.data.fill_(float(mean_rate))
        model_ts.w_pos.weight.data.mul_(0).add_(1e-5 * torch.randn_like(model_ts.w_pos.weight.data, device=DEVICE))
        model_ts.w_neg.weight.data.mul_(0).add_(1e-5 * torch.randn_like(model_ts.w_neg.weight.data, device=DEVICE))
        model_ts.alpha_pos.data.fill_(0.1)
        model_ts.alpha_neg.data.fill_(0.1)

    out_ts = model_ts(batch0)
    rhat = out_ts["rhat"].detach()
    pos_f, neg_f, _ = model_ts.get_pyr_feats(batch0)
    z = (model_ts.w_pos(pos_f) + model_ts.w_neg(neg_f)).detach()
    print(f"feat_scale: {model_ts.feat_scale.item():.4f}")
    print(f"pos_f (after scale): mean={pos_f.mean().item():.6f} std={pos_f.std().item():.6f}")
    print(f"z: min={z.min().item():.6f} max={z.max().item():.6f} mean={z.mean().item():.6f} std={z.std().item():.6f}")
    print(f"rhat: min={rhat.min().item():.6f} max={rhat.max().item():.6f} mean={rhat.mean().item():.6f} std={rhat.std().item():.6f}")
    agg = PoissonBPSAggregator(device=DEVICE)
    agg(out_ts)
    bps0 = agg.closure().item()
    print(f"Two-stage BPS (first batch): {bps0:.4f}")

    # --- LN model same batch for comparison ---
    print("\n--- LN model (STA init) same batch ---")
    kernel_shape = (n_lags, DIM, DIM)
    model_ln = LinearNonLinearModel(kernel_shape).to(DEVICE)
    sta = torch.from_numpy(sta_cell).float().to(DEVICE) / (torch.from_numpy(sta_cell).float().norm() + 1e-8)
    model_ln.kernel_.data.copy_(sta)
    stim_flat = train_c["stim"].view(train_c["stim"].shape[0], -1).to(DEVICE)
    gen = (stim_flat @ sta.view(-1)).detach()
    from tejas.model.cursor_ln_two_stage.models_ln import AffineSoftplus
    aff = AffineSoftplus(learn_bias=True).to(DEVICE)
    loss_fn = MaskedPoissonNLLLoss(pred_key="rhat", target_key="robs", mask_key="dfs")
    opt_aff = torch.optim.LBFGS(aff.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
    for _ in range(3):
        def cl():
            opt_aff.zero_grad()
            b = {"generator": gen.unsqueeze(1), "robs": robs_tr.to(DEVICE), "dfs": dfs_tr.to(DEVICE)}
            o = aff(b)
            loss_fn(o).backward()
            return loss_fn(o)
        opt_aff.step(cl)
    batch_ln = {"stim": stim_tr[:BATCH_SIZE].to(DEVICE), "robs": robs_tr[:BATCH_SIZE].to(DEVICE), "dfs": dfs_tr[:BATCH_SIZE].to(DEVICE)}
    with torch.no_grad():
        model_ln.scale.data.copy_(aff.weight)
        model_ln.bias.data.copy_(aff.bias)
        out_ln = model_ln(batch_ln)
    rhat_ln = out_ln["rhat"].detach()
    print(f"LN rhat: min={rhat_ln.min().item():.6f} max={rhat_ln.max().item():.6f} mean={rhat_ln.mean().item():.6f} std={rhat_ln.std().item():.6f}")
    agg2 = PoissonBPSAggregator(device=DEVICE)
    agg2(out_ln)
    print(f"LN BPS (first batch): {agg2.closure().item():.4f}")

    # --- Overfit test: 200 samples, high LR, 30 epochs ---
    print("\n--- Overfit test: 200 samples, lr=1e-2, 30 epochs ---")
    N_SUB = 200
    stim_sub = stim_tr[:N_SUB].to(DEVICE)
    robs_sub = robs_tr[:N_SUB].to(DEVICE)
    dfs_sub = dfs_tr[:N_SUB].to(DEVICE)
    model2 = TwoStage((DIM, DIM), 1, n_lags=n_lags, height=2, order=5).to(DEVICE)
    with torch.no_grad():
        b0 = {"stim": stim_sub[:64], "robs": robs_sub[:64], "dfs": dfs_sub[:64]}
        pf, nf, _ = model2.get_pyr_feats(b0)
        sc2 = (pf.std().item() + nf.std().item()) / 2.0 + 1e-6
        model2.feat_scale.fill_(sc2)
        model2.beta.data.fill_(float(mean_rate))
        # Larger init so z has range; no tiny scale
        model2.w_pos.weight.data.zero_()
        model2.w_neg.weight.data.zero_()
        model2.alpha_pos.data.fill_(1.0)
        model2.alpha_neg.data.fill_(1.0)
        # Small random so z has some range but rhat doesn't explode (feat_scale normalizes)
        model2.w_pos.weight.data.add_(1e-3 * torch.randn_like(model2.w_pos.weight.data, device=DEVICE))
        model2.w_neg.weight.data.add_(1e-3 * torch.randn_like(model2.w_neg.weight.data, device=DEVICE))

    opt = torch.optim.Adam(model2.parameters(), lr=1e-3)
    loss_fn = MaskedPoissonNLLLoss(pred_key="rhat", target_key="robs", mask_key="dfs")
    for ep in range(50):
        model2.train()
        opt.zero_grad()
        batch = {"stim": stim_sub, "robs": robs_sub, "dfs": dfs_sub}
        out = model2(batch)
        loss = loss_fn(out)
        loss.backward()
        opt.step()
        with torch.no_grad():
            agg = PoissonBPSAggregator(device=DEVICE)
            agg(out)
            bps = agg.closure().item()
        rhat = out["rhat"]
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Epoch {ep+1} train_bps={bps:.4f} rhat_std={rhat.std().item():.6f} loss={loss.item():.6f}")

    print("\nDone. If overfit BPS stayed negative, the readout cannot explain variance with current feature/init.")


if __name__ == "__main__":
    main()
