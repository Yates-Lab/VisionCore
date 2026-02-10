"""
Run 3 epochs max; print input/intermediate/output stats and gradient norms.
STOP if BPS unchanged or gradients near zero. Run from VisionCore root.
"""
import os
import sys
_VISIONCORE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _VISIONCORE_ROOT not in sys.path:
    sys.path.insert(0, _VISIONCORE_ROOT)

import torch
from torch.utils.data import DataLoader, TensorDataset

from tejas.model.cursor_ln_two_stage.data_utils import load_and_center_for_cell
from tejas.model.cursor_ln_two_stage.models_two_stage import TwoStage, sparsity_penalty, locality_penalty
from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator

CELL_IDX = 14
DIM = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
MAX_EPOCHS = 3
LAMBDA_REG = 5e-5
GAMMA_LOCALITY = 0.01


def stats(name, t):
    t = t.detach().float()
    return (
        f"{name}: mean={t.mean().item():.6f} std={t.std().item():.6f} "
        f"min={t.min().item():.6f} max={t.max().item():.6f}"
    )


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

    loss_fn = MaskedPoissonNLLLoss(pred_key="rhat", target_key="robs", mask_key="dfs")
    train_loader = DataLoader(TensorDataset(stim_tr, robs_tr, dfs_tr), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    model = TwoStage((DIM, DIM), 1, n_lags=n_lags, height=2, order=5).to(DEVICE)

    with torch.no_grad():
        b0 = {"stim": stim_tr[:BATCH_SIZE].to(DEVICE), "robs": robs_tr[:BATCH_SIZE].to(DEVICE), "dfs": dfs_tr[:BATCH_SIZE].to(DEVICE)}
        pos_f, neg_f, _ = model.get_pyr_feats(b0)
        sc = (pos_f.std().item() + neg_f.std().item()) / 2.0 + 1e-6
        model.feat_scale.fill_(sc)
        w_pos, w_neg = model.sta_in_pyramid_weights(sta_cell, DEVICE)
        model.w_pos.weight.data.copy_(w_pos.unsqueeze(0))
        model.w_neg.weight.data.copy_(w_neg.unsqueeze(0))
        mean_rate = (robs_tr * dfs_tr).sum().item() / (dfs_tr.sum().clamp(min=1e-8).item())
        model.beta.data.fill_(mean_rate)
        model.alpha_pos.data.fill_(0.5)
        model.alpha_neg.data.fill_(0.5)

    # ---- Single-batch diagnostic: full pipeline stats ----
    print("\n--- Single-batch stats (first batch) ---")
    with torch.no_grad():
        s = b0["stim"]
        print(stats("stim", s))
        print(stats("robs", b0["robs"]))
        pos_f, neg_f, _ = model.get_pyr_feats(b0)
        print(stats("pos_feats", pos_f))
        print(stats("neg_feats", neg_f))
        z = model.w_pos(pos_f) + model.w_neg(neg_f)
        print(stats("z", z))
        rhat = (model.beta + model.alpha_pos * torch.relu(z) + model.alpha_neg * torch.relu(-z)).clamp(min=1e-6)
        print(stats("rhat", rhat))

    rhat_std = rhat.std().item()
    if rhat_std < 1e-5:
        print("\n*** BOTTLENECK: rhat is constant (std < 1e-5). Stopping.")
        return
    # Cross-sample variance of feats (do feats differ across samples?)
    pos_std_per_sample = pos_f.std(dim=1)
    print(f"\npos_feats std per sample: mean={pos_std_per_sample.mean().item():.6f} min={pos_std_per_sample.min().item():.6f}")
    if pos_std_per_sample.min().item() < 1e-6:
        print("*** WARNING: some samples have near-constant pos_feats.")

    # ---- Gradient check (no step - just report norms) ----
    print("\n--- Gradient norms (one backward, no step) ---")
    model.train()
    stim_b, robs_b, dfs_b = next(iter(train_loader))
    batch = {"stim": stim_b.to(DEVICE), "robs": robs_b.to(DEVICE), "dfs": dfs_b.to(DEVICE)}
    out = model(batch)
    loss = loss_fn(out)
    loss.backward()
    gn_wp = model.w_pos.weight.grad.norm().item() if model.w_pos.weight.grad is not None else 0.0
    gn_wn = model.w_neg.weight.grad.norm().item() if model.w_neg.weight.grad is not None else 0.0
    print(f"grad norm w_pos={gn_wp:.6e} w_neg={gn_wn:.6e}")
    if gn_wp < 1e-10 and gn_wn < 1e-10:
        print("\n*** BOTTLENECK: gradients to readout are zero. Stopping.")
        return

    # ---- Short training: 3 epochs, stop if no change ----
    print("\n--- Training (max 3 epochs, stop if BPS flat) ---")
    optimizer = torch.optim.Adam([
        {"params": [model.w_pos.weight, model.w_neg.weight], "lr": 1e-6},
        {"params": [model.beta, model.alpha_pos, model.alpha_neg], "lr": 1e-2},
    ])
    best_train_bps = -float("inf")
    best_val_bps = -float("inf")
    prev_train_bps = None
    prev_val_bps = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_agg = PoissonBPSAggregator(device=DEVICE)
        for stim_b, robs_b, dfs_b in train_loader:
            batch = {"stim": stim_b.to(DEVICE), "robs": robs_b.to(DEVICE), "dfs": dfs_b.to(DEVICE)}
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out) + LAMBDA_REG * sparsity_penalty(model) * (1.0 + GAMMA_LOCALITY * locality_penalty(model))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_agg(out)
        train_bps = train_agg.closure().item()

        model.eval()
        val_agg = PoissonBPSAggregator(device=DEVICE)
        with torch.no_grad():
            for stim_b, robs_b, dfs_b in DataLoader(TensorDataset(stim_val, robs_val, dfs_val), batch_size=BATCH_SIZE):
                val_agg(model({"stim": stim_b.to(DEVICE), "robs": robs_b.to(DEVICE), "dfs": dfs_b.to(DEVICE)}))
        val_bps = val_agg.closure().item()

        with torch.no_grad():
            b = {"stim": stim_tr[:256].to(DEVICE), "robs": robs_tr[:256].to(DEVICE), "dfs": dfs_tr[:256].to(DEVICE)}
            r = model(b)["rhat"]
        print(f"Epoch {epoch+1} train_bps={train_bps:.4f} val_bps={val_bps:.4f} rhat_std={r.std().item():.6f}")

        if prev_train_bps is not None and abs(train_bps - prev_train_bps) < 1e-6 and abs(val_bps - prev_val_bps) < 1e-6:
            print("\n*** No change in BPS; stopping to save compute.")
            break
        prev_train_bps, prev_val_bps = train_bps, val_bps
        if train_bps > best_train_bps:
            best_train_bps = train_bps
        if val_bps > best_val_bps:
            best_val_bps = val_bps

    print(f"\nBest train_bps={best_train_bps:.4f} best_val_bps={best_val_bps:.4f}")


if __name__ == "__main__":
    main()
