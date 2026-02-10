"""
Train two-stage (pyramid + LN) model for cell 14.
Same data pipeline as train_ln.py (centered 30x30, robs/dfs for cell 14).
Run from VisionCore root: uv run python tejas/model/cursor_ln_two_stage/train_two_stage.py
"""
import os
import sys
from copy import deepcopy

_VISIONCORE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _VISIONCORE_ROOT not in sys.path:
    sys.path.insert(0, _VISIONCORE_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tejas.model.cursor_ln_two_stage.data_utils import load_and_center_for_cell
from tejas.model.cursor_ln_two_stage.models_two_stage import TwoStage, sparsity_penalty, locality_penalty
from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator

CELL_IDX = 14
DIM_CENTER = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 8
LAMBDA_REG = 5e-5  # sparsity+locality; avoid over-regularizing early
GAMMA_LOCALITY = 0.01  # paper: locality ~1% of sparsity magnitude
N_REG_SAMPLES = 3000  # samples for LN-regression readout init
RIDGE_LAM = 0.1  # ridge for readout regression


def _cg_ridge(X, y, ridge_lam, tol=1e-5, max_iter=500):
    """Solve (X'X + lam*I) w = X'y by conjugate gradient without forming X'X. X: (N, D), y: (N,)."""
    device = X.device
    Xty = X.T @ y
    w = torch.zeros(X.shape[1], device=device, dtype=X.dtype)
    r = Xty - (X.T @ (X @ w) + ridge_lam * w)
    p = r.clone()
    rs_old = (r * r).sum().item()
    for _ in range(max_iter):
        Ap = X.T @ (X @ p) + ridge_lam * p
        alpha = rs_old / ((p * Ap).sum().item() + 1e-12)
        w = w + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum().item()
        if rs_new < tol * tol * (Xty * Xty).sum().item():
            break
        beta = rs_new / (rs_old + 1e-12)
        p = r + beta * p
        rs_old = rs_new
    return w


def init_readout_from_ln_regression(model, sta_cell, stim_tr, device, n_samples=N_REG_SAMPLES, ridge_lam=RIDGE_LAM):
    """Initialize w_pos, w_neg by regressing pyramid features onto LN generator (stim @ sta). Uses CG to avoid forming X'X."""
    N = min(n_samples, stim_tr.shape[0])
    sta_flat = torch.from_numpy(sta_cell).float().to(device).reshape(-1)
    sta_flat = sta_flat / (sta_flat.norm() + 1e-8)
    stim_sub = stim_tr[:N].to(device)
    if stim_sub.ndim == 5:
        stim_flat = stim_sub.reshape(N, -1)
    else:
        stim_flat = stim_sub.reshape(N, -1)
    gen_ln = (stim_flat @ sta_flat).detach()  # (N,)
    batch = {"stim": stim_sub, "robs": None, "dfs": None}
    with torch.no_grad():
        pos_f, neg_f, _ = model.get_pyr_feats(batch)
    F = pos_f.shape[1]
    X = torch.cat([pos_f, neg_f], dim=1)  # (N, 2*F)
    y = gen_ln
    w_all = _cg_ridge(X, y, ridge_lam)
    w_pos = w_all[:F].clamp(min=0.0)
    w_neg = w_all[F:].clamp(min=0.0)
    # Scale so initial z has std ~1 (avoid huge/small rhat)
    z_init = (pos_f @ w_pos + neg_f @ w_neg)
    z_std = z_init.std().item() + 1e-6
    scale = 1.0 / z_std
    w_pos = w_pos * scale
    w_neg = w_neg * scale
    with torch.no_grad():
        model.w_pos.weight.data.copy_(w_pos.unsqueeze(0))
        model.w_neg.weight.data.copy_(w_neg.unsqueeze(0))
    return 1.0  # z_std after scaling is ~1


def main():
    print("Loading and centering data for cell", CELL_IDX, "...")
    train_centered, val_centered, sta_cell, dataset_config = load_and_center_for_cell(
        cell_idx=CELL_IDX, dim_for_centering=DIM_CENTER
    )
    n_lags = len(dataset_config["keys_lags"]["stim"])

    # Stim: (N, lags, 30, 30) -> (N, 1, lags, 30, 30) for TwoStage
    stim_tr = train_centered["stim"]
    if stim_tr.ndim == 4:
        stim_tr = stim_tr.unsqueeze(1)
    robs_tr = train_centered["robs"][:, [CELL_IDX]]
    dfs_tr = train_centered["dfs"][:, [CELL_IDX]]

    stim_val = val_centered["stim"]
    if stim_val.ndim == 4:
        stim_val = stim_val.unsqueeze(1)
    robs_val = val_centered["robs"][:, [CELL_IDX]]
    dfs_val = val_centered["dfs"][:, [CELL_IDX]]

    sta_flat = torch.from_numpy(sta_cell).float().to(DEVICE).reshape(-1)
    sta_flat = sta_flat / (sta_flat.norm() + 1e-8)

    def add_generator(batch_dict, stim_b):
        B = stim_b.shape[0]
        s = stim_b.reshape(B, -1).to(sta_flat.device)
        batch_dict["generator"] = (s @ sta_flat).detach()

    loss_fn = MaskedPoissonNLLLoss(pred_key="rhat", target_key="robs", mask_key="dfs")
    train_dset = TensorDataset(stim_tr, robs_tr, dfs_tr)
    train_loader = DataLoader(
        train_dset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    image_shape = (DIM_CENTER, DIM_CENTER)
    # 30x30 allows at most height=2 (plenoptic pyramid limit)
    model = TwoStage(
        image_shape=image_shape,
        n_neurons=1,
        n_lags=n_lags,
        height=2,
        order=5,
    ).to(DEVICE)

    # Set feature scale from first batch
    with torch.no_grad():
        first_batch = {
            "stim": stim_tr[:BATCH_SIZE].to(DEVICE),
            "robs": robs_tr[:BATCH_SIZE].to(DEVICE),
            "dfs": dfs_tr[:BATCH_SIZE].to(DEVICE),
        }
        pos_f, neg_f, _ = model.get_pyr_feats(first_batch)
        sc = ((pos_f.std().item() + neg_f.std().item()) / 2.0) + 1e-6
        model.feat_scale.fill_(sc)
    # Init readout by regressing pyramid features onto LN generator (stim @ sta)
    z_std_init = init_readout_from_ln_regression(model, sta_cell, stim_tr, DEVICE, n_samples=N_REG_SAMPLES, ridge_lam=RIDGE_LAM)
    print(f"LN-regression readout init done (z_std after scale={z_std_init:.4f})")
    # Set beta/alpha so initial rhat has mean ~ mean(robs) and non-zero variance (avoid collapse to constant)
    with torch.no_grad():
        b0 = {"stim": stim_tr[:1024].to(DEVICE), "robs": robs_tr[:1024].to(DEVICE), "dfs": dfs_tr[:1024].to(DEVICE)}
        pos_f, neg_f, _ = model.get_pyr_feats(b0)
        z0 = (model.w_pos(pos_f) + model.w_neg(neg_f)).squeeze()
        sp = F.softplus(z0)
        mean_robs = robs_tr[:1024].float().mean().item()
        mean_sp = sp.mean().item() + 1e-6
        model.beta.zero_()
        model.alpha_pos.data.fill_(max(0.1, mean_robs / mean_sp))
    # Set gen_scale and turn on LN path so model can get positive BPS
    with torch.no_grad():
        first_batch = {"stim": stim_tr[:BATCH_SIZE].to(DEVICE), "robs": robs_tr[:BATCH_SIZE].to(DEVICE), "dfs": dfs_tr[:BATCH_SIZE].to(DEVICE)}
        add_generator(first_batch, stim_tr[:BATCH_SIZE])
        g_std = first_batch["generator"].std().item() + 1e-6
        model.gen_scale.fill_(g_std)
        model.gamma_ln.data.fill_(1.0)
        # Re-set alpha so rhat scale is right with z_total = z_pyr + gen_ln
        pos_f, neg_f, _ = model.get_pyr_feats(first_batch)
        z_pyr = (model.w_pos(pos_f) + model.w_neg(neg_f)).squeeze()
        z_total = z_pyr + model.gamma_ln * (first_batch["generator"].to(DEVICE) / model.gen_scale)
        sp0 = F.softplus(z_total)
        model.beta.zero_()
        mean_robs0 = robs_tr[:BATCH_SIZE].float().mean().item()
        model.alpha_pos.data.fill_(max(0.1, mean_robs0 / (sp0.mean().item() + 1e-6)))
    # Fit output scale/bias with weights frozen
    for p in model.w_pos.parameters():
        p.requires_grad = False
    for p in model.w_neg.parameters():
        p.requires_grad = False
    # Freeze alpha_pos so we only fit beta (prevents collapse to constant)
    model.alpha_pos.requires_grad = False
    opt_init = torch.optim.Adam([model.beta], lr=1e-2)
    best_init_val = -float("inf")
    init_best_state = None
    for ep in range(10):
        model.train()
        for stim_b, robs_b, dfs_b in train_loader:
            opt_init.zero_grad()
            batch_init = {"stim": stim_b.to(DEVICE), "robs": robs_b.to(DEVICE), "dfs": dfs_b.to(DEVICE)}
            add_generator(batch_init, stim_b)
            out_init = model(batch_init)
            loss = loss_fn(out_init)
            loss.backward()
            opt_init.step()
        model.eval()
        with torch.no_grad():
            train_agg = PoissonBPSAggregator(device=DEVICE)
            for stim_b, robs_b, dfs_b in train_loader:
                b = {"stim": stim_b.to(DEVICE), "robs": robs_b.to(DEVICE), "dfs": dfs_b.to(DEVICE)}
                add_generator(b, stim_b)
                train_agg(model(b))
            tbps = train_agg.closure().item()
            val_agg = PoissonBPSAggregator(device=DEVICE)
            for stim_b, robs_b, dfs_b in DataLoader(TensorDataset(stim_val, robs_val, dfs_val), batch_size=BATCH_SIZE):
                b = {"stim": stim_b.to(DEVICE), "robs": robs_b.to(DEVICE), "dfs": dfs_b.to(DEVICE)}
                add_generator(b, stim_b)
                val_agg(model(b))
            vbps = val_agg.closure().item()
        if vbps > best_init_val:
            best_init_val = vbps
            init_best_state = deepcopy(model.state_dict())
        print(f"  Init fit epoch {ep+1} train_bps={tbps:.4f} val_bps={vbps:.4f} best_val={best_init_val:.4f}", flush=True)
        if ep >= 3 and vbps <= best_init_val - 1e-5:
            break
    if init_best_state is not None:
        model.load_state_dict(init_best_state)
    print(f"Fitted beta/alpha with frozen STA. Best val_bps={best_init_val:.4f}")

    # Unfreeze readout with very small lr; add variance penalty to prevent rhat collapse
    model.alpha_pos.requires_grad = True
    for p in model.w_pos.parameters():
        p.requires_grad = True
    for p in model.w_neg.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam([
        {"params": [model.w_pos.weight, model.w_neg.weight], "lr": 5e-7},
        {"params": [model.beta, model.alpha_pos, model.alpha_neg, model.gamma_ln], "lr": 1e-3},
    ])
    use_schedulefree = False
    TARGET_RHAT_STD = 0.01  # penalize if rhat std below this (keep predictions varying)
    LAMBDA_VAR = 1.0
    best_val_bps = -float("inf")
    best_state = None
    epochs_no_improve = 0
    last_train_bps = None
    last_val_bps = None

    for epoch in range(EPOCHS):
        if use_schedulefree:
            optimizer.train()
        model.train()
        train_agg = PoissonBPSAggregator(device=DEVICE)
        for stim_b, robs_b, dfs_b in train_loader:
            batch = {
                "stim": stim_b.to(DEVICE),
                "robs": robs_b.to(DEVICE),
                "dfs": dfs_b.to(DEVICE),
            }
            add_generator(batch, stim_b)
            optimizer.zero_grad()
            out = model(batch)
            poisson_loss = loss_fn(out)
            rhat_std_b = out["rhat"].std().clamp(min=1e-8)
            var_penalty = LAMBDA_VAR * F.relu(TARGET_RHAT_STD - rhat_std_b)
            loss = poisson_loss + var_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                train_agg(out)
        train_bps = train_agg.closure().item()

        model.eval()
        if use_schedulefree:
            optimizer.eval()
        val_agg = PoissonBPSAggregator(device=DEVICE)
        with torch.no_grad():
            val_loader = DataLoader(
                TensorDataset(stim_val, robs_val, dfs_val),
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
            for stim_b, robs_b, dfs_b in val_loader:
                val_batch = {
                    "stim": stim_b.to(DEVICE),
                    "robs": robs_b.to(DEVICE),
                    "dfs": dfs_b.to(DEVICE),
                }
                add_generator(val_batch, stim_b)
                out_val = model(val_batch)
                val_agg(out_val)
            val_bps = val_agg.closure().item()

        if val_bps > best_val_bps:
            best_val_bps = val_bps
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        with torch.no_grad():
            b = {"stim": stim_tr[:256].to(DEVICE), "robs": robs_tr[:256].to(DEVICE), "dfs": dfs_tr[:256].to(DEVICE)}
            add_generator(b, stim_tr[:256])
            r = model(b)["rhat"]
            rhat_std = r.std().item()
        print(f"Epoch {epoch+1} train_bps={train_bps:.4f} val_bps={val_bps:.4f} best_val_bps={best_val_bps:.4f} rhat_std={rhat_std:.6f}", flush=True)

        if rhat_std < 1e-6:
            print("rhat collapsed to constant; stopping.", flush=True)
            break
        if last_train_bps is not None and abs(train_bps - last_train_bps) < 1e-5 and abs(val_bps - last_val_bps) < 1e-5:
            epochs_no_improve += 1
            if epochs_no_improve >= 2:
                print("No BPS change for 2+ epochs; stopping.", flush=True)
                break
        last_train_bps, last_val_bps = train_bps, val_bps

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model with val_bps={best_val_bps:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
