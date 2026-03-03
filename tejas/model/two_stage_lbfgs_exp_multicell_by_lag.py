import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
from DataYatesV1 import enable_autoreload

enable_autoreload()
from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator
from tqdm import tqdm
from torch.utils.data import DataLoader
from util import get_dataset_info
import torch
import numpy as np
import math

# %%
dataset_configs_path = "/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml"
subject = "Allen"
date = "2022-04-13"
image_shape = (41, 41)
dataset_info = get_dataset_info(dataset_configs_path, subject, date, image_shape)
peak_lags = dataset_info["peak_lags"]
train_dset = dataset_info["train_dset"]
val_dset = dataset_info["val_dset"]
robs = dataset_info["robs"]
dfs = dataset_info["dfs"]
crop_size = dataset_info["crop_size"]

# %%
from two_stage_core import TwoStage
from two_stage_trainer import eval_step, train_step_lbfgs

spike_loss = MaskedPoissonNLLLoss(pred_key="rhat", target_key="robs", mask_key="dfs")

lambda_reg_lbfgs = 0.0
gamma_local_lbfgs = 0.0
circular_dims = {1}
locality_mode = "weighted_l21"
sparsity_mode = "prox_l1"
lbfgs_step_mode = "full_dataset_accum"
num_epochs = int(os.getenv("NUM_EPOCHS", "4"))
batch_size_lbfgs = int(os.getenv("BATCH_SIZE", "2048"))

cell_lambda_prox = {14: 1.00e-04, 15: 1.00e-04, 16: 1.00e-04, 66: 2.68e-05, 76: 1.93e-06}
cell_lambda_local = {14: 1.00e-04, 15: 1.00e-04, 16: 1.00e-04, 66: 1.00e-04, 76: 1.00e-04}


def parse_cell_ids(raw, n_total_cells):
    s = str(raw).strip().lower()
    if s in {"all", "*"}:
        return list(range(int(n_total_cells)))
    ids = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not ids:
        raise ValueError("CELL_IDS must be 'all' or comma-separated indices.")
    for cid in ids:
        if cid < 0 or cid >= n_total_cells:
            raise ValueError(f"cell id {cid} out of range [0, {n_total_cells - 1}]")
    return sorted(set(ids))


def crop_slice(c):
    return slice(c, -c) if int(c) > 0 else slice(None)


def prepare_batch_one_lag(batch, lag_index, crop_size, device="cuda"):
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    ys = crop_slice(crop_size)
    xs = crop_slice(crop_size)
    b["stim"] = b["stim"][:, :, [int(lag_index)], ys, xs]
    return b


def calibrate_exp_intercept_full_dataset(model, train_loader, lag_index, group_cell_ids, eps=1e-12):
    pred_sum = None
    target_sum = None
    with torch.no_grad():
        for batch in train_loader:
            batch = prepare_batch_one_lag(
                batch=batch,
                lag_index=lag_index,
                crop_size=crop_size,
                device="cuda",
            )
            out = eval_step(model=model, batch=batch, cell_ids=group_cell_ids, use_resolver=True)
            mask = out["dfs"]
            pred_vec = (out["rhat"] * mask).sum(dim=0)
            target_vec = (out["robs"] * mask).sum(dim=0)
            if pred_sum is None:
                pred_sum = pred_vec
                target_sum = target_vec
            else:
                pred_sum = pred_sum + pred_vec
                target_sum = target_sum + target_vec
    correction = torch.log((target_sum + eps) / (pred_sum + eps))
    with torch.no_grad():
        model.beta.add_(correction.to(device=model.beta.device, dtype=model.beta.dtype))


def cell_bps_table(bps_tensor, group_cell_ids):
    vals = np.asarray(bps_tensor).reshape(-1)
    return {int(cid): float(vals[i]) for i, cid in enumerate(group_cell_ids)}


n_total_cells = int(robs.shape[1])
cell_ids = parse_cell_ids(os.getenv("CELL_IDS", "all"), n_total_cells=n_total_cells)
peak_lags_np = np.asarray(peak_lags).reshape(-1)
_stim_shape = tuple(train_dset[0]["stim"].shape)
if len(_stim_shape) == 4:
    stim_lag_count = int(_stim_shape[1])
elif len(_stim_shape) == 5:
    stim_lag_count = int(_stim_shape[2])
else:
    raise ValueError(f"Unexpected stim shape for lag counting: {_stim_shape}")
max_valid_lag = max(0, stim_lag_count - 1)
print(f"stim lag count={stim_lag_count}, max valid lag index={max_valid_lag}")

lag_to_cells = {}
for cid in cell_ids:
    lag = int(np.clip(int(peak_lags_np[cid]), 0, max_valid_lag))
    lag_to_cells.setdefault(lag, []).append(int(cid))

print(f"selected {len(cell_ids)} cells across {len(lag_to_cells)} peak-lag groups")
for lag, ids in sorted(lag_to_cells.items()):
    print(f"lag={lag}: {len(ids)} cells")

train_dset.to("cpu")
val_dset.to("cpu")
train_loader_lbfgs_accum = DataLoader(
    train_dset,
    batch_size=batch_size_lbfgs,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=4,
)
train_loader_lbfgs_legacy = DataLoader(
    train_dset,
    batch_size=batch_size_lbfgs,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=4,
)
val_loader_lbfgs = DataLoader(
    val_dset,
    batch_size=batch_size_lbfgs,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=4,
)

all_best_val = {}

for lag_index, group_cell_ids in sorted(lag_to_cells.items()):
    print(f"\n[group] lag={lag_index} n_cells={len(group_cell_ids)} cells={group_cell_ids[:12]}")
    n_neurons = len(group_cell_ids)

    model = TwoStage(
        image_shape=image_shape,
        n_neurons=n_neurons,
        n_lags=1,
        height=3,
        order=5,
        lowest_cpd_target=1.0,
        ppd=train_dset.dsets[0].metadata["ppd"],
        rel_tolerance=0.3,
        validate_cpd=True,
        beta_init=0.0,
        init_weight_scale=1e-4,
        beta_as_parameter=True,
        clamp_beta_min=None,
        hann_window_power=2,
        output_nonlinearity="exp",
    )
    model.cuda()

    eps = 1e-12
    masked_rate_mean = (
        np.asarray((robs[:, group_cell_ids] * dfs[:, group_cell_ids]).sum(axis=0), dtype=np.float64)
        / np.maximum(np.asarray(dfs[:, group_cell_ids].sum(axis=0), dtype=np.float64), eps)
    )
    beta_init_vec = np.log(np.maximum(masked_rate_mean, 1e-8))
    with torch.no_grad():
        model.beta.copy_(torch.tensor(beta_init_vec, device=model.beta.device, dtype=model.beta.dtype))

    calibrate_exp_intercept_full_dataset(
        model=model,
        train_loader=train_loader_lbfgs_accum,
        lag_index=lag_index,
        group_cell_ids=group_cell_ids,
        eps=eps,
    )

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=float(os.getenv("LBFGS_LR", "0.1")),
        max_iter=int(os.getenv("LBFGS_MAX_ITER", "10")),
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    lambda_prox_vec = torch.tensor(
        [float(cell_lambda_prox.get(int(cid), 1e-4)) for cid in group_cell_ids],
        device=model.w_pos.weight.device,
        dtype=model.w_pos.weight.dtype,
    ).unsqueeze(1)
    lambda_local_vec = torch.tensor(
        [float(cell_lambda_local.get(int(cid), 1e-4)) for cid in group_cell_ids],
        device=model.w_pos.weight.device,
        dtype=model.w_pos.weight.dtype,
    )

    best_val_bps_mean = -1e9
    best_val_bps_cells = None
    best_epoch = -1

    for epoch in range(num_epochs):
        train_agg = PoissonBPSAggregator()
        val_agg = PoissonBPSAggregator()
        poisson_last = reg_last = prox_tau_last = 0.0

        active_train_loader = (
            train_loader_lbfgs_accum
            if lbfgs_step_mode == "full_dataset_accum"
            else train_loader_lbfgs_legacy
        )

        if lbfgs_step_mode != "full_dataset_accum":
            raise ValueError(f"Unsupported lbfgs_step_mode: {lbfgs_step_mode}")

        model.train()
        model.alpha_pos.requires_grad = False
        step_stats, _ = train_step_lbfgs(
            model=model,
            optimizer=optimizer_lbfgs,
            batch=None,
            batch_loader=active_train_loader,
            prepare_batch_fn=lambda b: prepare_batch_one_lag(
                batch=b,
                lag_index=lag_index,
                crop_size=crop_size,
                device="cuda",
            ),
            spike_loss=spike_loss,
            cell_ids=group_cell_ids,
            sparsity_mode=sparsity_mode,
            lambda_reg=lambda_reg_lbfgs,
            lambda_local_prox=0.0,
            circular_dims=circular_dims,
            gamma_mode="fixed",
            gamma_value=gamma_local_lbfgs,
            lambda_prox=lambda_prox_vec,
            use_resolver=True,
            locality_mode=locality_mode,
            poisson_aggregation_mode="sum_per_cell_means",
            lambda_local_per_neuron=lambda_local_vec,
        )
        poisson_last = step_stats["poisson"]
        reg_last = step_stats["reg"]
        prox_tau_last = step_stats["prox_tau"]

        for batch in tqdm(active_train_loader):
            model.eval()
            batch = prepare_batch_one_lag(
                batch=batch,
                lag_index=lag_index,
                crop_size=crop_size,
                device="cuda",
            )
            out = eval_step(model=model, batch=batch, cell_ids=group_cell_ids, use_resolver=True)
            train_agg(out)

        for batch in val_loader_lbfgs:
            model.eval()
            batch = prepare_batch_one_lag(
                batch=batch,
                lag_index=lag_index,
                crop_size=crop_size,
                device="cuda",
            )
            out = eval_step(model=model, batch=batch, cell_ids=group_cell_ids, use_resolver=True)
            val_agg(out)

        bps_train = train_agg.closure().detach().cpu().numpy()
        bps_val = val_agg.closure().detach().cpu().numpy()
        val_mean = float(np.asarray(bps_val).mean())
        if val_mean > best_val_bps_mean:
            best_val_bps_mean = val_mean
            best_val_bps_cells = np.asarray(bps_val).copy()
            best_epoch = epoch

        print(
            f"lag={lag_index:02d} epoch={epoch:03d} "
            f"train_mean={float(np.asarray(bps_train).mean()):.4f} "
            f"val_mean={val_mean:.4f} "
            f"poisson={poisson_last:.6f} reg={reg_last:.6f} prox_tau_mean={prox_tau_last:.6e}"
        )

    best_table = cell_bps_table(best_val_bps_cells, group_cell_ids)
    for cid, v in best_table.items():
        all_best_val[int(cid)] = float(v)
    print(
        f"[group final] lag={lag_index} best_epoch={best_epoch} "
        f"best_val_mean={best_val_bps_mean:.4f} "
        f"min_cell={min(best_table.values()):.4f} max_cell={max(best_table.values()):.4f}"
    )

summary_vals = [all_best_val[cid] for cid in sorted(all_best_val.keys())]
print(
    f"\n[final] trained_cells={len(all_best_val)} "
    f"val_bps_mean={float(np.mean(summary_vals)):.4f} "
    f"val_bps_median={float(np.median(summary_vals)):.4f} "
    f"val_bps_min={float(np.min(summary_vals)):.4f}"
)

