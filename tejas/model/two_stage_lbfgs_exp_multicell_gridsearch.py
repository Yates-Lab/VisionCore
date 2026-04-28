import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# %%
from DataYatesV1 import enable_autoreload

enable_autoreload()
from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator
from tqdm import tqdm
from torch.utils.data import DataLoader
from util import get_dataset_info
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import time
import pickle

# %%
dataset_configs_path = os.getenv(
    "DATASET_CONFIGS_PATH",
    "/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml",
)
subject = os.getenv("SUBJECT", "Allen")
date = os.getenv("DATE", "2022-04-13")
image_shape = tuple(
    int(x.strip()) for x in os.getenv("IMAGE_SHAPE", "41,41").split(",") if x.strip()
)
dataset_info = get_dataset_info(dataset_configs_path, subject, date, image_shape)
peak_lags = dataset_info["peak_lags"]
stas = dataset_info["stas"]
stes = dataset_info["stes"]
train_dset = dataset_info["train_dset"]
val_dset = dataset_info["val_dset"]
robs = dataset_info["robs"]
dfs = dataset_info["dfs"]
crop_size = dataset_info["crop_size"]

# %%
from two_stage_core import TwoStage
from two_stage_helpers import show_epoch_diagnostics
from two_stage_trainer import eval_step, train_step_lbfgs

spike_loss = MaskedPoissonNLLLoss(pred_key="rhat", target_key="robs", mask_key="dfs")

lambda_reg_lbfgs = float(os.getenv("LAMBDA_REG_LBFGS", "0.0"))
gamma_local_lbfgs = float(os.getenv("GAMMA_LOCAL_LBFGS", "0.0"))
circular_dims = {1}
locality_mode = os.getenv("LOCALITY_MODE", "weighted_l21")
sparsity_mode = os.getenv("SPARSITY_MODE", "prox_l1")
lbfgs_step_mode = "full_dataset_accum"
output_nonlinearity = os.getenv("OUTPUT_NONLINEARITY", "exp").strip().lower()
num_workers = int(os.getenv("NUM_WORKERS", "8"))
pin_memory = str(os.getenv("PIN_MEMORY", "1")).strip().lower() not in {"0", "false", "no"}
persistent_workers = str(os.getenv("PERSISTENT_WORKERS", "0")).strip().lower() in {"1", "true", "yes"}
prefetch_factor_env = os.getenv("PREFETCH_FACTOR")
prefetch_factor = int(prefetch_factor_env) if prefetch_factor_env is not None else 4
save_best_pngs = str(os.getenv("SAVE_BEST_PNGS", "1")).strip().lower() not in {"0", "false", "no"}
skip_train_eval = str(os.getenv("SKIP_TRAIN_EVAL", "0")).strip().lower() in {"1", "true", "yes"}
save_png_mode = os.getenv("SAVE_PNG_MODE", "best").strip().lower()  # "best" | "last_epoch"
precompute_pyramid = str(os.getenv("PRECOMPUTE_PYRAMID", "0")).strip().lower() in {"1", "true", "yes"}
precompute_denominator = str(os.getenv("PRECOMPUTE_DENOMINATOR", "0")).strip().lower() in {"1", "true", "yes"}
hann_window_power = float(os.getenv("HANN_WINDOW_POWER", "2"))
model_height = int(os.getenv("PYR_LEVELS", "3"))
cache_dtype_name = os.getenv("CACHE_DTYPE", "float16").strip().lower()
if cache_dtype_name == "float32":
    cache_dtype = torch.float32
elif cache_dtype_name == "float16":
    cache_dtype = torch.float16
else:
    raise ValueError(f"Unsupported CACHE_DTYPE: {cache_dtype_name}")

num_epochs = int(os.getenv("NUM_EPOCHS", "1"))
batch_size_lbfgs = int(os.getenv("BATCH_SIZE", "2048"))
lbfgs_lr = float(os.getenv("LBFGS_LR", "0.1"))
lbfgs_max_iter = int(os.getenv("LBFGS_MAX_ITER", "5"))
lbfgs_history_size = int(os.getenv("LBFGS_HISTORY_SIZE", "10"))
lbfgs_tol_grad = float(os.getenv("LBFGS_TOL_GRAD", "1e-7"))
lbfgs_tol_change = float(os.getenv("LBFGS_TOL_CHANGE", "1e-9"))
lbfgs_max_eval_env = os.getenv("LBFGS_MAX_EVAL")
lbfgs_max_eval = int(lbfgs_max_eval_env) if lbfgs_max_eval_env is not None else None
seed = int(os.getenv("SEED", "0"))
use_group_seed = str(os.getenv("USE_GROUP_SEED", "1")).strip().lower() not in {"0", "false", "no"}
time_breakdown = str(os.getenv("TIME_BREAKDOWN", "0")).strip().lower() in {"1", "true", "yes"}
init_torch_rng_state_path = os.getenv("INIT_TORCH_RNG_STATE_PATH")
init_numpy_rng_state_path = os.getenv("INIT_NUMPY_RNG_STATE_PATH")
save_best_checkpoints = str(os.getenv("SAVE_BEST_CHECKPOINTS", "0")).strip().lower() in {"1", "true", "yes"}
checkpoint_dir_env = os.getenv("CHECKPOINT_DIR")
output_dir = Path(
    os.getenv(
        "OUTPUT_DIR",
        "/home/tejas/VisionCore/tejas/model/all_cell_training_png",
    )
)
output_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = Path(checkpoint_dir_env) if checkpoint_dir_env else (output_dir / "checkpoints")
if save_best_checkpoints:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

timings = {
    "intercept_calibration_s": 0.0,
    "lbfgs_step_s": 0.0,
    "train_eval_pass_s": 0.0,
    "val_eval_pass_s": 0.0,
    "diagnostics_save_s": 0.0,
}
run_t0 = time.perf_counter()

prox_mults = [float(x) for x in os.getenv("PROX_MULTS", "0.5,1.0,2.0").split(",") if x.strip()]
local_mults = [float(x) for x in os.getenv("LOCAL_MULTS", "0.5,1.0,2.0").split(",") if x.strip()]

cell_lambda_prox = {14: 1.00e-04, 15: 1.00e-04, 16: 1.00e-04, 66: 2.68e-05, 76: 1.93e-06}
cell_lambda_local = {14: 1.00e-04, 15: 1.00e-04, 16: 1.00e-04, 66: 1.00e-04, 76: 1.00e-04}
default_lambda_prox = 1.00e-04
default_lambda_local = 1.00e-04
fixed_lambda_prox_env = os.getenv("FIXED_LAMBDA_PROX")
fixed_lambda_local_env = os.getenv("FIXED_LAMBDA_LOCAL")
fixed_lambda_prox = float(fixed_lambda_prox_env) if fixed_lambda_prox_env is not None else None
fixed_lambda_local = float(fixed_lambda_local_env) if fixed_lambda_local_env is not None else None


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


def _weighted_pearson_r2_per_cell(pred, target, mask, eps=1e-12):
    pred = pred.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32).clamp_min(0.0)
    sw = mask.sum(dim=0).clamp_min(eps)
    mx = (mask * pred).sum(dim=0) / sw
    my = (mask * target).sum(dim=0) / sw
    xc = pred - mx
    yc = target - my
    cov = (mask * xc * yc).sum(dim=0) / sw
    vx = (mask * xc * xc).sum(dim=0) / sw
    vy = (mask * yc * yc).sum(dim=0) / sw
    r = cov / torch.sqrt(vx * vy + eps)
    r2 = torch.clamp(r * r, min=0.0, max=1.0)
    return r2


def _extract_pos_neg_feats(model, batch):
    if "pos_feats" in batch and "neg_feats" in batch:
        return batch["pos_feats"], batch["neg_feats"]
    pos_feats, neg_feats, _ = model.get_pyr_feats(batch)
    return pos_feats, neg_feats


def _compute_linearity_tables(model, li_chunks, group_cell_ids, eps=1e-12):
    if not li_chunks:
        empty = {int(cid): float("nan") for cid in group_cell_ids}
        return (
            empty,
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
        )
    z_full = torch.cat([c["z_full"] for c in li_chunks], dim=0)
    z_lin = torch.cat([c["z_lin"] for c in li_chunks], dim=0)
    z_eng = torch.cat([c["z_eng"] for c in li_chunks], dim=0)
    robs = torch.cat([c["robs"] for c in li_chunks], dim=0)
    dfs = torch.cat([c["dfs"] for c in li_chunks], dim=0)

    r2_full = _weighted_pearson_r2_per_cell(z_full, robs, dfs, eps=eps)
    r2_lin = _weighted_pearson_r2_per_cell(z_lin, robs, dfs, eps=eps)
    r2_eng = _weighted_pearson_r2_per_cell(z_eng, robs, dfs, eps=eps)
    frac_lin = r2_lin / r2_full.clamp_min(eps)
    frac_eng = r2_eng / r2_full.clamp_min(eps)
    li = frac_lin - frac_eng

    def _table(vals):
        vals_np = vals.detach().cpu().numpy().reshape(-1)
        return {int(cid): float(vals_np[i]) for i, cid in enumerate(group_cell_ids)}

    return _table(li), _table(frac_lin), _table(frac_eng), _table(r2_full), _table(r2_lin), _table(r2_eng)


def save_cell_best_png(
    model,
    cell_id,
    neuron_idx,
    bps_train,
    bps_val,
    poisson_last,
    local_last,
    reg_last,
    prox_tau_last,
    phase,
    epoch,
    linearity_metrics=None,
    save_label="best",
):
    show_epoch_diagnostics(
        model=model,
        stas=stas,
        stes=stes,
        peak_lags=peak_lags,
        cell_id=int(cell_id),
        sparsity_mode=sparsity_mode,
        poisson_last=float(poisson_last),
        sparse_last=0.0,
        local_last=float(local_last),
        gamma_local=0.0,
        prox_tau_last=float(prox_tau_last),
        reg_last=float(reg_last),
        bps=bps_train,
        bps_val=bps_val,
        phase=phase,
        epoch=int(epoch),
        show_colorwheel=True,
        neuron_idx=int(neuron_idx),
        lag_idx=0,
        save_dir=str(output_dir),
        save_prefix=f"cell_{int(cell_id):03d}_{str(save_label)}",
        close_figs=True,
        show_plots=False,
        linearity_metrics=linearity_metrics,
    )


def save_cell_best_checkpoint(
    model,
    cell_id,
    neuron_idx,
    group_cell_ids,
    lag_index,
    phase,
    epoch,
    step_stats,
    train_bps_value,
    val_bps_value,
):
    ckpt_path = checkpoint_dir / f"cell_{int(cell_id):03d}_best.pt"
    model_state_cpu = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    torch.save(
        {
            "cell_id": int(cell_id),
            "group_cell_ids": [int(cid) for cid in group_cell_ids],
            "neuron_idx": int(neuron_idx),
            "lag_index": int(lag_index),
            "epoch": int(epoch),
            "phase": str(phase),
            "model_state": model_state_cpu,
            "model_kwargs": {
                "image_shape": tuple(image_shape),
                "n_neurons": int(len(group_cell_ids)),
                "n_lags": 1,
                "height": model_height,
                "order": 5,
                "lowest_cpd_target": 1.0,
                "ppd": float(train_dset.dsets[0].metadata["ppd"]),
                "rel_tolerance": 0.3,
                "validate_cpd": True,
                "beta_init": 0.0,
                "init_weight_scale": 1e-4,
                "beta_as_parameter": True,
                "clamp_beta_min": (None if output_nonlinearity == "exp" else 1e-6),
                "hann_window_power": float(hann_window_power),
                "output_nonlinearity": str(output_nonlinearity),
            },
            "step_stats": {k: float(v) if isinstance(v, (int, float)) else v for k, v in step_stats.items()},
            "best_train_bps": float(train_bps_value),
            "best_val_bps": float(val_bps_value),
        },
        ckpt_path,
    )


def _prepare_cached_batch(batch_cached, device, model_dtype):
    return {
        "pos_feats": batch_cached["pos_feats"].to(device=device, dtype=model_dtype, non_blocking=True),
        "neg_feats": batch_cached["neg_feats"].to(device=device, dtype=model_dtype, non_blocking=True),
        "robs": batch_cached["robs"].to(device=device, non_blocking=True),
        "dfs": batch_cached["dfs"].to(device=device, non_blocking=True),
    }


def build_feature_cache(model, loader, lag_index, crop_size, cache_dtype):
    cache = []
    with torch.no_grad():
        for batch in loader:
            batch = prepare_batch_one_lag(
                batch=batch,
                lag_index=lag_index,
                crop_size=crop_size,
                device="cuda",
            )
            pos_feats, neg_feats, _ = model.get_pyr_feats(batch)
            cache.append(
                {
                    "pos_feats": pos_feats.detach().to(device="cpu", dtype=cache_dtype),
                    "neg_feats": neg_feats.detach().to(device="cpu", dtype=cache_dtype),
                    "robs": batch["robs"].detach().to(device="cpu"),
                    "dfs": batch["dfs"].detach().to(device="cpu"),
                }
            )
    return cache


def precompute_total_den_from_cache(cache, group_cell_ids, per_cell=True):
    if per_cell:
        total_den = torch.zeros((len(group_cell_ids),), dtype=torch.float32)
    else:
        total_den = torch.zeros((), dtype=torch.float32)
    total_samples = 0
    total_batches = 0
    for b in cache:
        if per_cell:
            den = b["dfs"][:, group_cell_ids].sum(dim=0).to(dtype=torch.float32)
        else:
            den = b["dfs"][:, group_cell_ids].sum().to(dtype=torch.float32)
        total_den = total_den + den
        total_samples += int(b["robs"].shape[0])
        total_batches += 1
    return total_den.clamp_min(1.0), total_batches, total_samples


n_total_cells = int(robs.shape[1])
cell_ids = parse_cell_ids(os.getenv("CELL_IDS", "all"), n_total_cells=n_total_cells)
training_protocol = os.getenv("TRAINING_PROTOCOL", "group_by_lag").strip().lower()
if os.getenv("LBFGS_MAX_ITER") is None:
    # Empirically, grouped lag training needs a much larger inner solve than the
    # single-cell path; keeping protocol-specific defaults avoids the obviously
    # bad max_iter=5 regime while preserving explicit overrides.
    lbfgs_max_iter = 50 if training_protocol == "group_by_lag" else 10
if training_protocol == "group_by_lag" and lbfgs_max_iter < 50:
    print(
        f"[warn] TRAINING_PROTOCOL=group_by_lag with LBFGS_MAX_ITER={lbfgs_max_iter} "
        "has been empirically unstable; 50 is the safer starting point."
    )
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

if training_protocol == "group_by_lag":
    group_specs = []
    lag_to_cells = {}
    for cid in cell_ids:
        lag = int(np.clip(int(peak_lags_np[cid]), 0, max_valid_lag))
        lag_to_cells.setdefault(lag, []).append(int(cid))
    for lag, ids in sorted(lag_to_cells.items()):
        group_specs.append(
            {
                "label": f"lag={lag}",
                "lag_index": int(lag),
                "group_cell_ids": [int(cid) for cid in ids],
                "seed_key": int(lag),
            }
        )
    print(f"selected {len(cell_ids)} cells across {len(group_specs)} peak-lag groups")
    for spec in group_specs:
        print(f"{spec['label']}: {len(spec['group_cell_ids'])} cells")
elif training_protocol == "single_cell":
    group_specs = []
    for cid in cell_ids:
        lag = int(np.clip(int(peak_lags_np[cid]), 0, max_valid_lag))
        group_specs.append(
            {
                "label": f"cell={int(cid)} lag={lag}",
                "lag_index": int(lag),
                "group_cell_ids": [int(cid)],
                "seed_key": int(cid),
            }
        )
    print(f"selected {len(cell_ids)} cells as {len(group_specs)} single-cell groups")
    for spec in group_specs:
        print(spec["label"])
else:
    raise ValueError(
        f"Unknown TRAINING_PROTOCOL={training_protocol!r}. "
        "Expected 'group_by_lag' or 'single_cell'."
    )
print(f"grid size: {len(prox_mults)} x {len(local_mults)} = {len(prox_mults) * len(local_mults)}")

train_dset.to("cpu")
val_dset.to("cpu")
train_loader_lbfgs_accum = DataLoader(
    train_dset,
    batch_size=batch_size_lbfgs,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=(prefetch_factor if num_workers > 0 else None),
)
val_loader_lbfgs = DataLoader(
    val_dset,
    batch_size=batch_size_lbfgs,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=(prefetch_factor if num_workers > 0 else None),
)

best_global = {
    int(cid): {
        "best_val_bps": -1e9,
        "train_bps_at_best": float("nan"),
        "lag": None,
        "epoch": None,
        "prox_mult": None,
        "local_mult": None,
    }
    for cid in cell_ids
}

for prox_mult in prox_mults:
    for local_mult in local_mults:
        combo_name = f"pm{prox_mult:g}_lm{local_mult:g}"
        print(f"\n=== combo {combo_name} ===")
        for spec in group_specs:
            lag_index = int(spec["lag_index"])
            group_cell_ids = list(spec["group_cell_ids"])
            seed_key = int(spec["seed_key"])
            print(f"[group] {spec['label']} n_cells={len(group_cell_ids)}")
            n_neurons = len(group_cell_ids)
            # Optional per-group deterministic init independent of lag execution order.
            if use_group_seed:
                group_seed = int(seed) + int(seed_key)
                torch.manual_seed(group_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(group_seed)
                np.random.seed(group_seed)
            if init_torch_rng_state_path:
                torch.set_rng_state(torch.load(init_torch_rng_state_path, map_location="cpu"))
            if init_numpy_rng_state_path:
                with open(init_numpy_rng_state_path, "rb") as f:
                    np.random.set_state(pickle.load(f))

            model = TwoStage(
                image_shape=image_shape,
                n_neurons=n_neurons,
                n_lags=1,
                height=model_height,
                order=5,
                lowest_cpd_target=1.0,
                ppd=train_dset.dsets[0].metadata["ppd"],
                rel_tolerance=0.3,
                validate_cpd=True,
                beta_init=0.0,
                init_weight_scale=1e-4,
                beta_as_parameter=True,
                clamp_beta_min=(None if output_nonlinearity == "exp" else 1e-6),
                hann_window_power=hann_window_power,
                output_nonlinearity=output_nonlinearity,
            )
            model.cuda()

            eps = 1e-12
            masked_rate_mean = (
                np.asarray((robs[:, group_cell_ids] * dfs[:, group_cell_ids]).sum(axis=0), dtype=np.float64)
                / np.maximum(np.asarray(dfs[:, group_cell_ids].sum(axis=0), dtype=np.float64), eps)
            )
            if output_nonlinearity == "exp":
                beta_init_vec = np.log(np.maximum(masked_rate_mean, 1e-8))
            elif output_nonlinearity == "relu":
                beta_init_vec = np.maximum(masked_rate_mean, 1e-8)
            else:
                raise ValueError(f"Unsupported OUTPUT_NONLINEARITY: {output_nonlinearity}")
            with torch.no_grad():
                model.beta.copy_(torch.tensor(beta_init_vec, device=model.beta.device, dtype=model.beta.dtype))

            if output_nonlinearity == "exp":
                t0 = time.perf_counter()
                calibrate_exp_intercept_full_dataset(
                    model=model,
                    train_loader=train_loader_lbfgs_accum,
                    lag_index=lag_index,
                    group_cell_ids=group_cell_ids,
                    eps=eps,
                )
                timings["intercept_calibration_s"] += time.perf_counter() - t0

            optimizer_lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=lbfgs_lr,
                max_iter=lbfgs_max_iter,
                history_size=lbfgs_history_size,
                line_search_fn="strong_wolfe",
                tolerance_grad=lbfgs_tol_grad,
                tolerance_change=lbfgs_tol_change,
                max_eval=lbfgs_max_eval,
            )

            lambda_prox_vec = torch.tensor(
                [
                    (
                        float(fixed_lambda_prox)
                        if fixed_lambda_prox is not None
                        else float(cell_lambda_prox.get(int(cid), default_lambda_prox))
                    )
                    * float(prox_mult)
                    for cid in group_cell_ids
                ],
                device=model.w_pos.weight.device,
                dtype=model.w_pos.weight.dtype,
            ).unsqueeze(1)
            lambda_local_vec = torch.tensor(
                [
                    (
                        float(fixed_lambda_local)
                        if fixed_lambda_local is not None
                        else float(cell_lambda_local.get(int(cid), default_lambda_local))
                    )
                    * float(local_mult)
                    for cid in group_cell_ids
                ],
                device=model.w_pos.weight.device,
                dtype=model.w_pos.weight.dtype,
            )

            train_cache = None
            val_cache = None
            cached_total_den = None
            cached_num_batches = None
            cached_num_samples = None
            if precompute_pyramid:
                t0 = time.perf_counter()
                train_cache = build_feature_cache(
                    model=model,
                    loader=train_loader_lbfgs_accum,
                    lag_index=lag_index,
                    crop_size=crop_size,
                    cache_dtype=cache_dtype,
                )
                val_cache = build_feature_cache(
                    model=model,
                    loader=val_loader_lbfgs,
                    lag_index=lag_index,
                    crop_size=crop_size,
                    cache_dtype=cache_dtype,
                )
                timings["intercept_calibration_s"] += time.perf_counter() - t0
                if precompute_denominator:
                    cached_total_den, cached_num_batches, cached_num_samples = precompute_total_den_from_cache(
                        cache=train_cache,
                        group_cell_ids=group_cell_ids,
                        per_cell=True,
                    )

            for epoch in range(num_epochs):
                train_agg = PoissonBPSAggregator()
                val_agg = PoissonBPSAggregator()
                li_chunks = []

                model.train()
                model.alpha_pos.requires_grad = False
                if precompute_pyramid:
                    active_train_loader = train_cache
                    prepare_batch_fn = (
                        lambda b, _d=model.w_pos.weight.device, _dt=model.w_pos.weight.dtype: _prepare_cached_batch(
                            b, device=_d, model_dtype=_dt
                        )
                    )
                else:
                    active_train_loader = train_loader_lbfgs_accum
                    prepare_batch_fn = lambda b: prepare_batch_one_lag(
                        batch=b,
                        lag_index=lag_index,
                        crop_size=crop_size,
                        device="cuda",
                    )
                t0 = time.perf_counter()
                step_stats, _ = train_step_lbfgs(
                    model=model,
                    optimizer=optimizer_lbfgs,
                    batch=None,
                    batch_loader=active_train_loader,
                    prepare_batch_fn=prepare_batch_fn,
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
                    precomputed_total_den=(cached_total_den if precompute_denominator else None),
                    precomputed_num_batches=(cached_num_batches if precompute_denominator else None),
                    precomputed_num_samples=(cached_num_samples if precompute_denominator else None),
                )
                timings["lbfgs_step_s"] += time.perf_counter() - t0
                if time_breakdown:
                    print(
                        "[closure_timing] "
                        f"calls={int(step_stats.get('closure_calls', 0))} "
                        f"pass1_last_s={float(step_stats.get('closure_pass1_s_last', 0.0)):.3f} "
                        f"pass2_last_s={float(step_stats.get('closure_pass2_s_last', 0.0)):.3f} "
                        f"reg_last_s={float(step_stats.get('closure_reg_s_last', 0.0)):.3f}"
                    )

                if not skip_train_eval:
                    t0 = time.perf_counter()
                    train_eval_iter = train_cache if precompute_pyramid else train_loader_lbfgs_accum
                    for batch in tqdm(train_eval_iter):
                        model.eval()
                        if precompute_pyramid:
                            batch = _prepare_cached_batch(
                                batch_cached=batch,
                                device=model.w_pos.weight.device,
                                model_dtype=model.w_pos.weight.dtype,
                            )
                        else:
                            batch = prepare_batch_one_lag(
                                batch=batch,
                                lag_index=lag_index,
                                crop_size=crop_size,
                                device="cuda",
                            )
                        out = eval_step(model=model, batch=batch, cell_ids=group_cell_ids, use_resolver=True)
                        train_agg(out)
                    timings["train_eval_pass_s"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                val_eval_iter = val_cache if precompute_pyramid else val_loader_lbfgs
                for batch in val_eval_iter:
                    model.eval()
                    if precompute_pyramid:
                        batch = _prepare_cached_batch(
                            batch_cached=batch,
                            device=model.w_pos.weight.device,
                            model_dtype=model.w_pos.weight.dtype,
                        )
                    else:
                        batch = prepare_batch_one_lag(
                            batch=batch,
                            lag_index=lag_index,
                            crop_size=crop_size,
                            device="cuda",
                        )
                    out = eval_step(model=model, batch=batch, cell_ids=group_cell_ids, use_resolver=True)
                    val_agg(out)
                    with torch.no_grad():
                        pos_feats, neg_feats = _extract_pos_neg_feats(model, batch)
                        inv_sqrt2 = 1.0 / math.sqrt(2.0)
                        w_pos = model._windowed_weight(model.w_pos.weight)
                        w_neg = model._windowed_weight(model.w_neg.weight)
                        # Paper-style decomposition: keep stimulus activity fixed and
                        # rotate only the afferent weights into linear/energy axes.
                        w_l = inv_sqrt2 * (w_pos - w_neg)
                        w_e = inv_sqrt2 * (w_pos + w_neg)
                        z_l = F.linear(pos_feats, w_l) - F.linear(neg_feats, w_l)
                        z_e = F.linear(pos_feats, w_e) + F.linear(neg_feats, w_e)
                        beta = model.beta.unsqueeze(0)
                        if output_nonlinearity == "exp":
                            # Match the summary-metrics path and the paper-style
                            # variance-explained definition in response space.
                            pred_full = out["rhat"]
                            pred_lin = torch.exp(beta + z_l)
                            pred_eng = torch.exp(beta + z_e)
                        elif output_nonlinearity == "relu":
                            pred_full = out["rhat"]
                            pred_lin = (beta + F.relu(z_l)).clamp_min(1e-6)
                            pred_eng = (beta + F.relu(z_e)).clamp_min(1e-6)
                        else:
                            raise ValueError(f"Unsupported OUTPUT_NONLINEARITY: {output_nonlinearity}")
                        li_chunks.append(
                            {
                                "z_full": pred_full.detach().to(device="cpu"),
                                "z_lin": pred_lin.detach().to(device="cpu"),
                                "z_eng": pred_eng.detach().to(device="cpu"),
                                "robs": out["robs"].detach().to(device="cpu"),
                                "dfs": out["dfs"].detach().to(device="cpu"),
                            }
                        )
                timings["val_eval_pass_s"] += time.perf_counter() - t0

                bps_val = val_agg.closure().detach().cpu().numpy()
                if skip_train_eval:
                    bps_train = bps_val
                else:
                    bps_train = train_agg.closure().detach().cpu().numpy()
                train_table = cell_bps_table(bps_train, group_cell_ids)
                val_table = cell_bps_table(bps_val, group_cell_ids)
                (
                    li_table,
                    frac_lin_table,
                    frac_eng_table,
                    r2_full_table,
                    r2_lin_table,
                    r2_eng_table,
                ) = _compute_linearity_tables(model=model, li_chunks=li_chunks, group_cell_ids=group_cell_ids)

                print(
                    f"combo={combo_name} lag={lag_index:02d} epoch={epoch:03d} "
                    f"train_mean={float(np.mean(list(train_table.values()))):.4f} "
                    f"val_mean={float(np.mean(list(val_table.values()))):.4f}"
                )

                for local_idx, cid in enumerate(group_cell_ids):
                    val_now = float(val_table[int(cid)])
                    if val_now > best_global[int(cid)]["best_val_bps"]:
                        best_global[int(cid)]["best_val_bps"] = val_now
                        best_global[int(cid)]["train_bps_at_best"] = float(train_table[int(cid)])
                        best_global[int(cid)]["lag"] = int(lag_index)
                        best_global[int(cid)]["epoch"] = int(epoch)
                        best_global[int(cid)]["prox_mult"] = float(prox_mult)
                        best_global[int(cid)]["local_mult"] = float(local_mult)
                        if save_best_pngs and save_png_mode == "best":
                            t0 = time.perf_counter()
                            save_cell_best_png(
                                model=model,
                                cell_id=int(cid),
                                neuron_idx=int(local_idx),
                                bps_train=bps_train,
                                bps_val=bps_val,
                                poisson_last=step_stats["poisson"],
                                local_last=step_stats["local"],
                                reg_last=step_stats["reg"],
                                prox_tau_last=step_stats["prox_tau"],
                                phase=f"grid_{combo_name}",
                                epoch=epoch,
                                linearity_metrics={
                                    "linearity_index": li_table[int(cid)],
                                    "frac_linear": frac_lin_table[int(cid)],
                                    "frac_energy": frac_eng_table[int(cid)],
                                    "r2_full": r2_full_table[int(cid)],
                                    "r2_linear": r2_lin_table[int(cid)],
                                    "r2_energy": r2_eng_table[int(cid)],
                                },
                                save_label="best",
                            )
                            timings["diagnostics_save_s"] += time.perf_counter() - t0
                        if save_best_checkpoints:
                            save_cell_best_checkpoint(
                                model=model,
                                cell_id=int(cid),
                                neuron_idx=int(local_idx),
                                group_cell_ids=group_cell_ids,
                                lag_index=int(lag_index),
                                phase=f"grid_{combo_name}",
                                epoch=int(epoch),
                                step_stats=step_stats,
                                train_bps_value=float(train_table[int(cid)]),
                                val_bps_value=float(val_table[int(cid)]),
                            )

                    if save_best_pngs and save_png_mode == "last_epoch" and epoch == (num_epochs - 1):
                        t0 = time.perf_counter()
                        save_cell_best_png(
                            model=model,
                            cell_id=int(cid),
                            neuron_idx=int(local_idx),
                            bps_train=bps_train,
                            bps_val=bps_val,
                            poisson_last=step_stats["poisson"],
                            local_last=step_stats["local"],
                            reg_last=step_stats["reg"],
                            prox_tau_last=step_stats["prox_tau"],
                            phase=f"grid_{combo_name}",
                            epoch=epoch,
                            linearity_metrics={
                                "linearity_index": li_table[int(cid)],
                                "frac_linear": frac_lin_table[int(cid)],
                                "frac_energy": frac_eng_table[int(cid)],
                                "r2_full": r2_full_table[int(cid)],
                                "r2_linear": r2_lin_table[int(cid)],
                                "r2_energy": r2_eng_table[int(cid)],
                            },
                            save_label="last_epoch",
                        )
                        timings["diagnostics_save_s"] += time.perf_counter() - t0

            del train_cache, val_cache
            torch.cuda.empty_cache()

rows = []
for cid in sorted(best_global.keys()):
    d = best_global[cid]
    rows.append(
        {
            "cell_id": int(cid),
            "best_val_bps": float(d["best_val_bps"]),
            "train_bps_at_best": float(d["train_bps_at_best"]),
            "best_lag_group": int(d["lag"]) if d["lag"] is not None else None,
            "best_epoch": int(d["epoch"]) if d["epoch"] is not None else None,
            "best_prox_mult": float(d["prox_mult"]) if d["prox_mult"] is not None else None,
            "best_local_mult": float(d["local_mult"]) if d["local_mult"] is not None else None,
        }
    )
df = pd.DataFrame(rows)
csv_path = output_dir / "best_val_bps.csv"
df.to_csv(csv_path, index=False)
print(f"\nsaved per-cell best summary to: {csv_path}")
print(
    f"[final] cells={len(df)} mean_best_val={float(df['best_val_bps'].mean()):.4f} "
    f"median_best_val={float(df['best_val_bps'].median()):.4f} "
    f"min_best_val={float(df['best_val_bps'].min()):.4f} "
    f"max_best_val={float(df['best_val_bps'].max()):.4f}"
)
if time_breakdown:
    total_s = time.perf_counter() - run_t0
    accounted = sum(timings.values())
    print(
        "[timing] "
        f"total_s={total_s:.2f} "
        f"calibration_s={timings['intercept_calibration_s']:.2f} "
        f"lbfgs_step_s={timings['lbfgs_step_s']:.2f} "
        f"train_eval_s={timings['train_eval_pass_s']:.2f} "
        f"val_eval_s={timings['val_eval_pass_s']:.2f} "
        f"diagnostics_save_s={timings['diagnostics_save_s']:.2f} "
        f"other_s={max(total_s - accounted, 0.0):.2f}"
    )

