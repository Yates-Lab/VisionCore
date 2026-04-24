import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ray import air, tune
from ray.air import session
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataYatesV1 import enable_autoreload
from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator
from two_stage_core import TwoStage
from two_stage_helpers import show_epoch_diagnostics
from two_stage_trainer import eval_step, train_step_lbfgs
from util import get_dataset_info

enable_autoreload()


DATASET_CONFIGS_PATH = "/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml"
SUBJECT = "Allen"
DATE = "2022-04-13"
IMAGE_SHAPE = (41, 41)


def crop_slice(c):
    return slice(c, -c) if int(c) > 0 else slice(None)


def prepare_batch_one_lag(batch, lag_index, crop_size, device="cuda"):
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    ys = crop_slice(crop_size)
    xs = crop_slice(crop_size)
    b["stim"] = b["stim"][:, :, [int(lag_index)], ys, xs]
    return b


def calibrate_exp_intercept_full_dataset(model, train_loader, lag_index, group_cell_ids, crop_size, eps=1e-12):
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


def _prepare_cached_batch(batch_cached, device, model_dtype):
    return {
        "pos_feats": batch_cached["pos_feats"].to(device=device, dtype=model_dtype, non_blocking=True),
        "neg_feats": batch_cached["neg_feats"].to(device=device, dtype=model_dtype, non_blocking=True),
        "robs": batch_cached["robs"].to(device=device, non_blocking=True),
        "dfs": batch_cached["dfs"].to(device=device, non_blocking=True),
    }


def precompute_total_den_from_cache(cache, group_cell_ids):
    total_den = torch.zeros((len(group_cell_ids),), dtype=torch.float32)
    total_samples = 0
    total_batches = 0
    for b in cache:
        den = b["dfs"][:, group_cell_ids].sum(dim=0).to(dtype=torch.float32)
        total_den = total_den + den
        total_samples += int(b["robs"].shape[0])
        total_batches += 1
    return total_den.clamp_min(1.0), total_batches, total_samples


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


def make_lag_groups(train_dset, peak_lags, cell_ids):
    peak_lags_np = np.asarray(peak_lags).reshape(-1)
    stim_shape = tuple(train_dset[0]["stim"].shape)
    if len(stim_shape) == 4:
        stim_lag_count = int(stim_shape[1])
    elif len(stim_shape) == 5:
        stim_lag_count = int(stim_shape[2])
    else:
        raise ValueError(f"Unexpected stim shape for lag counting: {stim_shape}")
    max_valid_lag = max(0, stim_lag_count - 1)
    lag_to_cells = {}
    for cid in cell_ids:
        lag = int(np.clip(int(peak_lags_np[cid]), 0, max_valid_lag))
        lag_to_cells.setdefault(lag, []).append(int(cid))
    return lag_to_cells


def load_data():
    info = get_dataset_info(DATASET_CONFIGS_PATH, SUBJECT, DATE, IMAGE_SHAPE)
    return info


def trial_trainable(config):
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = load_data()
    peak_lags = data["peak_lags"]
    train_dset = data["train_dset"]
    val_dset = data["val_dset"]
    robs = data["robs"]
    dfs = data["dfs"]
    crop_size = data["crop_size"]
    cell_ids = parse_cell_ids(config["cell_ids"], int(robs.shape[1]))
    lag_to_cells = make_lag_groups(train_dset, peak_lags, cell_ids)

    train_dset.to("cpu")
    val_dset.to("cpu")
    train_loader = DataLoader(
        train_dset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]),
        persistent_workers=bool(config["persistent_workers"]),
        prefetch_factor=(int(config["prefetch_factor"]) if int(config["num_workers"]) > 0 else None),
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]),
        persistent_workers=bool(config["persistent_workers"]),
        prefetch_factor=(int(config["prefetch_factor"]) if int(config["num_workers"]) > 0 else None),
    )

    spike_loss = MaskedPoissonNLLLoss(pred_key="rhat", target_key="robs", mask_key="dfs")
    trial_dir = Path(session.get_trial_dir())
    ckpt_dir = trial_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Metrics rows for all epochs/cells/trial params
    rows = []
    # Best per cell across all epochs in this trial
    best_by_cell = {}
    running_best_mean = -1e9

    for lag_index, group_cell_ids in sorted(lag_to_cells.items()):
        if bool(config["use_group_seed"]):
            group_seed = int(seed) + int(lag_index)
            torch.manual_seed(group_seed)
            np.random.seed(group_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(group_seed)

        model = TwoStage(
            image_shape=IMAGE_SHAPE,
            n_neurons=len(group_cell_ids),
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
            train_loader=train_loader,
            lag_index=lag_index,
            group_cell_ids=group_cell_ids,
            crop_size=crop_size,
            eps=eps,
        )

        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=float(config["lbfgs_lr"]),
            max_iter=int(config["lbfgs_max_iter"]),
            history_size=int(config["lbfgs_history_size"]),
            line_search_fn="strong_wolfe",
            max_eval=int(config["lbfgs_max_eval"]),
            tolerance_grad=float(config["lbfgs_tol_grad"]),
            tolerance_change=float(config["lbfgs_tol_change"]),
        )

        lambda_prox_vec = torch.full(
            (len(group_cell_ids), 1),
            float(config["lambda_prox"]),
            device=model.w_pos.weight.device,
            dtype=model.w_pos.weight.dtype,
        )
        lambda_local_vec = torch.full(
            (len(group_cell_ids),),
            float(config["lambda_local"]),
            device=model.w_pos.weight.device,
            dtype=model.w_pos.weight.dtype,
        )

        train_cache = None
        val_cache = None
        cached_total_den = None
        cached_num_batches = None
        cached_num_samples = None
        if bool(config["precompute_pyramid"]):
            cache_dtype = torch.float32 if str(config["cache_dtype"]).lower() == "float32" else torch.float16
            train_cache = build_feature_cache(model, train_loader, lag_index, crop_size, cache_dtype)
            val_cache = build_feature_cache(model, val_loader, lag_index, crop_size, cache_dtype)
            if bool(config["precompute_denominator"]):
                cached_total_den, cached_num_batches, cached_num_samples = precompute_total_den_from_cache(
                    cache=train_cache,
                    group_cell_ids=group_cell_ids,
                )

        for epoch in range(int(config["num_epochs"])):
            model.train()
            model.alpha_pos.requires_grad = False
            if train_cache is not None:
                active_train_loader = train_cache
                prepare_batch_fn = (
                    lambda b, _d=model.w_pos.weight.device, _dt=model.w_pos.weight.dtype: _prepare_cached_batch(
                        b, _d, _dt
                    )
                )
            else:
                active_train_loader = train_loader
                prepare_batch_fn = lambda b: prepare_batch_one_lag(b, lag_index, crop_size, device="cuda")

            step_stats, _ = train_step_lbfgs(
                model=model,
                optimizer=optimizer,
                batch=None,
                batch_loader=active_train_loader,
                prepare_batch_fn=prepare_batch_fn,
                spike_loss=spike_loss,
                cell_ids=group_cell_ids,
                sparsity_mode="prox_l1",
                lambda_reg=0.0,
                lambda_local_prox=0.0,
                circular_dims={1},
                gamma_mode="fixed",
                gamma_value=0.0,
                lambda_prox=lambda_prox_vec,
                use_resolver=True,
                locality_mode="weighted_l21",
                poisson_aggregation_mode="sum_per_cell_means",
                lambda_local_per_neuron=lambda_local_vec,
                precomputed_total_den=(cached_total_den if bool(config["precompute_denominator"]) else None),
                precomputed_num_batches=(cached_num_batches if bool(config["precompute_denominator"]) else None),
                precomputed_num_samples=(cached_num_samples if bool(config["precompute_denominator"]) else None),
            )

            train_agg = PoissonBPSAggregator()
            val_agg = PoissonBPSAggregator()

            if not bool(config["skip_train_eval"]):
                train_eval_iter = train_cache if train_cache is not None else train_loader
                for b in train_eval_iter:
                    if train_cache is not None:
                        b = _prepare_cached_batch(b, model.w_pos.weight.device, model.w_pos.weight.dtype)
                    else:
                        b = prepare_batch_one_lag(b, lag_index, crop_size, device="cuda")
                    out = eval_step(model=model, batch=b, cell_ids=group_cell_ids, use_resolver=True)
                    train_agg(out)

            val_eval_iter = val_cache if val_cache is not None else val_loader
            for b in val_eval_iter:
                if val_cache is not None:
                    b = _prepare_cached_batch(b, model.w_pos.weight.device, model.w_pos.weight.dtype)
                else:
                    b = prepare_batch_one_lag(b, lag_index, crop_size, device="cuda")
                out = eval_step(model=model, batch=b, cell_ids=group_cell_ids, use_resolver=True)
                val_agg(out)

            # Guard against occasional empty eval aggregation in highly parallel Ray workers.
            if len(val_agg.robs) == 0:
                bps_val = np.zeros((len(group_cell_ids),), dtype=np.float32)
            else:
                bps_val = val_agg.closure().detach().cpu().numpy()
            if bool(config["skip_train_eval"]):
                bps_train = bps_val
            else:
                if len(train_agg.robs) == 0:
                    bps_train = np.zeros((len(group_cell_ids),), dtype=np.float32)
                else:
                    bps_train = train_agg.closure().detach().cpu().numpy()
            train_table = cell_bps_table(bps_train, group_cell_ids)
            val_table = cell_bps_table(bps_val, group_cell_ids)

            # checkpoint this epoch for post-hoc best-cell png reconstruction
            ckpt_path = ckpt_dir / f"lag{int(lag_index)}_epoch{int(epoch):03d}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "lag_index": int(lag_index),
                    "group_cell_ids": [int(c) for c in group_cell_ids],
                    "epoch": int(epoch),
                    "lambda_prox": float(config["lambda_prox"]),
                    "lambda_local": float(config["lambda_local"]),
                    "lbfgs_step_stats": step_stats,
                },
                ckpt_path,
            )

            for cid in group_cell_ids:
                cid = int(cid)
                tr = float(train_table[cid])
                va = float(val_table[cid])
                rows.append(
                    {
                        "trial_id": str(config["trial_id"]),
                        "epoch": int(epoch),
                        "lag_index": int(lag_index),
                        "cell_id": cid,
                        "lambda_prox": float(config["lambda_prox"]),
                        "lambda_local": float(config["lambda_local"]),
                        "train_bps": tr,
                        "val_bps": va,
                    }
                )
                prev = best_by_cell.get(cid)
                if (prev is None) or (va > prev["best_val_bps"]):
                    best_by_cell[cid] = {
                        "best_val_bps": va,
                        "train_bps_at_best": tr,
                        "epoch": int(epoch),
                        "lag_index": int(lag_index),
                        "checkpoint_path": str(ckpt_path),
                        "lambda_prox": float(config["lambda_prox"]),
                        "lambda_local": float(config["lambda_local"]),
                    }

            # Trial-level reported metric: mean val across all selected cells seen so far this epoch.
            epoch_vals = [float(val_table[int(c)]) for c in group_cell_ids]
            epoch_best_vals = [float(v["best_val_bps"]) for v in best_by_cell.values()] if best_by_cell else []
            running_best_mean = float(np.mean(epoch_best_vals)) if epoch_best_vals else -1e9
            session.report(
                {
                    "epoch": int(epoch),
                    "lag_index": int(lag_index),
                    "mean_val_bps": float(np.mean(epoch_vals)),
                    "mean_train_bps": float(np.mean([float(train_table[int(c)]) for c in group_cell_ids])),
                    "mean_best_val_bps": float(running_best_mean),
                    "lambda_prox": float(config["lambda_prox"]),
                    "lambda_local": float(config["lambda_local"]),
                }
            )

        del train_cache, val_cache
        torch.cuda.empty_cache()

    trial_metrics_path = trial_dir / "epoch_cell_bps.csv"
    pd.DataFrame(rows).to_csv(trial_metrics_path, index=False)
    best_cells_rows = [{"cell_id": int(k), **v} for k, v in sorted(best_by_cell.items())]
    best_cells_path = trial_dir / "best_cells.csv"
    pd.DataFrame(best_cells_rows).to_csv(best_cells_path, index=False)

    mean_best = float(np.mean([v["best_val_bps"] for v in best_by_cell.values()])) if best_by_cell else -1e9
    session.report(
        {
            "done": 1,
            "mean_best_val_bps": mean_best,
            "lambda_prox": float(config["lambda_prox"]),
            "lambda_local": float(config["lambda_local"]),
            "trial_metrics_path": str(trial_metrics_path),
            "best_cells_path": str(best_cells_path),
        }
    )


def consolidate_and_render_best_pngs(output_dir: Path, run_name: str, num_epochs: int):
    data = load_data()
    stas = data["stas"]
    peak_lags = data["peak_lags"]
    train_dset = data["train_dset"]

    run_dir = output_dir / run_name
    trial_dirs = list(run_dir.glob("trainable_with_trial_id_*")) + list(run_dir.glob("trial_*"))
    all_rows = []
    all_best_rows = []
    for tdir in trial_dirs:
        ep = tdir / "epoch_cell_bps.csv"
        bc = tdir / "best_cells.csv"
        if ep.exists():
            all_rows.append(pd.read_csv(ep))
        if bc.exists():
            dfb = pd.read_csv(bc)
            if "trial_dir" not in dfb.columns:
                dfb["trial_dir"] = str(tdir)
            all_best_rows.append(dfb)

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
    else:
        all_df = pd.DataFrame(columns=["trial_id", "epoch", "lag_index", "cell_id", "lambda_prox", "lambda_local", "train_bps", "val_bps"])
    all_df.to_csv(run_dir / "all_epoch_cell_bps.csv", index=False)

    if all_best_rows:
        best_df = pd.concat(all_best_rows, ignore_index=True)
    else:
        best_df = pd.DataFrame(columns=["cell_id", "best_val_bps", "train_bps_at_best", "epoch", "lag_index", "checkpoint_path", "lambda_prox", "lambda_local", "trial_dir"])
    best_df["cell_id"] = best_df["cell_id"].astype(int)
    best_global = best_df.sort_values("best_val_bps", ascending=False).groupby("cell_id", as_index=False).first()
    best_global.to_csv(run_dir / "best_val_bps.csv", index=False)

    # Render final best PNG per cell across all trials
    for _, r in best_global.iterrows():
        cid = int(r["cell_id"])
        ckpt = torch.load(r["checkpoint_path"], map_location="cpu")
        group_cell_ids = [int(x) for x in ckpt["group_cell_ids"]]
        lag_index = int(ckpt["lag_index"])
        neuron_idx = int(group_cell_ids.index(cid))

        model = TwoStage(
            image_shape=IMAGE_SHAPE,
            n_neurons=len(group_cell_ids),
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
        model.load_state_dict(ckpt["model_state"])
        model.cuda()
        model.eval()

        # Recreate compact scalar arrays for display
        bps_train = np.array([float(r["train_bps_at_best"])], dtype=np.float32)
        bps_val = np.array([float(r["best_val_bps"])], dtype=np.float32)
        phase = (
            f"ray_lprox{float(r['lambda_prox']):.2e}_llocal{float(r['lambda_local']):.2e}"
        )
        show_epoch_diagnostics(
            model=model,
            stas=stas,
            peak_lags=peak_lags,
            cell_id=cid,
            sparsity_mode="prox_l1",
            poisson_last=float(ckpt["lbfgs_step_stats"]["poisson"]),
            sparse_last=float(ckpt["lbfgs_step_stats"]["sparse"]),
            local_last=float(ckpt["lbfgs_step_stats"]["local"]),
            gamma_local=float(ckpt["lbfgs_step_stats"]["gamma_local"]),
            prox_tau_last=float(ckpt["lbfgs_step_stats"]["prox_tau"]),
            reg_last=float(ckpt["lbfgs_step_stats"]["reg"]),
            bps=bps_train,
            bps_val=bps_val,
            phase=phase,
            epoch=int(r["epoch"]),
            show_colorwheel=True,
            neuron_idx=neuron_idx,
            lag_idx=0,
            save_dir=str(run_dir),
            save_prefix=f"cell_{cid:03d}_best",
            close_figs=True,
            show_plots=False,
        )
        del model
        torch.cuda.empty_cache()


def main():
    output_root = Path(os.getenv("OUTPUT_ROOT", "/home/tejas/VisionCore/tejas/model"))
    run_name = os.getenv("RUN_NAME", f"fast_ray_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir = output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Time budget: until tomorrow 09:30 local unless overridden.
    if os.getenv("TIME_BUDGET_S"):
        time_budget_s = int(os.getenv("TIME_BUDGET_S"))
    else:
        now = datetime.now()
        tomorrow_930 = (now + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
        time_budget_s = max(int((tomorrow_930 - now).total_seconds()), 3600)

    num_epochs = int(os.getenv("NUM_EPOCHS", "3"))
    gpu_fraction = float(os.getenv("TRIAL_GPU_FRACTION", "0.5"))
    cpu_per_trial = int(os.getenv("TRIAL_CPU", "6"))
    num_samples = int(os.getenv("NUM_SAMPLES", "100000"))

    # Ray init
    ray_num_gpus = int(os.getenv("RAY_NUM_GPUS", str(torch.cuda.device_count())))
    import ray

    ray.init(
        ignore_reinit_error=True,
        num_gpus=ray_num_gpus,
        runtime_env={
            "py_executable": sys.executable,
            "excludes": [
                ".git/**",
                ".venv/**",
                "tejas/metrics/app/cloudflared-linux-amd64.deb",
                "tejas/model/final_pngs*/**",
                "tejas/model/fast_final_pngs*/**",
                "tejas/model/_timing*/**",
            ]
        },
    )

    base_cfg = {
        "trial_id": "",
        "seed": int(os.getenv("SEED", "0")),
        "cell_ids": os.getenv("CELL_IDS", "all"),
        "num_epochs": num_epochs,
        "batch_size": int(os.getenv("BATCH_SIZE", "2048")),
        "num_workers": int(os.getenv("NUM_WORKERS", "0")),
        "pin_memory": str(os.getenv("PIN_MEMORY", "1")).strip().lower() not in {"0", "false", "no"},
        "persistent_workers": str(os.getenv("PERSISTENT_WORKERS", "0")).strip().lower() in {"1", "true", "yes"},
        "prefetch_factor": int(os.getenv("PREFETCH_FACTOR", "4")),
        "use_group_seed": str(os.getenv("USE_GROUP_SEED", "1")).strip().lower() not in {"0", "false", "no"},
        "lbfgs_lr": float(os.getenv("LBFGS_LR", "0.1")),
        "lbfgs_max_iter": int(os.getenv("LBFGS_MAX_ITER", "50")),
        "lbfgs_history_size": int(os.getenv("LBFGS_HISTORY_SIZE", "10")),
        "lbfgs_max_eval": int(os.getenv("LBFGS_MAX_EVAL", "400")),
        "lbfgs_tol_grad": float(os.getenv("LBFGS_TOL_GRAD", "1e-7")),
        "lbfgs_tol_change": float(os.getenv("LBFGS_TOL_CHANGE", "1e-9")),
        "precompute_pyramid": str(os.getenv("PRECOMPUTE_PYRAMID", "1")).strip().lower() in {"1", "true", "yes"},
        "precompute_denominator": str(os.getenv("PRECOMPUTE_DENOMINATOR", "1")).strip().lower() in {"1", "true", "yes"},
        "cache_dtype": os.getenv("CACHE_DTYPE", "float32"),
        "skip_train_eval": str(os.getenv("SKIP_TRAIN_EVAL", "0")).strip().lower() in {"1", "true", "yes"},
    }

    search_space = {
        **base_cfg,
        "lambda_prox": tune.loguniform(1e-7, 1e-2),
        "lambda_local": tune.loguniform(1e-7, 1e-2),
    }

    def trainable_with_trial_id(cfg):
        cfg = dict(cfg)
        cfg["trial_id"] = session.get_trial_name()
        trial_trainable(cfg)

    tuner = tune.Tuner(
        tune.with_resources(trainable_with_trial_id, resources={"cpu": cpu_per_trial, "gpu": gpu_fraction}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="mean_best_val_bps",
            mode="max",
            num_samples=num_samples,
            time_budget_s=time_budget_s,
            max_concurrent_trials=int(os.getenv("MAX_CONCURRENT_TRIALS", "0")) or None,
        ),
        run_config=air.RunConfig(
            name=run_name,
            storage_path=str(output_root),
            verbose=1,
            failure_config=air.FailureConfig(max_failures=2),
            checkpoint_config=air.CheckpointConfig(checkpoint_at_end=False),
        ),
    )

    result_grid = tuner.fit()
    result_df = result_grid.get_dataframe()
    result_df.to_csv(out_dir / "ray_results_table.csv", index=False)
    consolidate_and_render_best_pngs(output_root, run_name, num_epochs=num_epochs)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "time_budget_s": time_budget_s,
                "num_samples": num_samples,
                "gpu_fraction": gpu_fraction,
                "cpu_per_trial": cpu_per_trial,
                "base_config": base_cfg,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
