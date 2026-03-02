from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Prevent Ray from cloning uv runtime env into worker sandboxes (breaks editable deps here).
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from ray import tune
from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator
from two_stage_core import TwoStage
from two_stage_helpers import show_epoch_diagnostics
from two_stage_trainer import eval_step, prepare_batch, train_step_lbfgs
from util import get_dataset_info


DEFAULT_DATASET_CONFIGS_PATH = "/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml"
DEFAULT_SUBJECT = "Allen"
DEFAULT_DATE = "2022-04-13"
DEFAULT_IMAGE_SHAPE = (41, 41)
DEFAULT_OUT_DIR = "/home/tejas/VisionCore/tejas/model/best_fits"


def parse_args():
    p = argparse.ArgumentParser(description="Ray Tune LBFGS-only single-cell sparsity sweep.")
    p.add_argument("--dataset-configs-path", type=str, default=DEFAULT_DATASET_CONFIGS_PATH)
    p.add_argument("--subject", type=str, default=DEFAULT_SUBJECT)
    p.add_argument("--date", type=str, default=DEFAULT_DATE)
    p.add_argument("--image-shape", type=int, nargs=2, default=list(DEFAULT_IMAGE_SHAPE))
    p.add_argument("--cell-ids", type=str, default="all", help="Comma list (e.g. 14,15) or 'all'.")
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--num-trials", type=int, default=15)
    p.add_argument("--lambda-reg-min", type=float, default=1e-6)
    p.add_argument("--lambda-reg-max", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=10024)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--height", type=int, default=3)
    p.add_argument("--order", type=int, default=5)
    p.add_argument("--lowest-cpd-target", type=float, default=1.0)
    p.add_argument("--rel-tolerance", type=float, default=0.3)
    p.add_argument("--sparsity-mode", type=str, default="ratio_l1_l2", choices=["ratio_l1_l2", "prox_l1"])
    p.add_argument("--lambda-local-prox", type=float, default=1e-1)
    p.add_argument("--lambda-prox", type=float, default=1e-4)
    p.add_argument("--lbfgs-lr", type=float, default=1.0)
    p.add_argument("--lbfgs-max-iter", type=int, default=5)
    p.add_argument("--lbfgs-history-size", type=int, default=10)
    p.add_argument("--cpu-per-trial", type=int, default=8)
    p.add_argument("--gpu-per-trial", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--ray-local-dir", type=str, default="/home/tejas/VisionCore/tejas/model/ray_results")
    return p.parse_args()


def parse_cell_ids(cell_ids: str, n_cells: int) -> list[int]:
    raw = cell_ids.strip().lower()
    if raw == "all":
        return list(range(int(n_cells)))
    parsed = [int(x.strip()) for x in cell_ids.split(",") if x.strip()]
    if not parsed:
        raise ValueError("No valid --cell-ids provided.")
    return parsed


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(train_dset, val_dset, batch_size: int, num_workers: int):
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_dset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def run_lbfgs_single_cell(
    *,
    dataset_info: dict,
    image_shape: tuple[int, int],
    cell_id: int,
    lambda_reg: float,
    args,
    save_png: bool = False,
    save_dir: str | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    peak_lags = dataset_info["peak_lags"]
    stas = dataset_info["stas"]
    robs = dataset_info["robs"]
    crop_size = int(dataset_info["crop_size"])
    train_dset = dataset_info["train_dset"]
    val_dset = dataset_info["val_dset"]
    sample_stim = train_dset[0]["stim"]
    max_valid_lag = int(sample_stim.shape[1]) - 1
    clipped_peak_lag = max(0, min(int(peak_lags[int(cell_id)]), max_valid_lag))
    peak_lags_safe = np.asarray(peak_lags).copy()
    peak_lags_safe[int(cell_id)] = clipped_peak_lag

    beta_init = float(robs[:, int(cell_id)].mean())
    model = TwoStage(
        image_shape=image_shape,
        n_neurons=1,
        n_lags=1,
        height=args.height,
        order=args.order,
        lowest_cpd_target=args.lowest_cpd_target,
        ppd=train_dset.dsets[0].metadata["ppd"],
        rel_tolerance=args.rel_tolerance,
        validate_cpd=True,
        beta_init=beta_init,
        init_weight_scale=1e-4,
        beta_as_parameter=True,
        clamp_beta_min=1e-6,
    ).to(device)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=args.lbfgs_lr,
        max_iter=args.lbfgs_max_iter,
        history_size=args.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )
    spike_loss = MaskedPoissonNLLLoss(pred_key="rhat", target_key="robs", mask_key="dfs")
    train_loader, val_loader = build_loaders(
        train_dset=train_dset,
        val_dset=val_dset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    gamma_local_fixed = float(lambda_reg) * 4.0 / 20.0
    best_val = -math.inf
    best_epoch = -1
    best_train = -math.inf

    poisson_last = sparse_last = local_last = reg_last = prox_tau_last = 0.0
    for epoch in range(int(args.num_epochs)):
        train_agg = PoissonBPSAggregator()
        val_agg = PoissonBPSAggregator()
        model.alpha_pos.requires_grad = False

        for batch in train_loader:
            model.train()
            batch = prepare_batch(
                batch,
                peak_lags=peak_lags_safe,
                cell_ids=[int(cell_id)],
                crop_size=crop_size,
                device=device,
            )
            step_stats, out = train_step_lbfgs(
                model=model,
                optimizer=optimizer,
                batch=batch,
                spike_loss=spike_loss,
                cell_ids=[int(cell_id)],
                sparsity_mode=args.sparsity_mode,
                lambda_reg=float(lambda_reg),
                lambda_local_prox=args.lambda_local_prox,
                circular_dims={1},
                gamma_mode="fixed",
                gamma_value=gamma_local_fixed,
                lambda_prox=args.lambda_prox,
                use_resolver=True,
            )
            poisson_last = step_stats["poisson"]
            sparse_last = step_stats["sparse"]
            local_last = step_stats["local"]
            reg_last = step_stats["reg"]
            prox_tau_last = step_stats["prox_tau"]
            with torch.no_grad():
                train_agg(out)

        for batch in val_loader:
            model.eval()
            batch = prepare_batch(
                batch,
                peak_lags=peak_lags_safe,
                cell_ids=[int(cell_id)],
                crop_size=crop_size,
                device=device,
            )
            out = eval_step(model=model, batch=batch, cell_ids=[int(cell_id)], use_resolver=True)
            val_agg(out)

        train_bps = float(train_agg.closure().detach().cpu().numpy().reshape(-1)[0])
        val_bps = float(val_agg.closure().detach().cpu().numpy().reshape(-1)[0])
        if val_bps > best_val:
            best_val = val_bps
            best_train = train_bps
            best_epoch = int(epoch)

    if save_png and save_dir is not None:
        show_epoch_diagnostics(
            model=model,
            stas=stas,
            peak_lags=peak_lags,
            cell_id=int(cell_id),
            sparsity_mode=args.sparsity_mode,
            poisson_last=poisson_last,
            sparse_last=sparse_last,
            local_last=local_last,
            gamma_local=gamma_local_fixed,
            prox_tau_last=prox_tau_last,
            reg_last=reg_last,
            bps=np.array([best_train], dtype=np.float32),
            bps_val=np.array([best_val], dtype=np.float32),
            phase=f"lbfgs_best_lam={lambda_reg:.2e}",
            epoch=best_epoch,
            show_colorwheel=True,
            neuron_idx=0,
            lag_idx=0,
            save_dir=save_dir,
            save_prefix=f"cell_{int(cell_id):03d}_lambda_{lambda_reg:.2e}",
            close_figs=True,
            show_plots=False,
        )

    model.cpu()
    del model
    torch.cuda.empty_cache()
    return {"best_val_bps": float(best_val), "best_train_bps": float(best_train), "best_epoch": int(best_epoch)}


def ray_trainable(config, fixed):
    set_seed(int(fixed["seed"]))
    dataset_info = get_dataset_info(
        fixed["dataset_configs_path"],
        fixed["subject"],
        fixed["date"],
        tuple(fixed["image_shape"]),
    )
    out = run_lbfgs_single_cell(
        dataset_info=dataset_info,
        image_shape=tuple(fixed["image_shape"]),
        cell_id=int(fixed["cell_id"]),
        lambda_reg=float(config["lambda_reg"]),
        args=fixed["args"],
        save_png=False,
        save_dir=None,
    )
    tune.report(
        {
            "best_val_bps": float(out["best_val_bps"]),
            "best_train_bps": float(out["best_train_bps"]),
            "best_epoch": int(out["best_epoch"]),
            "lambda_reg": float(config["lambda_reg"]),
        }
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    image_shape = tuple(args.image_shape)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = get_dataset_info(args.dataset_configs_path, args.subject, args.date, image_shape)
    all_cell_ids = parse_cell_ids(args.cell_ids, int(info["stas"].shape[0]))
    lambda_grid = np.logspace(
        np.log10(args.lambda_reg_min),
        np.log10(args.lambda_reg_max),
        int(args.num_trials),
    ).astype(np.float64)
    lambda_grid = [float(x) for x in lambda_grid]

    ray_num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    tune_kwargs = dict(
        config={"lambda_reg": tune.grid_search(lambda_grid)},
        resources_per_trial={"cpu": int(args.cpu_per_trial), "gpu": float(args.gpu_per_trial)},
        metric="best_val_bps",
        mode="max",
        storage_path=args.ray_local_dir,
        name="two_stage_lbfgs_single_cell_sparsity",
        raise_on_failed_trial=False,
        verbose=1,
    )

    tune_run_init = {"num_cpus": None, "num_gpus": None}
    try:
        import ray

        init_kwargs = dict(
            num_cpus=None,
            num_gpus=ray_num_gpus,
            include_dashboard=False,
            log_to_driver=True,
            ignore_reinit_error=True,
            _skip_env_hook=True,
        )
        ray.init(**init_kwargs)
        tune_run_init = {"num_cpus": "auto", "num_gpus": ray_num_gpus}
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Ray: {e}") from e

    print(f"Ray initialized with {tune_run_init}")
    print(f"Sweeping {len(lambda_grid)} lambdas per cell: {lambda_grid[0]:.2e} -> {lambda_grid[-1]:.2e}")

    summary_rows = []
    for cell_id in all_cell_ids:
        print(f"\n=== Tuning cell {int(cell_id)} ===")
        analysis = tune.run(
            tune.with_parameters(
                ray_trainable,
                fixed={
                    "dataset_configs_path": args.dataset_configs_path,
                    "subject": args.subject,
                    "date": args.date,
                    "image_shape": image_shape,
                    "seed": args.seed,
                    "cell_id": int(cell_id),
                    "args": args,
                },
            ),
            **tune_kwargs,
        )
        best_cfg = analysis.get_best_config(metric="best_val_bps", mode="max")
        if best_cfg is None or "lambda_reg" not in best_cfg:
            raise RuntimeError(f"Ray Tune did not return best config for cell {cell_id}")
        best_lambda = float(best_cfg["lambda_reg"])

        best_fit = run_lbfgs_single_cell(
            dataset_info=info,
            image_shape=image_shape,
            cell_id=int(cell_id),
            lambda_reg=best_lambda,
            args=args,
            save_png=True,
            save_dir=str(output_dir),
        )
        summary_rows.append((int(cell_id), best_lambda, best_fit["best_val_bps"], best_fit["best_train_bps"]))
        print(
            f"cell={int(cell_id)} best_lambda={best_lambda:.2e} "
            f"best_val_bps={best_fit['best_val_bps']:.4f} best_train_bps={best_fit['best_train_bps']:.4f}"
        )

    print("\n=== Best Fits Summary ===")
    print("cell | lambda_reg | best_val_bps | best_train_bps")
    for cid, lam, val_bps, train_bps in summary_rows:
        print(f"{cid:3d} | {lam: .2e} | {val_bps: .4f} | {train_bps: .4f}")

    try:
        import ray

        ray.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
