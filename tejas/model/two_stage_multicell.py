from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import schedulefree
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator
from two_stage_core import TwoStage
from two_stage_helpers import (
    locality_penalty_from_maps,
    prox_group_l21_,
    show_epoch_diagnostics,
    sparsity_penalty,
)
from util import get_dataset_info


@dataclass
class RunResult:
    best_val_bps: np.ndarray
    best_epoch: int
    reached_target: bool


def parse_args():
    p = argparse.ArgumentParser(description="Multicell two-stage fitting.")
    p.add_argument("--mode", type=str, default="adam", choices=["adam", "lbfgs", "hybrid"])
    p.add_argument(
        "--lag-mode",
        type=str,
        default="median_peak",
        choices=["median_peak", "peak_bank", "all"],
        help="How to choose lag indices for training. median_peak uses one shared lag.",
    )
    p.add_argument("--cell-ids", type=str, default="14,16")
    p.add_argument("--dataset-configs-path", type=str, default="/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml")
    p.add_argument("--subject", type=str, default="Allen")
    p.add_argument("--date", type=str, default="2022-04-13")
    p.add_argument("--image-shape", type=int, nargs=2, default=[41, 41])

    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--lbfgs-epochs", type=int, default=10)

    p.add_argument("--batch-size-adam", type=int, default=1024)
    p.add_argument("--batch-size-lbfgs", type=int, default=10024)
    p.add_argument("--num-workers", type=int, default=16)

    p.add_argument("--lambda-reg-adam", type=float, default=1e-2)
    p.add_argument("--lambda-reg-lbfgs", type=float, default=1e-4)
    p.add_argument("--gamma-local-lbfgs", type=float, default=None)
    p.add_argument("--lambda-prox", type=float, default=1e-4)
    p.add_argument("--lambda-local-prox", type=float, default=1e-1)
    p.add_argument("--sparsity-mode", type=str, default="ratio_l1_l2", choices=["ratio_l1_l2", "prox_l1"])
    p.add_argument("--circular-orientation", action="store_true", default=True)
    p.add_argument("--no-circular-orientation", action="store_false", dest="circular_orientation")

    p.add_argument("--target-bps", type=float, default=0.4)
    p.add_argument("--target-bps-lbfgs", type=float, default=0.1)

    p.add_argument("--height", type=int, default=3)
    p.add_argument("--order", type=int, default=5)
    p.add_argument("--lowest-cpd-target", type=float, default=1.0)
    p.add_argument("--rel-tolerance", type=float, default=0.3)
    p.add_argument("--init-weight-scale", type=float, default=1e-4)
    p.add_argument("--beta-init", type=float, default=0.0)
    p.add_argument("--clamp-beta-min", type=float, default=1e-6)
    p.add_argument("--beta-as-parameter", action="store_true", default=True)
    p.add_argument("--freeze-beta-adam", action="store_true", default=False)

    p.add_argument("--lbfgs-lr", type=float, default=1.0)
    p.add_argument("--lbfgs-max-iter", type=int, default=5)
    p.add_argument("--lbfgs-history-size", type=int, default=10)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-diagnostics-png", action="store_true", default=False)
    p.add_argument("--diagnostics-dir", type=str, default="./multicell_diagnostics")
    p.add_argument("--diagnostics-every", type=int, default=0, help="If >0, save diagnostics every N epochs.")
    p.add_argument(
        "--diagnostics-best-only",
        action="store_true",
        default=True,
        help="Save only best-so-far PNG per cell by val BPS (overwrites).",
    )
    p.add_argument(
        "--no-diagnostics-best-only",
        action="store_false",
        dest="diagnostics_best_only",
        help="Disable best-only behavior and allow per-epoch exports.",
    )
    p.add_argument(
        "--max-diag-cells",
        type=int,
        default=0,
        help="If >0, limit diagnostics export to first K selected cells (0 means all).",
    )
    p.add_argument(
        "--diagnostics-cell-ids",
        type=str,
        default="",
        help="Optional comma-separated absolute cell IDs to export diagnostics for (e.g. '16,21').",
    )
    p.add_argument(
        "--diagnostics-lag-mode",
        type=str,
        default="max_energy",
        choices=["peak", "max_energy"],
        help="Which lag to visualize per cell in diagnostics PNGs.",
    )
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_cell_ids(cell_ids_str: str, n_total_cells: int | None = None) -> list[int]:
    raw = cell_ids_str.strip().lower()
    if raw == "all":
        if n_total_cells is None:
            raise ValueError("n_total_cells is required when using --cell-ids all")
        return list(range(int(n_total_cells)))
    out = [int(x.strip()) for x in cell_ids_str.split(",") if x.strip()]
    if len(out) == 0:
        raise ValueError("No valid cell IDs passed.")
    return out


def unique_preserve_order(vals: list[int]) -> list[int]:
    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def crop_slice(crop_size: int):
    return slice(crop_size, -crop_size) if int(crop_size) > 0 else slice(None)


def prepare_batch_multicell(batch, lag_indices, crop_size, device="cuda"):
    y = crop_slice(crop_size)
    x = crop_slice(crop_size)
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    batch["stim"] = batch["stim"][:, :, lag_indices, y, x]
    return batch


def align_outputs_multicell(out, cell_ids):
    n_targets = int(out["robs"].shape[1])
    if n_targets != int(out["dfs"].shape[1]):
        raise ValueError(f"robs/dfs mismatch: {out['robs'].shape} vs {out['dfs'].shape}")
    if len(cell_ids) == n_targets and list(cell_ids) == list(range(n_targets)):
        # Fast path for "all cells": avoid unnecessary indexing on GPU.
        pass
    else:
        if len(cell_ids) == 0:
            raise ValueError("cell_ids cannot be empty")
        lo = min(cell_ids)
        hi = max(cell_ids)
        if lo < -n_targets or hi >= n_targets:
            raise IndexError(
                f"cell_ids out of bounds: min={lo}, max={hi}, target_width={n_targets}"
            )
        out["robs"] = out["robs"][:, cell_ids]
        out["dfs"] = out["dfs"][:, cell_ids]
    assert out["rhat"].shape == out["robs"].shape == out["dfs"].shape, (
        f"shape mismatch: rhat={out['rhat'].shape}, robs={out['robs'].shape}, dfs={out['dfs'].shape}"
    )
    return out


def multicell_locality(model, circular_dims):
    pos = model.positive_afferent_map
    neg = model.negative_afferent_map
    terms = []
    for n in range(pos.shape[0]):
        for l in range(pos.shape[1]):
            local_term, _ = locality_penalty_from_maps(pos[n, l], neg[n, l], circular_dims=circular_dims)
            terms.append(local_term)
    if len(terms) == 0:
        return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)
    return torch.stack(terms).mean()


def compute_reg(model, args, lambda_reg, gamma_mode, gamma_value, circular_dims):
    l_local = multicell_locality(model, circular_dims=circular_dims)
    if args.sparsity_mode == "ratio_l1_l2":
        l_sparse = sparsity_penalty(model)
        if gamma_mode == "adaptive_5pct":
            gamma_local = 0.05 / max(l_local.detach().item(), 1e-12)
        elif gamma_mode == "fixed":
            gamma_local = float(gamma_value)
        else:
            raise ValueError(f"Unknown gamma_mode: {gamma_mode}")
        reg = lambda_reg * l_sparse * (1.0 + gamma_local * l_local)
    elif args.sparsity_mode == "prox_l1":
        l_sparse = l_local.new_zeros(())
        gamma_local = 0.0
        reg = args.lambda_local_prox * l_local
    else:
        raise ValueError(f"Unknown sparsity_mode: {args.sparsity_mode}")
    return l_sparse, l_local, reg, gamma_local


def maybe_prox_step(model, optimizer, args):
    prox_tau = 0.0
    if args.sparsity_mode == "prox_l1":
        lr = float(optimizer.param_groups[0].get("lr", 1e-3))
        prox_tau = lr * args.lambda_prox
        prox_group_l21_(model.w_pos.weight, model.w_neg.weight, tau=prox_tau)
    return prox_tau


def bps_to_numpy(bps):
    arr = np.asarray(bps.detach().cpu().numpy() if torch.is_tensor(bps) else bps, dtype=float).reshape(-1)
    return arr


def all_cells_meet_target(arr: np.ndarray, target: float):
    return bool(np.all(arr >= float(target)))


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_shape = tuple(args.image_shape)

    dataset_info = get_dataset_info(args.dataset_configs_path, args.subject, args.date, image_shape)
    n_total_cells = int(dataset_info["robs"].shape[1])
    cell_ids = parse_cell_ids(args.cell_ids, n_total_cells=n_total_cells)
    peak_lags = dataset_info["peak_lags"]
    train_dset = dataset_info["train_dset"]
    val_dset = dataset_info["val_dset"]
    crop_size = dataset_info["crop_size"]

    stim_full = dataset_info["stim"]
    if stim_full.ndim != 5:
        raise ValueError(f"Expected dataset_info['stim'] to be 5D [N,C,L,H,W], got shape={tuple(stim_full.shape)}")
    max_valid_lag = int(stim_full.shape[2]) - 1
    raw_cell_lags = [int(peak_lags[cid]) for cid in cell_ids]
    clipped_cell_lags = [min(lg, max_valid_lag) for lg in raw_cell_lags]
    if args.lag_mode == "median_peak":
        median_lag = int(np.rint(np.median(clipped_cell_lags)))
        median_lag = max(0, min(median_lag, max_valid_lag))
        lag_indices = [median_lag]
    elif args.lag_mode == "peak_bank":
        lag_indices = unique_preserve_order(clipped_cell_lags)
    elif args.lag_mode == "all":
        lag_indices = list(range(max_valid_lag + 1))
    else:
        raise ValueError(f"Unknown lag_mode: {args.lag_mode}")
    if args.lag_mode == "median_peak":
        cell_lag_slots = [0 for _ in clipped_cell_lags]
    else:
        cell_lag_slots = [lag_indices.index(lg) for lg in clipped_cell_lags]
    if any(r != c for r, c in zip(raw_cell_lags, clipped_cell_lags)):
        print(
            f"Adjusted out-of-range peak lags to max_valid_lag={max_valid_lag}. "
            f"raw_unique={sorted(set(raw_cell_lags))}, clipped_unique={sorted(set(clipped_cell_lags))}"
        )
    print(
        f"cell_ids={cell_ids} peak_lags={raw_cell_lags} "
        f"lag_mode={args.lag_mode} selected_lags={lag_indices}"
    )

    n_neurons = len(cell_ids)
    n_lags = len(lag_indices)
    model = TwoStage(
        image_shape=image_shape,
        n_neurons=n_neurons,
        n_lags=n_lags,
        height=args.height,
        order=args.order,
        lowest_cpd_target=args.lowest_cpd_target,
        ppd=train_dset.dsets[0].metadata["ppd"],
        rel_tolerance=args.rel_tolerance,
        validate_cpd=True,
        beta_init=args.beta_init,
        init_weight_scale=args.init_weight_scale,
        beta_as_parameter=args.beta_as_parameter,
        clamp_beta_min=args.clamp_beta_min,
    ).to(device)

    torch.cuda.empty_cache()
    train_dset.to("cpu")
    val_dset.to("cpu")

    common_loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    # Remove invalid kwargs when num_workers=0
    if args.num_workers == 0:
        common_loader_kwargs.pop("persistent_workers")
        common_loader_kwargs.pop("prefetch_factor")

    train_loader_adam = DataLoader(
        train_dset, batch_size=args.batch_size_adam, shuffle=True, **common_loader_kwargs
    )
    val_loader_adam = DataLoader(
        val_dset, batch_size=args.batch_size_adam, shuffle=False, **common_loader_kwargs
    )
    train_loader_lbfgs = DataLoader(
        train_dset, batch_size=args.batch_size_lbfgs, shuffle=True, **common_loader_kwargs
    )
    val_loader_lbfgs = DataLoader(
        val_dset, batch_size=args.batch_size_lbfgs, shuffle=False, **common_loader_kwargs
    )

    spike_loss = MaskedPoissonNLLLoss(pred_key="rhat", target_key="robs", mask_key="dfs")
    circular_dims = {1} if args.circular_orientation else set()

    if args.gamma_local_lbfgs is None:
        gamma_local_lbfgs = args.lambda_reg_lbfgs * 4.0 / 20.0
    else:
        gamma_local_lbfgs = float(args.gamma_local_lbfgs)

    optimizer_adam = schedulefree.RAdamScheduleFree(model.parameters())
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=args.lbfgs_lr,
        max_iter=args.lbfgs_max_iter,
        history_size=args.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )

    best_val_bps = np.full((n_neurons,), -np.inf, dtype=float)
    best_val_mean = -np.inf
    best_epoch = -1
    target = args.target_bps_lbfgs if args.mode == "lbfgs" else args.target_bps
    best_diag_val = np.full((n_neurons,), -np.inf, dtype=float)

    for epoch in range(args.num_epochs):
        phase = args.mode
        if args.mode == "hybrid":
            phase = "lbfgs" if epoch < args.lbfgs_epochs else "adam"

        train_agg = PoissonBPSAggregator(device=device)
        val_agg = PoissonBPSAggregator(device=device)
        prox_tau_last = 0.0
        poisson_last = sparse_last = local_last = reg_last = gamma_local = 0.0

        if phase == "adam":
            active_train_loader = train_loader_adam
            active_val_loader = val_loader_adam
        else:
            active_train_loader = train_loader_lbfgs
            active_val_loader = val_loader_lbfgs

        for batch in tqdm(active_train_loader, desc=f"train epoch={epoch} phase={phase}"):
            batch = prepare_batch_multicell(batch, lag_indices=lag_indices, crop_size=crop_size, device=device)

            if phase == "adam":
                optimizer_adam.train()
                if args.freeze_beta_adam:
                    model.beta.requires_grad = False
                out = model(dict(batch))
                out = align_outputs_multicell(out, cell_ids)
                poisson = spike_loss(out)
                l_sparse, l_local, reg, gamma_local = compute_reg(
                    model=model,
                    args=args,
                    lambda_reg=args.lambda_reg_adam,
                    gamma_mode="adaptive_5pct",
                    gamma_value=0.0,
                    circular_dims=circular_dims,
                )
                loss = poisson + reg
                loss.backward()
                optimizer_adam.step()
                prox_tau_last = maybe_prox_step(model, optimizer_adam, args)
                optimizer_adam.zero_grad()
                poisson_last = float(poisson.detach().item())
                sparse_last = float(l_sparse.detach().item())
                local_last = float(l_local.detach().item())
                reg_last = float(reg.detach().item())
                with torch.no_grad():
                    train_agg(out)
            else:
                step_stats = {}

                def closure():
                    optimizer_lbfgs.zero_grad()
                    out_c = model(dict(batch))
                    out_c = align_outputs_multicell(out_c, cell_ids)
                    poisson_c = spike_loss(out_c)
                    l_sparse_c, l_local_c, reg_c, gamma_local_c = compute_reg(
                        model=model,
                        args=args,
                        lambda_reg=args.lambda_reg_lbfgs,
                        gamma_mode="fixed",
                        gamma_value=gamma_local_lbfgs,
                        circular_dims=circular_dims,
                    )
                    loss_c = poisson_c + reg_c
                    loss_c.backward()
                    step_stats["poisson"] = float(poisson_c.detach().item())
                    step_stats["sparse"] = float(l_sparse_c.detach().item())
                    step_stats["local"] = float(l_local_c.detach().item())
                    step_stats["reg"] = float(reg_c.detach().item())
                    step_stats["gamma_local"] = float(gamma_local_c)
                    return loss_c

                optimizer_lbfgs.step(closure)
                prox_tau_last = maybe_prox_step(model, optimizer_lbfgs, args)
                poisson_last = step_stats["poisson"]
                sparse_last = step_stats["sparse"]
                local_last = step_stats["local"]
                reg_last = step_stats["reg"]
                gamma_local = step_stats["gamma_local"]

                with torch.no_grad():
                    out = model(dict(batch))
                    out = align_outputs_multicell(out, cell_ids)
                    train_agg(out)

        for batch in active_val_loader:
            with torch.no_grad():
                batch = prepare_batch_multicell(batch, lag_indices=lag_indices, crop_size=crop_size, device=device)
                out = model(dict(batch))
                out = align_outputs_multicell(out, cell_ids)
                val_agg(out)

        bps_train = bps_to_numpy(train_agg.closure())
        bps_val = bps_to_numpy(val_agg.closure())

        val_mean = float(np.mean(bps_val))
        if val_mean > best_val_mean:
            best_val_mean = val_mean
            best_val_bps = bps_val.copy()
            best_epoch = epoch

        local_factor = gamma_local * local_last
        print(
            f"epoch={epoch:03d} phase={phase} "
            f"train_bps={np.round(bps_train, 4).tolist()} "
            f"val_bps={np.round(bps_val, 4).tolist()} "
            f"poisson={poisson_last:.6f} sparse={sparse_last:.6f} "
            f"local={local_last:.6f} gamma_local={gamma_local:.6f} "
            f"gamma*local={local_factor:.6f} prox_tau={prox_tau_last:.3e} reg={reg_last:.6f}"
        )

        should_export = False
        if args.save_diagnostics_png:
            if args.diagnostics_best_only:
                should_export = True
            else:
                if args.diagnostics_every > 0 and (epoch % args.diagnostics_every == 0):
                    should_export = True
                if epoch == args.num_epochs - 1:
                    should_export = True
                if all_cells_meet_target(bps_val, target):
                    should_export = True

        if should_export:
            if args.diagnostics_cell_ids.strip():
                requested = [int(x.strip()) for x in args.diagnostics_cell_ids.split(",") if x.strip()]
                requested_set = set(requested)
                diag_local_indices = [i for i, cid in enumerate(cell_ids) if int(cid) in requested_set]
            else:
                n_diag = len(cell_ids) if args.max_diag_cells <= 0 else min(len(cell_ids), int(args.max_diag_cells))
                diag_local_indices = list(range(n_diag))

            for local_idx in diag_local_indices:
                cell_id = int(cell_ids[local_idx])
                if args.diagnostics_best_only:
                    current_val = float(bps_val[local_idx])
                    if current_val <= float(best_diag_val[local_idx]):
                        continue
                    best_diag_val[local_idx] = current_val
                if args.diagnostics_lag_mode == "peak":
                    lag_slot = int(cell_lag_slots[local_idx])
                else:
                    with torch.no_grad():
                        pos_n = model.positive_afferent_map[local_idx]
                        neg_n = model.negative_afferent_map[local_idx]
                        lag_energy = (pos_n.pow(2) + neg_n.pow(2)).sum(dim=(1, 2, 3, 4))
                        lag_slot = int(torch.argmax(lag_energy).item())
                if args.diagnostics_best_only:
                    prefix = f"cell_{cell_id:03d}"
                else:
                    prefix = f"epoch_{epoch:03d}_cell_{cell_id:03d}"
                show_epoch_diagnostics(
                    model=model,
                    stas=dataset_info["stas"],
                    peak_lags=peak_lags,
                    cell_id=cell_id,
                    sparsity_mode=args.sparsity_mode,
                    poisson_last=poisson_last,
                    sparse_last=sparse_last,
                    local_last=local_last,
                    gamma_local=gamma_local,
                    prox_tau_last=prox_tau_last,
                    reg_last=reg_last,
                    bps=bps_train,
                    bps_val=bps_val,
                    phase=phase,
                    epoch=epoch,
                    neuron_idx=local_idx,
                    lag_idx=lag_slot,
                    save_dir=args.diagnostics_dir,
                    save_prefix=prefix,
                    close_figs=True,
                    show_plots=False,
                )

        if all_cells_meet_target(bps_val, target):
            print(
                f"SUCCESS mode={args.mode}: val_bps={np.round(bps_val, 4).tolist()} "
                f"met target>={target:.3f} for all cells at epoch {epoch}."
            )
            return RunResult(best_val_bps=best_val_bps, best_epoch=best_epoch, reached_target=True)

    print(
        f"DONE mode={args.mode}: best_val_bps={np.round(best_val_bps, 4).tolist()} "
        f"at epoch {best_epoch}, best_mean={best_val_mean:.6f}, "
        f"target={target:.3f}, reached={all_cells_meet_target(best_val_bps, target)}"
    )
    return RunResult(
        best_val_bps=best_val_bps,
        best_epoch=best_epoch,
        reached_target=all_cells_meet_target(best_val_bps, target),
    )


if __name__ == "__main__":
    args = parse_args()
    result = run(args)
    if not result.reached_target:
        raise SystemExit(2)
