from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator
from two_stage_core import TwoStage
from two_stage_helpers import locality_penalty_from_maps, show_epoch_diagnostics
from util import get_dataset_info


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "All-cells LBFGS run with side-by-side single-cell baselines for "
            "train/val and receptive-field similarity checks."
        )
    )
    p.add_argument(
        "--dataset-configs-path",
        type=str,
        default="/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml",
    )
    p.add_argument("--subject", type=str, default="Allen")
    p.add_argument("--date", type=str, default="2022-04-13")
    p.add_argument("--image-shape", type=int, nargs=2, default=[41, 41])
    p.add_argument("--cell-ids", type=str, default="14,15,16,66,75,76")
    p.add_argument(
        "--lag-mode",
        type=str,
        default="peak_bank",
        choices=["peak_bank", "median_peak", "all"],
        help="Lag selection for all-cell run. peak_bank best matches per-cell peak-lag single runs.",
    )
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=10024)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--height", type=int, default=3)
    p.add_argument("--order", type=int, default=5)
    p.add_argument("--lowest-cpd-target", type=float, default=1.0)
    p.add_argument("--rel-tolerance", type=float, default=0.3)
    p.add_argument("--lambda-reg", type=float, default=1e-4)
    p.add_argument("--lambda-local-prox", type=float, default=1e-1)
    p.add_argument("--sparsity-mode", type=str, default="ratio_l1_l2", choices=["ratio_l1_l2", "prox_l1"])
    p.add_argument("--lambda-prox", type=float, default=1e-4)
    p.add_argument("--lbfgs-lr", type=float, default=1.0)
    p.add_argument("--lbfgs-max-iter", type=int, default=5)
    p.add_argument("--lbfgs-history-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--training-mode",
        type=str,
        default="independent",
        choices=["independent", "joint"],
        help=(
            "independent: optimize each cell in one automated run (matches single-cell behavior). "
            "joint: optimize all cells in one shared model."
        ),
    )
    p.add_argument(
        "--joint-within-lag-mode",
        type=str,
        default="cellwise",
        choices=["shared", "cellwise"],
        help=(
            "How to optimize cells that share a lag in joint mode. "
            "shared uses one multicell LBFGS model; cellwise runs one LBFGS fit per cell "
            "to avoid shared line-search coupling."
        ),
    )
    p.add_argument("--save-diagnostics", action="store_true", default=False)
    p.add_argument("--diagnostics-dir", type=str, default="/home/tejas/VisionCore/tejas/model/lbfgs_all_cells_diagnostics")
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_cell_ids(cell_ids_str: str) -> list[int]:
    cell_ids = [int(x.strip()) for x in cell_ids_str.split(",") if x.strip()]
    if not cell_ids:
        raise ValueError("No valid --cell-ids provided.")
    return cell_ids


def crop_slice(crop_size: int):
    return slice(crop_size, -crop_size) if int(crop_size) > 0 else slice(None)


def unique_preserve_order(vals: list[int]) -> list[int]:
    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def prepare_batch_with_lags(batch, lag_indices: list[int], crop_size: int, device: str):
    ys = crop_slice(crop_size)
    xs = crop_slice(crop_size)
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    batch["stim"] = batch["stim"][:, :, lag_indices, ys, xs]
    return batch


def align_outputs(out, cell_ids: list[int]):
    out["robs"] = out["robs"][:, cell_ids]
    out["dfs"] = out["dfs"][:, cell_ids]
    if out["rhat"].shape != out["robs"].shape:
        raise ValueError(
            f"Output shape mismatch: rhat={tuple(out['rhat'].shape)} "
            f"robs={tuple(out['robs'].shape)} dfs={tuple(out['dfs'].shape)}"
        )
    return out


def multicell_locality(model: TwoStage, circular_dims: set[int]) -> torch.Tensor:
    pos = model.positive_afferent_map
    neg = model.negative_afferent_map
    terms = []
    for neuron_idx in range(pos.shape[0]):
        for lag_idx in range(pos.shape[1]):
            local_term, _ = locality_penalty_from_maps(
                pos[neuron_idx, lag_idx],
                neg[neuron_idx, lag_idx],
                circular_dims=circular_dims,
            )
            terms.append(local_term)
    if not terms:
        return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)
    return torch.stack(terms).mean()


def multicell_sparsity_ratio_l1_l2(model: TwoStage, eps: float = 1e-12) -> torch.Tensor:
    # Match single-cell behavior more closely by computing ratio per neuron, then averaging.
    per_neuron = []
    for neuron_idx in range(model.w_pos.weight.shape[0]):
        w_star = torch.sqrt(
            model.w_pos.weight[neuron_idx].pow(2) + model.w_neg.weight[neuron_idx].pow(2) + eps
        )
        l2 = w_star.norm(2).clamp_min(eps)
        per_neuron.append(w_star.norm(1) / l2)
    return torch.stack(per_neuron).mean()


def compute_regularization(
    model: TwoStage,
    sparsity_mode: str,
    lambda_reg: float,
    lambda_local_prox: float,
    circular_dims: set[int],
    gamma_value: float,
):
    l_local = multicell_locality(model, circular_dims=circular_dims)
    if sparsity_mode == "ratio_l1_l2":
        l_sparse = multicell_sparsity_ratio_l1_l2(model)
        gamma_local = float(gamma_value)
        reg_term = lambda_reg * l_sparse * (1.0 + gamma_local * l_local)
    elif sparsity_mode == "prox_l1":
        l_sparse = l_local.new_zeros(())
        gamma_local = 0.0
        reg_term = lambda_local_prox * l_local
    else:
        raise ValueError(f"Unknown sparsity_mode: {sparsity_mode}")
    return l_sparse, l_local, reg_term, gamma_local


def prox_step_if_needed(model: TwoStage, optimizer: torch.optim.Optimizer, sparsity_mode: str, lambda_prox: float) -> float:
    if sparsity_mode != "prox_l1":
        return 0.0
    lr = float(optimizer.param_groups[0].get("lr", 1e-3))
    tau = lr * float(lambda_prox)
    with torch.no_grad():
        norm = torch.sqrt(model.w_pos.weight.pow(2) + model.w_neg.weight.pow(2) + 1e-12)
        scale = (1.0 - tau / norm).clamp_min(0.0)
        model.w_pos.weight.mul_(scale)
        model.w_neg.weight.mul_(scale)
    return float(tau)


def masked_poisson_loss_equal_cells(out: dict[str, torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    # Match single-cell behavior in multicell runs by averaging per-cell losses equally.
    loss_elem = F.poisson_nll_loss(
        out["rhat"],
        out["robs"],
        log_input=False,
        full=False,
        reduction="none",
    )
    if "dfs" in out:
        mask = out["dfs"]
        masked = loss_elem * mask
        per_cell_den = mask.sum(dim=0).clamp_min(eps)
        per_cell = masked.sum(dim=0) / per_cell_den
        return per_cell.mean()
    return loss_elem.mean(dim=0).mean()


@dataclass
class TrainResult:
    cell_ids: list[int]
    lag_indices: list[int]
    cell_lag_slots: list[int]
    train_bps_hist: list[np.ndarray]
    val_bps_hist: list[np.ndarray]
    model: TwoStage


def pick_lag_indices(cell_ids: list[int], peak_lags: np.ndarray, max_valid_lag: int, lag_mode: str):
    raw_lags = [int(peak_lags[cid]) for cid in cell_ids]
    clipped_lags = [min(lg, max_valid_lag) for lg in raw_lags]
    if lag_mode == "median_peak":
        med = int(np.rint(np.median(clipped_lags)))
        selected_lags = [max(0, min(med, max_valid_lag))]
        lag_slots = [0 for _ in clipped_lags]
    elif lag_mode == "all":
        selected_lags = list(range(max_valid_lag + 1))
        lag_slots = [selected_lags.index(lg) for lg in clipped_lags]
    else:
        selected_lags = unique_preserve_order(clipped_lags)
        lag_slots = [selected_lags.index(lg) for lg in clipped_lags]
    return raw_lags, clipped_lags, selected_lags, lag_slots


def train_lbfgs(
    dataset_info: dict,
    image_shape: tuple[int, int],
    cell_ids: list[int],
    lag_indices: list[int],
    cell_lag_slots: list[int],
    args,
    run_name: str,
    device: str,
) -> TrainResult:
    peak_lags = dataset_info["peak_lags"]
    stas = dataset_info["stas"]
    train_dset = dataset_info["train_dset"]
    val_dset = dataset_info["val_dset"]
    crop_size = int(dataset_info["crop_size"])
    robs = dataset_info["robs"]

    n_neurons = len(cell_ids)
    n_lags = len(lag_indices)
    beta_init = torch.as_tensor(
        np.asarray(robs[:, cell_ids], dtype=np.float32).mean(axis=0),
        dtype=torch.float32,
    )

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
        beta_init=beta_init,
        init_weight_scale=1e-4,
        beta_as_parameter=True,
        clamp_beta_min=1e-6,
    ).to(device)

    if device == "cuda":
        torch.cuda.empty_cache()
    train_dset.to("cpu")
    val_dset.to("cpu")
    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True, shuffle=False)
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, **{k: v for k, v in loader_kwargs.items() if k != "shuffle"})
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, **loader_kwargs)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=args.lbfgs_lr,
        max_iter=args.lbfgs_max_iter,
        history_size=args.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )
    spike_loss = MaskedPoissonNLLLoss(pred_key="rhat", target_key="robs", mask_key="dfs")
    gamma_local = float(args.lambda_reg) * 4.0 / 20.0
    circular_dims = {1}

    train_bps_hist: list[np.ndarray] = []
    val_bps_hist: list[np.ndarray] = []
    for epoch in range(args.num_epochs):
        train_agg = PoissonBPSAggregator()
        val_agg = PoissonBPSAggregator()
        poisson_last = sparse_last = local_last = reg_last = prox_tau_last = 0.0

        for batch in tqdm(train_loader, desc=f"{run_name} train epoch={epoch}"):
            model.train()
            model.alpha_pos.requires_grad = False
            batch = prepare_batch_with_lags(batch, lag_indices=lag_indices, crop_size=crop_size, device=device)
            step_stats = {}

            def closure():
                optimizer.zero_grad()
                out_c = model(dict(batch))
                out_c = align_outputs(out_c, cell_ids=cell_ids)
                poisson = masked_poisson_loss_equal_cells(out_c)
                l_sparse, l_local, reg_term, gamma_val = compute_regularization(
                    model=model,
                    sparsity_mode=args.sparsity_mode,
                    lambda_reg=args.lambda_reg,
                    lambda_local_prox=args.lambda_local_prox,
                    circular_dims=circular_dims,
                    gamma_value=gamma_local,
                )
                loss = poisson + reg_term
                loss.backward()
                step_stats["poisson"] = float(poisson.detach().item())
                step_stats["sparse"] = float(l_sparse.detach().item())
                step_stats["local"] = float(l_local.detach().item())
                step_stats["reg"] = float(reg_term.detach().item())
                step_stats["gamma_local"] = float(gamma_val)
                return loss

            optimizer.step(closure)
            prox_tau_last = prox_step_if_needed(model, optimizer, args.sparsity_mode, args.lambda_prox)
            poisson_last = step_stats["poisson"]
            sparse_last = step_stats["sparse"]
            local_last = step_stats["local"]
            reg_last = step_stats["reg"]

            with torch.no_grad():
                out = model(dict(batch))
                out = align_outputs(out, cell_ids=cell_ids)
                train_agg(out)

        for batch in val_loader:
            model.eval()
            batch = prepare_batch_with_lags(batch, lag_indices=lag_indices, crop_size=crop_size, device=device)
            with torch.no_grad():
                out = model(dict(batch))
                out = align_outputs(out, cell_ids=cell_ids)
                val_agg(out)

        bps_train = np.asarray(train_agg.closure().detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        bps_val = np.asarray(val_agg.closure().detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        train_bps_hist.append(bps_train)
        val_bps_hist.append(bps_val)

        print(
            f"{run_name} epoch={epoch:03d} "
            f"train_bps={np.round(bps_train, 4).tolist()} "
            f"val_bps={np.round(bps_val, 4).tolist()} "
            f"poisson={poisson_last:.6f} sparse={sparse_last:.6f} "
            f"local={local_last:.6f} gamma*local={(gamma_local * local_last):.6f} "
            f"prox_tau={prox_tau_last:.3e} reg={reg_last:.6f}"
        )

        if args.save_diagnostics:
            for local_idx, cid in enumerate(cell_ids):
                show_epoch_diagnostics(
                    model=model,
                    stas=stas,
                    peak_lags=peak_lags,
                    cell_id=int(cid),
                    sparsity_mode=args.sparsity_mode,
                    poisson_last=poisson_last,
                    sparse_last=sparse_last,
                    local_last=local_last,
                    gamma_local=gamma_local,
                    prox_tau_last=prox_tau_last,
                    reg_last=reg_last,
                    bps=bps_train,
                    bps_val=bps_val,
                    phase=run_name,
                    epoch=epoch,
                    neuron_idx=int(local_idx),
                    lag_idx=int(cell_lag_slots[local_idx]),
                    save_dir=args.diagnostics_dir,
                    save_prefix=f"{run_name}_epoch_{epoch:03d}_cell_{int(cid):03d}",
                    close_figs=True,
                    show_plots=False,
                )

    return TrainResult(
        cell_ids=list(cell_ids),
        lag_indices=list(lag_indices),
        cell_lag_slots=list(cell_lag_slots),
        train_bps_hist=train_bps_hist,
        val_bps_hist=val_bps_hist,
        model=model,
    )


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    av = a.detach().reshape(-1).float()
    bv = b.detach().reshape(-1).float()
    den = av.norm() * bv.norm()
    if float(den) <= eps:
        return float("nan")
    return float(torch.dot(av, bv).item() / den.item())


def compare_models(
    all_result: TrainResult,
    single_results: dict[int, TrainResult],
):
    print("\n=== Final Per-Cell Comparison (all-cell vs single-cell) ===")
    header = (
        "cell | train_bps_all | train_bps_single | d_train | "
        "val_bps_all | val_bps_single | d_val | afferent_cos | linear_rf_cos"
    )
    print(header)
    print("-" * len(header))

    all_last_train = all_result.train_bps_hist[-1]
    all_last_val = all_result.val_bps_hist[-1]

    for local_idx, cell_id in enumerate(all_result.cell_ids):
        single = single_results[int(cell_id)]
        single_last_train = float(single.train_bps_hist[-1][0])
        single_last_val = float(single.val_bps_hist[-1][0])
        all_train = float(all_last_train[local_idx])
        all_val = float(all_last_val[local_idx])

        all_lag_slot = int(all_result.cell_lag_slots[local_idx])
        all_pos = all_result.model.positive_afferent_map[local_idx, all_lag_slot]
        all_neg = all_result.model.negative_afferent_map[local_idx, all_lag_slot]
        single_pos = single.model.positive_afferent_map[0, 0]
        single_neg = single.model.negative_afferent_map[0, 0]
        aff_cos = cosine_similarity(
            torch.cat([all_pos.reshape(-1), all_neg.reshape(-1)]),
            torch.cat([single_pos.reshape(-1), single_neg.reshape(-1)]),
        )

        all_rf = all_result.model.linear_receptive_field_at(neuron_idx=local_idx, lag_idx=all_lag_slot)
        single_rf = single.model.linear_receptive_field_at(neuron_idx=0, lag_idx=0)
        rf_cos = cosine_similarity(all_rf, single_rf)
        print(
            f"{int(cell_id):3d} | {all_train: .4f} | {single_last_train: .4f} | {all_train - single_last_train: .4f} | "
            f"{all_val: .4f} | {single_last_val: .4f} | {all_val - single_last_val: .4f} | {aff_cos: .4f} | {rf_cos: .4f}"
        )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_shape = tuple(args.image_shape)

    dataset_info = get_dataset_info(args.dataset_configs_path, args.subject, args.date, image_shape)
    cell_ids = parse_cell_ids(args.cell_ids)
    stim = dataset_info["stim"]
    if stim.ndim == 5:
        lag_dim = 2
    elif stim.ndim == 4:
        lag_dim = 1
    else:
        raise ValueError(f"Unexpected stim shape: {tuple(stim.shape)}")
    max_valid_lag = int(stim.shape[lag_dim]) - 1
    peak_lags = dataset_info["peak_lags"]

    raw_lags, clipped_lags, selected_lags, lag_slots = pick_lag_indices(
        cell_ids=cell_ids,
        peak_lags=peak_lags,
        max_valid_lag=max_valid_lag,
        lag_mode=args.lag_mode,
    )
    print(
        f"all-cell setup: cell_ids={cell_ids} raw_peak_lags={raw_lags} "
        f"clipped_peak_lags={clipped_lags} selected_lags={selected_lags}"
    )

    if args.training_mode == "joint":
        # Peak-lag-only joint mode: do not mix lag banks in one model.
        lag_to_cells: dict[int, list[int]] = {}
        for cid in cell_ids:
            raw = int(peak_lags[int(cid)])
            lag = max(0, min(raw, max_valid_lag))
            lag_to_cells.setdefault(lag, []).append(int(cid))

        print("\nRunning joint peak-lag groups (each group uses exactly one lag).")
        grouped_results: dict[int, TrainResult] = {}
        for lag, group_cells in sorted(lag_to_cells.items()):
            print(f"\n--- joint group lag={lag} cells={group_cells} mode={args.joint_within_lag_mode} ---")
            if args.joint_within_lag_mode == "shared" or len(group_cells) == 1:
                grouped_results[lag] = train_lbfgs(
                    dataset_info=dataset_info,
                    image_shape=image_shape,
                    cell_ids=group_cells,
                    lag_indices=[int(lag)],
                    cell_lag_slots=[0 for _ in group_cells],
                    args=args,
                    run_name=f"joint_lag_{int(lag)}",
                    device=device,
                )
            else:
                per_cell_results = []
                for cid in group_cells:
                    per_cell_results.append(
                        train_lbfgs(
                            dataset_info=dataset_info,
                            image_shape=image_shape,
                            cell_ids=[int(cid)],
                            lag_indices=[int(lag)],
                            cell_lag_slots=[0],
                            args=args,
                            run_name=f"joint_lag_{int(lag)}_cell_{int(cid)}",
                            device=device,
                        )
                    )
                # Compose one lag result object for unified reporting.
                composed_train = np.stack([r.train_bps_hist[-1][0] for r in per_cell_results], axis=0).astype(np.float32)
                composed_val = np.stack([r.val_bps_hist[-1][0] for r in per_cell_results], axis=0).astype(np.float32)
                grouped_results[lag] = TrainResult(
                    cell_ids=[int(cid) for cid in group_cells],
                    lag_indices=[int(lag)],
                    cell_lag_slots=[0 for _ in group_cells],
                    train_bps_hist=[composed_train],
                    val_bps_hist=[composed_val],
                    model=per_cell_results[0].model,
                )

        print("\n=== Joint Peak-Lag Group Metrics ===")
        print("lag | cell | train_bps | val_bps")
        print("----------------------------------")
        for lag, res in sorted(grouped_results.items()):
            train_last = res.train_bps_hist[-1]
            val_last = res.val_bps_hist[-1]
            for local_idx, cid in enumerate(res.cell_ids):
                print(
                    f"{int(lag):3d} | {int(cid):3d} | "
                    f"{float(train_last[local_idx]): .4f} | {float(val_last[local_idx]): .4f}"
                )
        return

    print("\nRunning independent-cell optimization (single command, all requested cells automated).")
    independent_results: dict[int, TrainResult] = {}
    for cid in cell_ids:
        cid_raw = int(peak_lags[int(cid)])
        cid_lag = max(0, min(cid_raw, max_valid_lag))
        print(f"\n--- independent all-cells run: cell={cid} lag={cid_lag} (raw={cid_raw}) ---")
        independent_results[int(cid)] = train_lbfgs(
            dataset_info=dataset_info,
            image_shape=image_shape,
            cell_ids=[int(cid)],
            lag_indices=[int(cid_lag)],
            cell_lag_slots=[0],
            args=args,
            run_name=f"all_cells_independent_{int(cid)}",
            device=device,
        )

    print("\n=== Independent-Mode Final Metrics ===")
    print("cell | train_bps | val_bps")
    print("---------------------------")
    for cid in cell_ids:
        res = independent_results[int(cid)]
        tr = float(res.train_bps_hist[-1][0])
        va = float(res.val_bps_hist[-1][0])
        print(f"{int(cid):3d} | {tr: .4f} | {va: .4f}")


if __name__ == "__main__":
    main()
