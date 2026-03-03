import torch
import torch.nn.functional as F

from two_stage_helpers import (
    convex_locality_weighted_l21_from_maps,
    _resolve_output_indices,
    locality_penalty_from_maps,
    prox_group_l21_,
    sparsity_penalty,
)


def _crop_slice(crop_size):
    return slice(crop_size, -crop_size) if int(crop_size) > 0 else slice(None)


def prepare_batch(batch, peak_lags, cell_ids, crop_size, device="cuda"):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    ys = _crop_slice(crop_size)
    xs = _crop_slice(crop_size)
    batch["stim"] = batch["stim"][:, :, peak_lags[cell_ids], ys, xs]
    return batch


def align_outputs(out, cell_ids, use_resolver=False):
    if use_resolver:
        pred_idx, target_idx = _resolve_output_indices(cell_ids, out)
        out["rhat"] = out["rhat"][:, pred_idx]
        if "eta" in out:
            out["eta"] = out["eta"][:, pred_idx]
        out["robs"] = out["robs"][:, target_idx]
        out["dfs"] = out["dfs"][:, target_idx]
    else:
        out["rhat"] = out["rhat"][:, cell_ids]
        if "eta" in out:
            out["eta"] = out["eta"][:, cell_ids]
        out["robs"] = out["robs"][:, cell_ids]
        out["dfs"] = out["dfs"][:, cell_ids]
    assert out["dfs"].shape == out["rhat"].shape == out["robs"].shape
    return out


def compute_regularization(
    model,
    sparsity_mode,
    lambda_reg,
    lambda_local_prox,
    circular_dims,
    gamma_mode,
    gamma_value,
    locality_mode="legacy_fft",
    lambda_local_per_neuron=None,
):
    pos_maps = model.positive_afferent_map
    neg_maps = model.negative_afferent_map
    # Aggregate locality over all neurons/lags for true multicell behavior.
    l_local = pos_maps.new_zeros(())
    l_local_per_neuron = pos_maps.new_zeros((int(pos_maps.shape[0]),))
    for neuron_idx in range(int(pos_maps.shape[0])):
        l_local_this_neuron = pos_maps.new_zeros(())
        for lag_idx in range(int(pos_maps.shape[1])):
            pos_map = pos_maps[neuron_idx, lag_idx]
            neg_map = neg_maps[neuron_idx, lag_idx]
            if locality_mode == "legacy_fft":
                l_local_nl, _ = locality_penalty_from_maps(pos_map, neg_map, circular_dims=circular_dims)
            elif locality_mode == "weighted_l21":
                l_local_nl, _ = convex_locality_weighted_l21_from_maps(
                    pos_map,
                    neg_map,
                    circular_dims=circular_dims,
                )
            else:
                raise ValueError(f"Unknown locality_mode: {locality_mode}")
            l_local = l_local + l_local_nl
            l_local_this_neuron = l_local_this_neuron + l_local_nl
        l_local_per_neuron[neuron_idx] = l_local_this_neuron
    if sparsity_mode == "ratio_l1_l2":
        l_sparse = sparsity_penalty(model)
        if gamma_mode == "adaptive_5pct":
            gamma_local = 0.05 / l_local.detach().item()
        elif gamma_mode == "fixed":
            gamma_local = float(gamma_value)
        else:
            raise ValueError(f"Unknown gamma_mode: {gamma_mode}")
        reg_term = lambda_reg * l_sparse * (1.0 + gamma_local * l_local)
    elif sparsity_mode == "prox_l1":
        l_sparse = l_local.new_zeros(())
        gamma_local = 0.0
        if lambda_local_per_neuron is not None:
            lambda_local_per_neuron = lambda_local_per_neuron.to(
                device=l_local_per_neuron.device,
                dtype=l_local_per_neuron.dtype,
            )
            reg_term = torch.sum(lambda_local_per_neuron * l_local_per_neuron)
        else:
            reg_term = lambda_local_prox * l_local
    else:
        raise ValueError(f"Unknown sparsity_mode: {sparsity_mode}")
    return l_sparse, l_local, reg_term, gamma_local


def _masked_poisson_num_den(out):
    loss_elem = F.poisson_nll_loss(
        out["rhat"],
        out["robs"],
        log_input=False,
        full=False,
        reduction="none",
    )
    if "dfs" in out:
        mask = out["dfs"]
        numerator = (loss_elem * mask).sum()
        denominator = mask.sum()
    else:
        numerator = loss_elem.sum()
        denominator = torch.tensor(
            float(loss_elem.numel()),
            device=loss_elem.device,
            dtype=loss_elem.dtype,
        )
    return numerator, denominator


def _masked_poisson_num_den_per_cell(out):
    loss_elem = F.poisson_nll_loss(
        out["rhat"],
        out["robs"],
        log_input=False,
        full=False,
        reduction="none",
    )
    if "dfs" in out:
        mask = out["dfs"]
        numerator = (loss_elem * mask).sum(dim=0)
        denominator = mask.sum(dim=0)
    else:
        numerator = loss_elem.sum(dim=0)
        denominator = torch.full(
            (loss_elem.shape[1],),
            float(loss_elem.shape[0]),
            device=loss_elem.device,
            dtype=loss_elem.dtype,
        )
    return numerator, denominator


def _masked_poisson_den(out):
    if "dfs" in out:
        return out["dfs"].sum()
    return torch.tensor(
        float(out["robs"].numel()),
        device=out["robs"].device,
        dtype=out["robs"].dtype,
    )


def train_step_adam(
    model,
    optimizer,
    batch,
    spike_loss,
    cell_ids,
    sparsity_mode,
    lambda_reg,
    lambda_local_prox,
    circular_dims,
    gamma_mode,
    gamma_value=0.0,
    freeze_beta=False,
    use_resolver=False,
    lambda_prox=0.0,
    locality_mode="legacy_fft",
):
    optimizer.train()
    if freeze_beta:
        model.beta.requires_grad = False
    out = model(batch)
    out = align_outputs(out, cell_ids=cell_ids, use_resolver=use_resolver)
    poisson_loss = spike_loss(out)
    l_sparse, l_local, reg_term, gamma_local = compute_regularization(
        model=model,
        sparsity_mode=sparsity_mode,
        lambda_reg=lambda_reg,
        lambda_local_prox=lambda_local_prox,
        circular_dims=circular_dims,
        gamma_mode=gamma_mode,
        gamma_value=gamma_value,
        locality_mode=locality_mode,
    )
    loss = poisson_loss + reg_term
    loss.backward()
    optimizer.step()

    prox_tau_last = 0.0
    if sparsity_mode == "prox_l1":
        lr = float(optimizer.param_groups[0].get("lr", 1e-3))
        prox_tau_last = lr * lambda_prox
        prox_group_l21_(model.w_pos.weight, model.w_neg.weight, tau=prox_tau_last)
    optimizer.zero_grad()

    return {
        "loss": float(loss.detach().item()),
        "poisson": float(poisson_loss.detach().item()),
        "sparse": float(l_sparse.detach().item()),
        "local": float(l_local.detach().item()),
        "reg": float(reg_term.detach().item()),
        "gamma_local": float(gamma_local),
        "prox_tau": float(prox_tau_last.detach().mean().item()) if torch.is_tensor(prox_tau_last) else float(prox_tau_last),
    }, out


def train_step_lbfgs(
    model,
    optimizer,
    batch,
    spike_loss,
    cell_ids,
    sparsity_mode,
    lambda_reg,
    lambda_local_prox,
    circular_dims,
    gamma_mode,
    gamma_value=0.0,
    lambda_prox=0.0,
    use_resolver=True,
    batch_loader=None,
    prepare_batch_fn=None,
    locality_mode="legacy_fft",
    poisson_aggregation_mode="global_mask_mean",
    lambda_local_per_neuron=None,
):
    step_stats = {}

    if batch_loader is not None and batch is not None:
        raise ValueError("Provide either `batch` or `batch_loader`, not both.")
    if batch_loader is None and batch is None:
        raise ValueError("Must provide `batch` (single-batch) or `batch_loader` (full-dataset).")

    if batch_loader is not None and prepare_batch_fn is None:
        prepare_batch_fn = lambda x: x

    if batch_loader is not None:
        device = next(model.parameters()).device

        def closure():
            optimizer.zero_grad()

            # Pass 1: compute the global denominator across micro-batches so each
            # batch contributes with true full-dataset weighting.
            if poisson_aggregation_mode == "sum_per_cell_means":
                total_den = torch.zeros((len(cell_ids),), device=device)
            else:
                total_den = torch.zeros((), device=device)
            total_samples = 0
            total_batches = 0
            with torch.no_grad():
                for batch_raw in batch_loader:
                    batch_local = prepare_batch_fn(batch_raw)
                    if poisson_aggregation_mode == "sum_per_cell_means":
                        if "dfs" in batch_local:
                            den_local = batch_local["dfs"][:, cell_ids].sum(dim=0)
                        else:
                            den_local = torch.full(
                                (len(cell_ids),),
                                float(batch_local["robs"].shape[0]),
                                device=device,
                                dtype=batch_local["robs"].dtype,
                            )
                    else:
                        if "dfs" in batch_local:
                            den_local = batch_local["dfs"][:, cell_ids].sum()
                        else:
                            den_local = torch.tensor(
                                float(batch_local["robs"][:, cell_ids].numel()),
                                device=device,
                                dtype=batch_local["robs"].dtype,
                            )
                    total_den = total_den + den_local.to(device=device)
                    total_samples += int(batch_local["robs"].shape[0])
                    total_batches += 1

            total_den = total_den.clamp_min(1.0)

            # Pass 2: accumulate gradients over micro-batches with global scaling.
            if poisson_aggregation_mode == "sum_per_cell_means":
                poisson_num_total = torch.zeros((len(cell_ids),), device=device)
            else:
                poisson_num_total = torch.zeros((), device=device)
            for batch_raw in batch_loader:
                batch_local = prepare_batch_fn(batch_raw)
                out = model(batch_local)
                out_local = align_outputs(out, cell_ids=cell_ids, use_resolver=use_resolver)
                if poisson_aggregation_mode == "sum_per_cell_means":
                    batch_num_vec, _ = _masked_poisson_num_den_per_cell(out_local)
                    (batch_num_vec / total_den).sum().backward()
                    poisson_num_total = poisson_num_total + batch_num_vec.detach()
                else:
                    batch_num, _ = _masked_poisson_num_den(out_local)
                    (batch_num / total_den).backward()
                    poisson_num_total = poisson_num_total + batch_num.detach()

            if poisson_aggregation_mode == "sum_per_cell_means":
                poisson_loss = (poisson_num_total / total_den.detach()).sum()
            else:
                poisson_loss = poisson_num_total / total_den.detach()
            l_sparse, l_local, reg_term, gamma_local = compute_regularization(
                model=model,
                sparsity_mode=sparsity_mode,
                lambda_reg=lambda_reg,
                lambda_local_prox=lambda_local_prox,
                circular_dims=circular_dims,
                gamma_mode=gamma_mode,
                gamma_value=gamma_value,
                locality_mode=locality_mode,
                lambda_local_per_neuron=lambda_local_per_neuron,
            )
            reg_term.backward()

            loss = poisson_loss + reg_term.detach()
            step_stats["loss"] = float(loss.detach().item())
            step_stats["poisson"] = float(poisson_loss.detach().item())
            step_stats["sparse"] = float(l_sparse.detach().item())
            step_stats["local"] = float(l_local.detach().item())
            step_stats["reg"] = float(reg_term.detach().item())
            step_stats["gamma_local"] = float(gamma_local)
            if torch.is_tensor(total_den) and total_den.ndim > 0:
                step_stats["accum_denominator"] = float(total_den.detach().sum().item())
            else:
                step_stats["accum_denominator"] = float(total_den.detach().item())
            step_stats["accum_num_batches"] = int(total_batches)
            step_stats["accum_num_samples"] = int(total_samples)
            return loss

        optimizer.step(closure)

        prox_tau_last = 0.0
        if sparsity_mode == "prox_l1":
            lr = float(optimizer.param_groups[0].get("lr", 1e-3))
            prox_tau_last = lr * lambda_prox
            prox_group_l21_(model.w_pos.weight, model.w_neg.weight, tau=prox_tau_last)

        step_stats["prox_tau"] = (
            float(prox_tau_last.detach().mean().item())
            if torch.is_tensor(prox_tau_last)
            else float(prox_tau_last)
        )
        return step_stats, None

    def closure():
        optimizer.zero_grad()
        out = model(batch)
        out_local = align_outputs(out, cell_ids=cell_ids, use_resolver=use_resolver)
        poisson_loss = spike_loss(out_local)
        l_sparse, l_local, reg_term, gamma_local = compute_regularization(
            model=model,
            sparsity_mode=sparsity_mode,
            lambda_reg=lambda_reg,
            lambda_local_prox=lambda_local_prox,
            circular_dims=circular_dims,
            gamma_mode=gamma_mode,
            gamma_value=gamma_value,
            locality_mode=locality_mode,
            lambda_local_per_neuron=lambda_local_per_neuron,
        )
        loss = poisson_loss + reg_term
        loss.backward()
        step_stats["loss"] = float(loss.detach().item())
        step_stats["poisson"] = float(poisson_loss.detach().item())
        step_stats["sparse"] = float(l_sparse.detach().item())
        step_stats["local"] = float(l_local.detach().item())
        step_stats["reg"] = float(reg_term.detach().item())
        step_stats["gamma_local"] = float(gamma_local)
        return loss

    optimizer.step(closure)

    prox_tau_last = 0.0
    if sparsity_mode == "prox_l1":
        lr = float(optimizer.param_groups[0].get("lr", 1e-3))
        prox_tau_last = lr * lambda_prox
        prox_group_l21_(model.w_pos.weight, model.w_neg.weight, tau=prox_tau_last)

    with torch.no_grad():
        out = model(batch)
        out = align_outputs(out, cell_ids=cell_ids, use_resolver=use_resolver)

    step_stats["prox_tau"] = (
        float(prox_tau_last.detach().mean().item())
        if torch.is_tensor(prox_tau_last)
        else float(prox_tau_last)
    )
    return step_stats, out


def eval_step(model, batch, cell_ids, use_resolver=False):
    with torch.no_grad():
        out = model(batch)
        out = align_outputs(out, cell_ids=cell_ids, use_resolver=use_resolver)
    return out
