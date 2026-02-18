import torch

from two_stage_helpers import (
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
        out["robs"] = out["robs"][:, target_idx]
        out["dfs"] = out["dfs"][:, target_idx]
    else:
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
):
    pos_map = model.positive_afferent_map[0, 0]
    neg_map = model.negative_afferent_map[0, 0]
    l_local, _ = locality_penalty_from_maps(pos_map, neg_map, circular_dims=circular_dims)
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
        reg_term = lambda_local_prox * l_local
    else:
        raise ValueError(f"Unknown sparsity_mode: {sparsity_mode}")
    return l_sparse, l_local, reg_term, gamma_local


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
        "prox_tau": float(prox_tau_last),
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
):
    step_stats = {}

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

    step_stats["prox_tau"] = float(prox_tau_last)
    return step_stats, out


def eval_step(model, batch, cell_ids, use_resolver=False):
    with torch.no_grad():
        out = model(batch)
        out = align_outputs(out, cell_ids=cell_ids, use_resolver=use_resolver)
    return out
