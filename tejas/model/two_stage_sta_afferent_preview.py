#%%

"""
STA afferent initialization and preview export helpers.

Run directly for standalone preview PNG export:
    uv run python /home/tejas/VisionCore/tejas/model/two_stage_sta_afferent_preview.py
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from two_stage_core import TwoStage
from two_stage_helpers import (
    build_rf_sta_figure,
    save_stacked_figures_png,
    visualize_afferent_map,
)
from util import get_dataset_info

ExportMode = Literal["all", "single"]
LagMode = Literal["peak_bank", "median_peak", "all"]


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


def resolve_cell_ids(n_all_cells: int, requested_cell_ids: list[int] | None):
    if requested_cell_ids is None:
        return list(range(int(n_all_cells)))
    if len(requested_cell_ids) == 0:
        raise ValueError("cell_ids cannot be empty when provided.")
    return [int(cid) for cid in requested_cell_ids]


def resolve_lag_selection(
    peak_lags,
    cell_ids: list[int],
    max_valid_lag: int,
    lag_mode: LagMode = "peak_bank",
    force_all_lags: bool = False,
):
    raw_lags = [int(peak_lags[cid]) for cid in cell_ids]
    clipped_lags = [min(max(int(lg), 0), int(max_valid_lag)) for lg in raw_lags]

    if force_all_lags or lag_mode == "all":
        selected_lags = list(range(int(max_valid_lag) + 1))
        lag_slots = [selected_lags.index(lg) for lg in clipped_lags]
    elif lag_mode == "median_peak":
        selected_lags = [int(np.rint(np.median(clipped_lags)))]
        selected_lags[0] = max(0, min(selected_lags[0], int(max_valid_lag)))
        lag_slots = [0 for _ in clipped_lags]
    else:
        selected_lags = unique_preserve_order(clipped_lags)
        lag_slots = [selected_lags.index(lg) for lg in clipped_lags]
    return selected_lags, lag_slots


def compute_sta_feature_weights(
    preview_model: TwoStage,
    dset,
    cell_ids: list[int],
    lag_indices: list[int],
    crop_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
):
    loader_kwargs = dict(num_workers=num_workers, pin_memory=True, shuffle=False)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dset, batch_size=batch_size, **loader_kwargs)

    n_cells = len(cell_ids)
    n_lags = len(lag_indices)
    n_feat = preview_model.w_pos.weight.shape[1]
    acc_pos = torch.zeros((n_cells, n_lags, n_feat), device=device)
    acc_neg = torch.zeros((n_cells, n_lags, n_feat), device=device)
    acc_den = torch.zeros((n_cells, n_lags), device=device)

    ys = crop_slice(crop_size)
    xs = crop_slice(crop_size)
    lag_tensors = [torch.tensor([int(lg)], device=device, dtype=torch.long) for lg in lag_indices]

    preview_model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing STA afferent weights"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            robs = batch["robs"][:, cell_ids]
            dfs = batch["dfs"][:, cell_ids]
            weights = (robs * dfs).clamp_min(0.0)
            for lag_slot, lag_tensor in enumerate(lag_tensors):
                batch_lag = dict(batch)
                batch_lag["stim"] = batch["stim"][:, :, lag_tensor, ys, xs]
                pos_feats, neg_feats, _ = preview_model.get_pyr_feats(batch_lag)
                acc_pos[:, lag_slot] += weights.transpose(0, 1) @ pos_feats
                acc_neg[:, lag_slot] += weights.transpose(0, 1) @ neg_feats
                acc_den[:, lag_slot] += weights.sum(dim=0)

    den = acc_den.clamp_min(1e-8).unsqueeze(-1)
    w_pos_init = acc_pos / den
    w_neg_init = acc_neg / den
    return w_pos_init, w_neg_init


def apply_sta_weights_to_model(
    model: TwoStage,
    w_pos: torch.Tensor,
    w_neg: torch.Tensor,
    activation_softmax: bool = False,
    temp: float = 0.35,
    neuron_indices: list[int] | None = None,
):
    w_pos = w_pos.unsqueeze(0) if w_pos.ndim == 1 else w_pos
    w_neg = w_neg.unsqueeze(0) if w_neg.ndim == 1 else w_neg
    if w_pos.shape != w_neg.shape:
        raise ValueError(f"w_pos and w_neg must share shape, got {w_pos.shape} and {w_neg.shape}.")

    n_rows = int(w_pos.shape[0])
    if neuron_indices is None:
        if n_rows == int(model.w_pos.weight.shape[0]):
            neuron_indices = list(range(n_rows))
        elif n_rows == 1:
            neuron_indices = [0]
        else:
            raise ValueError(
                f"Cannot infer neuron_indices for {n_rows} row(s) and model rows={int(model.w_pos.weight.shape[0])}."
            )
    if len(neuron_indices) != n_rows:
        raise ValueError("neuron_indices length must match number of weight rows.")

    with torch.no_grad():
        for row, neuron_idx in enumerate(neuron_indices):
            w_pos_vec = w_pos[row].clone()
            w_neg_vec = w_neg[row].clone()
            if activation_softmax:
                softmax_temp = max(float(temp), 1e-6)
                act = (w_pos_vec + w_neg_vec).clamp_min(0.0)
                gate = torch.softmax(act / softmax_temp, dim=0)
                w_pos_vec = w_pos_vec * gate
                w_neg_vec = w_neg_vec * gate
            scale = torch.maximum(w_pos_vec.abs().amax(), w_neg_vec.abs().amax()).clamp_min(1e-8)
            model.w_pos.weight[int(neuron_idx)].copy_(w_pos_vec / scale)
            model.w_neg.weight[int(neuron_idx)].copy_(w_neg_vec / scale)


def initialize_model_afferents_from_sta(
    model: TwoStage,
    train_dset,
    peak_lags,
    cell_ids: list[int],
    crop_size: int,
    lag_mode: LagMode = "peak_bank",
    batch_size: int = 2048,
    num_workers: int = 8,
    activation_softmax: bool = False,
    activation_softmax_temp: float = 0.35,
    device: str | None = None,
    export_all_lags: bool = False,
):
    if len(cell_ids) == 0:
        raise ValueError("cell_ids cannot be empty.")
    run_device = device or str(next(model.parameters()).device)
    sample_stim = train_dset[0]["stim"]
    if sample_stim.ndim != 4:
        raise ValueError(f"Expected sample stim to be 4D [C, L, H, W], got shape={tuple(sample_stim.shape)}.")
    max_valid_lag = int(sample_stim.shape[1]) - 1
    selected_lags, lag_slots = resolve_lag_selection(
        peak_lags=peak_lags,
        cell_ids=cell_ids,
        max_valid_lag=max_valid_lag,
        lag_mode=lag_mode,
        force_all_lags=export_all_lags,
    )
    w_pos_by_lag, w_neg_by_lag = compute_sta_feature_weights(
        preview_model=model,
        dset=train_dset,
        cell_ids=cell_ids,
        lag_indices=selected_lags,
        crop_size=crop_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=run_device,
    )
    row_pos = torch.stack([w_pos_by_lag[i, lag_slots[i]] for i in range(len(cell_ids))], dim=0)
    row_neg = torch.stack([w_neg_by_lag[i, lag_slots[i]] for i in range(len(cell_ids))], dim=0)

    model_rows = int(model.w_pos.weight.shape[0])
    if model_rows == len(cell_ids):
        apply_rows = list(range(len(cell_ids)))
        apply_pos = row_pos
        apply_neg = row_neg
        applied_cell_ids = [int(cid) for cid in cell_ids]
    elif model_rows == 1:
        apply_rows = [0]
        apply_pos = row_pos[0]
        apply_neg = row_neg[0]
        applied_cell_ids = [int(cell_ids[0])]
    else:
        raise ValueError(
            f"Cannot map cell_ids (n={len(cell_ids)}) to model rows (n={model_rows})."
        )

    apply_sta_weights_to_model(
        model=model,
        w_pos=apply_pos,
        w_neg=apply_neg,
        activation_softmax=activation_softmax,
        temp=activation_softmax_temp,
        neuron_indices=apply_rows,
    )
    return {
        "selected_lags": selected_lags,
        "lag_slots": lag_slots,
        "w_pos_by_lag": w_pos_by_lag,
        "w_neg_by_lag": w_neg_by_lag,
        "applied_cell_ids": applied_cell_ids,
    }


def save_combined_preview(
    model: TwoStage,
    stas: np.ndarray,
    cell_id: int,
    lag_idx: int,
    lag_value: int,
    neuron_idx: int,
    out_png: Path,
    lag_mode: str,
    activation_softmax: bool,
):
    aff_fig, _ = visualize_afferent_map(
        model,
        title=(
            f"STA Afferent Preview | cell={cell_id} | lag_mode={lag_mode} "
            f"| lag={lag_value} | softmax={'on' if activation_softmax else 'off'}"
        ),
        show_colorwheel=True,
        neuron_idx=neuron_idx,
        lag_idx=lag_idx,
    )

    sta_rf = np.asarray(stas[cell_id, lag_value], dtype=np.float32)
    sta_fig, _ = build_rf_sta_figure(
        model=model,
        sta_rf=sta_rf,
        cell_id=cell_id,
        neuron_idx=neuron_idx,
        lag_idx=lag_idx,
        suptitle=f"cell={cell_id} | lag={lag_value}",
    )
    save_stacked_figures_png(aff_fig, sta_fig, out_png)
    plt.close(aff_fig)
    plt.close(sta_fig)


def render_and_save_cell_previews(
    preview_model: TwoStage,
    stas: np.ndarray,
    cell_ids: list[int],
    selected_lags: list[int],
    lag_slots: list[int],
    w_pos_by_lag: torch.Tensor,
    w_neg_by_lag: torch.Tensor,
    output_dir: str,
    lag_mode: LagMode,
    activation_softmax: bool,
    activation_softmax_temp: float,
    export_mode: ExportMode = "all",
    example_cell_id: int = 14,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if export_mode == "single":
        if int(example_cell_id) not in cell_ids:
            raise ValueError(f"example_cell_id={example_cell_id} is not in cell_ids={cell_ids}")
        export_local_indices = [cell_ids.index(int(example_cell_id))]
    else:
        export_local_indices = list(range(len(cell_ids)))

    for local_idx in tqdm(export_local_indices, desc="Saving cell PNGs"):
        cell_id = int(cell_ids[local_idx])
        lag_slot = int(lag_slots[local_idx])
        lag_value = int(selected_lags[lag_slot])
        apply_sta_weights_to_model(
            model=preview_model,
            w_pos=w_pos_by_lag[local_idx, lag_slot],
            w_neg=w_neg_by_lag[local_idx, lag_slot],
            activation_softmax=activation_softmax,
            temp=activation_softmax_temp,
            neuron_indices=[0],
        )
        out_png = out_dir / f"sta_afferent_cell_{cell_id:03d}.png"
        save_combined_preview(
            model=preview_model,
            stas=stas,
            cell_id=cell_id,
            lag_idx=0,
            lag_value=lag_value,
            neuron_idx=0,
            out_png=out_png,
            lag_mode=lag_mode,
            activation_softmax=activation_softmax,
        )
        # tqdm.write(f"Saved: {out_png}")


def run_sta_afferent_preview(
    cell_ids: list[int] | None = None,
    example_cell_id: int = 14,
    export_mode: ExportMode = "all",
    dataset_configs_path: str = "/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml",
    subject: str = "Allen",
    date: str = "2022-04-13",
    image_shape: tuple[int, int] = (41, 41),
    lag_mode: LagMode = "peak_bank",
    batch_size: int = 2048,
    num_workers: int = 8,
    height: int = 3,
    order: int = 5,
    lowest_cpd_target: float = 1.0,
    rel_tolerance: float = 0.3,
    output_dir: str = "/home/tejas/VisionCore/tejas/model",
    seed: int = 0,
    activation_softmax: bool = False,
    activation_softmax_temp: float = 0.35,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = get_dataset_info(dataset_configs_path, subject, date, tuple(image_shape))
    resolved_cell_ids = resolve_cell_ids(int(info["stas"].shape[0]), cell_ids)
    train_dset = info["train_dset"]
    train_dset.to("cpu")

    preview_model = TwoStage(
        image_shape=tuple(image_shape),
        n_neurons=1,
        n_lags=1,
        height=height,
        order=order,
        lowest_cpd_target=lowest_cpd_target,
        ppd=train_dset.dsets[0].metadata["ppd"],
        rel_tolerance=rel_tolerance,
        validate_cpd=True,
        init_weight_scale=1e-4,
        beta_init=0.0,
        beta_as_parameter=True,
        clamp_beta_min=1e-6,
    ).to(device)

    init_out = initialize_model_afferents_from_sta(
        model=preview_model,
        train_dset=train_dset,
        peak_lags=info["peak_lags"],
        cell_ids=resolved_cell_ids,
        crop_size=info["crop_size"],
        lag_mode=lag_mode,
        batch_size=batch_size,
        num_workers=num_workers,
        activation_softmax=activation_softmax,
        activation_softmax_temp=activation_softmax_temp,
        device=device,
        export_all_lags=(export_mode == "all"),
    )
    render_and_save_cell_previews(
        preview_model=preview_model,
        stas=info["stas"],
        cell_ids=resolved_cell_ids,
        selected_lags=init_out["selected_lags"],
        lag_slots=init_out["lag_slots"],
        w_pos_by_lag=init_out["w_pos_by_lag"],
        w_neg_by_lag=init_out["w_neg_by_lag"],
        output_dir=output_dir,
        lag_mode=lag_mode,
        activation_softmax=activation_softmax,
        activation_softmax_temp=activation_softmax_temp,
        export_mode=export_mode,
        example_cell_id=example_cell_id,
    )
    return preview_model, init_out


if __name__ == "__main__":
    run_sta_afferent_preview(
        cell_ids=None,
        example_cell_id=14,
        export_mode="all",
        dataset_configs_path="/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml",
        subject="Allen",
        date="2022-04-13",
        image_shape=(41, 41),
        lag_mode="peak_bank",
        batch_size=2048,
        num_workers=8,
        height=3,
        order=5,
        lowest_cpd_target=1.0,
        rel_tolerance=0.3,
        output_dir="sta_afferent_maps_0.005_TEST",
        seed=0,
        activation_softmax=True,
        activation_softmax_temp=0.005,
    )
#%%