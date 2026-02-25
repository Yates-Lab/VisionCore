from __future__ import annotations

"""
Interactive STA afferent preview script.

Edit `DEFAULT_CONFIG`, then run:
    uv run python /home/tejas/VisionCore/tejas/model/two_stage_sta_afferent_preview.py

For reuse from other scripts:
    - `compute_initial_sta_weights(config)` returns (context, w_pos_by_lag, w_neg_by_lag)
    - `run_sta_afferent_preview(config)` computes weights and exports PNG previews
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from two_stage_core import TwoStage
from two_stage_helpers import render_energy_component_rgb, visualize_afferent_map
from util import get_dataset_info


ExportMode = Literal["all", "single"]
LagMode = Literal["peak_bank", "median_peak", "all"]


@dataclass
class StaPreviewConfig:
    """
    Editable run configuration for interactive use.
    """

    cell_ids: list[int] | None = None
    example_cell_id: int = 14
    export_mode: ExportMode = "all"
    dataset_configs_path: str = (
        "/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml"
    )
    subject: str = "Allen"
    date: str = "2022-04-13"
    image_shape: tuple[int, int] = (41, 41)
    lag_mode: LagMode = "peak_bank"
    batch_size: int = 2048
    num_workers: int = 8
    height: int = 3
    order: int = 5
    lowest_cpd_target: float = 1.0
    rel_tolerance: float = 0.3
    output_dir: str = "/home/tejas/VisionCore/tejas/model"
    seed: int = 0
    activation_softmax: bool = False
    activation_softmax_temp: float = 0.35


@dataclass
class StaRunContext:
    config: StaPreviewConfig
    device: str
    image_shape: tuple[int, int]
    info: dict
    cell_ids: list[int]
    selected_lags: list[int]
    lag_slots: list[int]
    train_dset: object
    crop_size: int
    feature_model: TwoStage
    render_model: TwoStage


# Edit these values for interactive runs.
DEFAULT_CONFIG = StaPreviewConfig()


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


def build_run_context(config: StaPreviewConfig) -> StaRunContext:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_shape = tuple(config.image_shape)

    info = get_dataset_info(config.dataset_configs_path, config.subject, config.date, image_shape)
    n_all_cells = int(info["stas"].shape[0])
    if config.cell_ids is None:
        cell_ids = list(range(n_all_cells))
    else:
        if len(config.cell_ids) == 0:
            raise ValueError("cell_ids cannot be empty when provided.")
        cell_ids = [int(cid) for cid in config.cell_ids]

    peak_lags = info["peak_lags"]
    train_dset = info["train_dset"]
    crop_size = info["crop_size"]
    max_valid_lag = int(info["stim"].shape[2]) - 1

    raw_lags = [int(peak_lags[cid]) for cid in cell_ids]
    clipped_lags = [min(lg, max_valid_lag) for lg in raw_lags]
    if config.export_mode == "all":
        # Batch export mode: compute all lag maps, then pick each cell's own peak lag.
        selected_lags = list(range(max_valid_lag + 1))
        lag_slots = [selected_lags.index(lg) for lg in clipped_lags]
    elif config.lag_mode == "median_peak":
        selected_lags = [int(np.rint(np.median(clipped_lags)))]
        selected_lags[0] = max(0, min(selected_lags[0], max_valid_lag))
        lag_slots = [0 for _ in clipped_lags]
    elif config.lag_mode == "all":
        selected_lags = list(range(max_valid_lag + 1))
        lag_slots = [selected_lags.index(lg) for lg in clipped_lags]
    else:
        selected_lags = unique_preserve_order(clipped_lags)
        lag_slots = [selected_lags.index(lg) for lg in clipped_lags]

    feature_model = TwoStage(
        image_shape=image_shape,
        n_neurons=1,
        n_lags=1,
        height=config.height,
        order=config.order,
        lowest_cpd_target=config.lowest_cpd_target,
        ppd=train_dset.dsets[0].metadata["ppd"],
        rel_tolerance=config.rel_tolerance,
        validate_cpd=True,
        init_weight_scale=1e-4,
        beta_init=0.0,
        beta_as_parameter=True,
        clamp_beta_min=1e-6,
    ).to(device)

    render_model = TwoStage(
        image_shape=image_shape,
        n_neurons=1,
        n_lags=1,
        height=config.height,
        order=config.order,
        lowest_cpd_target=config.lowest_cpd_target,
        ppd=train_dset.dsets[0].metadata["ppd"],
        rel_tolerance=config.rel_tolerance,
        validate_cpd=True,
        init_weight_scale=1e-4,
        beta_init=0.0,
        beta_as_parameter=True,
        clamp_beta_min=1e-6,
    ).to(device)

    return StaRunContext(
        config=config,
        device=device,
        image_shape=image_shape,
        info=info,
        cell_ids=cell_ids,
        selected_lags=selected_lags,
        lag_slots=lag_slots,
        train_dset=train_dset,
        crop_size=crop_size,
        feature_model=feature_model,
        render_model=render_model,
    )


def compute_sta_feature_weights(
    feature_model: TwoStage,
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
    n_feat = feature_model.w_pos.weight.shape[1]
    acc_pos = torch.zeros((n_cells, n_lags, n_feat), device=device)
    acc_neg = torch.zeros((n_cells, n_lags, n_feat), device=device)
    acc_den = torch.zeros((n_cells, n_lags), device=device)

    ys = crop_slice(crop_size)
    xs = crop_slice(crop_size)

    feature_model.eval()
    lag_tensors = [torch.tensor([int(lg)], device=device, dtype=torch.long) for lg in lag_indices]
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing STA afferent weights"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            robs = batch["robs"][:, cell_ids]
            dfs = batch["dfs"][:, cell_ids]
            weights = (robs * dfs).clamp_min(0.0)
            for lag_slot, lag_tensor in enumerate(lag_tensors):
                batch_lag = dict(batch)
                batch_lag["stim"] = batch["stim"][:, :, lag_tensor, ys, xs]
                pos_feats, neg_feats, _ = feature_model.get_pyr_feats(batch_lag)
                acc_pos[:, lag_slot] += weights.transpose(0, 1) @ pos_feats
                acc_neg[:, lag_slot] += weights.transpose(0, 1) @ neg_feats
                acc_den[:, lag_slot] += weights.sum(dim=0)

    den = acc_den.clamp_min(1e-8).unsqueeze(-1)
    w_pos_init = acc_pos / den
    w_neg_init = acc_neg / den
    return w_pos_init, w_neg_init


def fig_to_rgb_arr(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return rgba[..., :3]


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

    model_lin = (
        model.linear_receptive_field_at(neuron_idx=neuron_idx, lag_idx=lag_idx)[0, 0]
        .detach()
        .cpu()
        .numpy()
    )
    model_exc, model_inh = model.energy_receptive_fields_at(neuron_idx=neuron_idx, lag_idx=lag_idx)
    model_exc = model_exc[0, 0].detach().cpu().numpy()
    model_inh = model_inh[0, 0].detach().cpu().numpy()

    model_joint_abs = np.concatenate([np.abs(model_exc).reshape(-1), np.abs(model_inh).reshape(-1)])
    model_amp_scale = float(np.percentile(model_joint_abs, 99)) if model_joint_abs.size else 1.0
    model_carrier_scale = float(model_joint_abs.max()) if model_joint_abs.size else 1.0
    model_exc_rgb = render_energy_component_rgb(
        model_exc,
        hue_rgb=(0.95, 0.70, 0.35),
        amp_scale=model_amp_scale,
        carrier_scale=model_carrier_scale,
    )
    model_inh_rgb = render_energy_component_rgb(
        model_inh,
        hue_rgb=(0.45, 0.70, 0.95),
        amp_scale=model_amp_scale,
        carrier_scale=model_carrier_scale,
    )

    sta_rf = np.asarray(stas[cell_id, lag_value], dtype=np.float32)
    sta_fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(model_lin, cmap="coolwarm_r")
    axes[0].set_title("Linear RF")
    axes[0].axis("off")
    axes[1].imshow(model_exc_rgb)
    axes[1].set_title("Energy Exc RF")
    axes[1].axis("off")
    axes[2].imshow(model_inh_rgb)
    axes[2].set_title("Energy Inh RF")
    axes[2].axis("off")
    axes[3].imshow(sta_rf, cmap="coolwarm_r")
    axes[3].set_title(f"STA (cell {cell_id})")
    axes[3].axis("off")
    sta_fig.suptitle(f"cell={cell_id} | lag={lag_value}", y=0.98)
    plt.tight_layout()

    aff_img = fig_to_rgb_arr(aff_fig)
    sta_img = fig_to_rgb_arr(sta_fig)
    pad = 16
    out_w = max(aff_img.shape[1], sta_img.shape[1])

    def _pad_to_width(img: np.ndarray, width: int):
        if img.shape[1] == width:
            return img
        left = (width - img.shape[1]) // 2
        right = width - img.shape[1] - left
        return np.pad(img, ((0, 0), (left, right), (0, 0)), mode="constant", constant_values=255)

    aff_pad = _pad_to_width(aff_img, out_w)
    sta_pad = _pad_to_width(sta_img, out_w)
    gap = np.full((pad, out_w, 3), 255, dtype=np.uint8)
    combined = np.concatenate([aff_pad, gap, sta_pad], axis=0)
    plt.imsave(out_png, combined)
    plt.close(aff_fig)
    plt.close(sta_fig)


def compute_initial_sta_weights(config: StaPreviewConfig):
    ctx = build_run_context(config)
    ctx.train_dset.to("cpu")
    w_pos_by_lag, w_neg_by_lag = compute_sta_feature_weights(
        feature_model=ctx.feature_model,
        dset=ctx.train_dset,
        cell_ids=ctx.cell_ids,
        lag_indices=ctx.selected_lags,
        crop_size=ctx.crop_size,
        batch_size=ctx.config.batch_size,
        num_workers=ctx.config.num_workers,
        device=ctx.device,
    )
    return ctx, w_pos_by_lag, w_neg_by_lag


def render_and_save_cell_previews(ctx: StaRunContext, w_pos_by_lag: torch.Tensor, w_neg_by_lag: torch.Tensor):
    out_dir = Path(ctx.config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if ctx.config.export_mode == "single":
        if ctx.config.example_cell_id not in ctx.cell_ids:
            raise ValueError(
                f"example_cell_id={ctx.config.example_cell_id} is not in cell_ids={ctx.cell_ids}"
            )
        export_local_indices = [ctx.cell_ids.index(ctx.config.example_cell_id)]
    else:
        export_local_indices = list(range(len(ctx.cell_ids)))

    for local_idx in tqdm(export_local_indices, desc="Saving cell PNGs"):
        cell_id = int(ctx.cell_ids[local_idx])
        lag_idx = int(ctx.lag_slots[local_idx])
        w_pos_vec = w_pos_by_lag[local_idx, lag_idx].clone()
        w_neg_vec = w_neg_by_lag[local_idx, lag_idx].clone()
        if ctx.config.activation_softmax:
            temp = max(float(ctx.config.activation_softmax_temp), 1e-6)
            act = (w_pos_vec + w_neg_vec).clamp_min(0.0)
            gate = torch.softmax(act / temp, dim=0)
            w_pos_vec = w_pos_vec * gate
            w_neg_vec = w_neg_vec * gate
        scale = torch.maximum(w_pos_vec.abs().amax(), w_neg_vec.abs().amax()).clamp_min(1e-8)
        w_pos_vec = w_pos_vec / scale
        w_neg_vec = w_neg_vec / scale
        with torch.no_grad():
            ctx.render_model.w_pos.weight.copy_(w_pos_vec.unsqueeze(0))
            ctx.render_model.w_neg.weight.copy_(w_neg_vec.unsqueeze(0))
        out_png = out_dir / f"sta_afferent_cell_{cell_id:03d}.png"
        save_combined_preview(
            model=ctx.render_model,
            stas=ctx.info["stas"],
            cell_id=cell_id,
            lag_idx=0,
            lag_value=ctx.selected_lags[lag_idx],
            neuron_idx=0,
            out_png=out_png,
            lag_mode=ctx.config.lag_mode,
            activation_softmax=ctx.config.activation_softmax,
        )
        tqdm.write(f"Saved: {out_png}")


def run_sta_afferent_preview(config: StaPreviewConfig = DEFAULT_CONFIG):
    ctx, w_pos_by_lag, w_neg_by_lag = compute_initial_sta_weights(config)
    render_and_save_cell_previews(ctx, w_pos_by_lag, w_neg_by_lag)
    return ctx, w_pos_by_lag, w_neg_by_lag


if __name__ == "__main__":
    run_sta_afferent_preview(DEFAULT_CONFIG)