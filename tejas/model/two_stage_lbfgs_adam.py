# NOTE: LBFGS variant of two_stage.py (optimizer/training loop changed).
#%%
from DataYatesV1 import get_gaborium_sta_ste, get_session, plot_stas, enable_autoreload,calc_sta
enable_autoreload()
from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator, MaskedLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import schedulefree
import numpy as np
#%%

from util import get_dataset_from_config
dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml'
train_dset, val_dset, dataset_config = get_dataset_from_config('Allen', '2022-04-13', dataset_configs_path)
cids = dataset_config['cids']

#%%
train_dset_loaded = train_dset[:]


stim = train_dset_loaded['stim']
robs = train_dset_loaded['robs']
dfs = train_dset_loaded['dfs']

n_lags = 5
# Calculate spike-triggered averages (STAs)
stas = calc_sta(stim.detach().cpu().squeeze()[:, 0, 5:-5, 5:-5],
                robs.cpu(),
                range(n_lags),
                dfs=dfs.cpu().squeeze(),
                progress=True).cpu().squeeze().numpy()

# # Calculate spike-triggered second moments (STEs)
# # Uses squared stimulus values via stim_modifier
stes = calc_sta(stim.detach().cpu().squeeze()[:, 0, 5:-5, 5:-5],
                robs.cpu(),
                range(n_lags),
                dfs=dfs.cpu().squeeze(),
                stim_modifier=lambda x: x**2,
                progress=True).cpu().squeeze().numpy()

# plot_stas(stas[:, :, None, :, :])
# plt.show()
# plot_stas(stes[:, :, None, :, :])
# plt.show()
peak_lags = np.array([stes[cc].std((1,2)).argmax() for cc in range(stes.shape[0])])


#%%
from pyr_utils import (
    find_pyr_size_and_height_for_lowest_cpd,
)
from two_stage_core import TwoStage

# Example:
cfg = find_pyr_size_and_height_for_lowest_cpd(
    lowest_cpd_target=1.0,
    ppd=train_dset.dsets[0].metadata["ppd"],
    order=3,
    rel_tolerance=0.3,
    validate=True,
)
print(cfg)
#%%

import torch
from two_stage_helpers import (
    _resolve_output_indices,
    locality_penalty_from_maps,
    prox_group_l21_,
    render_energy_component_rgb,
    sparsity_penalty,
    visualize_afferent_map,
)


# Phase-specific regularization settings (matched to source files).
# LBFGS stage mirrors two_stage_lbfgs.py behavior.
lambda_reg_lbfgs = 1e-4
gamma_local_lbfgs = lambda_reg_lbfgs * 4 / 20
# Adam stage mirrors two_stage.py behavior.
lambda_reg_adam = 1e-4 #1e-5
sparsity_mode = "ratio_l1_l2"  # options: "ratio_l1_l2", "prox_l1"
lambda_prox = 1e-4  # used only when sparsity_mode == "prox_l1"
lambda_local_prox = 1e-1  # optional locality weight in prox mode
circular_dims = {1}
# circular_dims = {}
losses = []
crop_size = 5
cell_ids = [16]
num_epochs = 100
lbfgs_epochs = 10

spike_loss = MaskedPoissonNLLLoss(pred_key='rhat', target_key='robs', mask_key='dfs')
# spike_loss =  MaskedLoss(nn.MSELoss(reduction='none'), pred_key='rhat', target_key='robs', mask_key='dfs')
# n_lags = len(dataset_config['keys_lags']['stim'])
n_lags = 1
# image_shape = train_dset[0]['stim'].shape[2:]
image_shape = (41, 41)
# num_neurons = len(dataset_config['cids'])
num_neurons = 1
beta_init = robs[:, cell_ids[0]].mean().item()
# beta_init = 0.0
model = TwoStage(
    image_shape=image_shape,
    n_neurons=num_neurons,
    n_lags=n_lags,
    height=3,
    order=5,
    lowest_cpd_target=1.0,
    ppd=train_dset.dsets[0].metadata["ppd"],
    rel_tolerance=0.3,
    validate_cpd=True,
    beta_init=beta_init,
    init_weight_scale=1e-4,
    beta_as_parameter=False,
    clamp_beta_min=1e-6,
)


model.cuda()
torch.cuda.empty_cache()
train_dset.to('cpu')
val_dset.to('cpu')
batch_size_lbfgs = 10024  # matched to two_stage_lbfgs.py
batch_size_adam = 10024  # matched to two_stage.py

train_loader_lbfgs = DataLoader(
    train_dset,
    batch_size=batch_size_lbfgs,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
val_loader_lbfgs = DataLoader(
    val_dset,
    batch_size=batch_size_lbfgs,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
train_loader_adam = DataLoader(
    train_dset,
    batch_size=batch_size_adam,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
val_loader_adam = DataLoader(
    val_dset,
    batch_size=batch_size_adam,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1,
    max_iter=5,
    history_size=10,
    line_search_fn="strong_wolfe",
)
optimizer_adam = schedulefree.RAdamScheduleFree(model.parameters())
# optimizer_adam = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-4, warmup_steps = 30)

for epoch in range(num_epochs):
    train_agg = PoissonBPSAggregator()
    val_agg = PoissonBPSAggregator()
    prox_tau_last = 0.0
    phase = "lbfgs" if epoch < lbfgs_epochs else "adam"
    active_train_loader = train_loader_lbfgs if phase == "lbfgs" else train_loader_adam
    active_val_loader = val_loader_lbfgs if phase == "lbfgs" else val_loader_adam
    
    for i, batch in enumerate(tqdm(active_train_loader)):
        model.train()
        batch = {k: v.cuda() for k, v in batch.items()}
        batch['stim'] = batch['stim'][:, :, peak_lags[cell_ids], crop_size:-crop_size, crop_size:-crop_size]
        if phase == "lbfgs":
            step_stats = {}

            def closure():
                optimizer_lbfgs.zero_grad()
                out = model(batch)
                pred_idx, target_idx = _resolve_output_indices(cell_ids, out)
                out['rhat'] = out['rhat'][:, pred_idx]
                out['robs'] = out['robs'][:, target_idx]
                out['dfs'] = out['dfs'][:, target_idx]
                assert out['dfs'].shape == out['rhat'].shape == out['robs'].shape

                poisson_loss = spike_loss(out)
                pos_map = model.positive_afferent_map[0, 0]
                neg_map = model.negative_afferent_map[0, 0]
                l_local, _ = locality_penalty_from_maps(pos_map, neg_map, circular_dims=circular_dims)
                if sparsity_mode == "ratio_l1_l2":
                    l_sparse = sparsity_penalty(model)
                    gamma_local_local = gamma_local_lbfgs
                    reg_term = lambda_reg_lbfgs * l_sparse * (1.0 + gamma_local_local * l_local)
                elif sparsity_mode == "prox_l1":
                    l_sparse = l_local.new_zeros(())
                    gamma_local_local = 0.0
                    reg_term = lambda_local_prox * l_local
                else:
                    raise ValueError(f"Unknown sparsity_mode: {sparsity_mode}")
                loss = poisson_loss + reg_term
                loss.backward()
                step_stats["poisson"] = float(poisson_loss.detach().item())
                step_stats["sparse"] = float(l_sparse.detach().item())
                step_stats["local"] = float(l_local.detach().item())
                step_stats["reg"] = float(reg_term.detach().item())
                step_stats["gamma_local"] = float(gamma_local_local)
                step_stats["loss"] = float(loss.detach().item())
                return loss

            optimizer_lbfgs.step(closure)
            if sparsity_mode == "prox_l1":
                lr = float(optimizer_lbfgs.param_groups[0].get("lr", 1e-3))
                prox_tau_last = lr * lambda_prox
                prox_group_l21_(model.w_pos.weight, model.w_neg.weight, tau=prox_tau_last)
            losses.append(step_stats["loss"])

            poisson_last = step_stats["poisson"]
            sparse_last = step_stats["sparse"]
            local_last = step_stats["local"]
            reg_last = step_stats["reg"]
            gamma_local = step_stats["gamma_local"]

            with torch.no_grad():
                out = model(batch)
                pred_idx, target_idx = _resolve_output_indices(cell_ids, out)
                out['rhat'] = out['rhat'][:, pred_idx]
                out['robs'] = out['robs'][:, target_idx]
                out['dfs'] = out['dfs'][:, target_idx]
                train_agg(out)
        else:
            optimizer_adam.train()
            #don't allow beta to train in adam stage
            model.beta.requires_grad = False

            out = model(batch)
            out['robs'] = out['robs'][:, cell_ids]
            out['dfs'] = out['dfs'][:, cell_ids]
            assert out['dfs'].shape == out['rhat'].shape == out['robs'].shape

            poisson_loss = spike_loss(out)
            pos_map = model.positive_afferent_map[0, 0]
            neg_map = model.negative_afferent_map[0, 0]
            l_local, _ = locality_penalty_from_maps(pos_map, neg_map, circular_dims=circular_dims)
            if sparsity_mode == "ratio_l1_l2":
                l_sparse = sparsity_penalty(model)
                gamma_local = 0.05 / l_local.detach().item()
                reg_term = lambda_reg_adam * l_sparse * (1.0 + gamma_local * l_local)
            elif sparsity_mode == "prox_l1":
                l_sparse = l_local.new_zeros(())
                reg_term = lambda_local_prox * l_local
                gamma_local = 0.0
            else:
                raise ValueError(f"Unknown sparsity_mode: {sparsity_mode}")
            loss = poisson_loss + reg_term
            poisson_last = poisson_loss.detach()
            sparse_last = l_sparse.detach()
            local_last = l_local.detach()
            reg_last = reg_term.detach()
            losses.append(loss.item())

            loss.backward()
            optimizer_adam.step()
            if sparsity_mode == "prox_l1":
                lr = float(optimizer_adam.param_groups[0].get("lr", 1e-3))
                prox_tau_last = lr * lambda_prox
                prox_group_l21_(model.w_pos.weight, model.w_neg.weight, tau=prox_tau_last)
            optimizer_adam.zero_grad()

            with torch.no_grad():
                train_agg(out)
    
    for batch in active_val_loader:
        if phase == "adam":
            optimizer_adam.eval()
        with torch.no_grad():
            model.eval()
            batch = {k: v.cuda() for k, v in batch.items()}
            batch['stim'] = batch['stim'][:, :, peak_lags[cell_ids], crop_size:-crop_size, crop_size:-crop_size]
            out = model(batch)
            if phase == "lbfgs":
                pred_idx, target_idx = _resolve_output_indices(cell_ids, out)
                out['rhat'] = out['rhat'][:, pred_idx]
                out['robs'] = out['robs'][:, target_idx]
                out['dfs'] = out['dfs'][:, target_idx]
            else:
                out['robs'] = out['robs'][:, cell_ids]
                out['dfs'] = out['dfs'][:, cell_ids]
            assert out['dfs'].shape == out['rhat'].shape == out['robs'].shape
            val_agg(out)
    bps = train_agg.closure().cpu().numpy()
    bps_val = val_agg.closure().cpu().numpy()

    if epoch % 1 == 0:
        fig, axes = visualize_afferent_map(model, title=f"Cell {cell_ids[0]}")
        plt.show()
        sta_img = stas[cell_ids[0], peak_lags[cell_ids[0]]]
        energy_exc_rf, energy_inh_rf = model.energy_receptive_fields
        energy_exc_np = energy_exc_rf[0, 0].detach().cpu().numpy()
        energy_inh_np = energy_inh_rf[0, 0].detach().cpu().numpy()
        joint_abs = np.concatenate([np.abs(energy_exc_np).reshape(-1), np.abs(energy_inh_np).reshape(-1)])
        joint_amp_scale = float(np.percentile(joint_abs, 99))
        joint_carrier_scale = float(joint_abs.max())
        exc_rgb = render_energy_component_rgb(
            energy_exc_np,
            hue_rgb=(0.95, 0.70, 0.35),
            amp_scale=joint_amp_scale,
            carrier_scale=joint_carrier_scale,
        )
        inh_rgb = render_energy_component_rgb(
            energy_inh_np,
            hue_rgb=(0.45, 0.70, 0.95),
            amp_scale=joint_amp_scale,
            carrier_scale=joint_carrier_scale,
        )
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(model.linear_receptive_field[0, 0].detach().cpu().numpy(), cmap='coolwarm_r')
        axes[0].set_title("Linear RF")
        axes[0].axis('off')
        axes[1].imshow(exc_rgb)
        axes[1].set_title("Energy Exc RF")
        axes[1].axis('off')
        axes[2].imshow(inh_rgb)
        axes[2].set_title("Energy Inh RF")
        axes[2].axis('off')
        axes[3].imshow(sta_img, cmap='coolwarm_r')
        axes[3].set_title(f"STA (cell {cell_ids[0]})")
        axes[3].axis('off')
        plt.tight_layout()
        plt.show()
        locality_factor = gamma_local * (local_last if isinstance(local_last, float) else local_last.item())
        print(
            f"mode={sparsity_mode}, poisson={(poisson_last if isinstance(poisson_last, float) else poisson_last.item()):.6f}, "
            f"L_sparse={(sparse_last if isinstance(sparse_last, float) else sparse_last.item()):.6f}, "
            f"L_local={(local_last if isinstance(local_last, float) else local_last.item()):.6f}, "
            f"gamma*L_local={locality_factor:.6f} ({100.0 * locality_factor:.2f}%), "
            f"prox_tau={prox_tau_last:.6e}, reg={(reg_last if isinstance(reg_last, float) else reg_last.item()):.6f}"
        ) 
        # plt.plot(losses)
        # plt.show()

        print("beta:", model.beta.item())
        print(bps.item())
        print(bps_val.item())
        print(f"phase={phase}, epoch={epoch}")
    

# %%
