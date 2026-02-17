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
    locality_penalty_from_maps,
    prox_group_l21_,
    render_energy_component_rgb,
    sparsity_penalty,
    visualize_afferent_map,
)


lambda_reg = 1e-2
gamma_local = lambda_reg * 4/20#4/20 #4/20#1/10 
sparsity_mode = "ratio_l1_l2"  # options: "ratio_l1_l2", "prox_l1"
lambda_prox = 1e-4  # used only when sparsity_mode == "prox_l1"
lambda_local_prox = 1e-1  # optional locality weight in prox mode
circular_dims = {1}
# circular_dims = {}
losses = []
crop_size = 5
cell_ids = [66]

spike_loss = MaskedPoissonNLLLoss(pred_key='rhat', target_key='robs', mask_key='dfs')
# spike_loss =  MaskedLoss(nn.MSELoss(reduction='none'), pred_key='rhat', target_key='robs', mask_key='dfs')
# n_lags = len(dataset_config['keys_lags']['stim'])
n_lags = 1
# image_shape = train_dset[0]['stim'].shape[2:]
image_shape = (41, 41)
# num_neurons = len(dataset_config['cids'])
num_neurons = 1
# beta_init = robs[:, cell_ids[0]].mean().item()
beta_init = 0.0
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
)

model.cuda()
torch.cuda.empty_cache()
train_dset.to('cpu')
val_dset.to('cpu')
batch_size = 1024  # 64

train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = schedulefree.RAdamScheduleFree(model.parameters())
# optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-5)

for epoch in range(100):
    train_agg = PoissonBPSAggregator()
    val_agg = PoissonBPSAggregator()
    prox_tau_last = 0.0
    
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.train()
        batch = {k: v.cuda() for k, v in batch.items()}
        batch['stim'] = batch['stim'][:, :, peak_lags[cell_ids], crop_size:-crop_size, crop_size:-crop_size]
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
            gamma_local = 0.05/l_local.detach().item()
            reg_term = lambda_reg * l_sparse * (1.0 + gamma_local * l_local)
        elif sparsity_mode == "prox_l1":
            l_sparse = l_local.new_zeros(())
            reg_term = lambda_local_prox * l_local
        else:
            raise ValueError(f"Unknown sparsity_mode: {sparsity_mode}")
        loss = poisson_loss + reg_term
        poisson_last = poisson_loss.detach()
        sparse_last = l_sparse.detach()
        local_last = l_local.detach()
        reg_last = reg_term.detach()
        losses.append(loss.item())
        
        loss.backward()

        optimizer.step()
        if sparsity_mode == "prox_l1":
            lr = float(optimizer.param_groups[0].get("lr", 1e-3))
            prox_tau_last = lr * lambda_prox
            prox_group_l21_(model.w_pos.weight, model.w_neg.weight, tau=prox_tau_last)
        optimizer.zero_grad()


        with torch.no_grad():
            train_agg(out)
    
    for batch in val_loader:
        optimizer.eval()
        with torch.no_grad():
            batch = {k: v.cuda() for k, v in batch.items()}
            batch['stim'] = batch['stim'][:, :, peak_lags[cell_ids], crop_size:-crop_size, crop_size:-crop_size]
            out = model(batch)
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
        locality_factor = gamma_local * local_last.item()
        print(
            f"mode={sparsity_mode}, poisson={poisson_last.item():.6f}, "
            f"L_sparse={sparse_last.item():.6f}, L_local={local_last.item():.6f}, "
            f"gamma*L_local={locality_factor:.6f} ({100.0 * locality_factor:.2f}%), "
            f"prox_tau={prox_tau_last:.6e}, reg={reg_last.item():.6f}"
        ) 
        # plt.plot(losses)
        # plt.show()

        print("beta:", model.beta.item())
        print(bps.item())
        print(bps_val.item())
    

# %%
