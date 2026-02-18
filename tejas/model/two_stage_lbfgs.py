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
from util import get_dataset_info
import torch
#%%
dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml'
subject = 'Allen'
date = '2022-04-13'
image_shape = (41, 41)
dataset_info = get_dataset_info(dataset_configs_path, subject, date, image_shape)
peak_lags = dataset_info['peak_lags']
stas = dataset_info['stas']
stes = dataset_info['stes']
cids = dataset_info['cids']
train_dset = dataset_info['train_dset']
val_dset = dataset_info['val_dset']
dataset_config = dataset_info['dataset_config']
robs = dataset_info['robs']
#%%
from two_stage_core import TwoStage
#%%

from two_stage_helpers import (
    _resolve_output_indices,
    locality_penalty_from_maps,
    prox_group_l21_,
    render_energy_component_rgb,
    sparsity_penalty,
    visualize_afferent_map,
)


lambda_reg = 1e-4 
gamma_local = lambda_reg * 4/20#4/20 #4/20#1/10 
sparsity_mode = "ratio_l1_l2"  # options: "ratio_l1_l2", "prox_l1"
lambda_prox = 1e-4  # used only when sparsity_mode == "prox_l1"
lambda_local_prox = 1e-1  # optional locality weight in prox mode
circular_dims = {1}
# circular_dims = {}
losses = []
crop_size = 5
cell_ids = [16]
num_epochs = 80
target_bps_val = 2
plot_every = 1  # set >0 to enable periodic plotting

spike_loss = MaskedPoissonNLLLoss(pred_key='rhat', target_key='robs', mask_key='dfs')
# spike_loss =  MaskedLoss(nn.MSELoss(reduction='none'), pred_key='rhat', target_key='robs', mask_key='dfs')
# n_lags = len(dataset_config['keys_lags']['stim'])
n_lags = 1
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
    clamp_beta_min = 1e-6

)

model.cuda()
torch.cuda.empty_cache()
train_dset.to('cpu')
val_dset.to('cpu')
batch_size = 10024  # 64

train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.LBFGS(
    model.parameters(),
    # lr=0.2,
    # max_iter=5,
    # history_size=10,
    # line_search_fn="strong_wolfe",
    # lr=1,
    # max_iter=10000,
    # tolerance_grad=1e-6,
    # # tolerance_change=1e-8,
    # tolerance_change=1e-6,
    # # line_search_fn='strong_wolfe'
    # history_size=10,
    lr=1,
    max_iter=5,
    history_size=10,
    line_search_fn="strong_wolfe",
)

for epoch in range(num_epochs):
    train_agg = PoissonBPSAggregator()
    val_agg = PoissonBPSAggregator()
    prox_tau_last = 0.0
    
    for i, batch in enumerate(tqdm(train_loader)):
        model.train()
        batch = {k: v.cuda() for k, v in batch.items()}
        batch['stim'] = batch['stim'][:, :, peak_lags[cell_ids], crop_size:-crop_size, crop_size:-crop_size]
        step_stats = {}

        def closure():
            optimizer.zero_grad()
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
                # gamma_local_local = 0.05 / max(l_local.detach().item(), 1e-12)
                gamma_local_local = gamma_local
                reg_term = lambda_reg * l_sparse * (1.0 + gamma_local_local * l_local)
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

        optimizer.step(closure)
        if sparsity_mode == "prox_l1":
            lr = float(optimizer.param_groups[0].get("lr", 1e-3))
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
    
    for batch in val_loader:
        model.eval()
        with torch.no_grad():
            batch = {k: v.cuda() for k, v in batch.items()}
            batch['stim'] = batch['stim'][:, :, peak_lags[cell_ids], crop_size:-crop_size, crop_size:-crop_size]
            out = model(batch)
            pred_idx, target_idx = _resolve_output_indices(cell_ids, out)
            out['rhat'] = out['rhat'][:, pred_idx]
            out['robs'] = out['robs'][:, target_idx]
            out['dfs'] = out['dfs'][:, target_idx]
            assert out['dfs'].shape == out['rhat'].shape == out['robs'].shape
            val_agg(out)
    bps = train_agg.closure().cpu().numpy()
    bps_val = val_agg.closure().cpu().numpy()

    if plot_every > 0 and (epoch % plot_every == 0):
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
        locality_factor = gamma_local * local_last
        print(
            f"mode={sparsity_mode}, poisson={poisson_last:.6f}, "
            f"L_sparse={sparse_last:.6f}, L_local={local_last:.6f}, "
            f"gamma*L_local={locality_factor:.6f} ({100.0 * locality_factor:.2f}%), "
            f"prox_tau={prox_tau_last:.6e}, reg={reg_last:.6f}"
        ) 
        # plt.plot(losses)
        # plt.show()

        print("beta:", model.beta.item())
        print(bps.item())
        print(bps_val.item())

    bps_train_scalar = float(np.asarray(bps).reshape(-1)[0])
    bps_val_scalar = float(np.asarray(bps_val).reshape(-1)[0])
    print(
        f"epoch={epoch:03d} bps_train={bps_train_scalar:.4f} "
        f"bps_val={bps_val_scalar:.4f} poisson={poisson_last:.6f} reg={reg_last:.6f}"
    )
    if bps_val_scalar >= target_bps_val:
        print(f"Reached target validation BPS {bps_val_scalar:.4f} >= {target_bps_val:.4f} at epoch {epoch}.")
        break
    

# %%
