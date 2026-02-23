# NOTE: LBFGS-Adam hybrid variant of two_stage.py.
#%%
from DataYatesV1 import get_gaborium_sta_ste, get_session, plot_stas, enable_autoreload,calc_sta
enable_autoreload()
from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator, MaskedLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
import schedulefree
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
crop_size = dataset_info['crop_size']
#%%

from two_stage_core import TwoStage



from two_stage_helpers import (
    show_epoch_diagnostics,
)
from two_stage_trainer import eval_step, prepare_batch, train_step_adam, train_step_lbfgs


# Phase-specific regularization settings (matched to source files).
# LBFGS stage mirrors two_stage_lbfgs.py behavior.
lambda_reg_lbfgs = 1e-4
gamma_local_lbfgs = lambda_reg_lbfgs * 4 / 20
# Adam stage mirrors two_stage.py behavior.
lambda_reg_adam = 1e-2 #1e-5
sparsity_mode = "ratio_l1_l2"  # options: "ratio_l1_l2", "prox_l1"
lambda_prox = 1e-4  # used only when sparsity_mode == "prox_l1"
lambda_local_prox = 1e-1  # optional locality weight in prox mode
circular_dims = {1}
# circular_dims = {}
losses = []
cell_ids = [16]
num_epochs = 100
lbfgs_epochs = 2

spike_loss = MaskedPoissonNLLLoss(pred_key='rhat', target_key='robs', mask_key='dfs')
# spike_loss =  MaskedLoss(nn.MSELoss(reduction='none'), pred_key='rhat', target_key='robs', mask_key='dfs')
# n_lags = len(dataset_config['keys_lags']['stim'])
n_lags = 1
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
    poisson_last = sparse_last = local_last = reg_last = gamma_local = 0.0
    phase = "lbfgs" if epoch < lbfgs_epochs else "adam"
    active_train_loader = train_loader_lbfgs if phase == "lbfgs" else train_loader_adam
    active_val_loader = val_loader_lbfgs if phase == "lbfgs" else val_loader_adam
    
    for i, batch in enumerate(tqdm(active_train_loader)):
        model.train()
        batch = prepare_batch(batch, peak_lags=peak_lags, cell_ids=cell_ids, crop_size=crop_size)
        if phase == "lbfgs":
            step_stats, out = train_step_lbfgs(
                model=model,
                optimizer=optimizer_lbfgs,
                batch=batch,
                spike_loss=spike_loss,
                cell_ids=cell_ids,
                sparsity_mode=sparsity_mode,
                lambda_reg=lambda_reg_lbfgs,
                lambda_local_prox=lambda_local_prox,
                circular_dims=circular_dims,
                gamma_mode="fixed",
                gamma_value=gamma_local_lbfgs,
                lambda_prox=lambda_prox,
                use_resolver=True,
            )
            losses.append(step_stats["loss"])
            prox_tau_last = step_stats["prox_tau"]

            poisson_last = step_stats["poisson"]
            sparse_last = step_stats["sparse"]
            local_last = step_stats["local"]
            reg_last = step_stats["reg"]
            gamma_local = step_stats["gamma_local"]
            with torch.no_grad():
                train_agg(out)
        else:
            step_stats, out = train_step_adam(
                model=model,
                optimizer=optimizer_adam,
                batch=batch,
                spike_loss=spike_loss,
                cell_ids=cell_ids,
                sparsity_mode=sparsity_mode,
                lambda_reg=lambda_reg_adam,
                lambda_local_prox=lambda_local_prox,
                circular_dims=circular_dims,
                gamma_mode="adaptive_5pct",
                lambda_prox=lambda_prox,
                freeze_beta=True,
                use_resolver=False,
            )
            poisson_last = step_stats["poisson"]
            sparse_last = step_stats["sparse"]
            local_last = step_stats["local"]
            reg_last = step_stats["reg"]
            gamma_local = step_stats["gamma_local"]
            prox_tau_last = step_stats["prox_tau"]
            losses.append(step_stats["loss"])
            with torch.no_grad():
                train_agg(out)
    
    for batch in active_val_loader:
        if phase == "adam":
            optimizer_adam.eval()
        model.eval()
        batch = prepare_batch(batch, peak_lags=peak_lags, cell_ids=cell_ids, crop_size=crop_size)
        out = eval_step(model=model, batch=batch, cell_ids=cell_ids, use_resolver=(phase == "lbfgs"))
        val_agg(out)
    bps = train_agg.closure().cpu().numpy()
    bps_val = val_agg.closure().cpu().numpy()

    if epoch % 1 == 0:
        show_epoch_diagnostics(
            model=model,
            stas=stas,
            peak_lags=peak_lags,
            cell_id=cell_ids[0],
            sparsity_mode=sparsity_mode,
            poisson_last=poisson_last,
            sparse_last=sparse_last,
            local_last=local_last,
            gamma_local=gamma_local,
            prox_tau_last=prox_tau_last,
            reg_last=reg_last,
            bps=bps,
            bps_val=bps_val,
            phase=phase,
            epoch=epoch,
            show_colorwheel=True,
        )
    

# %%
