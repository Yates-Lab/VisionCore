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
#%%

from two_stage_helpers import show_epoch_diagnostics
from two_stage_trainer import eval_step, prepare_batch, train_step_adam


lambda_reg = 1e-2
gamma_local = lambda_reg * 4/20#4/20 #4/20#1/10 
sparsity_mode = "ratio_l1_l2"  # options: "ratio_l1_l2", "prox_l1"
lambda_prox = 1e-4  # used only when sparsity_mode == "prox_l1"
lambda_local_prox = 1e-1  # optional locality weight in prox mode
circular_dims = {1}
# circular_dims = {}
losses = []
cell_ids = [14]

spike_loss = MaskedPoissonNLLLoss(pred_key='rhat', target_key='robs', mask_key='dfs')
# spike_loss =  MaskedLoss(nn.MSELoss(reduction='none'), pred_key='rhat', target_key='robs', mask_key='dfs')
# n_lags = len(dataset_config['keys_lags']['stim'])
n_lags = 1
# image_shape = train_dset[0]['stim'].shape[2:]

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
    poisson_last = sparse_last = local_last = reg_last = gamma_local = 0.0
    
    for i, batch in enumerate(tqdm(train_loader)):
        batch = prepare_batch(batch, peak_lags=peak_lags, cell_ids=cell_ids, crop_size=crop_size)
        step_stats, out = train_step_adam(
            model=model,
            optimizer=optimizer,
            batch=batch,
            spike_loss=spike_loss,
            cell_ids=cell_ids,
            sparsity_mode=sparsity_mode,
            lambda_reg=lambda_reg,
            lambda_local_prox=lambda_local_prox,
            circular_dims=circular_dims,
            gamma_mode="adaptive_5pct",
            lambda_prox=lambda_prox,
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
    
    for batch in val_loader:
        optimizer.eval()
        batch = prepare_batch(batch, peak_lags=peak_lags, cell_ids=cell_ids, crop_size=crop_size)
        out = eval_step(model=model, batch=batch, cell_ids=cell_ids, use_resolver=False)
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
        )
    

# %%
