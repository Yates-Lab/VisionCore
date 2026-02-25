# NOTE: LBFGS-only variant with STA afferent initialization.
#%%
from DataYatesV1 import get_gaborium_sta_ste, get_session, plot_stas, enable_autoreload, calc_sta
enable_autoreload()
from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator, MaskedLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from util import get_dataset_info
import torch
#%%
dataset_configs_path = "/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml"
subject = "Allen"
date = "2022-04-13"
image_shape = (41, 41)
dataset_info = get_dataset_info(dataset_configs_path, subject, date, image_shape)
peak_lags = dataset_info["peak_lags"]
stas = dataset_info["stas"]
stes = dataset_info["stes"]
cids = dataset_info["cids"]
train_dset = dataset_info["train_dset"]
val_dset = dataset_info["val_dset"]
dataset_config = dataset_info["dataset_config"]
robs = dataset_info["robs"]
crop_size = dataset_info["crop_size"]
#%%

from two_stage_core import TwoStage
from two_stage_helpers import show_epoch_diagnostics
from two_stage_sta_afferent_preview import initialize_model_afferents_from_sta
from two_stage_trainer import eval_step, prepare_batch, train_step_lbfgs


# Phase-A calibration: no regularization, only beta/alphas train.
phase_a_epochs = 1
# Keep total length close to two_stage_lbfgs_adam.py.
num_epochs = 100

# Phase-B regularization settings (match existing LBFGS settings).
lambda_reg_lbfgs = 5e-4
gamma_local_lbfgs = lambda_reg_lbfgs * 4 / 20
sparsity_mode = "ratio_l1_l2"  # options: "ratio_l1_l2", "prox_l1"
lambda_prox = 1e-4  # used only when sparsity_mode == "prox_l1"
lambda_local_prox = 1e-1  # optional locality weight in prox mode
circular_dims = {1}
losses = []
cell_ids = [66]

spike_loss = MaskedPoissonNLLLoss(pred_key="rhat", target_key="robs", mask_key="dfs")
# spike_loss = MaskedLoss(nn.MSELoss(reduction='none'), pred_key='rhat', target_key='robs', mask_key='dfs')
n_lags = 1
num_neurons = 1
beta_init = robs[:, cell_ids[0]].mean().item()

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
train_dset.to("cpu")
val_dset.to("cpu")


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.detach().pow(2).mean().sqrt()


def match_sta_scale_to_init_regime(model: TwoStage, target_pos_rms: torch.Tensor, target_neg_rms: torch.Tensor):
    # Keep STA geometry but match initial magnitude regime so LBFGS starts stably.
    with torch.no_grad():
        cur_pos_rms = _rms(model.w_pos.weight).clamp_min(1e-12)
        cur_neg_rms = _rms(model.w_neg.weight).clamp_min(1e-12)
        model.w_pos.weight.mul_(target_pos_rms / cur_pos_rms)
        model.w_neg.weight.mul_(target_neg_rms / cur_neg_rms)


target_pos_rms = _rms(model.w_pos.weight)
target_neg_rms = _rms(model.w_neg.weight)
sta_init_out = initialize_model_afferents_from_sta(
    model=model,
    train_dset=train_dset,
    peak_lags=peak_lags,
    cell_ids=cell_ids,
    crop_size=crop_size,
    lag_mode="peak_bank",
    batch_size=2048,
    num_workers=8,
    activation_softmax=False,
    activation_softmax_temp=0.1,
    device="cuda",
    export_all_lags=False,
)
match_sta_scale_to_init_regime(model, target_pos_rms=target_pos_rms, target_neg_rms=target_neg_rms)

batch_size = 10024
train_loader = DataLoader(
    train_dset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
val_loader = DataLoader(
    val_dset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

optimizer_phase_a = torch.optim.LBFGS(
    [model.alpha_pos, model.alpha_neg],
    lr=1,
    max_iter=5,
    history_size=10,
    line_search_fn="strong_wolfe",
)

optimizer_phase_b = torch.optim.LBFGS(
    model.parameters(),
    lr=1,
    max_iter=5,
    history_size=10,
    line_search_fn="strong_wolfe",
)


for epoch in range(num_epochs):
    train_agg = PoissonBPSAggregator()
    val_agg = PoissonBPSAggregator()
    prox_tau_last = 0.0
    poisson_last = sparse_last = local_last = reg_last = gamma_local = 0.0
    in_phase_a = epoch < phase_a_epochs
    phase = "lbfgs_calibration" if in_phase_a else "lbfgs_regularized"

    if in_phase_a:
        model.w_pos.weight.requires_grad = False
        model.w_neg.weight.requires_grad = False
        optimizer = optimizer_phase_a
        lambda_reg = 0.0
        lambda_local = 0.0
        gamma_value = 0.0
    else:
        model.w_pos.weight.requires_grad = True
        model.w_neg.weight.requires_grad = True
        #freeze model.alpha_pos for phase b
        model.alpha_pos.requires_grad = False
        optimizer = optimizer_phase_b
        lambda_reg = lambda_reg_lbfgs
        lambda_local = lambda_local_prox
        gamma_value = gamma_local_lbfgs

    for i, batch in enumerate(tqdm(train_loader)):
        model.train()
        batch = prepare_batch(batch, peak_lags=peak_lags, cell_ids=cell_ids, crop_size=crop_size)
        step_stats, out = train_step_lbfgs(
            model=model,
            optimizer=optimizer,
            batch=batch,
            spike_loss=spike_loss,
            cell_ids=cell_ids,
            sparsity_mode=sparsity_mode,
            lambda_reg=lambda_reg,
            lambda_local_prox=lambda_local,
            circular_dims=circular_dims,
            gamma_mode="fixed",
            gamma_value=gamma_value,
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

    for batch in val_loader:
        model.eval()
        batch = prepare_batch(batch, peak_lags=peak_lags, cell_ids=cell_ids, crop_size=crop_size)
        out = eval_step(model=model, batch=batch, cell_ids=cell_ids, use_resolver=True)
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
