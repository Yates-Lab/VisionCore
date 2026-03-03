import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%%
from DataYatesV1 import enable_autoreload

enable_autoreload()
from models.losses import MaskedPoissonNLLLoss, PoissonBPSAggregator
from tqdm import tqdm
from torch.utils.data import DataLoader
from util import get_dataset_info
import torch
import numpy as np
import math

#%%
dataset_configs_path = "/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml"
subject = "Allen"
date = "2022-04-13"
image_shape = (41, 41)
dataset_info = get_dataset_info(dataset_configs_path, subject, date, image_shape)
peak_lags = dataset_info["peak_lags"]
stas = dataset_info["stas"]
train_dset = dataset_info["train_dset"]
val_dset = dataset_info["val_dset"]
robs = dataset_info["robs"]
dfs = dataset_info["dfs"]
crop_size = dataset_info["crop_size"]

#%%
from two_stage_core import TwoStage
from two_stage_helpers import show_epoch_diagnostics
from two_stage_trainer import eval_step, prepare_batch, train_step_lbfgs

lambda_reg_lbfgs = 0.0
gamma_local_lbfgs = 0.0
circular_dims = {1}
losses = []
cell_ids = [int(os.getenv("CELL_ID", "16"))]
num_epochs = int(os.getenv("NUM_EPOCHS", "4"))
lbfgs_step_mode = "full_dataset_accum"
locality_mode = "weighted_l21"

spike_loss = MaskedPoissonNLLLoss(pred_key="rhat", target_key="robs", mask_key="dfs")
sparsity_mode = "prox_l1"
phase_name = "lbfgs_exp_convex_sparse_local"

# Convex sparse settings
cell_lambda_prox = {
    14: 1.00e-04,
    15: 1.00e-04,
    16: 1.00e-04,
    66: 2.68e-05,
    76: 1.93e-06,
}
cell_lambda_local = {
    14: 1.00e-04,
    15: 1.00e-04,
    16: 1.00e-04,
    66: 1.00e-04,
    76: 1.00e-04,
}
lambda_prox = float(
    os.getenv("LAMBDA_PROX", str(cell_lambda_prox.get(int(cell_ids[0]), 1e-2)))
)
lambda_local_prox = float(
    os.getenv("LAMBDA_LOCAL", str(cell_lambda_local.get(int(cell_ids[0]), 1e-2)))
)

n_lags = 1
num_neurons = 1

eps = 1e-12
masked_rate_mean = float((robs[:, cell_ids[0]] * dfs[:, cell_ids[0]]).sum().item()) / max(
    float(dfs[:, cell_ids[0]].sum().item()),
    eps,
)
beta_init = float(np.log(max(masked_rate_mean, 1e-8)))
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
    beta_as_parameter=True,
    clamp_beta_min=None,
    hann_window_power=2,
    output_nonlinearity="exp",
)

model.cuda()
torch.cuda.empty_cache()
train_dset.to("cpu")
val_dset.to("cpu")
batch_size_lbfgs = int(os.getenv("BATCH_SIZE", "2048"))

train_loader_lbfgs_accum = DataLoader(
    train_dset,
    batch_size=batch_size_lbfgs,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=4,
)
train_loader_lbfgs_legacy = DataLoader(
    train_dset,
    batch_size=batch_size_lbfgs,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=4,
)
val_loader_lbfgs = DataLoader(
    val_dset,
    batch_size=batch_size_lbfgs,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=4,
)


def calibrate_exp_intercept_full_dataset():
    pred_num = 0.0
    target_num = 0.0
    with torch.no_grad():
        for batch in train_loader_lbfgs_accum:
            batch = prepare_batch(
                batch,
                peak_lags=peak_lags,
                cell_ids=cell_ids,
                crop_size=crop_size,
            )
            out = eval_step(model=model, batch=batch, cell_ids=cell_ids, use_resolver=True)
            mask = out["dfs"]
            pred_num += float((out["rhat"] * mask).sum().item())
            target_num += float((out["robs"] * mask).sum().item())
    correction = math.log(max(target_num, eps) / max(pred_num, eps))
    with torch.no_grad():
        model.beta.add_(correction)


calibrate_exp_intercept_full_dataset()

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=float(os.getenv("LBFGS_LR", "0.1")),
    max_iter=int(os.getenv("LBFGS_MAX_ITER", "10")),
    history_size=10,
    line_search_fn="strong_wolfe",
)

best_val_bps = -1e9
best_epoch = -1

for epoch in range(num_epochs):
    train_agg = PoissonBPSAggregator()
    val_agg = PoissonBPSAggregator()
    prox_tau_last = 0.0
    poisson_last = sparse_last = local_last = reg_last = gamma_local = 0.0
    accum_den_last = accum_batches_last = accum_samples_last = 0.0

    active_train_loader = (
        train_loader_lbfgs_accum
        if lbfgs_step_mode == "full_dataset_accum"
        else train_loader_lbfgs_legacy
    )
    active_val_loader = val_loader_lbfgs

    if lbfgs_step_mode == "full_dataset_accum":
        model.train()
        model.alpha_pos.requires_grad = False
        step_stats, _ = train_step_lbfgs(
            model=model,
            optimizer=optimizer_lbfgs,
            batch=None,
            batch_loader=active_train_loader,
            prepare_batch_fn=lambda b: prepare_batch(
                b,
                peak_lags=peak_lags,
                cell_ids=cell_ids,
                crop_size=crop_size,
            ),
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
            locality_mode=locality_mode,
        )
        losses.append(step_stats["loss"])
        prox_tau_last = step_stats["prox_tau"]
        poisson_last = step_stats["poisson"]
        sparse_last = step_stats["sparse"]
        local_last = step_stats["local"]
        reg_last = step_stats["reg"]
        gamma_local = step_stats["gamma_local"]
        accum_den_last = step_stats.get("accum_denominator", 0.0)
        accum_batches_last = step_stats.get("accum_num_batches", 0)
        accum_samples_last = step_stats.get("accum_num_samples", 0)
    else:
        raise ValueError(f"Unknown lbfgs_step_mode: {lbfgs_step_mode}")

    for batch in tqdm(active_train_loader):
        model.eval()
        batch = prepare_batch(batch, peak_lags=peak_lags, cell_ids=cell_ids, crop_size=crop_size)
        out = eval_step(model=model, batch=batch, cell_ids=cell_ids, use_resolver=True)
        train_agg(out)

    for batch in active_val_loader:
        model.eval()
        batch = prepare_batch(batch, peak_lags=peak_lags, cell_ids=cell_ids, crop_size=crop_size)
        out = eval_step(model=model, batch=batch, cell_ids=cell_ids, use_resolver=True)
        val_agg(out)
    bps = train_agg.closure().cpu().numpy()
    bps_val = val_agg.closure().cpu().numpy()
    val_bps_scalar = float(np.asarray(bps_val).reshape(-1)[0])
    if val_bps_scalar > best_val_bps:
        best_val_bps = val_bps_scalar
        best_epoch = epoch

    print(
        f"epoch={epoch:03d} lbfgs_fullbatch "
        f"micro_batches={int(accum_batches_last)} "
        f"samples={int(accum_samples_last)} "
        f"denominator={float(accum_den_last):.3f} "
        f"poisson={poisson_last:.6f} reg={reg_last:.6f}"
    )
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
        phase=phase_name,
        epoch=epoch,
        show_colorwheel=True,
    )

print(
    f"[final] cell={cell_ids[0]} lambda_prox={lambda_prox:.3e} "
    f"lambda_local={lambda_local_prox:.3e} best_val_bps={best_val_bps:.4f} best_epoch={best_epoch}"
)

# %%
