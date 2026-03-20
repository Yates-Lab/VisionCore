# %% [markdown]
# # Learning Rate Range Finding (Multi-Step per LR Level)
#
# **What:** Test a range of learning rates by training for N steps at each LR
# level (with fresh model weights each time), recording the loss improvement.
# The optimal LR is the one that reduces loss the most in a fixed step budget.
#
# **Why:** The standard 1-step-per-LR range test (Smith 2017) doesn't work well
# for multi-dataset training — each step only updates a subset of readout heads,
# so the model needs multiple steps to show meaningful signal at any given LR.
#
# **References:**
# - Smith (2017) "Cyclical Learning Rates for Training Neural Networks"
#   https://arxiv.org/abs/1506.01438
# - Smith & Topin (2019) "Super-Convergence: Very Fast Training of Neural Networks
#   Using Large Learning Rates" https://arxiv.org/abs/1708.07120
# - fastai implementation: https://docs.fast.ai/callback.schedule.html#learner.lr_find
#
# **How this variant works:**
# 1. Save the initial model weights (with bias initialization)
# 2. Compute null loss (forward pass only, no training) as a baseline
# 3. For each LR level (log-spaced from LR_MIN to LR_MAX):
#    a. Restore the initial weights (fresh start — no crosstalk between levels)
#    b. Create a fresh optimizer (fresh Adam momentum/variance)
#    c. Train for STEPS_PER_LR steps at this fixed LR
#    d. Record the loss trajectory
# 4. Compare final losses across LR levels against the null baseline
# 5. The best LR is the one with the lowest final loss (or steepest improvement)
#
# **Why reinitialize?** With many steps per LR level, the model changes
# significantly. Without reinitialization, by the last LR level the model has
# seen thousands of cumulative steps — you'd be measuring crosstalk, not LR
# sensitivity.

# %% Imports and setup
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# __file__ isn't defined in IPython/Jupyter — use a hardcoded fallback
try:
    _project_root = str(Path(__file__).resolve().parent.parent)
except NameError:
    _project_root = str(Path.home() / "VisionCore")
sys.path.insert(0, _project_root)

from models.config_loader import load_config, load_dataset_configs
from models import build_model
from models.losses import MaskedLoss
from training.pl_modules import MultiDatasetDM

torch.set_float32_matmul_precision('medium')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# %% Configuration — adjust these to match your training setup
#
# These should mirror your train_digital_twin_120_long.sh settings.

MODEL_CONFIG = "VisionCore/experiments/model_configs/learned_resnet_concat_convgru_gaussian.yaml"
DATASET_CONFIGS = "VisionCore/experiments/dataset_configs/multi_basic_120_long.yaml"
MAX_DATASETS = 30
BATCH_SIZE = 256            # large batch needed for informative gradients (most spike bins are 0)
ACCUM_STEPS = 4             # gradient accumulation — effective batch = BATCH_SIZE * ACCUM_STEPS
NUM_WORKERS = 32
DSET_DTYPE = "bfloat16"

# Range finder parameters
LR_MIN = 1e-4             # starting LR (should be too small to learn much)
LR_MAX = 1e-1             # ending LR (should be too large — will diverge)
NUM_LR_LEVELS = 12        # number of LR values to test (log-spaced)
STEPS_PER_LR = 1536       # 3 "epochs" of 512 steps (2 warmup + 1 real) per LR level
TAIL_STEPS = 64           # average loss over last N steps as the "final loss" at each LR

# Optimizer settings (match your training)
WEIGHT_DECAY = 1e-5
CORE_LR_SCALE = 1.0      # core gets lr * this scale factor

# %% Load data

MICRO_BATCHES_PER_LEVEL = STEPS_PER_LR * ACCUM_STEPS
TOTAL_MICRO_BATCHES = NUM_LR_LEVELS * MICRO_BATCHES_PER_LEVEL
print(f"Loading datasets... ({MICRO_BATCHES_PER_LEVEL} micro-batches per level, {NUM_LR_LEVELS} levels)")
dm = MultiDatasetDM(
    cfg_dir=DATASET_CONFIGS,
    max_ds=MAX_DATASETS,
    batch=BATCH_SIZE,
    workers=NUM_WORKERS,
    steps_per_epoch=MICRO_BATCHES_PER_LEVEL,
    dset_dtype=DSET_DTYPE,
)
dm.setup()

train_loader = dm.train_dataloader()
print(f"Train loader: {len(train_loader)} batches available (need {MICRO_BATCHES_PER_LEVEL} per LR level)")

# %% Build model

print("Building model...")
model_cfg = load_config(MODEL_CONFIG)
dataset_cfgs = load_dataset_configs(DATASET_CONFIGS)[:MAX_DATASETS]
for cfg in dataset_cfgs:
    cfg["_dataset_name"] = cfg['session']

model = build_model(model_cfg, dataset_cfgs).to(device)

# Detect activation type for loss
log_input = isinstance(model.activation, nn.Identity)
loss_fn = MaskedLoss(nn.PoissonNLLLoss(log_input=log_input, reduction="none"))
print(f"Model built. log_input={log_input}")

# Initialize readout biases from empirical firing rates
# (mirrors MultiDatasetModel.on_fit_start — without this, the model starts
# with zero/random biases and the initial loss is artificially high)
names = [cfg['session'] for cfg in dataset_cfgs]
print("Initializing readout biases from empirical firing rates...")
for idx, name in enumerate(names):
    if name not in dm.train_dsets:
        continue
    dset = dm.train_dsets[name]
    combined = dset.base if hasattr(dset, 'base') else dset

    total_rate, total_weight = None, None
    for sub_dset in combined.dsets:
        robs = sub_dset['robs'].float()
        dfs = sub_dset['dfs'].float() if 'dfs' in sub_dset else torch.ones_like(robs)
        weighted_sum = (robs * dfs).sum(dim=0)
        weight_sum = dfs.sum(dim=0)
        if total_rate is None:
            total_rate, total_weight = weighted_sum, weight_sum
        else:
            total_rate += weighted_sum
            total_weight += weight_sum

    if total_rate is None:
        continue

    mean_rate = total_rate / total_weight.clamp(min=1.0)
    if log_input:
        bias = torch.log(mean_rate.clamp(min=1e-8))
    else:
        clamped = mean_rate.clamp(min=1e-6)
        bias = torch.where(clamped > 20.0, clamped, torch.log(torch.expm1(clamped)))

    readout = model.readouts[idx]
    if hasattr(readout, 'bias') and readout.bias is not None:
        readout.bias.data = bias.to(dtype=readout.bias.dtype, device=readout.bias.device)
        print(f"  {name}: bias [{bias.min():.3f}, {bias.max():.3f}], "
              f"mean_rate [{mean_rate.min():.4f}, {mean_rate.max():.4f}]")

print("Readout biases initialized.")

# %% Model summary
print("Model summary:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Size on disk (float32): {total_params * 4 / 1e6:.2f} MB")

# %% Param group helper
#
# Mirrors the param group structure from MultiDatasetModel so the LR range
# test reflects the actual training dynamics (core vs head LR).

def make_param_groups(named_params, base_lr, wd, core_lr_scale):
    """Create param groups matching MultiDatasetModel.configure_optimizers."""
    core_keys = ("frontend", "convnet", "modulator")
    core_lr = base_lr * core_lr_scale

    core_wd, core_no, head_wd, head_no = [], [], [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        apply_wd = name.endswith(".weight") and (p.ndim > 1)
        is_core = any(k in name for k in core_keys)
        if is_core:
            (core_wd if apply_wd else core_no).append(p)
        else:
            (head_wd if apply_wd else head_no).append(p)

    groups = []
    if core_wd: groups.append({"params": core_wd, "lr": core_lr, "weight_decay": wd,  "lr_ratio": core_lr_scale})
    if core_no: groups.append({"params": core_no, "lr": core_lr, "weight_decay": 0.0, "lr_ratio": core_lr_scale})
    if head_wd: groups.append({"params": head_wd, "lr": base_lr, "weight_decay": wd,  "lr_ratio": 1.0})
    if head_no: groups.append({"params": head_no, "lr": base_lr, "weight_decay": 0.0, "lr_ratio": 1.0})
    return groups


# Save the initial model + bias state so we can restore for each LR level.
# This must happen AFTER bias initialization above.
init_state = copy.deepcopy(model.state_dict())
print("Saved initial model state (for reinitialization at each LR level).")

# %% Forward pass helper

def forward_batch(batch):
    """Run a single micro-batch through the model, return scalar loss or None."""
    batch_list = [batch] if isinstance(batch, dict) else batch
    losses = []
    for b in batch_list:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        ds_idx = b["dataset_idx"][0]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            rhat = model(stimulus=b["stim"], dataset_idx=ds_idx, behavior=b.get("behavior"))
            rhat = torch.clamp(rhat, min=-20 if log_input else 1e-8)
            loss = loss_fn({'rhat': rhat.float(), 'robs': b["robs"].float(), 'dfs': b["dfs"].float()})
        if torch.isfinite(loss):
            losses.append(loss)
    return losses


# %% Deterministic data iterator
#
# Every LR level (and the null baseline) sees the exact same data.
# Instead of pre-fetching all batches (which OOMs on pin_memory), we seed
# the RNG identically before each pass so the dataloader produces the same
# batch sequence every time.

DATA_SEED = 42

def make_data_iter():
    """Create a deterministic iterator over train_loader.

    Seeds Python, NumPy, and PyTorch RNGs so that the RandomSampler inside the
    dataloader yields the same index order every time this is called.
    """
    import random
    random.seed(DATA_SEED)
    np.random.seed(DATA_SEED)
    torch.manual_seed(DATA_SEED)
    return iter(train_loader)

# %% Compute null loss (no training baseline)
#
# Run the initialized model over the same batches without any parameter
# updates.  The tail-averaged null loss serves as a divergence threshold:
# any LR whose final loss exceeds this is doing worse than not training.

print("Computing null loss (forward pass only, no training)...")
model.load_state_dict(init_state)
model.eval()

null_step_losses = []
data_iter = make_data_iter()
with torch.no_grad():
    for step in range(STEPS_PER_LR):
        accum_losses = []
        for accum_idx in range(ACCUM_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)  # wrap around (no reseed — same as pre-fetch)
                batch = next(data_iter)
            losses = forward_batch(batch)
            accum_losses.extend(l.item() for l in losses)
        if accum_losses:
            null_step_losses.append(np.mean(accum_losses))

null_loss = np.mean(null_step_losses[-TAIL_STEPS:]) if len(null_step_losses) >= TAIL_STEPS else np.mean(null_step_losses)
print(f"  Null loss (tail avg): {null_loss:.6f}")

# %% LR range sweep

lr_levels = np.logspace(np.log10(LR_MIN), np.log10(LR_MAX), NUM_LR_LEVELS)

all_curves = []       # list of (lr, [loss_step0, loss_step1, ...])
tail_losses = []      # tail-averaged loss at each LR level

print(f"\nRunning LR range test: {LR_MIN:.0e} -> {LR_MAX:.0e}")
print(f"  {NUM_LR_LEVELS} LR levels x {STEPS_PER_LR} steps (x{ACCUM_STEPS} accum) = {TOTAL_MICRO_BATCHES} total micro-batches")
print(f"  Tail averaging over last {TAIL_STEPS} steps\n")

for level_idx, lr in enumerate(tqdm(lr_levels, desc="LR sweep", unit="level")):
    # 1. Restore initial weights
    model.load_state_dict(init_state)
    model.train()

    # 2. Fresh optimizer
    param_groups = make_param_groups(
        list(model.named_parameters()), base_lr=lr, wd=WEIGHT_DECAY,
        core_lr_scale=CORE_LR_SCALE,
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # 3. Train for STEPS_PER_LR steps (same batches for every LR — seeded identically)
    data_iter = make_data_iter()
    step_losses = []
    for step in range(STEPS_PER_LR):
        optimizer.zero_grad()
        accum_losses = []

        for accum_idx in range(ACCUM_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            losses = forward_batch(batch)
            for l in losses:
                (l / ACCUM_STEPS).backward()
                accum_losses.append(l.item())

        if not accum_losses:
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        step_losses.append(np.mean(accum_losses))

    # 4. Record
    all_curves.append((lr, step_losses))
    tail_avg = np.mean(step_losses[-TAIL_STEPS:]) if len(step_losses) >= TAIL_STEPS else np.mean(step_losses)
    tail_losses.append(tail_avg)

    delta = step_losses[0] - tail_avg if step_losses else float('nan')
    print(f"  [{level_idx+1:2d}/{NUM_LR_LEVELS}]  lr={lr:.2e}  "
          f"init={step_losses[0]:.4f}  final={tail_avg:.4f}  "
          f"delta={delta:+.4f}")

print(f"\nDone. Tested {NUM_LR_LEVELS} LR levels.")

# %% Analysis: find the suggested LR

lr_arr = np.array(lr_levels)
tail_arr = np.array(tail_losses)

# Flag diverged LRs: all LRs above the first one worse than null
diverged_mask = tail_arr > null_loss
first_diverged = np.argmax(diverged_mask) if diverged_mask.any() else NUM_LR_LEVELS
# Everything at and above the first diverged index is flagged
diverged_mask[first_diverged:] = True

# Best by final loss (among non-diverged)
valid_mask = ~diverged_mask
if valid_mask.any():
    best_final_idx = np.where(valid_mask)[0][np.argmin(tail_arr[valid_mask])]
else:
    best_final_idx = np.argmin(tail_arr)
lr_best_final = lr_arr[best_final_idx]

# Steepest loss gradient (smoothed finite differences)
n_smooth = 3
tail_diff = np.diff(tail_arr)
tail_diff = np.convolve(tail_diff, np.ones(n_smooth) / n_smooth, mode='same')
best_gradient_idx = np.argmax(-tail_diff) + 1  # +1 because diff shifts indices
lr_best_gradient = lr_arr[best_gradient_idx]
lr_scale = 1 / 3
lr_suggested = lr_best_gradient * lr_scale

n_diverged = diverged_mask.sum()
print(f"\n{'='*60}")
print(f"LR Range Test Results")
print(f"{'='*60}")
print(f"  Null loss (no training): {null_loss:.4f}")
print(f"  Best final loss:         lr={lr_best_final:.2e}  (loss={tail_arr[best_final_idx]:.4f})")
print(f"  Steepest improvement:    lr={lr_best_gradient:.2e}")
print(f"  Suggested LR (x{lr_scale:.2f}):  lr={lr_suggested:.2e}")
print(f"  Diverged LR levels:      {n_diverged}/{NUM_LR_LEVELS}", end="")
if n_diverged > 0:
    print(f"  (first diverged: {lr_arr[first_diverged]:.2e})")
else:
    print()
print(f"{'='*60}")

# %% Plot: Loss vs Learning Rate
#
# Two-panel plot:
#   Left:  Final (tail) loss at each LR with null loss baseline
#   Right: Loss gradient (negative = improving)

# Auto-scale: 10% buffer above null loss, 10% below best loss
best_loss = tail_arr[best_final_idx]
loss_range = null_loss - best_loss
y_top = null_loss + 0.1 * loss_range
y_bottom = best_loss - 0.1 * loss_range

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Left panel: Final loss vs LR ---
ax = axes[0]
ax.plot(lr_arr[~diverged_mask], tail_arr[~diverged_mask], 'o-', color='#2196F3', linewidth=2, markersize=5)
if diverged_mask.any():
    ax.plot(lr_arr[diverged_mask], tail_arr[diverged_mask], 'x', color='#9E9E9E', markersize=7, label='Diverged')
ax.axhline(null_loss, color='#F44336', linestyle=':', linewidth=1.5, label=f'Null loss: {null_loss:.4f}')
ax.axvline(lr_best_gradient, color='#4CAF50', linestyle='--', label=f'Steepest improvement: {lr_best_gradient:.2e}')
ax.axvline(lr_suggested, color='#FF5722', linestyle='--', label=f'Suggested LR: {lr_suggested:.2e}')

ax.set_xscale('log')
ax.set_xlabel('Learning Rate')
ax.set_ylabel(f'Loss (avg of last {TAIL_STEPS} steps)')
ax.set_title('Final Loss vs Learning Rate')
ax.legend(fontsize=9)
ax.set_ylim(y_bottom, y_top)
ax.grid(True, alpha=0.3)

# --- Right panel: loss gradient vs LR ---
ax = axes[1]
ax.plot(lr_arr[1:], -tail_diff, 'o-', color='#E91E63', linewidth=2, markersize=5)
ax.axhline(0, color='#9E9E9E', linestyle=':', linewidth=1)
ax.axvline(lr_best_gradient, color='#4CAF50', linestyle='--', label=f'Steepest improvement: {lr_best_gradient:.2e}')
ax.axvline(lr_suggested, color='#FF5722', linestyle='--', label=f'Suggested LR: {lr_suggested:.2e}')
ax.set_xscale('log')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Loss Improvement (smoothed)')
ax.set_title('Loss Gradient vs Learning Rate')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ryan/lr_range_test_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plot to ryan/lr_range_test_results.png")

# %% Summary & next steps
#
# HOW TO INTERPRET:
#
#   1. Left plot: Should show the classic U-shape (or check-mark shape).
#      Low LRs -> flat (no learning). Moderate LRs -> low loss (learning).
#      High LRs -> diverged (grey X markers above the red null-loss line).
#      The minimum is the best LR.
#
#   2. Right plot: The peak is the LR with the steepest loss reduction.
#      Negative values mean the LR caused the loss to increase.
#
# HOW TO USE THE RESULT:
#
#   Use the suggested LR as your --lr flag:
#     bash experiments/train_digital_twin_120_long.sh --lr <suggested_lr>
#
#   The core LR is automatically scaled by CORE_LR_SCALE (0.5), so you
#   only need to set the base (head) LR.
#
# CAVEATS:
#
#   - The range test uses a fresh model with random init. If you're
#     fine-tuning from a checkpoint, re-run with loaded weights.
#
#   - Different optimizers have different optimal LR ranges. This test
#     assumes AdamW (matching our training pipeline).
