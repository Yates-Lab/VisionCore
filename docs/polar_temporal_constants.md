# Polar Model Temporal Constants - Guide

This document explains the temporal constants in the Polar-V1 model and how to make them learnable.

## Overview

The Polar model has 4 key temporal constants that control how features evolve over time:

1. **`lambda_fix`** - Amplitude decay rate during fixation
2. **`lambda_sac`** - Amplitude decay rate during saccade
3. **`alpha_fast`** - Fast EMA decay rate for temporal integration
4. **`alpha_slow`** - Slow EMA decay rate for temporal integration

---

## 1. Amplitude Relaxation Rates (`lambda_fix`, `lambda_sac`)

### What They Do

These control how quickly feature amplitudes decay back to baseline:

```python
# In PolarDynamics.forward() (line 124-128)
lam = (1 - q) * lambda_fix + q * lambda_sac  # Blend based on saccade gate
A_next = a_bar + (A - a_bar) * exp(-lam * dt)  # Exponential decay
```

- **During fixation** (`q=0`): Uses `lambda_fix`
- **During saccade** (`q=1`): Uses `lambda_sac`
- **Time constant**: `τ = 1 / lambda`

### Original Values

```yaml
lambda_fix: 10.0   # τ = 100ms (fixation persistence)
lambda_sac: 40.0   # τ = 25ms (saccadic suppression)
```

**Interpretation:**
- Features persist for ~100ms during fixation
- Features decay 4x faster during saccades (saccadic suppression)

### Are They Learnable?

**YES!** They're already `nn.Parameter` in the code (line 76-77):

```python
self.lambda_fix = nn.Parameter(torch.tensor(lambda_fix))
self.lambda_sac = nn.Parameter(torch.tensor(lambda_sac))
```

The model will optimize these during training!

### Relaxed Initialization

**Original:**
```yaml
lambda_fix: 10.0   # τ = 100ms
lambda_sac: 40.0   # τ = 25ms
```

**Relaxed (recommended):**
```yaml
lambda_fix: 5.0    # τ = 200ms (2x slower - more persistent)
lambda_sac: 20.0   # τ = 50ms (2x slower - less suppression)
```

**Why?**
- Original values were hand-tuned for specific assumptions
- Your data may have different temporal statistics
- Starting slower gives the optimizer room to speed up if needed
- Easier to learn to go faster than slower

---

## 2. EMA Decay Rates (`alpha_fast`, `alpha_slow`)

### What They Do

These control temporal integration windows for creating temporal summaries:

```python
# In TemporalSummarizer._ema() (line 177)
acc = alpha * acc + (1 - alpha) * x[t]  # Exponential moving average
```

- **Higher alpha** (closer to 1) = longer memory, slower decay
- **Lower alpha** (closer to 0) = shorter memory, faster decay
- **Effective time constant**: `τ ≈ 1 / (1 - alpha)` frames

### Original Values

```yaml
alpha_fast: 0.74   # τ ≈ 3.8 frames ≈ 32ms @ 120Hz
alpha_slow: 0.95   # τ ≈ 20 frames ≈ 167ms @ 120Hz
```

**Interpretation:**
- Fast trace captures transient responses (~30ms window)
- Slow trace captures sustained responses (~170ms window)

### Were They Learnable?

**NO** (originally) - they were just floats.

**NOW YES!** After our modification, they're learnable parameters stored in logit space:

```python
# New code (line 144-159)
self.alpha_fast_logit = nn.Parameter(torch.logit(torch.tensor(alpha_fast)))
self.alpha_slow_logit = nn.Parameter(torch.logit(torch.tensor(alpha_slow)))

@property
def alpha_fast(self):
    return torch.sigmoid(self.alpha_fast_logit)  # Constrained to [0, 1]
```

### Relaxed Initialization

**Original:**
```yaml
alpha_fast: 0.74   # τ ≈ 32ms
alpha_slow: 0.95   # τ ≈ 167ms
```

**Relaxed (recommended):**
```yaml
alpha_fast: 0.80   # τ ≈ 42ms (slightly slower)
alpha_slow: 0.90   # τ ≈ 83ms (faster, more responsive)
learnable_alphas: true  # Enable learning
```

**Why?**
- Original values may not match your data's temporal structure
- Starting with intermediate values gives more flexibility
- Model can learn to be faster or slower as needed

---

## 3. Configuration Options

### Option 1: Original (Fixed Values)

```yaml
# experiments/model_configs/polar_v1.yaml
recurrent:
  type: polar
  params:
    lambda_fix: 10.0
    lambda_sac: 40.0
    alpha_fast: 0.74
    alpha_slow: 0.95
    learnable_alphas: false  # Disable learning of alphas
```

**Use when:** You're confident in the hand-tuned values.

---

### Option 2: Relaxed & Learnable (Recommended)

```yaml
# experiments/model_configs/polar_v1_relaxed.yaml
recurrent:
  type: polar
  params:
    lambda_fix: 5.0     # 2x slower than original
    lambda_sac: 20.0    # 2x slower than original
    alpha_fast: 0.80    # Slightly slower than original
    alpha_slow: 0.90    # Faster than original
    learnable_alphas: true  # Enable learning (default)
```

**Use when:** You want the model to learn optimal temporal dynamics from data.

---

### Option 3: Very Relaxed (Exploratory)

```yaml
recurrent:
  type: polar
  params:
    lambda_fix: 2.0     # τ = 500ms (very persistent)
    lambda_sac: 10.0    # τ = 100ms (minimal suppression)
    alpha_fast: 0.85    # τ ≈ 56ms (slower)
    alpha_slow: 0.85    # τ ≈ 56ms (much faster, same as fast!)
    learnable_alphas: true
```

**Use when:** You have no prior knowledge and want maximum flexibility.

---

## 4. Monitoring During Training

To track how these parameters evolve during training, add logging:

```python
# In your training loop or Lightning module
if hasattr(model.recurrent, 'dynamics'):
    # Log amplitude relaxation rates
    wandb.log({
        'temporal/lambda_fix': model.recurrent.dynamics.lambda_fix.item(),
        'temporal/lambda_sac': model.recurrent.dynamics.lambda_sac.item(),
        'temporal/tau_fix_ms': 1000.0 / model.recurrent.dynamics.lambda_fix.item(),
        'temporal/tau_sac_ms': 1000.0 / model.recurrent.dynamics.lambda_sac.item(),
    })

if hasattr(model.recurrent, 'summarizer'):
    # Log EMA decay rates
    wandb.log({
        'temporal/alpha_fast': model.recurrent.summarizer.alpha_fast.item(),
        'temporal/alpha_slow': model.recurrent.summarizer.alpha_slow.item(),
        'temporal/tau_fast_frames': 1.0 / (1.0 - model.recurrent.summarizer.alpha_fast.item()),
        'temporal/tau_slow_frames': 1.0 / (1.0 - model.recurrent.summarizer.alpha_slow.item()),
    })
```

This will show you:
- How the parameters change over training
- Whether they converge to reasonable values
- If the model is learning meaningful temporal dynamics

---

## 5. Expected Behavior

### What to Expect During Training

1. **Early epochs:** Parameters may fluctuate as the model explores
2. **Mid training:** Parameters should stabilize around optimal values
3. **Late training:** Parameters should converge and stop changing

### Sanity Checks

**Lambda values:**
- Should be positive (enforced by initialization)
- `lambda_sac` should typically be > `lambda_fix` (faster decay during saccades)
- Typical range: 1.0 - 100.0 (time constants: 10ms - 1000ms)

**Alpha values:**
- Should be in (0, 1) (enforced by sigmoid)
- `alpha_slow` should typically be > `alpha_fast` (longer memory)
- Typical range: 0.5 - 0.99 (time constants: 2 - 100 frames)

### If Parameters Go Crazy

If you see unreasonable values (e.g., `lambda_fix = 1000.0` or `alpha_fast = 0.99`):

1. **Check learning rate:** May be too high for these parameters
2. **Add constraints:** Clamp values to reasonable ranges
3. **Use separate LR:** Give temporal params a lower learning rate
4. **Increase regularization:** Add L2 penalty on temporal params

---

## 6. Advanced: Per-Parameter Learning Rates

If temporal parameters are unstable, you can give them a lower learning rate:

```python
# In configure_optimizers()
param_groups = [
    # Core parameters (normal LR)
    {'params': core_params, 'lr': core_lr},
    
    # Temporal parameters (lower LR)
    {'params': [
        model.recurrent.dynamics.lambda_fix,
        model.recurrent.dynamics.lambda_sac,
        model.recurrent.summarizer.alpha_fast_logit,
        model.recurrent.summarizer.alpha_slow_logit,
    ], 'lr': core_lr * 0.1},  # 10x lower LR
]

optimizer = torch.optim.AdamW(param_groups)
```

---

## 7. Summary & Recommendations

### For Most Users (Recommended)

Use `polar_v1_relaxed.yaml`:
- ✅ Learnable temporal constants (enabled by default)
- ✅ Relaxed initialization (2x slower decay rates)
- ✅ Model learns optimal dynamics from your data
- ✅ Monitor parameters during training

### For Conservative Users

Use `polar_v1.yaml` with `learnable_alphas: true`:
- ✅ Original initialization (hand-tuned values)
- ✅ Learnable alphas (new feature)
- ✅ Lambda values still learnable (already were)
- ⚠️ May be suboptimal if your data differs from original assumptions

### For Exploratory Users

Create custom config with very relaxed values:
- ✅ Maximum flexibility
- ✅ Let model find optimal dynamics
- ⚠️ May need more training time
- ⚠️ Monitor for instability

---

## 8. Code Changes Summary

### Modified Files

1. **`models/modules/polar_recurrent.py`**
   - Made `alpha_fast` and `alpha_slow` learnable parameters
   - Stored in logit space for unconstrained optimization
   - Constrained to [0, 1] via sigmoid
   - Added `learnable_alphas` flag to `TemporalSummarizer`

2. **`experiments/model_configs/polar_v1_relaxed.yaml`**
   - New config with relaxed temporal constants
   - Learnable alphas enabled by default
   - Extensive documentation

### Backward Compatibility

✅ **Fully backward compatible!**
- Old configs still work (alphas default to learnable)
- Can disable with `learnable_alphas: false`
- Lambda values were already learnable

---

## Questions?

- **Q: Should I always use learnable temporal constants?**
  - A: Yes, unless you have strong prior knowledge about optimal values.

- **Q: Will this slow down training?**
  - A: No, negligible overhead (4 extra parameters).

- **Q: Can I fix some and learn others?**
  - A: Yes, but requires code modification. Set specific params as buffers instead of Parameters.

- **Q: What if learned values are unreasonable?**
  - A: Add logging, check learning rates, consider constraints or regularization.

