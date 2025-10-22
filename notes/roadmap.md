# VisionCore Development Roadmap

## Current Status
✅ Big refactor complete - models are fitting!
✅ Training pipeline working with `experiments/run_all_models_backimage.sh`
✅ Multi-dataset training (20 datasets) with shared core + per-dataset readouts

---

## Major Goals

### 1. Logging & Monitoring
**Problem**: Currently only logging loss - need richer monitoring during training.

**Requirements**:
- **Fast logging (every N=2-5 epochs)**: < 1 minute
  - Weight statistics/distributions
  - Basic performance metrics
  - Training dynamics
  
- **Slow logging (every J=10-15 epochs)**: Minutes allowed
  - Receptive field visualizations
  - Example predictions vs ground truth
  - Detailed model analysis

**Thoughts**: This is very tractable. Most frameworks have hooks for this. The key is:
- Separate fast vs slow logging clearly
- Make logging modular so you can add/remove metrics easily
- Use existing tools (TensorBoard, wandb) rather than reinventing

---

### 2. Strategic Model Comparison
**Problem**: Need systematic way to compare models and training regimes across different conditions.

**Dimensions to explore**:
- **Architecture**: ResNet vs DenseNet
- **Architecture details**: norm type, activation, pooling, kernel sizes, etc.
- **Training scope**: 
  - All 20 datasets (current)
  - Single dataset
  - Only single units
  - Other subsets

**Goal**: Understand which architectural choices matter and validate multi-dataset training approach.

**Thoughts**: This is actually simpler than it seems:
- You already have the YAML config system - this is perfect for this!
- The hard part is organizing the experiments and results, not running them
- Need:
  1. A way to generate/manage config combinations
  2. A way to track which experiments have been run
  3. A way to compare results across experiments
- Could be as simple as a spreadsheet + naming convention, or as complex as a full experiment tracking system

---

### 3. Multi-stage Fitting
**Problem**: Concerned about readout quality and generalization.

**Specific concerns**:
- Readouts may favor some neurons over others
- No normalization by spike rate in loss
- Potential underfitting of some neurons

**Desired capabilities**:
- **Readout refinement**: Freeze core, refit readouts
- **Transfer learning**: Test if learned cores generalize to new neurons/datasets
- **Per-neuron diagnostics**: Understand which neurons are well/poorly fit

**Thoughts**: This is moderately complex but very doable:
- Readout refinement is straightforward: freeze core parameters, continue training
- The harder question is: what's the right objective for readout refinement?
- Transfer learning requires careful data splits (hold out neurons or datasets)
- This connects strongly to Goal #4 (evaluation)

---

### 4. Model Evaluation
**Problem**: Need standardized evaluation metrics to assess model quality.

**Requirements**:
- Summary metrics for overall model performance
- Per-neuron diagnostics
- Connects to slow logging (Goal #1)
- Informs multi-stage fitting (Goal #3)

**Potential metrics**:
- Correlation (per neuron, aggregated)
- Log-likelihood (normalized by spike rate?)
- Receptive field quality
- Prediction examples
- Generalization metrics (train vs val vs test)

**Thoughts**: This is the foundation for everything else:
- Without good eval, you can't compare models (Goal #2)
- Without good eval, you can't know if readout refinement helps (Goal #3)
- Start simple: correlation + log-likelihood per neuron
- Build up from there

---

## Initial Assessment: Is This Daunting?

**Short answer**: It's a lot, but it's not as bad as it seems!

**Why it's manageable**:
1. **You have the infrastructure**: YAML configs, multi-dataset training, modular architecture
2. **Goals are interconnected**: Solving one helps the others
3. **Can be incremental**: Don't need to solve everything at once

**Suggested priority order**:
1. **Start with Goal #4 (Eval)**: Define 2-3 key metrics. This unblocks everything else.
2. **Then Goal #1 (Logging)**: Implement fast logging of those metrics. Slow logging can wait.
3. **Then Goal #2 (Comparison)**: With logging in place, run systematic comparisons.
4. **Finally Goal #3 (Multi-stage)**: Once you understand what works, refine it.

**Key simplifications**:
- Don't build a fancy experiment tracking system - start with simple naming + spreadsheet
- Don't implement all possible metrics - pick 2-3 that matter most
- Don't try to compare everything - pick a few key architectural choices first
- Readout refinement can be a simple script that loads a checkpoint and continues training

---

## Current Success: Generalization from Natural Images to Gabors

**What's working**:
- Training on 'backimage' (static natural images with free viewing)
- Testing on 'gaborium' (rapidly flashed spatial gabors)
- **Spatial RF structure generalizes!** Model captures spatial shapes at different time lags
- This validates that natural images + eye movements provide enough spatiotemporal modulation

**What's failing**:

### Issue 1: Temporal Resolution
**Problem**: Model temporal responses are too fast, integration too slow
- Learned temporal filters put too much weight on 0th time lag
- Makes sense given high autocorrelation in natural images
- Need: faster responses with slightly higher latencies

**Context**:
- Frontend is feature extraction (potentially retina-like), NOT trying to be V1
- `gaussian_derivatives` init already puts mass at center, but it drifts to lag-0 during training
- Tried biphasic initialization - didn't help
- Tried windowing temporal kernels to reduce aliasing - hurt fitting substantially
- **Core issue**: Need to force first layer to be more highpass and have higher latency

**Challenge**: Initialization alone is not enough. The optimization drives filters toward lag-0 because of natural image statistics.

**Potential solutions**:
1. **Temporal regularization**: -- cool idea, opens up a lot of hyperparameters
   - Penalize low temporal frequencies (encourage highpass)
   - Penalize 0-lag dominance explicitly
   - Could use FFT-based regularization on temporal kernels
   - Question: How to balance with fitting performance?

2. **Temporal dilation in first layer**: -- unlikely to work
   - Force temporal sampling to skip frames
   - Prevents relying on lag-0
   - May conflict with causal padding

3. **Temporal pooling after frontend**: --> unlikely to work
   - Explicit temporal integration layer
   - Forces model to look at longer windows
   - Could be learnable

4. **Hard constraints on temporal kernels**: -- should try masking again. we already have the aa_signal option
   - Enforce minimum latency (e.g., zero out first N lags)
   - Enforce highpass structure (e.g., constrain to be derivative-like)
   - More invasive but guarantees desired properties

**Most straightforward options to try**:

1. **Hard temporal mask** (PRIORITY):
   - Zero out first N time lags in frontend filters (e.g., first 2-3 lags always 0)
   - Forces minimum latency
   - Simple: just apply mask in forward pass
   - Previously tried windowing (hurt fitting) - revisit masking approach
   - **Action**: Try this first

2. **Temporal frequency regularization**:
   - Compute 1D FFT along temporal dimension of filters
   - Penalize DC and low-frequency components (L2 on low-freq bins)
   - Encourages highpass structure
   - Tunable via regularization weight

3. **Temporal derivative constraint**:
   - Constrain filters to sum to ~0 across time (derivative-like)
   - Inherently highpass
   - Could be soft constraint (regularization) or hard constraint (projection)

**Reality check**: No free lunch - all of these trade fitting performance for better temporal structure. Question is whether the tradeoff is worth it.

**Constraint**: Each full training run takes ~8 hours. Must be very selective about what to try.

### Issue 2: Gain Control Across Stimuli
**Problem**: Model rates need affine transformation (gain + offset per cell) on withheld stimuli
- 2 parameters per cell substantially improves bits/spike
- Model learns correct **structure** (spatial/temporal RF) but wrong **scale**
- Suggests normalization isn't handling changes in stimulus statistics well

**Context**:
- Currently using **Global Response Normalization (GRN)** in convnet
- GRN has divisive normalization flavor (biologically motivated)
- Divisive norm models are typically steady-state models
- We're building **dynamic** models - need to learn the nonlinearity
- Hope: data would support learning it, but need right inductive biases

**Normalization types explained**:
- **Instance Normalization**: Normalizes each sample independently (per sample, per channel)
  - Computes mean/std per sample (not across batch like BatchNorm)
  - Used in style transfer to remove instance-specific contrast/brightness
  - Could help because each stimulus type would be normalized independently

- **Local Response Normalization (LRN)**: Divisive norm across spatial locations
  - Normalizes spatially (across nearby spatial positions)

- **Global Response Normalization (GRN)**: Current approach
  - Normalizes across channels
  - Has divisive norm flavor

**Core challenge**:
- Need normalization that responds **dynamically per sample** with **temporal context**
- GRN and LRN have this flavor, but may need to push further
- Normalization layers are typically static (fixed statistics or running averages)
- Stimulus statistics change dramatically between training (natural images) and test (gabors)
- Need normalization that adapts to stimulus statistics **dynamically**
- But also need to learn this from data, not hand-tune

**Potential directions to explore**:

1. **Stimulus-conditioned normalization**:
   - Normalization parameters depend on input statistics
   - Small network takes stimulus features (mean, variance, temporal freq) → outputs norm parameters
   - Learnable but guided by stimulus properties
   - Adapts per sample based on actual stimulus statistics

2. **Temporal context in normalization**:
   - Current norms operate per-frame or with running stats
   - Could normalize based on recent temporal window
   - Running statistics computed over time, not just space/channels
   - Adapts to changing stimulus statistics over time

3. **Hybrid normalization**:
   - Combine multiple normalization types
   - E.g., Instance Norm (per-sample) + GRN (cross-channel)
   - Different stages of network use different norms

4. **Learnable normalization mixing**:
   - Learn to mix different normalization strategies
   - Network learns when to apply which type of normalization
   - More parameters but more flexible

5. **Per-neuron output calibration** (last resort):
   - Learnable gain/bias at readout (regularized to stay near identity)
   - Allows adaptation without overfitting
   - Feels like a band-aid - doesn't address root cause

**Priority**: Stimulus-conditioned normalization seems most promising - addresses dynamic statistics while being learnable.

**Constraint**: Each experiment takes 8 hours. Need to be very selective.

**Note on evaluation**: Pearson correlation is blind to affine transformations - good metric for structure independent of gain.

---

## Connection to Loss Function and Neuron Favoritism

**Concern**: Current Poisson log-likelihood loss may favor some neurons over others
- No normalization by spike rate
- High-firing neurons may dominate gradient
- Could lead to underfitting of low-firing neurons

**Analysis**:
- Gain control issue is likely about **normalization**, not loss function
- Correlation loss probably won't help with gain issue (different problem)
- But correlation loss might help with neuron favoritism (low-firing neurons)

**Most promising solution**:
**Inverse spike-rate weighting of Poisson LL** (PRIORITY):
- Weight each neuron's loss by inverse of its mean spike rate
- Ensures all neurons contribute equally to gradient
- Simple and principled
- May prevent high-firing neurons from dominating
- **Action**: Try this - could be a simple config change

**Alternative approaches** (lower priority):
1. **Augment loss with correlation term**:
   - Combine Poisson LL with Pearson correlation
   - Correlation is scale-invariant, treats all neurons equally
   - Question: How to weight the two terms? Adds hyperparameter tuning

2. **Multi-objective optimization**:
   - Optimize for both LL and correlation
   - Pareto frontier of solutions
   - More complex, harder to implement

**Constraint**: Each experiment = 8 hours. Inverse spike-rate weighting is simplest to try first.

---

## Summary: Prioritized Actions to Try

Given the **8-hour training constraint**, we need to be very selective. Here are the most promising experiments ranked by ease of implementation and likelihood of impact:

### Tier 1: Simple config changes (try first)
1. **Inverse spike-rate weighting in loss**
   - Simple config/code change
   - Addresses neuron favoritism directly
   - Low risk, potentially high reward

2. **Temporal kernel masking**
   - Mask first 2-3 lags in frontend filters
   - Forces minimum latency
   - Previously tried windowing (hurt fitting) - revisit masking
   - Moderate risk

### Tier 2: Moderate changes (try if Tier 1 shows promise)
3. **Temporal frequency regularization**
   - Penalize low frequencies in frontend filters
   - Tunable via regularization weight
   - Can start with weak regularization and increase

4. **Instance normalization experiments**
   - Try replacing some GRN layers with Instance Norm
   - Per-sample normalization may help with stimulus statistics changes
   - Systematic comparison needed

### Tier 3: Larger architectural changes (only if needed)
5. **Stimulus-conditioned normalization**
   - Most promising for gain control
   - But requires new architecture
   - Higher implementation cost

6. **Temporal context in normalization**
   - Running stats over temporal windows
   - Requires rethinking normalization layers
   - High implementation cost

---

## Immediate Next Steps

1. **Focus on evaluation stack** (as planned)
   - Define core metrics: Pearson correlation, normalized Poisson LL, per-neuron diagnostics
   - Build logging infrastructure
   - This unblocks everything else

2. **Then try Tier 1 experiments**
   - Inverse spike-rate weighting (easiest)
   - Temporal kernel masking (if temporal issue is critical)

3. **Use evaluation to guide further experiments**
   - With good metrics, can systematically compare approaches
   - Avoid wasting 8-hour runs on low-impact changes

---

## Key Constraints to Remember

- **8 hours per training run** - must be very selective
- **Limited compute budget** - can't try everything
- **Need good evaluation first** - can't compare without metrics
- **Incremental changes preferred** - minimize risk of breaking what works

