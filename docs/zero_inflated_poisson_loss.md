# Zero-Inflated Poisson (ZIP) Loss

## Overview

The Zero-Inflated Poisson (ZIP) loss is now available for training neural encoding models. This loss function is useful when your neural data has excess zeros beyond what a standard Poisson distribution would predict.

## What is Zero-Inflated Poisson?

The ZIP distribution is a mixture model that combines:
1. A point mass at zero (with probability π)
2. A Poisson distribution (with probability 1-π and rate λ)

This is particularly useful for modeling neural responses that may have:
- Periods of complete silence (true zeros)
- Poisson-distributed spike counts when active

## Mathematical Formulation

The probability mass function is:

```
P(Y = y | λ, π) = {
    π + (1-π) * exp(-λ)                           if y = 0
    (1-π) * [λ^y * exp(-λ) / y!]                  if y > 0
}
```

The negative log-likelihood (what we minimize) is:

```
NLL = {
    -log(π + (1-π) * exp(-λ))                                    if y = 0
    -log(1-π) - y*log(λ) + λ + log(y!)                          if y > 0
}
```

## Usage

### 1. Basic Configuration

To use ZIP loss, add `loss_type: zip` to your model configuration YAML file:

```yaml
# Model configuration
model_type: v1multi

# ... other model settings ...

# Specify ZIP loss
loss_type: zip  # or 'zero_inflated_poisson'

# Output activation should be 'none' for log-space predictions
output_activation: none
```

### 2. Model Output Requirements

**Important**: When using ZIP loss, your model needs to output **two** parameters:
- `lambda` (λ): The Poisson rate parameter
- `pi` (π): The zero-inflation probability

Currently, the implementation supports two modes:

#### Mode 1: Single Output (Fallback)
If your model only outputs `rhat` (lambda), the loss will fall back to standard Poisson NLL. This maintains backward compatibility.

#### Mode 2: Dual Output (Full ZIP)
To use full ZIP functionality, you need to modify your model to output both parameters. This requires:

1. A custom readout layer that outputs both λ and π
2. Modifications to the forward pass to return both values

**Example readout modification** (advanced users):

```python
class ZIPReadout(BaseFactorizedReadout):
    """Readout that outputs both lambda and pi for ZIP loss."""
    
    def __init__(self, in_channels, n_units, bias=True):
        super().__init__(in_channels, n_units, bias)
        # Additional head for pi parameter
        self.pi_head = nn.Conv2d(in_channels, n_units, kernel_size=1, bias=False)
        
    def forward(self, x):
        # ... spatial pooling for lambda ...
        lam = # ... your lambda computation ...
        
        # Compute pi (zero-inflation probability)
        pi_logits = self.pi_head(x)
        # ... spatial pooling ...
        pi = torch.sigmoid(pi_logits)  # Ensure pi in [0, 1]
        
        return lam, pi  # Return tuple
```

### 3. Example Configuration File

Here's a complete example based on `res_small_gru.yaml`:

```yaml
# V1 multidataset model with ZIP loss
model_type: v1multi

# Model dimensions
sampling_rate: 240
initial_input_channels: 1

# ... adapter, frontend, convnet, modulator, recurrent configs ...

# Readout configuration
readout:
  type: gaussian  # Standard readout (outputs lambda only)
  params:
    n_units: 8
    bias: true
    initial_std: 5.0
    initial_mean_scale: 0.1

# Use identity activation for log-space predictions
output_activation: none

# Specify ZIP loss
loss_type: zip

# ... regularization ...
```

## Implementation Details

### Loss Function Classes

Three new classes have been added to `models/losses/poisson.py`:

1. **`ZeroInflatedPoissonNLLLoss`**: Core ZIP loss implementation
   - Handles both log-space and linear-space inputs
   - Supports tuple input `(lambda, pi)` or single tensor
   - Configurable epsilon for numerical stability

2. **`MaskedZIPNLLLoss`**: Masked version for use with data filters
   - Wraps `ZeroInflatedPoissonNLLLoss`
   - Applies masking like `MaskedLoss`
   - Handles missing `pi` gracefully (falls back to Poisson)

### Training Module Integration

The `MultiDatasetModel` in `training/pl_modules/multidataset_model.py` now:
- Checks for `loss_type` in model config
- Initializes appropriate loss function
- Prints loss type during initialization

### Backward Compatibility

The implementation is fully backward compatible:
- Default `loss_type` is `'poisson'` (standard Poisson NLL)
- If `loss_type: zip` is specified but model doesn't output `pi`, falls back to Poisson
- Existing models and configs work without modification

## Current Limitations

1. **Model Architecture**: The current readout modules (`DynamicGaussianReadout`, `DynamicGaussianReadoutEI`, etc.) only output a single tensor (lambda). To use full ZIP functionality, you need to:
   - Create a custom readout that outputs both λ and π
   - Modify the model's forward pass to handle tuple outputs
   - Update the batch dictionary to include `'pi'` key

2. **BPS Metrics**: The bits-per-spike calculation currently uses standard Poisson likelihood. For ZIP models, this should be updated to use ZIP likelihood.

## Future Enhancements

Potential improvements for full ZIP support:

1. **ZIP-specific readout modules**: Pre-built readout layers that output both parameters
2. **Automatic model modification**: Config option to automatically add pi head to existing readouts
3. **ZIP BPS calculation**: Update metrics to use ZIP likelihood
4. **Visualization tools**: Plot learned zero-inflation probabilities per neuron

## Testing

To test the ZIP loss implementation:

```python
import torch
from models.losses import MaskedZIPNLLLoss

# Create loss function
loss_fn = MaskedZIPNLLLoss(log_input=False)

# Create sample batch
batch = {
    'rhat': torch.rand(32, 10) * 5,      # lambda (N, n_units)
    'pi': torch.rand(32, 10) * 0.3,       # pi in [0, 1] (N, n_units)
    'robs': torch.poisson(torch.rand(32, 10) * 5),  # observed counts
    'dfs': torch.ones(32, 10)             # mask
}

# Compute loss
loss = loss_fn(batch)
print(f"ZIP Loss: {loss.item():.4f}")
```

## References

- Lambert, D. (1992). "Zero-inflated Poisson regression, with an application to defects in manufacturing". Technometrics.
- For neural data: Mullahy, J. (1986). "Specification and testing of some modified count data models". Journal of Econometrics.

## Questions?

For questions or issues with ZIP loss implementation, please open an issue or contact the development team.

