# Polar-V1 Integration Checklist

Use this checklist to track your progress integrating Polar-V1 Core into VisionCore.

---

## Phase 1: Core Module Implementation (Week 1)

### Day 1-2: Setup and Code Extraction
- [ ] Create branch `feature/polar-v1-integration`
- [ ] Create file `models/modules/polar_v1_core.py`
- [ ] Copy these classes from `scripts/devel_pyrConv.py`:
  - [ ] `MinimalGazeEncoder` (lines 440-541)
  - [ ] `QuadratureFilterBank2D` (lines 86-132)
  - [ ] `PolarDecompose` (lines 137-162)
  - [ ] `BehaviorEncoder` (lines 167-191)
  - [ ] `PolarDynamics` (lines 196-268)
  - [ ] `TemporalSummarizer` (lines 273-317)
  - [ ] `GaussianReadout` (lines 322-362)
  - [ ] `MultiLevelReadout` (lines 365-381)
  - [ ] `JEPAModule` (lines 386-434)
  - [ ] `PolarV1Core` (lines 758-841)
  - [ ] `PyramidAdapter` (lines 546-569)
  - [ ] `init_kxy` (lines 611-632)
- [ ] Add imports and docstrings

### Day 3-4: Wrapper Implementation
- [ ] Implement `PolarV1ConvNet` class
  - [ ] `__init__()` - Parse config, create components
  - [ ] `_lazy_init()` - Initialize pyramid and core on first forward
  - [ ] `_initialize_weights()` - Conservative initialization
  - [ ] `forward()` - Main forward pass
  - [ ] `_forward_standard()` - Standard mode (no JEPA)
  - [ ] `_forward_jepa()` - JEPA mode (context/target split)
  - [ ] `get_jepa_loss()` - Compute JEPA auxiliary loss
  - [ ] `get_output_channels()` - Return output channel count
- [ ] Add `requires_behavior = True` flag
- [ ] Add docstrings and type hints

### Day 5-6: Unit Tests
- [ ] Create `tests/test_polar_v1_core.py`
- [ ] Test `MinimalGazeEncoder`:
  - [ ] Forward pass with dummy gaze
  - [ ] Output shape `[B, T, D]`
  - [ ] Velocity/acceleration computation
- [ ] Test `PolarV1ConvNet`:
  - [ ] Forward pass with dummy data
  - [ ] Output shape `[B, C_out, H, W]`
  - [ ] Lazy initialization
  - [ ] JEPA mode (enable_jepa=True)
  - [ ] JEPA loss computation
- [ ] Test edge cases:
  - [ ] Short sequences (T < jepa_delta)
  - [ ] Missing behavior
  - [ ] Different batch sizes
- [ ] Run tests: `pytest tests/test_polar_v1_core.py -v`

### Day 7: Code Review and Cleanup
- [ ] Review code for clarity
- [ ] Add missing docstrings
- [ ] Check for TODOs
- [ ] Run linter: `ruff check models/modules/polar_v1_core.py`
- [ ] Commit: `git commit -m "Add Polar-V1 Core module"`

---

## Phase 2: Pipeline Integration (Week 2)

### Day 8: Factory Registration
- [ ] Open `models/factory.py`
- [ ] Find `create_convnet()` function (around line 100)
- [ ] Add after line 128:
  ```python
  if convnet_type.lower() == 'polar_v1':
      from .modules.polar_v1_core import PolarV1ConvNet
      core = PolarV1ConvNet(cfg)
      return core, core.get_output_channels()
  ```
- [ ] Test import: `python -c "from models.factory import create_convnet"`
- [ ] Commit: `git commit -m "Register Polar-V1 in factory"`

### Day 9: Model Forward Pass
- [ ] Open `models/modules/models.py`
- [ ] Find `MultiDatasetV1Model.forward()` (around line 418)
- [ ] Replace line 451 with:
  ```python
  # Process through shared convnet
  if hasattr(self.convnet, 'requires_behavior') and self.convnet.requires_behavior:
      x_conv = self.convnet(x, behavior=behavior)
  else:
      x_conv = self.convnet(x)
  ```
- [ ] Test: `python -c "from models.modules.models import MultiDatasetV1Model"`
- [ ] Commit: `git commit -m "Pass behavior to convnet in model forward"`

### Day 10: JEPA Auxiliary Loss
- [ ] Open `training/pl_modules/multidataset_model.py`
- [ ] Find `_compute_auxiliary_loss()` (around line 256)
- [ ] Add after line 276:
  ```python
  # JEPA loss for Polar-V1
  if hasattr(self.model, 'convnet') and hasattr(self.model.convnet, 'get_jepa_loss'):
      lambda_jepa = self.model_config.get('lambda_jepa', 0.5)
      jepa_loss = self.model.convnet.get_jepa_loss()
      
      if jepa_loss.item() > 0:
          if aux_loss is None:
              aux_loss = lambda_jepa * jepa_loss
          else:
              aux_loss = aux_loss + lambda_jepa * jepa_loss
          
          if self.global_rank == 0:
              self.log('jepa_loss', jepa_loss.item(), 
                      on_step=True, on_epoch=True, prog_bar=False, sync_dist=False)
  ```
- [ ] Commit: `git commit -m "Add JEPA auxiliary loss"`

### Day 11: Model Config
- [ ] Copy `experiments/model_configs/res_small_gru.yaml`
- [ ] Rename to `experiments/model_configs/polar_v1_core.yaml`
- [ ] Modify:
  ```yaml
  convnet:
    type: polar_v1
    params:
      n_pyramid_levels: 4
      n_pairs: 16
      dt: 0.004166667  # 1/240
      beh_dim: 128
      enable_jepa: false
  modulator:
    type: none
  recurrent:
    type: none
  ```
- [ ] Validate YAML: `python -c "import yaml; yaml.safe_load(open('experiments/model_configs/polar_v1_core.yaml'))"`
- [ ] Commit: `git commit -m "Add Polar-V1 model config"`

### Day 12: Dataset Config
- [ ] Copy `experiments/dataset_configs/multi_cones_120_backimage_all_eyepos.yaml`
- [ ] Rename to `experiments/dataset_configs/multi_polar_v1_eyepos.yaml`
- [ ] Modify transforms:
  ```yaml
  transforms:
    stim:
      source: stim
      ops:
        - pixelnorm: {}
      expose_as: stim
    
    eye_pos:
      source: eyepos
      ops: []  # No transforms!
      expose_as: behavior
  ```
- [ ] Validate YAML
- [ ] Commit: `git commit -m "Add Polar-V1 dataset config"`

---

## Phase 3: Testing and Validation (Week 3)

### Day 13-14: Integration Test Script
- [ ] Create `scripts/test_polar_v1_integration.py`
- [ ] Implement tests:
  - [ ] Load model from config
  - [ ] Load dataset from config
  - [ ] Create dummy batch
  - [ ] Forward pass
  - [ ] Backward pass
  - [ ] JEPA loss computation
- [ ] Run: `python scripts/test_polar_v1_integration.py`
- [ ] Commit: `git commit -m "Add integration test script"`

### Day 15: Dummy Data Test
- [ ] Create synthetic data:
  ```python
  stim = torch.randn(4, 1, 15, 51, 51)
  behavior = torch.randn(4, 15, 2)
  robs = torch.randint(0, 10, (4, 8))
  ```
- [ ] Test full pipeline:
  - [ ] Model creation
  - [ ] Forward pass
  - [ ] Loss computation
  - [ ] Backward pass
  - [ ] Optimizer step
- [ ] Verify no errors

### Day 16-17: Small Dataset Test
- [ ] Select small dataset (1 session, ~5k samples)
- [ ] Train for 10 epochs:
  ```bash
  python training/train_ddp_multidataset.py \
      --model_config experiments/model_configs/polar_v1_core.yaml \
      --dataset_configs_path experiments/dataset_configs/multi_polar_v1_eyepos.yaml \
      --max_epochs 10 \
      --steps_per_epoch 50 \
      --num_gpus 1
  ```
- [ ] Monitor:
  - [ ] Loss decreases
  - [ ] No NaNs
  - [ ] Memory usage < 4GB
  - [ ] Speed > 50 steps/sec
- [ ] Save checkpoint

### Day 18-19: Validation and BPS
- [ ] Compute validation BPS
- [ ] Compare to random baseline
- [ ] Visualize predictions vs actual
- [ ] Check for overfitting
- [ ] Document results

### Day 20-21: Hyperparameter Tuning
- [ ] Try different learning rates: [1e-4, 3e-4, 1e-3]
- [ ] Try different n_pairs: [8, 16, 24]
- [ ] Try different pyramid levels: [3, 4, 5]
- [ ] Try enabling JEPA
- [ ] Document best config

---

## Phase 4: Production Deployment (Week 4)

### Day 22-24: Full Dataset Training
- [ ] Train on full dataset (20 sessions):
  ```bash
  python training/train_ddp_multidataset.py \
      --model_config experiments/model_configs/polar_v1_core.yaml \
      --dataset_configs_path experiments/dataset_configs/multi_polar_v1_eyepos.yaml \
      --max_epochs 100 \
      --num_gpus 4
  ```
- [ ] Monitor training:
  - [ ] Loss curves
  - [ ] BPS metrics
  - [ ] JEPA loss (if enabled)
  - [ ] GPU utilization
- [ ] Save best checkpoint

### Day 25: Baseline Comparison
- [ ] Train ResNet baseline on same data
- [ ] Compare metrics:
  - [ ] Validation BPS
  - [ ] Training time
  - [ ] Memory usage
  - [ ] Parameter count
- [ ] Create comparison table
- [ ] Visualize results

### Day 26-27: Documentation
- [ ] Update README with Polar-V1 section
- [ ] Add usage examples
- [ ] Document hyperparameters
- [ ] Add troubleshooting guide
- [ ] Create visualization notebook
- [ ] Update API docs

### Day 28: Code Review and Merge
- [ ] Self-review all changes
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Check code coverage
- [ ] Create pull request
- [ ] Address review comments
- [ ] Merge to main: `git merge feature/polar-v1-integration`
- [ ] Tag release: `git tag v1.0.0-polar-v1`

---

## Post-Deployment Checklist

### Monitoring
- [ ] Set up training monitoring dashboard
- [ ] Track BPS over time
- [ ] Monitor memory usage
- [ ] Track training speed

### Documentation
- [ ] Add to model zoo
- [ ] Create tutorial notebook
- [ ] Record demo video
- [ ] Update changelog

### Future Work
- [ ] Experiment with different pyramid types
- [ ] Try learned spatial frequencies
- [ ] Explore multi-stream inputs
- [ ] Benchmark on other datasets

---

## Success Criteria

### Must Have ✅
- [ ] Model trains without errors
- [ ] BPS within 10% of ResNet baseline
- [ ] Memory usage < 4GB per GPU
- [ ] All tests pass

### Should Have 🎯
- [ ] BPS comparable to ResNet
- [ ] Training speed within 20% of baseline
- [ ] JEPA loss decreases over training
- [ ] Features are interpretable

### Nice to Have 🌟
- [ ] BPS better than ResNet
- [ ] Faster convergence
- [ ] Better generalization
- [ ] Visualizations in paper

---

## Troubleshooting

### Common Issues

**Issue**: `ImportError: cannot import name 'PolarV1ConvNet'`
- [ ] Check file exists: `ls models/modules/polar_v1_core.py`
- [ ] Check import in factory: `grep -n "PolarV1ConvNet" models/factory.py`

**Issue**: `ValueError: PolarV1ConvNet requires behavior`
- [ ] Check model forward passes behavior
- [ ] Verify dataset config has raw gaze

**Issue**: `RuntimeError: CUDA out of memory`
- [ ] Reduce batch size
- [ ] Reduce pyramid levels (4 → 3)
- [ ] Disable JEPA
- [ ] Use gradient checkpointing

**Issue**: JEPA loss is 0
- [ ] Check `enable_jepa: true` in config
- [ ] Verify `T >= jepa_delta + 5`
- [ ] Ensure model is in training mode

---

## Notes

- Keep this checklist updated as you progress
- Mark items complete with `[x]`
- Add notes for any deviations from plan
- Document any issues encountered

---

**Started**: ___________
**Completed**: ___________
**Total Time**: ___________

