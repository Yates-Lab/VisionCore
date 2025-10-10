# Polar-V1 Implementation Quick Checklist

## Week 1: ConvNet + Modulator

### Days 1-3: ConvNet
- [ ] Create `models/modules/polar_convnet.py`
- [ ] Copy: `PyramidAdapter`, `QuadratureFilterBank2D`, `PolarDecompose`
- [ ] Implement `PolarConvNet` wrapper
- [ ] Create `tests/test_polar_convnet.py`
- [ ] Test: forward pass, output shapes, amplitude/phase properties
- [ ] Register in `models/factory.py`
- [ ] Run: `pytest tests/test_polar_convnet.py -v`
- [ ] Commit: "Add PolarConvNet module"

### Days 4-6: Modulator
- [ ] Create `models/modules/polar_modulator.py`
- [ ] Copy: `MinimalGazeEncoder`, `BehaviorEncoder`
- [ ] Implement `PolarModulator` wrapper
- [ ] Create `tests/test_polar_modulator.py`
- [ ] Test: behavior encoding, params shapes, features pass-through
- [ ] Register in `models/factory.py`
- [ ] Run: `pytest tests/test_polar_modulator.py -v`
- [ ] Commit: "Add PolarModulator module"

---

## Week 2: Recurrent + Readout

### Days 7-9: Recurrent
- [ ] Create `models/modules/polar_recurrent.py`
- [ ] Copy: `PolarDynamics`, `TemporalSummarizer`, `init_kxy`
- [ ] Implement `PolarRecurrent` wrapper
- [ ] Add `set_modulator()` method
- [ ] Create `tests/test_polar_recurrent.py`
- [ ] Test: dynamics, summarization, output shapes
- [ ] Register in `models/factory.py`
- [ ] Run: `pytest tests/test_polar_recurrent.py -v`
- [ ] Commit: "Add PolarRecurrent module"

### Days 10-11: Readout
- [ ] Create `models/modules/polar_readout.py`
- [ ] Copy: `GaussianReadout`, `MultiLevelReadout`
- [ ] Implement `PolarMultiLevelReadout` wrapper
- [ ] Create `tests/test_polar_readout.py`
- [ ] Test: multi-level pooling, output shape
- [ ] Register in `models/factory.py`
- [ ] Run: `pytest tests/test_polar_readout.py -v`
- [ ] Commit: "Add PolarReadout module"

---

## Week 3: Integration + Testing

### Days 12-14: Integration
- [ ] Modify `models/modules/models.py`:
  - [ ] Handle tuple outputs from convnet
  - [ ] Call `set_modulator()` for recurrent
  - [ ] Handle list outputs from recurrent
- [ ] Create `experiments/model_configs/polar_v1.yaml`
- [ ] Create `experiments/dataset_configs/polar_v1_test.yaml`
- [ ] Create `tests/test_polar_integration.py`
- [ ] Test: full pipeline forward pass
- [ ] Create `scripts/test_polar_training.py`
- [ ] Test: 10 training steps on dummy data
- [ ] Run: `python scripts/test_polar_training.py`
- [ ] Commit: "Add full pipeline integration"

### Days 15-17: Validation
- [ ] Train on small dataset (1 session, 5 epochs)
- [ ] Monitor: loss, memory, speed
- [ ] Create `scripts/validate_polar_outputs.py`
- [ ] Validate: amplitudes, phases, behavior params
- [ ] Train ResNet baseline on same data
- [ ] Compare: BPS, time, memory
- [ ] Document results
- [ ] Commit: "Validate Polar-V1 on real data"

---

## Quick Test Commands

```bash
# Unit tests
pytest tests/test_polar_convnet.py -v
pytest tests/test_polar_modulator.py -v
pytest tests/test_polar_recurrent.py -v
pytest tests/test_polar_readout.py -v

# Integration test
pytest tests/test_polar_integration.py -v

# Training test
python scripts/test_polar_training.py

# Small dataset
python training/train_ddp_multidataset.py \
    --model_config experiments/model_configs/polar_v1.yaml \
    --dataset_configs_path experiments/dataset_configs/polar_v1_test.yaml \
    --max_epochs 5 --steps_per_epoch 20 --num_gpus 1
```

---

## Files to Create (7 new files)

1. `models/modules/polar_convnet.py` (~300 lines)
2. `models/modules/polar_modulator.py` (~200 lines)
3. `models/modules/polar_recurrent.py` (~200 lines)
4. `models/modules/polar_readout.py` (~150 lines)
5. `experiments/model_configs/polar_v1.yaml`
6. `experiments/dataset_configs/polar_v1_test.yaml`
7. `scripts/test_polar_training.py`

## Files to Modify (2 files)

1. `models/factory.py` (~40 lines added)
2. `models/modules/models.py` (~20 lines modified)

---

## Success Criteria

### Must Have ✅
- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] Training runs without errors
- [ ] Loss decreases on dummy data
- [ ] Trains on real data

### Should Have 🎯
- [ ] BPS > random baseline
- [ ] Memory < 4GB per GPU
- [ ] Speed > 30 steps/sec
- [ ] No NaNs or Infs

### Nice to Have 🌟
- [ ] BPS competitive with ResNet
- [ ] Visualizations work
- [ ] JEPA integration

---

## Troubleshooting

**NaN in loss**
→ Check `PolarDecompose` eps values

**Shape mismatch**
→ Print shapes at each stage

**CUDA OOM**
→ Reduce batch size or pyramid levels

**Modulator not set**
→ Check `set_modulator()` is called

**Slow training**
→ Profile with `torch.profiler`

---

## Daily Progress Tracker

| Day | Task | Status | Notes |
|-----|------|--------|-------|
| 1 | ConvNet setup | ⬜ | |
| 2 | ConvNet tests | ⬜ | |
| 3 | ConvNet factory | ⬜ | |
| 4 | Modulator setup | ⬜ | |
| 5 | Modulator tests | ⬜ | |
| 6 | Modulator factory | ⬜ | |
| 7 | Recurrent setup | ⬜ | |
| 8 | Recurrent tests | ⬜ | |
| 9 | Recurrent factory | ⬜ | |
| 10 | Readout setup | ⬜ | |
| 11 | Readout tests | ⬜ | |
| 12 | Model integration | ⬜ | |
| 13 | Configs + tests | ⬜ | |
| 14 | Training test | ⬜ | |
| 15 | Small dataset | ⬜ | |
| 16 | Validation | ⬜ | |
| 17 | Baseline compare | ⬜ | |

---

**Started**: ___________
**Completed**: ___________
**Total Time**: ___________

