# cursor_ln_two_stage

LN and two-stage (steerable pyramid + readout) models for **cell 14** on the same data pipeline: VisionCore Allen 2022-04-13, `multi_basic_240_gaborium_20lags`, centered 30×30 patch on RF (STE-based), `robs`/`dfs` sliced to `[14]`.

## Data

- **data_utils.py**: `load_and_center_for_cell(cell_idx=14)` loads train/val via `get_dataset_from_config`, computes RF center from STE for that cell, returns centered stim (30×30), and STA for that cell (for LN init).

## LN model

- **models_ln.py**: `LinearNonLinearModel` (one spatiotemporal kernel + scale/bias, softplus), `AffineSoftplus` for initial scale/bias fit.
- **train_ln.py**: STA init (unit-norm), freeze kernel and fit scale/bias with LBFGS on STA generator, then full LBFGS; best model by **val BPS** is restored.
- Run: `uv run python tejas/model/cursor_ln_two_stage/train_ln.py` (from VisionCore root).
- Typical best val BPS: ~0.43.

## Two-stage model

- **models_two_stage.py**: `TwoStage` (plenoptic steerable pyramid, height=2 for 30×30, order=5; pos/neg ReLU features, linear readout, beta + alpha_pos*softplus(z) + alpha_neg*softplus(-z)), `sparsity_penalty`.
- **train_two_stage.py**: Same data as LN (centered, cell 14). Batched train/val (batch size 64), ScheduleFree RAdam, sparsity penalty 1e-4; best model by **val BPS** restored. Validation is batched to avoid OOM.
- Run: `uv run python tejas/model/cursor_ln_two_stage/train_two_stage.py` (from VisionCore root).
- Epochs: 25 by default; change `EPOCHS` in the script for longer runs (~22 s/epoch).

## Comparison

- **LN**: STA-initialized linear–nonlinear, full-batch LBFGS.
- **Two-stage**: Fixed pyramid features, learned readout; same loss (MaskedPoissonNLL) and BPS aggregation.
