# M77 production workflow

This is the minimal path for training and auditing the selected native-240-Hz
model, reproducing Figures 3 and 4, and rendering the Twin-versus-M77
prediction and subspace supplements. Generated caches, checkpoints, plots, and
PDFs stay under `outputs/` or `/mnt/ssd` and are not committed.

## Frozen model

The selected checkpoint and its hashes are recorded in `M77_MODEL_CARD.md`.
All commands below assume the repository root and the `yatesfv` Conda
environment.

```bash
M77_CKPT=/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/dekel240/D240M77c_dekel_native240_freqmasked_floor0p5_gnlrnalpha0p1_presplit_s201/analysis_candidates/epoch=279-val_bps_overall=0.5912.ckpt
```

## Train M77

This is the exact command recorded in the completed run manifest. Checkpoint
selection is external-generalization-first; do not silently replace epoch 279
with the last epoch.

```bash
conda run -n yatesfv python training/train_multidataset.py \
  --model_config experiments/model_configs/dekel_capacitymatched_freqmasked_readoutfloor0p5_groupnorm_lrnalpha0p1_mlp_behavior.yaml \
  --dataset_configs_path paper/model_selection/configs/multi_240_long_split3_dekel35.yaml \
  --max_datasets 30 \
  --batch_size 128 \
  --learning_rate 0.0005 \
  --core_lr_scale 1.0 \
  --weight_decay 1e-5 \
  --lr_scheduler cosine_warmup \
  --warmup_epochs 2 \
  --max_epochs 488 \
  --accumulate_grad_batches 1 \
  --steps_per_epoch 512 \
  --gradient_clip_val 10.0 \
  --precision bf16-mixed \
  --dset_dtype uint8 \
  --num_workers 16 \
  --limit_val_batches 0.1 \
  --check_val_every_n_epoch 4 \
  --seed 201 \
  --gpu 0 \
  --checkpoint_dir /mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/dekel240 \
  --project_name model_selection \
  --experiment_name D240M77c_dekel_native240_freqmasked_floor0p5_gnlrnalpha0p1_presplit_s201 \
  --no-early_stopping \
  --homogeneous_batches \
  --enable_logging \
  --fast_log_interval 5 \
  --slow_log_interval 10000 \
  --log_dataset_idx 9
```

## Four-filter spatial-smoothness ablation

The streamlined follow-up changes only the first-layer filter count and its
spatial Laplacian coefficient. Both arms retain native 240-Hz spike
supervision, the 42-variable behavior path, frequency masks, GroupNorm then
LRN, the 84/84/84 spatial core, Gaussian readouts, and M77's training schedule.
They minimize Poisson NLL plus AdamW weight decay and the configured positive
regularization penalties; neither arm loads a checkpoint or distillation target.

| label | `RUN` | `CONFIG` | first-layer spatial Laplacian | `GPU` |
|---|---|---|---:|---:|
| M79 | `D240M79c_dekel_native240_4temporal_spatialsmooth5e4_gnlrnalpha0p1_s201` | `dekel_m77_4temporal_spatialsmooth_low_mlp_behavior.yaml` | `5e-4` | 0 |
| M80 | `D240M80c_dekel_native240_4temporal_spatialsmooth1e3_gnlrnalpha0p1_s201` | `dekel_m77_4temporal_spatialsmooth_high_mlp_behavior.yaml` | `1e-3` | 1 |

Set `CONFIG`, `RUN`, and `GPU` from one row, then use the shared command:

```bash
conda run --no-capture-output -n yatesfv python training/train_multidataset.py \
  --model_config "experiments/model_configs/$CONFIG" \
  --dataset_configs_path paper/model_selection/configs/multi_240_long_split3_dekel35.yaml \
  --max_datasets 30 --batch_size 128 \
  --learning_rate 0.0005 --core_lr_scale 1.0 --weight_decay 1e-5 \
  --lr_scheduler cosine_warmup --warmup_epochs 2 --max_epochs 488 \
  --accumulate_grad_batches 1 --steps_per_epoch 512 \
  --gradient_clip_val 10.0 --precision bf16-mixed --dset_dtype uint8 \
  --num_workers 16 --limit_val_batches 0.1 --check_val_every_n_epoch 4 \
  --seed 201 --gpu "$GPU" \
  --checkpoint_dir /mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/dekel240 \
  --project_name model_selection --experiment_name "$RUN" \
  --no-early_stopping --homogeneous_batches \
  --enable_logging --fast_log_interval 5 --slow_log_interval 10000 \
  --log_dataset_idx 9
```

## FixRSVP and Figure 3

The evaluator scores native 240-Hz predictions and exact adjacent-bin sums on
the canonical 120-Hz Figure-3 frame. It rejects session, neuron-order,
observation, support, and reliability mismatches.

```bash
conda run -n yatesfv python paper/model_selection/evaluate_true240_fixrsvp.py \
  "$M77_CKPT" \
  --gpu 0 \
  --candidate-label M77 \
  --figure3-cache outputs/cache/fig3_digitaltwin.pkl \
  --out outputs/dekel240_paper/m77_epoch279/figure3_predictions/full_true240.csv \
  --trace-cache outputs/dekel240_paper/m77_epoch279/figure3_predictions/full_true240_trace.pkl \
  --examples-out outputs/dekel240_paper/m77_epoch279/figure3_predictions/full_true240_examples.png
```

To make M77 the Figure-3 model without overwriting the canonical Twin cache,
point the existing production loaders at a separate cache tree:

```bash
FIG3_TWIN_CHECKPOINT="$M77_CKPT" \
FIG3_DATASET_CONFIGS=paper/model_selection/configs/multi_240_long_split3_dekel35.yaml \
FIG3_REFERENCE_CACHE=outputs/cache/fig3_digitaltwin.pkl \
FIG3_CACHE_PATH=outputs/dekel240_paper/m77_epoch279/production_figure3/cache/fig3_digitaltwin.pkl \
FIG3_ABLATION_CACHE_PATH=outputs/dekel240_paper/m77_epoch279/production_figure3/cache/fig3_ablation_inference.pkl \
FIG3_GPU=0 \
conda run -n yatesfv python paper/model_selection/regen_fig3_caches.py
```

Use the same environment variables when running
`paper/fig3/generate_figure3.py`. The schema-v7 audit is mandatory before a
paper render:

```bash
conda run -n yatesfv python paper/fig3/audit_ablation_cache.py \
  outputs/dekel240_paper/m77_epoch279/production_figure3/cache/fig3_ablation_inference.pkl \
  outputs/dekel240_paper/m77_epoch279/production_figure3/cache/fig3_digitaltwin.pkl \
  --out outputs/dekel240_paper/m77_epoch279/production_figure3/cache/ablation_audit.json
```

## Per-session prediction atlases

This writes one multipage PDF for every session at each display rate. At 240
Hz, data and M77 are native; the Twin curve is explicitly a repeated 120-Hz
reference and is never described as a native prediction.

```bash
conda run -n yatesfv python \
  paper/model_selection/render_m77_fixrsvp_session_atlases.py
```

Outputs are under
`outputs/dekel240_paper/m77_epoch279/fixrsvp_session_prediction_atlases/{120hz,240hz}/`.

## Figure 4 and the two-stage mechanism

First emit the native-240 production plan with the selected checkpoint and
dataset config. Run the two GPU shard commands recorded by the plan, keeping
their output under
`outputs/dekel240_paper/m77_epoch279/figure4_real_trace_production_corrected/shards/`.

```bash
FIG4_TWIN_CHECKPOINT="$M77_CKPT" \
FIG4_DATASET_CONFIGS=paper/model_selection/configs/multi_240_long_split3_dekel35.yaml \
conda run -n yatesfv python paper/fig4/upstream/run_real_trace_matrix.py \
  --profile production240
```

After both shards finish, merge, score the stabilized baseline, and build the
correct trajectory-phase and renderer-faithful analyses:

```bash
bash paper/fig4/spatiotemporal_tuning/run_m77_merge_baseline_after_shards.sh
bash paper/fig4/spatiotemporal_tuning/run_m77_corrected_full_after_merge.sh
```

The production interpretation has two separately tested claims:

1. Complete eye trajectories move image power into each unit's measured joint
   SF-by-TF passband and increase predicted activity. The causal conclusion
   comes from exact rendered replay; the trajectory-phase spectrum is a
   validated second-order cross-check.
2. Later spatial nonlinearities transform that drive into SSI sharpening.
   Passband power alone is not claimed to explain the nonlinear sharpening.

Only local joint quadratic peaks that pass the explicit trust gates are used
as resolved SF/TF preferences. Boundary maxima and unstable fits remain
censored.

## Paired Twin-versus-M77 subspaces

Reuse the Twin analysis's frozen unit selection so both models are evaluated
on the same RR100 units:

```bash
conda run -n yatesfv python paper/model_selection/run_m77_response_subspace.py \
  --selection-csv outputs/figures/fig4/nonlinear_phase_causal_v1/response_subspace_pilot/unit_selection.csv \
  --output-dir outputs/dekel240_paper/m77_epoch279/response_subspace_matched \
  --device cuda:0 \
  --n-train 1024 \
  --n-test 256 \
  --n-gradient 128 \
  --ranks 1,2,4,8,16 \
  --overwrite-bank

conda run -n yatesfv python \
  paper/model_selection/render_twin_m77_subspace_supplement.py
```

The supplement compares response fidelity and cumulative Jacobian energy
within each model and shows native physical basis filters. It deliberately
does not compute principal angles between the incompatible native feature
grids.
