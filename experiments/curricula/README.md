# Native-240-Hz production curriculum

## Readout simplification comparison

`native240_no_phase_rank1_v1.yaml` and `native240_no_phase_rank2_v1.yaml`
repeat the complete 488/24/64-epoch curriculum with seed 201 and no separate
signed stage-1 readout. The rank applies to the ordinary 568-channel,
9-by-9 multiscale/behavioral readout. Each run independently fits the initial
Gaussian-head model from scratch; rank differs at the sparse-readout stage.
Stage 2 freezes everything except `readouts`, and stage 3 unfreezes the model.
The rank-two handoff preserves the Gaussian model's predictions exactly and
seeds a second spatial factor with zero feature weights so it can learn.

Both sparse stages retain proximal L1 on spatial weights (0.02) and feature
weights (0.1), spatial normalization over rank and space, and a shared learned
Gaussian envelope per neuron with width clamped to [0.75, 2.0] grid pixels.
L1 ramps over epochs 2--12 of stage 2 and is constant during stage 3. Stage 1
has Gaussian readouts and no proximal L1. Total parameter counts for the 2,790
units are 5,295,486 (rank one) and 7,106,196 (rank two).

Launch either run with its explicit spec:

```bash
conda run --no-capture-output -n yatesfv python training/run_three_stage_curriculum.py \
  --spec experiments/curricula/native240_no_phase_rank1_v1.yaml \
  --run-id NO_PHASE_R1_s201 --gpu 0 --execute
```

Use the rank-two spec and `NO_PHASE_R2_s201 --gpu 1` for the second run.
The 2026-09-10 comparison, logs, checkpoint manifests, queued Figure 3/4
workflows, and per-run manuscript-statistics exports live under
`outputs/no_phase_readout_comparison_20260910/`. The historical selected model
remains pinned until the new results have been evaluated.

## Previous model with a separate stage-1 branch

`native240_sparse_phase_v1.yaml` is the complete predictive training story:

1. Train the visual core, behavior pathway, and ordinary Gaussian
   readout from scratch.
2. Freeze the shared representation, exactly embed the old readout in a
   speckled sparse readout, add a zero-output signed stage-1 rank-four branch,
   and refine both heads with proximal L1.
3. Unfreeze the entire model and fine-tune with the core learning 20 times
   more slowly than the readouts.

Every stage uses Poisson likelihood on recorded spikes. Gratings are sampled
more frequently in stages 2 and 3, but no condition receives a different loss
and there is no teacher, calibration, or tuning objective.

The handoff contract is deliberate:

- Stage 1 hands off the checkpoint with the best validation bits/spike.
- Stage 2 hands off its final checkpoint because the scheduled proximal path,
  rather than a transient likelihood maximum, defines the sparse solution.
- Stage 3 again selects the best validation bits/spike after end-to-end
  fine-tuning.

`paper/model_selection/production_model.yaml` freezes the retained checkpoint,
its training lineage, and the rank-two capacity comparator. A later curriculum
result must pass the same validation, fixed-RSVP, grating, and interpretability
checks before replacing it.

Inspect the complete resolved commands without training:

```bash
conda run -n yatesfv python training/run_three_stage_curriculum.py \
  --run-id RUN_NAME
```

Run persistently in tmux:

```bash
tmux new-session -d -s vc_native240 \
  "cd /home/jake/repos/VisionCore && conda run --no-capture-output -n yatesfv \
  python training/run_three_stage_curriculum.py --run-id RUN_NAME \
  --gpu 0 --execute"
```

The run directory contains a machine-readable `manifest.json` with the git
state, hashes of every config and training entry point, resolved commands,
handoff policy, selected checkpoint, and checkpoint hash. Re-running the same
command skips completed stages and resumes an interrupted stage from its own
`last.ckpt`.

For an intentional partial run, use `--stop-after-stage`; to restart from a
specific handoff, use `--start-stage`. The runner validates that the preceding
checkpoint exists before it will execute a dependent stage.
