# Native-240-Hz production curriculum

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
