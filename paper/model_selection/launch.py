"""Stage 0 run family: declare the arms, launch them, stamp the protocol.

Every arm is a short run at a fixed *sample* budget, not a fixed wall-clock
budget, so arms remain comparable across widths and batch sizes. Selection is
on validation BPS under the three-way split; the test split is not touched in
Stage 0.

The paper model's cosine horizon was set to 9999 epochs and it stopped at 374,
so its learning rate never annealed. Here `max_epochs` is derived from the
sample budget and passed as the cosine horizon, so every arm anneals.

    uv run python paper/model_selection/launch.py --list
    uv run python paper/model_selection/launch.py E1a --dry-run
    uv run python paper/model_selection/launch.py E1a --gpu 0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
VISIONCORE_ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from protocol import PROTOCOL_HASH, protocol_dict  # noqa: E402

CONFIGS = HERE / "configs"
CKPT_ROOT = Path("/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/model_selection")

# One optimizer step sees batch_size * accumulate samples; one epoch is
# steps_per_epoch batches. 512 * 256 = 131,072 samples per epoch, so the paper
# model's 374 epochs was ~49M samples. A Stage 0 short run is ~8M.
SAMPLE_BUDGET = 8_000_000
STEPS_PER_EPOCH = 512
EFFECTIVE_BATCH = 1024
LIMIT_VAL_BATCHES = 0.1
CHECK_VAL_EVERY = 4

JOINT = "multi_120_long_split3.yaml"
ALLEN = "multi_120_long_split3_allen.yaml"
LOGAN = "multi_120_long_split3_logan.yaml"

MODEL_BASE = "learned_resnet_concat_convgru_gaussian"


BASE_SEED = 101


def _defaults(**over):
    """Stage 0 (frozen) defaults: cross-session batching, adapter on, width 1.0."""
    d = dict(
        config=JOINT,
        model_config=f"experiments/model_configs/{MODEL_BASE}.yaml",
        width=1.0,
        batch_size=256,
        lr=1e-3,
        core_lr_scale=1.0,
        wd=1e-5,
        homogeneous=False,
        adapter=True,
        samples=SAMPLE_BUDGET,
        seed=BASE_SEED,
    )
    d.update(over)
    return d


# ---------------------------------------------------------------------------
# Stage 0b defaults
# ---------------------------------------------------------------------------
# Two decisions carried over from Stage 0, both taken on evidence:
#
# * **Homogeneous batching.** E1b lost 0.023 test BPS to E1a but gained 0.050
#   held-out fixrsvp CC_norm (16 of 17 sessions), the metric figures 3 and 4
#   rest on, and it is 1.3x faster per step at small widths. The in-domain BPS
#   loss may itself be a tuning artifact -- every Stage 0 lr and batch arm was
#   run under cross-session batching -- which is what F2/F3 below test.
# * **No adapter.** The `AffineAdapter` existed to reconcile datasets recorded
#   at different spatial scales and is unnecessary now. Read from E1a's
#   weights, it had largely learned its way back to the identity resample
#   (mean scale 0.99 from an init of 0.693). See `gen_configs.strip_adapter`.
#
# Width 0.5 for screening: measured at 1.21 ms/sample against width 1.0's
# 2.00, so an 8M-sample arm is ~2.7 h rather than ~5.5 h. Settings tuned here
# must be confirmed at the width actually shipped -- the ladder spans 152x and
# transfer is not guaranteed, which is what arm C1 exists for.
STAGE0B_SEED = 201


def _b_defaults(**over):
    d = _defaults(
        width=0.5,
        homogeneous=True,
        adapter=False,
        seed=STAGE0B_SEED,
    )
    d.update(over)
    return d


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------
# E5 is deliberately NOT "balancing off/on". `data_census.py` shows Allen holds
# 51.8% of training samples to Logan's 48.2%, Logan contributes more sessions,
# and the loss weights sessions rather than units -- so the imbalance the
# original plan set out to fix does not exist, and per unit Logan is already
# weighted ~2.5x more than Allen. `subject_gap.py` further shows the held-out
# gap is not distinguishable from zero once sessions are the resampling unit,
# and largely explained by unit reliability. What remains untested is whether
# joint training *interferes*, which single-subject controls measure directly.
FROZEN_RUNS = {
    # E1 -- batch composition. This is also the subject-weighting manipulation:
    # homogeneous batching draws a session with p proportional to its size,
    # which is the only setting under which sample-count imbalance matters.
    # E1a is also the baseline configuration every other arm is differenced
    # against: `_defaults` already carries homogeneous=False, effective batch
    # 1024 and lr 1e-3, so E1a, E2b and E3b are the *same* configuration. They
    # are kept, at distinct seeds, as replicates R1/R2/R3 of the baseline --
    # without a run-to-run spread there is no scale on which to read any other
    # arm's delta in validation BPS. See --seed in train_multidataset.py for
    # what a seed does and does not vary: weight init and GPU nondeterminism,
    # not the trials or their order.
    "E1a": _defaults(homogeneous=False, seed=101,
                     note="cross-session batches (current default); baseline R1"),
    "E1b": _defaults(homogeneous=True, note="homogeneous (one session per step)"),

    # E2 -- effective batch size, via accumulation.
    "E2a": _defaults(effective_batch=256, note="effective batch 256"),
    "E2b": _defaults(effective_batch=1024, seed=102,
                     note="effective batch 1024 (= baseline; replicate R2)"),
    "E2c": _defaults(effective_batch=4096, note="effective batch 4096"),

    # E3 -- learning rate and whether the core wants a different one.
    "E3a": _defaults(lr=3e-4, note="lr 3e-4"),
    "E3b": _defaults(lr=1e-3, seed=103,
                     note="lr 1e-3 (paper model) (= baseline; replicate R3)"),
    "E3c": _defaults(lr=3e-3, note="lr 3e-3"),
    "E3d": _defaults(lr=1e-3, core_lr_scale=0.3, note="core lr scaled 0.3"),
    "E3e": _defaults(lr=1e-3, core_lr_scale=3.0, note="core lr scaled 3.0"),

    # E4 -- compute budget. Same settings, different sample budgets, to find
    # where validation BPS stops moving under a horizon that actually anneals.
    "E4a": _defaults(samples=4_000_000, note="4M samples"),
    "E4b": _defaults(samples=16_000_000, note="16M samples"),

    # E5 -- interference control. Compare each single-subject model against the
    # joint model on that subject's validation sessions. Joint ~= single means
    # no interference and the gap is intrinsic; joint < single means genuine
    # negative transfer, and only then is a weighting change warranted.
    "E5a": _defaults(config=ALLEN, note="Allen-only"),
    "E5b": _defaults(config=LOGAN, note="Logan-only"),

    # Width-2.0 confirmations of the E1/E3 winners; a setting tuned at one
    # capacity need not transfer across a 152x ladder.
    "C1": _defaults(width=2.0, note="width 2.0, default settings"),
}


# ---------------------------------------------------------------------------
# Stage 0b -- the live family
# ---------------------------------------------------------------------------
# Everything here is homogeneous, adapter-off, width 0.5. The Stage 0 arms above
# are kept as FROZEN_RUNS so that E1a's and E1b's manifests stay resolvable and
# their numbers stay interpretable, but they describe an architecture and a
# batching mode we are no longer training: **they are not a baseline for
# anything below.** The floor has to be rebuilt here, which is what F1a/F1b/F1c
# are for. Until at least two of them finish, every F2/F3 delta is unreadable
# and `stability.py` will say so rather than print a verdict.
RUNS = {
    # F0 -- the adapter control, added 2026-08-05 after F1a evaluated.
    #
    # Stage 0b adopted adapter-off without a dedicated arm, so F1a differs from
    # E1b in *two* things (width 1.0 -> 0.5 and adapter on -> off) and neither
    # is isolated. F1a came in at fixrsvp CC_norm 0.465 against E1b's 0.639 --
    # a 27% drop, while test BPS fell only 8.7%. A loss concentrated in
    # out-of-domain generalisation rather than in-domain fit is what removing a
    # fixed sigma=1 pre-blur would look like, so it is worth one run to find
    # out: -0.174 CC_norm dwarfs the +0.050 that motivated switching batching
    # in the first place.
    #
    # Identical to F1a in every respect including seed 201; only the adapter
    # differs. Not a replicate of anything -- `model_config` differs, so
    # `config_signature` keeps it out of the F1 group.
    "F0": _b_defaults(adapter=True,
                      note="adapter ON control (w0.5, homog) -- vs F1a"),

    # F1 -- baseline replicates. Three seeds at one configuration, to measure
    # what two identical runs do. Note the caveat that carried over from Stage
    # 0: `split_inds_by_trial*` re-seeds globally, so replicates differ in
    # weight init and GPU nondeterminism but not in data order, and the floor
    # they give is an underestimate of true run-to-run spread.
    "F1a": _b_defaults(seed=201, note="baseline (homog, no adapter, w0.5); R1"),
    "F1b": _b_defaults(seed=202, note="baseline replicate R2"),
    "F1c": _b_defaults(seed=203, note="baseline replicate R3"),

    # F2 -- effective batch, the knob that acts directly on the mechanism.
    # Under homogeneous batching each micro-batch is one session, so
    # accumulation sets how many sessions an optimizer step averages: the
    # baseline's 1024 is 4 sessions, against cross-session's ~30. If E1b's BPS
    # deficit is gradient variance, this is the arm that should close it, and
    # 4096 (16 sessions) is the one to watch.
    "F2a": _b_defaults(effective_batch=256, note="eff batch 256 (1 session/step)"),
    "F2c": _b_defaults(effective_batch=4096, note="eff batch 4096 (16 sessions/step)"),

    # F3 -- learning rate. Every Stage 0 lr arm ran under cross-session
    # batching, so 1e-3 is tuned for a gradient this family no longer has.
    "F3a": _b_defaults(lr=3e-4, note="lr 3e-4"),
    "F3c": _b_defaults(lr=3e-3, note="lr 3e-3"),
}


# ---------------------------------------------------------------------------
# The final model
# ---------------------------------------------------------------------------
# Stage 0 exists to fill this dict in. Until it is filled, `resolve` refuses to
# build a command rather than quietly training the defaults, because "the
# defaults were probably right" is how the paper checkpoint ended up with flags
# recorded nowhere -- the defect this whole directory exists to remove.
#
# It lives here, beside the arms, rather than in `train_final.sh`, so that the
# final model's flags are produced by the same `build_command` as every arm it
# was selected against. A shell script with its own copy of the flag list is
# exactly how `train_digital_twin_120_long.sh` drifted away from the checkpoint
# it supposedly produced.
FINAL_RUN = "FINAL"
FINAL_SEED = 1
FINAL_SETTINGS = {
    "homogeneous": None,       # E1: cross-session vs one session per step
    "effective_batch": None,   # E2
    "lr": None,                # E3
    "core_lr_scale": None,     # E3
    "samples": None,           # E4: the per-run sample budget
    "width": None,             # capacity; 1.0 reproduces the paper model
}


def unsettled_final():
    """Which of the final model's settings the sweep has not yet decided."""
    return [k for k, v in FINAL_SETTINGS.items() if v is None]


def resolve_final():
    missing = unsettled_final()
    if missing:
        raise SystemExit(
            "The final model is not settled. Unset: " + ", ".join(missing) +
            "\nFill FINAL_SETTINGS in launch.py from the sweep result "
            "(collect.py / stability.py) before training the pinned model.")
    spec = _defaults(seed=FINAL_SEED, note="final pinned model",
                     **FINAL_SETTINGS)
    spec["name"] = FINAL_RUN
    return spec


def resolve(name):
    if name == FINAL_RUN:
        return _finish(resolve_final())
    table = RUNS if name in RUNS else FROZEN_RUNS
    if name not in table:
        raise SystemExit(f"unknown run {name!r}; try --list")
    spec = dict(table[name])
    spec["name"] = name
    return _finish(spec)


def _finish(spec):
    """Derive accumulation, epoch count and model config from an arm's spec.

    Shared by the declared arms and by the final model so that both are built
    the same way; the final model must be trained by the same derivation as the
    arms it was selected against.
    """
    eff = spec.pop("effective_batch", EFFECTIVE_BATCH)
    bs = spec["batch_size"]
    spec["accumulate"] = max(1, eff // bs)
    spec["effective_batch"] = spec["accumulate"] * bs
    # Samples per epoch does not depend on accumulation: it is the number of
    # batches per epoch times the batch size.
    per_epoch = STEPS_PER_EPOCH * bs
    spec["max_epochs"] = max(1, round(spec["samples"] / per_epoch))
    spec["samples_actual"] = spec["max_epochs"] * per_epoch

    # Ask gen_configs for the name rather than rebuilding it here. The
    # hand-rolled version was `str(width).replace(".", "p")`, which gives
    # "2p0" for width 2.0 while gen_configs writes "width2" (`f"{2.0:g}"` is
    # "2", so there is no "." to replace). C1 therefore pointed at a file that
    # has never existed, and --dry-run could not tell: only the dataset config
    # was checked for existence.
    #
    # Width 1.0 *with* the adapter keeps pointing at the unmodified
    # experiments/ config, so the frozen arms reproduce byte-identically.
    # Everything else resolves to a generated rung, with the architecture in
    # the filename so a manifest states which one it trained.
    from gen_configs import width_name

    adapter = spec.get("adapter", True)
    if not adapter or spec["width"] != 1.0:
        suffix = "" if adapter else "_noadapter"
        spec["model_config"] = (
            f"paper/model_selection/configs/"
            f"{width_name(spec['width'], suffix)}.yaml")
    return spec


def build_command(spec, gpu, resume=None):
    cfg = CONFIGS / spec["config"]
    if not cfg.exists():
        raise SystemExit(f"missing dataset config: {cfg}")

    # Check the model config too. Only the dataset config was checked, so an
    # arm pointing at a non-existent model config passed --dry-run and failed
    # after the ~19 min dataset load -- which is how the width-name mismatch
    # stayed hidden.
    model_cfg = VISIONCORE_ROOT / spec["model_config"]
    if not model_cfg.exists():
        raise SystemExit(
            f"missing model config: {model_cfg}\n"
            f"Ladder rungs are written by gen_configs.py; run it first "
            f"(add --no-adapter for the identity-adapter variant).")

    # train_multidataset.py appends --experiment_name to --checkpoint_dir, so
    # passing CKPT_ROOT (not CKPT_ROOT/name) lands checkpoints in the same
    # directory as the manifest rather than one level below it.
    ckpt_dir = CKPT_ROOT / spec["name"]
    cmd = [
        "uv", "run", "python", "training/train_multidataset.py",
        "--model_config", spec["model_config"],
        "--dataset_configs_path", str(cfg),
        "--max_datasets", "30",
        "--batch_size", str(spec["batch_size"]),
        "--learning_rate", str(spec["lr"]),
        "--core_lr_scale", str(spec["core_lr_scale"]),
        "--weight_decay", str(spec["wd"]),
        "--lr_scheduler", "cosine_warmup",
        "--warmup_epochs", "2",
        "--max_epochs", str(spec["max_epochs"]),
        "--accumulate_grad_batches", str(spec["accumulate"]),
        "--steps_per_epoch", str(STEPS_PER_EPOCH),
        "--gradient_clip_val", "10.0",
        "--precision", "bf16-mixed",
        "--dset_dtype", "bfloat16",
        "--num_workers", "16",
        # A full validation pass is ~5900 batches (~22 min). Selection needs a
        # stable BPS estimate, not an exhaustive one, so sample a tenth of it
        # and validate every 4th epoch: ~15 validations over a 61-epoch run.
        "--limit_val_batches", str(LIMIT_VAL_BATCHES),
        "--check_val_every_n_epoch", str(CHECK_VAL_EVERY),
        "--seed", str(spec["seed"]),
        "--gpu", str(gpu),
        "--checkpoint_dir", str(CKPT_ROOT),
        "--project_name", "model_selection",
        "--experiment_name", spec["name"],
        # Arms are sample-budgeted and the cosine horizon is max_epochs, so
        # stopping early would leave the learning rate un-annealed and the arms
        # no longer compute-matched. save_top_k=3 on val BPS still selects.
        "--no-early_stopping",
    ]
    # train_multidataset.py defaults --homogeneous_batches to True, so the
    # cross-session arm has to turn it off explicitly.
    cmd += ["--homogeneous_batches" if spec["homogeneous"]
            else "--no-homogeneous_batches"]
    if resume is not None:
        cmd += ["--ckpt_path", str(resume)]
    return cmd, ckpt_dir


def write_manifest(spec, ckpt_dir, cmd):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run": spec["name"],
        "protocol_hash": PROTOCOL_HASH,
        "protocol": protocol_dict(),
        "spec": {k: v for k, v in spec.items()},
        "command": cmd,
        "launched": datetime.now().isoformat(timespec="seconds"),
    }
    path = ckpt_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="?", help="Run id, e.g. E1a")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="Also list the frozen Stage 0 arms")
    ap.add_argument("--resume", type=str, default=None,
                    help="Checkpoint to resume this arm from, e.g. its last.ckpt")
    args = ap.parse_args()

    if args.list or not args.run:
        header = (f"{'id':<6}{'width':>6}{'bs':>6}{'eff':>7}{'lr':>9}"
                  f"{'epochs':>8}{'Msamp':>7}{'seed':>6}  note")

        def show(name):
            s = resolve(name)
            print(f"{name:<6}{s['width']:>6}{s['batch_size']:>6}"
                  f"{s['effective_batch']:>7}{s['lr']:>9.0e}{s['max_epochs']:>8}"
                  f"{s['samples_actual']/1e6:>7.1f}{s['seed']:>6}  {s['note']}")

        print(f"protocol {PROTOCOL_HASH}\n")
        print("Stage 0b -- live family (homogeneous, no adapter, width 0.5)")
        print(header)
        for name in RUNS:
            show(name)

        if args.all:
            print("\nStage 0 -- FROZEN (cross-session, adapter on). Completed "
                  "runs stay interpretable;\n  these are not a baseline for "
                  "Stage 0b -- different architecture and batching.")
            print(header)
            for name in FROZEN_RUNS:
                show(name)
        else:
            print(f"\n({len(FROZEN_RUNS)} frozen Stage 0 arms hidden; --all to "
                  f"show)")

        # The final model is not one of the arms; it is what the arms decide.
        missing = unsettled_final()
        if missing:
            print(f"\n{FINAL_RUN}: unsettled ({', '.join(missing)}) -- "
                  f"fill FINAL_SETTINGS from the sweep result")
        else:
            s = resolve(FINAL_RUN)
            print(f"\n{FINAL_RUN:<6}{s['width']:>6}{s['batch_size']:>6}"
                  f"{s['effective_batch']:>7}{s['lr']:>9.0e}{s['max_epochs']:>8}"
                  f"{s['samples_actual']/1e6:>7.1f}{s['seed']:>6}  {s['note']}")
        return

    spec = resolve(args.run)
    cmd, ckpt_dir = build_command(spec, args.gpu, resume=args.resume)

    print(f"[{spec['name']}] {spec['note']}")
    print(f"  {spec['max_epochs']} epochs = {spec['samples_actual']/1e6:.1f}M samples, "
          f"effective batch {spec['effective_batch']}, protocol {PROTOCOL_HASH}")
    print("  " + " ".join(cmd))

    if args.dry_run:
        return

    path = write_manifest(spec, ckpt_dir, cmd)
    print(f"  manifest -> {path}")
    raise SystemExit(subprocess.call(cmd, cwd=str(VISIONCORE_ROOT)))


if __name__ == "__main__":
    main()
