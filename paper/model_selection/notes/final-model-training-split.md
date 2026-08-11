# The final model trains on train + val

**Decision (2026-08-11).** Once selection is frozen, the selected configuration
is retrained on **train + val (85% of trials)** rather than on the 70% every
sweep arm sees. The 15% test split stays untouched.

**Not implemented.** Recorded now, built later, once the sweep has actually
selected a model. `FINAL_SETTINGS` in `launch.py` is still unset.

## Why

The comparison in [production-baseline.md](production-baseline.md) exposed the
problem. The fig3 production twin trained on 80% of trials; every sweep arm
gets 70%, because the three-way split is what stops selection and reporting
drawing on the same trials. That 14% is a real cost, and it is paid by the
*deliverable*, not just by the sweep.

So a genuinely better configuration could still ship a **worse twin** than the
one it replaces, purely on training data — and nothing in the current design
would let us tell which cause was which. Retraining on train + val removes the
handicap (85% > production's 80%) at the cost of one run, and it does not
compromise the split's purpose: selection is over by the time it happens, so
the val trials have no remaining job.

This is standard practice — select on validation, then refit the winner on
everything except test — and it is the reason the test split was carved out
separately in the first place.

## The mechanism, and why the test set stays identical

`protocol.py` **must not change.** Its constants feed `PROTOCOL_HASH`, and
changing the hash invalidates every run in the family by design. The retrain is
a property of the final run, not of the protocol, so it belongs in a dataset
config and in `FINAL_RUN` — never in `protocol.py`.

`models/data/splitting.py:130` shuffles trials once from `seed` and slices:

```python
train_end = int(len(trials) * train_split)
val_end   = int(len(trials) * (train_split + val_split))
test_trials = trials[rand_trials[val_end:]]
```

The test trials depend on `train_split + val_split` only — not on how that sum
is divided. So a final config declaring `train_val_split: 0.85` with
`test_split: 0.15` at `seed: 1002` yields **exactly the test trials the sweep
has been holding out**, while folding the val trials into training. The
integrity of the held-out set survives the change, which is what makes this
safe rather than merely convenient.

**Verify this rather than trusting the reading above** when it is implemented:
assert the resolved test indices are identical under (0.70, 0.15) and
(0.85, 0.15) for at least a few sessions before training anything.

## The open implementation question

With val folded into train, `val_inds` is empty and there is no validation
dataloader — which is what `ModelCheckpoint` monitors, what `check_val_every_n_epoch`
drives, and what every sweep arm selected its best epoch on. The final run
therefore needs a checkpoint policy that does not depend on a validation set.

The relevant fact from the sweep: **every long run in experiments 03-05 peaked
before its final epoch** (`05_lr5e-4` peaked at epoch 471 of 488, ending 0.0072
lower). So "just take the last epoch" is not free — it reliably costs something,
though the drop has stayed inside the replicate floor so far.

Options, undecided:

- Take the final epoch, and accept a cost the sweep has bounded at ~0.007.
- Hold out a small slice of the *val* trials purely for checkpointing, spending
  a little of the recovered data to keep epoch selection working.
- Fix the epoch count from where the selected configuration peaked in its sweep
  run, and train exactly that long.

Decide when the model is selected, not before — the right answer depends on how
large the peak-to-final drop is at the winning width.
