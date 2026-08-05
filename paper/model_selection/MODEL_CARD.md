# Model card — digital twin, Stage 0 (validation)

What has been settled, and the evidence for it. Tier 1 starts from the frozen
configuration recorded here. Nothing in this file is a paper result; these are
configuration choices that the paper does not report.

Protocol hash at time of writing: **`78ea1cc91f00`** (`protocol.py`). Runs under
a different hash are not poolable.

---

## Settled by measurement (no training required)

### The readout population already covers fig2 — no retraining

`fig2_analyzed ⊆ twin_readout` holds by construction: fig2 and the twin both
derive their populations from the same per-session YAML `cids`
(`paper/covariance_decomposition/data_loading.py:95` runs `prepare_data` on the
session config, with `min_total_spikes=0`), and the twin's readout is
`len(cids)`. Measured under current inclusion (rate > 2 Hz, PSTH R² > 0.10,
10-unit session floor): **1022 fig2 units across 19 sessions, of which 28 (2.7%)
were absent from the fig3 population — all of them present in the readout** and
excluded only by fig3's `MIN_TOTAL_SPIKES = 200` analysis threshold.

**Decision:** no `sessions_v2/` config set, no retraining for population
reasons. The invariant is asserted at load time in `eval/load_twin.py`.

`MIN_TOTAL_SPIKES` stays at **200** in `paper/fig3/_fig3_data.py`, and at 200 in
`protocol.py` for model-selection evaluation. It was briefly lowered to 0 on
2026-08-04 to recover those 28 cells, then reverted the same day: panel D scores
every cell in a session on one shared window set built as
`np.isfinite(robs).all(axis=2)` (`_fig3_explainable_variance.py:98`), so a cell
present for only part of a session deletes its absent bins for every other cell.
Measured over both caches, 10 of 24 sessions lost base windows and 8 lost more
than a quarter (Allen_2022-04-01 1260 → 539; Logan_2020-02-28 5034 → 2626 on two
added units), for 25 panel C cells. The superset invariant holds at the readout
level regardless of this analysis threshold, so nothing depends on it being 0.

**Consequence:** none. fig3 reports its published numbers, which
`generate_figure3.py` and `main.tex` already state.

### Subject balancing is not warranted

The premise behind TWIN_IMPROVEMENTS item 3 — "far more samples/sessions from
Allen than from Logan" — is false. From `data_census.py` over all 30 sessions:

| subject | sessions | units | train samples | session share | sample share | unit share |
|---|---|---|---|---|---|---|
| Allen | 14 | 1989 | 4,227,251 | 46.7% | 51.8% | 71.3% |
| Logan | 16 | 801 | 3,935,261 | 53.3% | 48.2% | 28.7% |

Unit count is the only lopsided quantity and it does not enter the weighting:
the loss is a masked mean over (samples × units) within a session, then a mean
over the sessions present in the batch. Under the default cross-session
batching a 256-sample batch contains essentially all 30 sessions, so each
session carries equal weight per step. **Per unit, Logan is already weighted
~2.5× more than Allen.**

From `subject_gap.py`, with sessions as the resampling unit:

| contrast | raw | matched on rate + PSTH R² |
|---|---|---|
| CC_norm (Logan − Allen) | −0.083 [−0.181, +0.041] | −0.057 [−0.153, +0.074] |
| CC_abs (Logan − Allen) | −0.131 [−0.214, −0.020] | **−0.016 [−0.091, +0.058]** |

Logan's units are genuinely harder (10.3 vs 19.3 Hz; PSTH R² 0.066 vs 0.178),
and matching on that removes essentially all of the absolute deficit. The
CC_norm gap is not distinguishable from zero: Logan's session medians span
0.398–0.860.

**Decision:** do not implement per-subject balancing as a default. The one
untested mechanism is interference from joint training, which arms E5a/E5b
measure directly. Revisit only if those show negative transfer.

### Batching and subject weighting are the same knob

`ByDatasetBatchSampler` (`homogeneous_batches=True`) draws a session with
p ∝ its size, making *sample* share the effective per-session weight; the
cross-session default makes it uniform per session. Arm E1 therefore measures
batching and subject weighting together, and they cannot be decided separately.

### Capacity ladder operating points

From `probe_capacity.py` on real batches (see `README.md`): bs = 256 for
width ≤ 2, bs = 128 for widths 3–4, gradient accumulation holding the effective
batch at 1024. Memory is activation-dominated and grows linearly in width, so
71.4M parameters fits in 18 GiB at bs = 64.

Gradient checkpointing is **not implemented** — the import at
`models/modules/convnet.py` was unused and the `checkpointing:` config key and
`use_checkpointing` assignments were no-ops. All removed. Reducing the
micro-batch covers the whole ladder, so no escape hatch is needed.

---

## Settled by design

### Three-way split, wired additively

`protocol.py` declares 70/15/15 by trial, seed 1002. The paper model used a
two-way 80/20 split, so selection and reporting drew on the same trials —
tolerable for one model, optimistically biased in N for a family.

Implementation: a `test_split` key in the dataset config switches
`prepare_data` to `split_inds_by_trial_train_val_test`. **Absent or null, the
two-way path is byte-identical**, which is what protects figures 1–4. Verified
two ways: unit tests asserting the served indices equal `split_inds_by_trial`
exactly (`tests/test_three_way_split.py`), and a real-data before/after hash of
every served index set across three sessions (`data_census.py --diff`).

The 70/15/15 train set nests inside the old 80/20 train set under the shared
seed, so the new protocol never trains on anything the old one held out.

`MultiDatasetDM.test_dataloader()` raises rather than falling back to
validation when no `test_split` is configured — silently reporting a selection
split as a test score is the exact bias the third split exists to remove.

### Run family

`launch.py` derives `max_epochs` from a **sample** budget (default 8M ≈ 61
epochs at 512 steps × bs 256), not wall clock, so arms stay comparable across
widths and batch sizes, and passes it as the cosine horizon. The paper model's
horizon was 9999 epochs and it stopped at 374, so its learning rate never
annealed; every Stage 0 arm anneals. Each run writes a `manifest.json` stamped
with `PROTOCOL_HASH`.

**No early stopping.** Arms are sample-budgeted and the cosine horizon *is*
`max_epochs`, so stopping early would leave the learning rate un-annealed and
the arms no longer compute-matched — the exact defect E4 exists to avoid.
`save_top_k=3` on validation BPS still supplies the selection checkpoint.
Note the semantics that made the old default a tripwire: Lightning's
`EarlyStopping` patience counts *validation calls*, so `patience=10` with
`check_val_every_n_epoch=4` was 40 of a 61-epoch run.

**Baseline replicates R1/R2/R3.** `_defaults` already carries
`homogeneous=False`, effective batch 1024 and lr 1e-3, so E1a, E2b and E3b are
the same configuration. They are kept, at seeds 101/102/103, as replicates of
the baseline rather than deleted: every other arm reports a ΔBPS, and without a
run-to-run spread there is no scale on which to judge whether any delta is real.
What a seed varies is narrower than it looks — `split_inds_by_trial*` calls
`set_seeds(SPLIT_SEED)` internally, so the trials and their order are fixed
regardless, and replicates differ in weight initialisation and GPU
nondeterminism only. Verified: a census run under `seed_everything(101)` is
bitwise identical to `census_baseline.json` across all 30 sessions.

Dataset configs live in `configs/` and reference the **unmodified** session
YAMLs, so the trained population is identical to the figure population.

### Defect fixed en route

`train_multidataset.py --homogeneous_batches` was `store_true` with
`default=True`, so it could never be switched off and the cross-session path —
the one the paper model was trained under — was unreachable from the single-GPU
script. Now `BooleanOptionalAction`; E1a passes `--no-homogeneous_batches`.
Without this, E1 would have compared an arm against itself.

---

## Pending

- **E1–E5.** Not yet run. ~16 arms × ~4.6 GPU-h ≈ 80 GPU-h on 2× RTX 6000 Ada.
- **Batch size, learning rate, compute budget.** Open until E2/E3/E4 report.
- **Width 2.0 confirmation** of whatever E1/E3 select (arm C1).

## Not settled, and why

- **Whether one-session-per-step batching is acceptable in principle.** E1
  measures the cost in validation BPS and throughput; whether the changed
  gradient is acceptable is a judgment call, not a measurement.
- **Whether to keep width 0.25 in Tier 1.** It is latency-bound (slower per
  step than width 1.0 despite 10× fewer parameters) and costs ~35 GPU-h for the
  least informative rung.
