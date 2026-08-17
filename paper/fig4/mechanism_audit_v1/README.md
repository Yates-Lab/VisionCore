# Figure 4 mechanism audit v1

This directory contains the new, isolated mechanism-audit code requested for
Figure 4.  It does not modify the model, the RR100 population, the crossed
movie bank, SSI, stabilized baselines, or any existing Figure 4 output.

## Current status: complete primary true-history correction; mechanism pending

The first audit found that the production movie scorer constructs its
32-frame prefix by copying samples 0--31 of the same 40-sample trajectory that
is subsequently scored.  After the scorer drops its extra first output, 31 of
the 40 retained outputs therefore contain a circular wrap from trajectory
sample 31 to sample 0 and include samples that are future relative to the
nominal output time.  This is not a causal prehistory.

The blocking audit was followed by an authorized correction. The exact causal
convention was established, all 1,000 real source histories were recovered and
validated, and the complete 100-image × 1,000-trajectory primary true-history
bank was rescored for the frozen 100-unit population. The core qualitative
interaction is `SURVIVES-LIKE`: correction modestly lowers effect size but the
low-SF progressive curve and high-SF intermediate optimum remain.

The held-prefix trajectories have been built and validated but their model
responses were deferred for runtime. The full temporal-frequency, retinal
space-to-time, event-history, frontend, step-and-hold, layerwise, intervention,
and q audits therefore remain pending. See the detailed methods and
interpretation report at
`outputs/figures/fig4/mechanism_audit_v1/corrected_history_v1/preliminary_true_only/TRUE_HISTORY_PRIMARY_REPORT.md`.

## Reproduce the blocking audit

From the VisionCore repository root:

```bash
python paper/fig4/mechanism_audit_v1/audit_history_prefix.py
```

Outputs are written to:

```text
outputs/figures/fig4/mechanism_audit_v1/
```

The script is CPU-only and reads the frozen crossed-bank metadata and the
previous joint-tuning analysis.  It does not run or change the model.
