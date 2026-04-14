"""Analysis C — adaptation ablation.

2x2 factorial on {real FEM, no FEM} x {GRU memory on, off}. Uses the
gru_memory_off context manager to zero the recurrent hidden state at every
step. The flattened-temporal decoder is fit at T_MAX.

Caveat: hidden-state reset leaves GRU input-dependent gating in place. A
retrained no-recurrence variant would be stronger — see README §TODOs.
"""
# %% imports and config
from __future__ import annotations

import sys

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR
_HERE = VISIONCORE_ROOT / "ryan" / "digital-twin-fem"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import matplotlib.pyplot as plt
import dill
from _common import (
    FIG_DIR, N_LAGS,
    load_digital_twin, build_population, extract_fixrsvp_eye_traces,
    render_tumbling_e, make_eye_trace, generate_trials,
    fit_eval_decoder, gru_memory_off,
)

SEED = 0
N_POP = 256
SIZE_DEG = 0.3
ORIENTATIONS = (0.0, 90.0, 180.0, 270.0)
N_TRIALS_PER_CLASS_PER_COND = 100
T_MAX = 60
L2 = 1.0
TRAIN_FRAC = 0.8
FEM_CONDS_C = ("real", "none")
GRU_CONDS = ("on", "off")
CACHE_PATH = CACHE_DIR / "fig4_c.pkl"

rng = np.random.default_rng(SEED)


# %% model, traces, population, stimuli
model, _, device = load_digital_twin()
eye_traces, durations = extract_fixrsvp_eye_traces(model, min_fix_dur=T_MAX + 1)
population = build_population(model, N=N_POP, rng=rng)
e_images = {phi: render_tumbling_e(phi, size_deg=SIZE_DEG) for phi in ORIENTATIONS}
print(f"Population size: {population.N}; grid_shape = {population.grid_shape}")


# %% generate trials across 2x2 cells
def make_trials(model, conditions, rng):
    out = {}
    valid = np.where(durations >= T_MAX + N_LAGS + 1)[0]
    for cond in conditions:
        for phi in ORIENTATIONS:
            ys = []
            for _ in range(N_TRIALS_PER_CLASS_PER_COND):
                idx = rng.choice(valid)
                trace = make_eye_trace(cond, eye_traces[idx][:T_MAX])
                y = generate_trials(
                    model, population, e_images[phi], trace,
                    n_trials=1, rng=rng, device=device,
                )
                ys.append(y[0])
            out[(cond, phi)] = np.stack(ys, axis=0)
    return out


# Sanity check the GRU memory-off intervention changes outputs before the
# expensive loop. Same stimulus, same eye trace — outputs must differ.
with gru_memory_off(model):
    y_off = generate_trials(
        model, population, e_images[ORIENTATIONS[0]],
        eye_traces[0][:T_MAX], n_trials=1,
        rng=np.random.default_rng(SEED + 99), device=device,
    )
y_on = generate_trials(
    model, population, e_images[ORIENTATIONS[0]],
    eye_traces[0][:T_MAX], n_trials=1,
    rng=np.random.default_rng(SEED + 99), device=device,
)
assert not np.allclose(y_on, y_off), \
    "gru_memory_off produced identical output to GRU-on; patch did not take effect."
print("GRU memory-off intervention verified (outputs differ from baseline).")


all_trials = {}
all_trials["on"] = make_trials(model, FEM_CONDS_C, rng)
with gru_memory_off(model):
    all_trials["off"] = make_trials(model, FEM_CONDS_C, np.random.default_rng(SEED + 4))


# %% decode
def stack_and_label(d, conditions):
    by_cond = {}
    for cond in conditions:
        Xs, ys = [], []
        for li, phi in enumerate(ORIENTATIONS):
            trials = d[(cond, phi)]
            Xs.append(trials)
            ys.append(np.full(trials.shape[0], li))
        by_cond[cond] = (np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0))
    return by_cond


acc = np.zeros((len(GRU_CONDS), len(FEM_CONDS_C)))
for i, gcond in enumerate(GRU_CONDS):
    by_cond = stack_and_label(all_trials[gcond], FEM_CONDS_C)
    for j, fcond in enumerate(FEM_CONDS_C):
        X, y = by_cond[fcond]
        n = X.shape[0]
        perm = np.random.default_rng(SEED + 10 + i).permutation(n)
        n_tr = int(TRAIN_FRAC * n)
        tr, te = perm[:n_tr], perm[n_tr:]
        a, _, _ = fit_eval_decoder(
            X[tr], y[tr], X[te], y[te],
            kind="flattened_temporal", t_window=T_MAX, l2=L2,
        )
        acc[i, j] = a
        print(f"  gru={gcond}  fem={fcond}  acc={a:.3f}")


# %% cache
with open(CACHE_PATH, "wb") as f:
    dill.dump({
        "accuracy": acc,
        "gru_conds": list(GRU_CONDS),
        "fem_conds": list(FEM_CONDS_C),
        "N": N_POP,
        "size_deg": SIZE_DEG,
    }, f)
print(f"Saved cache to {CACHE_PATH}")


# %% 2x2 bar plot
fig, ax = plt.subplots(figsize=(4, 3.5))
x = np.arange(len(FEM_CONDS_C))
width = 0.35
for i, gcond in enumerate(GRU_CONDS):
    ax.bar(x + i * width, acc[i], width=width, label=f"GRU {gcond}")
ax.set_xticks(x + width / 2)
ax.set_xticklabels(FEM_CONDS_C)
ax.axhline(0.25, color="k", ls="--", lw=0.5, label="chance")
ax.set_ylabel("Accuracy")
ax.set_title(f"Analysis C — adaptation ablation (N={N_POP})")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "c_adaptation_ablation.png", dpi=200)
