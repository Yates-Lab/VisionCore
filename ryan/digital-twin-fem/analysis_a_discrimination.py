"""Analysis A — Discrimination vs integration time.

Fits a flattened-temporal linear decoder on tumbling-E orientation, across four
FEM conditions, for a grid of integration windows T. A diagnostic
population-size sweep pins N for Analyses B and C.
"""
# %% imports and config
from __future__ import annotations

import sys

# Ensure _common.py is importable regardless of CWD (works in IPython too,
# where __file__ is not defined).
from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR
_HERE = VISIONCORE_ROOT / "ryan" / "digital-twin-fem"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import matplotlib.pyplot as plt
import dill
from tqdm import tqdm
from _common import (
    FIG_DIR, FEM_CONDITIONS, DT, N_LAGS,
    load_digital_twin, build_population, extract_fixrsvp_eye_traces,
    render_tumbling_e, make_eye_trace, generate_trials,
    fit_eval_decoder,
)

RECOMPUTE = False

# Hyperparameters.
SEED = 0
N_POP = 256
N_POP_SWEEP = (64, 128, 256, 512)
SIZE_DEG = 0.5
ORIENTATIONS = (0.0, 90.0, 180.0, 270.0)
N_TRIALS_PER_CLASS_PER_COND = 100
T_MAX = 60
T_GRID = (4, 8, 16, 32, 60)
L2 = 1.0
TRAIN_FRAC = 0.8
CACHE_PATH = CACHE_DIR / "fig4_a.pkl"

rng = np.random.default_rng(SEED)


# %% load model and eye traces
model, model_info, device = load_digital_twin()
eye_traces, durations = extract_fixrsvp_eye_traces(model, min_fix_dur=T_MAX + 1)
print(f"Loaded {len(eye_traces)} fixrsvp eye traces, "
      f"median dur {np.median(durations):.0f} frames.")


# %% build simulated population (main run)
population = build_population(model, N=N_POP, rng=rng)
print(f"Population size: {population.N}; grid_shape = {population.grid_shape}; "
      f"unit_ids.shape = {population.unit_ids.shape}")


# %% pre-render the four E stimuli
e_images = {phi: render_tumbling_e(phi, size_deg=SIZE_DEG) for phi in ORIENTATIONS}


# %% trial generation
def make_dataset(
    model, population, eye_traces, durations,
    conditions=FEM_CONDITIONS, n_trials=N_TRIALS_PER_CLASS_PER_COND,
    rng=rng,
) -> dict:
    """Returns {(condition, orientation): y[n_trials, T_MAX, N_pop]}."""
    out = {}
    valid = np.where(durations >= T_MAX + N_LAGS + 1)[0]
    if valid.size == 0:
        raise RuntimeError(
            f"No eye traces meet min duration T_MAX+N_LAGS+1={T_MAX + N_LAGS + 1}."
        )
    for cond in conditions:
        for phi in ORIENTATIONS:
            ys = []
            for _ in tqdm(range(n_trials), desc=f"{cond}/{int(phi)}", leave=False):
                idx = rng.choice(valid)
                trace = make_eye_trace(cond, eye_traces[idx][:T_MAX])
                y = generate_trials(
                    model, population, e_images[phi], trace,
                    n_trials=1, rng=rng, device=device,
                )
                ys.append(y[0])
            out[(cond, phi)] = np.stack(ys, axis=0)
    return out


if CACHE_PATH.exists() and not RECOMPUTE:
    with open(CACHE_PATH, "rb") as f:
        _cache = dill.load(f)
    dataset = _cache["dataset"]
    print(f"Loaded cached dataset from {CACHE_PATH}")
else:
    dataset = make_dataset(model, population, eye_traces, durations)


# %% decoder curve over T
def stack_and_label(dataset, conditions, orientations):
    by_cond = {}
    for cond in conditions:
        Xs, ys = [], []
        for li, phi in enumerate(orientations):
            trials = dataset[(cond, phi)]
            Xs.append(trials)
            ys.append(np.full(trials.shape[0], li))
        by_cond[cond] = (np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0))
    return by_cond


def split_train_test(X, y, rng, train_frac=TRAIN_FRAC):
    n = X.shape[0]
    perm = rng.permutation(n)
    n_tr = int(train_frac * n)
    return X[perm[:n_tr]], y[perm[:n_tr]], X[perm[n_tr:]], y[perm[n_tr:]]


by_cond = stack_and_label(dataset, FEM_CONDITIONS, ORIENTATIONS)
accuracy = np.zeros((len(FEM_CONDITIONS), len(T_GRID)))
for i, cond in enumerate(FEM_CONDITIONS):
    X, y = by_cond[cond]
    X_tr, y_tr, X_te, y_te = split_train_test(X, y, np.random.default_rng(SEED + 1))
    for j, T in enumerate(T_GRID):
        acc, _, _ = fit_eval_decoder(
            X_tr, y_tr, X_te, y_te,
            kind="flattened_temporal", t_window=T, l2=L2,
        )
        accuracy[i, j] = acc
        print(f"  {cond:12s}  T={T:3d}  acc={acc:.3f}")


# %% population-size sweep (diagnostic; real FEM at T_MAX)
sweep_acc = {}
for N_try in N_POP_SWEEP:
    pop_try = build_population(model, N=N_try, rng=np.random.default_rng(SEED + 2))
    ds_try = make_dataset(
        model, pop_try, eye_traces, durations,
        conditions=("real",),
        n_trials=N_TRIALS_PER_CLASS_PER_COND,
        rng=np.random.default_rng(SEED + 5),
    )
    bc_try = stack_and_label(ds_try, ("real",), ORIENTATIONS)
    X, y = bc_try["real"]
    X_tr, y_tr, X_te, y_te = split_train_test(X, y, np.random.default_rng(SEED + 3))
    acc, _, _ = fit_eval_decoder(
        X_tr, y_tr, X_te, y_te,
        kind="flattened_temporal", t_window=T_MAX, l2=L2,
    )
    sweep_acc[N_try] = acc
    print(f"  N={N_try:4d}  real FEM acc={acc:.3f}")


# %% save cache
with open(CACHE_PATH, "wb") as f:
    dill.dump({
        "dataset": dataset,
        "accuracy": accuracy,
        "T_grid": np.array(T_GRID),
        "conditions": list(FEM_CONDITIONS),
        "N": N_POP,
        "size_deg": SIZE_DEG,
        "pop_sweep": sweep_acc,
    }, f)
print(f"Saved cache to {CACHE_PATH}")


# %% plot: accuracy vs T
fig, ax = plt.subplots(figsize=(5, 4))
for i, cond in enumerate(FEM_CONDITIONS):
    ax.plot(np.array(T_GRID) * DT * 1000, accuracy[i], marker="o", label=cond)
ax.axhline(0.25, color="k", ls="--", lw=0.5, label="chance")
ax.set_xlabel("Integration window (ms)")
ax.set_ylabel("4-way E accuracy")
ax.set_title(f"Analysis A — discrimination vs T (N={N_POP}, size={SIZE_DEG}°)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "a_accuracy_vs_T.png", dpi=200)


# %% plot: population-size sweep
fig, ax = plt.subplots(figsize=(4, 3.5))
Ns = sorted(sweep_acc)
ax.plot(Ns, [sweep_acc[n] for n in Ns], marker="o")
ax.set_xscale("log")
ax.set_xlabel("Population size N")
ax.set_ylabel("Accuracy (real FEM, T_max)")
ax.set_title("Analysis A — population-size sweep")
fig.tight_layout()
fig.savefig(FIG_DIR / "a_popsize_sweep.png", dpi=200)
