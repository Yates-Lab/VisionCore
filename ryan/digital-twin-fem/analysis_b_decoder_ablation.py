"""Analysis B — decoder ablation.

Re-uses the cached trial tensors from Analysis A. Fits three decoder classes
(instantaneous, time-averaged, flattened-temporal) on the same data and compares
the FEM advantage across them.
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
from _common import FIG_DIR, FEM_CONDITIONS, DECODER_KINDS, fit_eval_decoder

SEED = 0
T_MAX = 60
L2 = 1.0
TRAIN_FRAC = 0.8
CACHE_PATH_A = CACHE_DIR / "fig4_a.pkl"
CACHE_PATH = CACHE_DIR / "fig4_b.pkl"


# %% load cached trials from Analysis A
with open(CACHE_PATH_A, "rb") as f:
    a_cache = dill.load(f)
dataset = a_cache["dataset"]
N_POP = a_cache["N"]
SIZE_DEG = a_cache["size_deg"]
ORIENTATIONS = tuple(sorted({phi for (_, phi) in dataset}))


# %% stack and split per condition
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


# %% fit the three decoders
acc = {kind: {cond: np.nan for cond in FEM_CONDITIONS} for kind in DECODER_KINDS}
for cond in FEM_CONDITIONS:
    X, y = by_cond[cond]
    X_tr, y_tr, X_te, y_te = split_train_test(X, y, np.random.default_rng(SEED + 1))
    for kind in DECODER_KINDS:
        if kind == "instantaneous":
            best = -np.inf
            for tb in range(T_MAX):
                a, _, _ = fit_eval_decoder(
                    X_tr, y_tr, X_te, y_te,
                    kind="instantaneous", t_window=T_MAX, t_bin=tb, l2=L2,
                )
                best = max(best, a)
            acc[kind][cond] = best
        else:
            a, _, _ = fit_eval_decoder(
                X_tr, y_tr, X_te, y_te,
                kind=kind, t_window=T_MAX, l2=L2,
            )
            acc[kind][cond] = a
        print(f"  {cond:12s}  {kind:18s}  acc={acc[kind][cond]:.3f}")


# %% cache
with open(CACHE_PATH, "wb") as f:
    dill.dump({
        "accuracy": acc,
        "conditions": list(FEM_CONDITIONS),
        "decoders": list(DECODER_KINDS),
        "N": N_POP,
        "size_deg": SIZE_DEG,
    }, f)
print(f"Saved cache to {CACHE_PATH}")


# %% grouped bar plot
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(FEM_CONDITIONS))
width = 0.25
for i, kind in enumerate(DECODER_KINDS):
    vals = [acc[kind][c] for c in FEM_CONDITIONS]
    ax.bar(x + i * width, vals, width=width, label=kind)
ax.set_xticks(x + width)
ax.set_xticklabels(FEM_CONDITIONS, rotation=20)
ax.axhline(0.25, color="k", ls="--", lw=0.5, label="chance")
ax.set_ylabel("Accuracy")
ax.set_title(f"Analysis B — decoder ablation (N={N_POP})")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "b_decoder_ablation.png", dpi=200)
