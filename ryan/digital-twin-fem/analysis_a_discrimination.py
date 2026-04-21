"""Analysis A — Discrimination vs integration time (iteration-focused).

Focused on demonstrating FEM > stationary discrimination. Conditions are
restricted to (real, none); three decoders are compared side-by-side:
flattened-temporal linear, plug-in Bayesian Poisson (ideal observer under the
generative noise model), and a small GRU. Verification cells render jshtml
animations of the eye-shifted stimulus and of model activations.
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
import matplotlib.animation as manim
import dill
import torch
from IPython.display import HTML, display
from tqdm import tqdm
from _common import (
    FIG_DIR, DT, N_LAGS, IMAGE_SHAPE, OUT_SIZE, PPD,
    load_digital_twin, build_population, extract_fixrsvp_eye_traces,
    render_tumbling_e, make_eye_trace, generate_trials,
    make_counterfactual_stim, compute_rate_map_batched,
    fit_eval_decoder, fit_eval_bayes_poisson, fit_eval_gru_decoder,
)

RECOMPUTE = True

# Hyperparameters.
SEED = 0
N_POP = 256
SIZE_DEG = 0.5
ORIENTATIONS = (0.0, 90.0, 180.0, 270.0)
N_TRIALS_PER_CLASS_PER_COND = 100
T_MAX = 60
T_GRID = (8, 16, 32, 60) # in frames 120 Hz -> (67, 133, 267, 500 ms)
L2 = 1.0
TRAIN_FRAC = 0.8

CONDITIONS = ("real", "none")
DECODERS = ("linear_flattened", "bayes_poisson", "gru_rnn")

CACHE_PATH = CACHE_DIR / "fig4_a.pkl"
rng = np.random.default_rng(SEED)


# %% load model and eye traces
model, model_info, device = load_digital_twin()
eye_traces, durations = extract_fixrsvp_eye_traces(model, min_fix_dur=T_MAX + 1)
print(f"Loaded {len(eye_traces)} fixrsvp eye traces, "
      f"median dur {np.median(durations):.0f} frames.")


# %% build simulated population
population = build_population(model, N=N_POP, rng=rng)
print(f"Population size: {population.N}; grid_shape = {population.grid_shape}; "
      f"unit_ids.shape = {population.unit_ids.shape}")


# %% pre-render the four E stimuli
e_images = {phi: render_tumbling_e(phi, size_deg=SIZE_DEG) for phi in ORIENTATIONS}


# %% trial generation (each trial draws its own eye trace)
def make_dataset(
    model, population, eye_traces, durations,
    conditions=CONDITIONS, n_trials=N_TRIALS_PER_CLASS_PER_COND, rng=rng,
) -> tuple[dict, dict]:
    """Returns (dataset, trace_ids).

    dataset: {(condition, orientation): y[n_trials, T_MAX, N_pop]}.
    trace_ids: {(condition, orientation): np.ndarray[int]} — per-trial eye-trace
    index into `eye_traces`, recorded so we can verify traces vary across trials.
    """
    out: dict = {}
    trace_ids: dict = {}
    valid = np.where(durations >= T_MAX + N_LAGS + 1)[0]
    if valid.size == 0:
        raise RuntimeError(
            f"No eye traces meet min duration T_MAX+N_LAGS+1={T_MAX + N_LAGS + 1}."
        )
    for cond in conditions:
        for phi in ORIENTATIONS:
            ys, ids = [], []
            for _ in tqdm(range(n_trials), desc=f"{cond}/{int(phi)}", leave=False):
                idx = int(rng.choice(valid))
                ids.append(idx)
                trace = make_eye_trace(cond, eye_traces[idx][:T_MAX])
                y = generate_trials(
                    model, population, e_images[phi], trace,
                    n_trials=1, rng=rng, device=device,
                )
                ys.append(y[0])
            out[(cond, phi)] = np.stack(ys, axis=0)
            trace_ids[(cond, phi)] = np.array(ids)
    return out, trace_ids


if CACHE_PATH.exists() and not RECOMPUTE:
    with open(CACHE_PATH, "rb") as f:
        _cache = dill.load(f)
    dataset = _cache["dataset"]
    trace_ids = _cache.get("trace_ids", {})
    print(f"Loaded cached dataset from {CACHE_PATH}")
else:
    dataset, trace_ids = make_dataset(model, population, eye_traces, durations)


# %% verification — per-trial eye-trace variability
# Eye traces should differ across trials (the claim is about FEMs in general,
# not a single specific trajectory).
for (cond, phi), ids in trace_ids.items():
    print(f"{cond:6s} phi={int(phi):3d}  "
          f"n_trials={len(ids)}  n_unique_traces={len(np.unique(ids))}")


# %% verification — sample the stimulus fed to the model for each condition
def make_sample_stim(condition: str, orientation: float, seed: int = SEED + 42):
    """Build one (T, H, W) eye-shifted stimulus movie for a sample trial."""
    local_rng = np.random.default_rng(seed)
    valid = np.where(durations >= T_MAX + N_LAGS + 1)[0]
    idx = int(local_rng.choice(valid))
    trace_np = make_eye_trace(condition, eye_traces[idx][:T_MAX])
    T_fix = int(np.sum(~np.isnan(trace_np[:, 0])))
    trace = torch.from_numpy(trace_np[:T_fix]).float()
    full_stack = np.broadcast_to(
        e_images[orientation][None], (T_fix + N_LAGS + 1, *IMAGE_SHAPE),
    ).copy()
    eye_stim = make_counterfactual_stim(
        full_stack, trace, ppd=PPD, scale_factor=1.0,
        n_lags=N_LAGS, out_size=OUT_SIZE,
    )
    frames = eye_stim[:, 0, -1, :, :].numpy()  # lag-0 frame per timepoint
    return frames, eye_stim


def animate_frames(frames: np.ndarray, title: str, vmin=None, vmax=None,
                   cmap: str = "gray", fps: int = 15) -> manim.FuncAnimation:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(frames[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")

    def update(t):
        im.set_data(frames[t])
        ax.set_title(f"{title}  t={t}/{len(frames)}")
        return (im,)

    ani = manim.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=False,
    )
    plt.close(fig)
    return ani


stim_frames_real, eye_stim_real = make_sample_stim("real", 0.0)
stim_frames_none, eye_stim_none = make_sample_stim("none", 0.0)
ani_stim_real = animate_frames(
    stim_frames_real, "stim (real FEM, phi=0)", vmin=0, vmax=255,
)
ani_stim_none = animate_frames(
    stim_frames_none, "stim (none, phi=0)", vmin=0, vmax=255,
)
ani_stim_real.save(FIG_DIR / "a_stim_real.mp4",
                   writer=manim.FFMpegWriter(fps=15, codec="libx264", bitrate=4000))
ani_stim_none.save(FIG_DIR / "a_stim_none.mp4",
                   writer=manim.FFMpegWriter(fps=15, codec="libx264", bitrate=4000))
print(f"Saved stimulus animations to {FIG_DIR}/a_stim_*.mp4")
display(HTML(ani_stim_real.to_jshtml()))
display(HTML(ani_stim_none.to_jshtml()))


# %% verification — model activations for the same sample trials
def compute_activations(eye_stim: torch.Tensor) -> torch.Tensor:
    stim_norm = (eye_stim - 127.0) / 255.0
    readout_dev = population.readout.to(device)
    rate_map = compute_rate_map_batched(model, readout_dev, stim_norm)
    population.readout.cpu()
    return rate_map  # (T, n_units, H, W)


def animate_activations(rate_map: torch.Tensor, title: str,
                        n_show: int = 16, fps: int = 15) -> manim.FuncAnimation:
    from torchvision.utils import make_grid

    rm = rate_map.detach().cpu()
    temporal_var = rm.mean(dim=(2, 3)).var(dim=0)
    units_to_show = torch.topk(temporal_var, k=min(n_show, rm.shape[1])).indices
    rm_sub = rm[:, units_to_show]
    mu = rm_sub.mean(dim=(0, 2, 3), keepdim=True)
    sd = rm_sub.std(dim=(0, 2, 3), keepdim=True)
    rm_sub = (rm_sub - mu) / (sd + 1e-8)
    T = rm_sub.shape[0]
    nrow = int(np.ceil(np.sqrt(rm_sub.shape[1])))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    grid0 = make_grid(rm_sub[0].unsqueeze(1), nrow=nrow, padding=1, pad_value=0.0)
    im = ax.imshow(grid0[0].numpy(), cmap="gray", vmin=-4, vmax=4)

    def update(t):
        grid_t = make_grid(rm_sub[t].unsqueeze(1), nrow=nrow, padding=1, pad_value=0.0)
        im.set_data(grid_t[0].numpy())
        ax.set_title(f"{title}  t={t}/{T}")
        return (im,)

    ani = manim.FuncAnimation(
        fig, update, frames=T, interval=1000 / fps, blit=False,
    )
    plt.close(fig)
    return ani


rate_map_real = compute_activations(eye_stim_real)
rate_map_none = compute_activations(eye_stim_none)
ani_act_real = animate_activations(rate_map_real, "activations (real FEM, phi=0)")
ani_act_none = animate_activations(rate_map_none, "activations (none, phi=0)")
ani_act_real.save(FIG_DIR / "a_act_real.mp4",
                  writer=manim.FFMpegWriter(fps=15, codec="libx264", bitrate=8000))
ani_act_none.save(FIG_DIR / "a_act_none.mp4",
                  writer=manim.FFMpegWriter(fps=15, codec="libx264", bitrate=8000))
print(f"Saved activation animations to {FIG_DIR}/a_act_*.mp4")
display(HTML(ani_act_real.to_jshtml()))
display(HTML(ani_act_none.to_jshtml()))


# %% decoder curves: (linear, bayes, GRU) × conditions × T
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


by_cond = stack_and_label(dataset, CONDITIONS, ORIENTATIONS)

accuracy = {d: np.zeros((len(CONDITIONS), len(T_GRID))) for d in DECODERS}
for i, cond in enumerate(CONDITIONS):
    X, y = by_cond[cond]
    X_tr, y_tr, X_te, y_te = split_train_test(X, y, np.random.default_rng(SEED + 1))
    for j, T in enumerate(T_GRID):
        acc_lin, _, _ = fit_eval_decoder(
            X_tr, y_tr, X_te, y_te,
            kind="flattened_temporal", t_window=T, l2=L2,
        )
        acc_bay, _ = fit_eval_bayes_poisson(
            X_tr, y_tr, X_te, y_te,
            t_window=T, n_classes=len(ORIENTATIONS),
        )
        acc_gru = fit_eval_gru_decoder(
            X_tr, y_tr, X_te, y_te, t_window=T,
            hidden=64, epochs=200, lr=1e-3, device=device, seed=SEED + j,
        )
        accuracy["linear_flattened"][i, j] = acc_lin
        accuracy["bayes_poisson"][i, j] = acc_bay
        accuracy["gru_rnn"][i, j] = acc_gru
        print(f"  {cond:6s}  T={T:3d}  linear={acc_lin:.3f}  "
              f"bayes={acc_bay:.3f}  gru={acc_gru:.3f}")


# %% save cache
with open(CACHE_PATH, "wb") as f:
    dill.dump({
        "dataset": dataset,
        "trace_ids": trace_ids,
        "accuracy": accuracy,
        "T_grid": np.array(T_GRID),
        "conditions": list(CONDITIONS),
        "decoders": list(DECODERS),
        "N": N_POP,
        "size_deg": SIZE_DEG,
    }, f)
print(f"Saved cache to {CACHE_PATH}")


# %% plot: accuracy vs T, one subplot per decoder
fig, axes = plt.subplots(
    1, len(DECODERS), figsize=(4 * len(DECODERS), 3.5), sharey=True,
)
T_ms = np.array(T_GRID) * DT * 1000
for k, decoder in enumerate(DECODERS):
    ax = axes[k]
    for i, cond in enumerate(CONDITIONS):
        ax.plot(T_ms, accuracy[decoder][i], marker="o", label=cond)
    ax.axhline(0.25, color="k", ls="--", lw=0.5, label="chance")
    ax.set_xlabel("Integration window (ms)")
    ax.set_title(decoder)
    if k == 0:
        ax.set_ylabel("4-way E accuracy")
    ax.legend(fontsize=8)
fig.suptitle(f"Analysis A — FEM vs none, by decoder (N={N_POP}, size={SIZE_DEG}°)")
fig.tight_layout()
fig.savefig(FIG_DIR / "a_accuracy_vs_T.png", dpi=200)
