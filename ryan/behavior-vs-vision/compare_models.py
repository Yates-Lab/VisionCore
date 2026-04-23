"""
Compare behavior vs vision-only digital twin models.

Figure 1: LLR histograms (bps_behavior - bps_vision) — 2 panels (Allen, Logan),
each with 3 overlaid histograms for backimage / gaborium / gratings.

Figures 2–7: Perisaccadic response imshows — one figure per (subject x stim),
3 panels each (data, behavior, vision). Rows are cells sorted by peak time of
the data response; values are baseline-subtracted, peak-normalized mean
responses.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from VisionCore.paths import FIGURES_DIR
from eval.eval_stack_multidataset import evaluate_model_multidataset


BEHAVIOR_DIR = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120/"
    "2026-03-31_11-33-32_learned_resnet_concat_convgru_gaussian/"
    "learned_resnet_concat_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4"
)
VISION_DIR = Path(
    "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/digital_twin_120/"
    "2026-04-13_11-10-15_learned_resnet_none_convgru_gaussian/"
    "learned_resnet_none_convgru_gaussian_lr1e-3_wd1e-5_cls1.0_bs256_ga4"
)

FIG_DIR = FIGURES_DIR / "behavior-vs-vision"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STIM_TYPES = ["backimage", "gaborium", "gratings"]
STIM_LABELS = {
    "backimage": "natural images",
    "gaborium": "gaborium",
    "gratings": "flashed gratings",
}
DT_MS = 1000.0 / 120.0


def find_best_ckpt(ckpt_dir: Path) -> Path:
    ckpts = list(ckpt_dir.glob("epoch=*-val_bps_overall=*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No val_bps_overall checkpoints in {ckpt_dir}")
    return max(ckpts, key=lambda p: float(p.stem.split("val_bps_overall=")[1]))


def load_results(ckpt_path: Path, model_type: str):
    return evaluate_model_multidataset(
        model_type=model_type,
        checkpoint_path=str(ckpt_path),
        analyses=["bps", "saccade"],
        recalc=False,
        batch_size=64,
    )


def split_subject_indices(dataset_names):
    allen = [i for i, n in enumerate(dataset_names) if n.startswith("Allen")]
    logan = [i for i, n in enumerate(dataset_names) if n.startswith("Logan")]
    return {"Allen": allen, "Logan": logan}


def concat_bps(res, model_type, stim, ds_idxs):
    parts = [np.asarray(res[model_type]["bps"][stim]["bps"][i]) for i in ds_idxs]
    return np.concatenate(parts)


def concat_rbar(res, model_type, stim, ds_idxs, key):
    """Concatenate (T, n_cells) arrays across datasets along axis=1."""
    parts = []
    for i in ds_idxs:
        arr = np.asarray(res[model_type]["saccade"][stim][key][i])
        parts.append(arr)
    return np.concatenate(parts, axis=1)


def _bs_peak_norm(R, t_ms, pre_window=(-80, 0)):
    """Baseline-subtract using pre-saccade window then divide by abs-peak."""
    R = R.astype(np.float64).copy()
    R = gaussian_filter1d(R, sigma=1.0, axis=0, mode="nearest")
    base_mask = (t_ms >= pre_window[0]) & (t_ms < pre_window[1])
    if base_mask.sum() == 0:
        base_mask = t_ms < 0
    baseline = np.nanmean(R[base_mask], axis=0)
    R = R - baseline[np.newaxis, :]
    peak = np.nanmax(np.abs(R), axis=0)
    peak[peak == 0] = 1.0
    return R / peak[np.newaxis, :]


def peak_time_order(R_norm, t_ms, window=(0, 200)):
    mask = (t_ms >= window[0]) & (t_ms <= window[1])
    masked = np.where(mask[:, None], R_norm, -np.inf)
    peak_idx = np.argmax(masked, axis=0)
    return np.argsort(peak_idx)


# ---------------------------------------------------------------- run analyses

beh_ckpt = find_best_ckpt(BEHAVIOR_DIR)
vis_ckpt = find_best_ckpt(VISION_DIR)
print(f"Behavior ckpt:    {beh_ckpt.name}")
print(f"Vision-only ckpt: {vis_ckpt.name}")

beh_res = load_results(beh_ckpt, "behavior")
vis_res = load_results(vis_ckpt, "vision")

dataset_names = beh_res["behavior"]["dataset_names"]
assert dataset_names == vis_res["vision"]["dataset_names"], (
    "Datasets differ between runs — cannot align cells."
)
subj_idx = split_subject_indices(dataset_names)
print(f"Allen sessions: {len(subj_idx['Allen'])}, Logan sessions: {len(subj_idx['Logan'])}")


#%% ---------------------------------------------------------------- Figure 1

fig1, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
bins = np.linspace(-0.2, 0.5, 31)
colors = {"backimage": "#1f77b4", "gaborium": "#ff7f0e", "gratings": "#2ca02c"}

for ax, subject in zip(axes, ["Allen", "Logan"]):
    idxs = subj_idx[subject]
    medians = {}
    max_count = 0
    for stim in STIM_TYPES:
        bps_b = concat_bps(beh_res, "behavior", stim, idxs)
        bps_v = concat_bps(vis_res, "vision", stim, idxs)
        diff = bps_b - bps_v
        diff = diff[np.isfinite(diff)]
        counts, _, _ = ax.hist(
            diff,
            bins=bins,
            alpha=0.5,
            color=colors[stim],
            label=f"{STIM_LABELS[stim]} (n={len(diff)})",
        )
        medians[stim] = np.median(diff)
        max_count = max(max_count, counts.max())
    y_tri = max_count * 1.05
    for stim in STIM_TYPES:
        ax.plot(
            medians[stim],
            y_tri,
            marker="v",
            color=colors[stim],
            markersize=10,
            markeredgecolor="k",
            markeredgewidth=0.5,
            clip_on=False,
        )
    ax.axvline(0, ls="--", color="k", lw=1)
    ax.set_xlim(-0.2, 0.5)
    ax.set_xlabel(r"$\Delta$BPS (behavior − vision)")
    ax.set_title(subject)
    ax.legend(loc="upper right", fontsize=9)
axes[0].set_ylabel("cell count")

fig1.suptitle("Validation-set LLR: behavior vs. vision-only models", y=1.02)
fig1.tight_layout()
fig1.savefig(FIG_DIR / "fig1_llr_histograms.pdf", bbox_inches="tight")
fig1.savefig(FIG_DIR / "fig1_llr_histograms.png", dpi=150, bbox_inches="tight")
print(f"Saved {FIG_DIR / 'fig1_llr_histograms.pdf'}")


#%% ---------------------------------------------------------------- Figures 2–7

# All datasets share the same saccade window (set in eval_stack_multidataset.py)
win_bins = beh_res["behavior"]["saccade"][STIM_TYPES[0]]["win"][0]
t_ms = np.arange(win_bins[0], win_bins[1]) * DT_MS


def make_perisaccadic_figure(subject, stim):
    idxs = subj_idx[subject]

    rbar_data = concat_rbar(beh_res, "behavior", stim, idxs, "rbar")
    rbarhat_beh = concat_rbar(beh_res, "behavior", stim, idxs, "rbarhat")
    rbarhat_vis = concat_rbar(vis_res, "vision", stim, idxs, "rbarhat")

    Rd = _bs_peak_norm(rbar_data, t_ms)
    Rb = _bs_peak_norm(rbarhat_beh, t_ms)
    Rv = _bs_peak_norm(rbarhat_vis, t_ms)

    valid = (
        np.isfinite(Rd).all(axis=0)
        & np.isfinite(Rb).all(axis=0)
        & np.isfinite(Rv).all(axis=0)
    )
    Rd, Rb, Rv = Rd[:, valid], Rb[:, valid], Rv[:, valid]
    if Rd.shape[1] == 0:
        print(f"  {subject} / {stim}: no valid cells — skipping")
        return

    order = peak_time_order(Rd, t_ms)
    Rd, Rb, Rv = Rd[:, order], Rb[:, order], Rv[:, order]

    fig, axs = plt.subplots(1, 3, figsize=(13, 6), sharey=True, sharex=True)
    panels = [("Data", Rd), ("Behavior model", Rb), ("Vision only", Rv)]
    extent = [t_ms[0], t_ms[-1], 0, Rd.shape[1]]
    vmin, vmax = -1.0, 1.0
    for ax, (title, R) in zip(axs, panels):
        im = ax.imshow(
            R.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        ax.axvline(0, ls="--", color="k", lw=1)
        ax.set_xlim(-200, 400)
        ax.set_title(title)
        ax.set_xlabel("Time from saccade (ms)")
    axs[0].set_ylabel("Cell (sorted by peak time of data)")
    cbar = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02)
    cbar.set_label("normalized response")
    fig.suptitle(f"{subject} — {STIM_LABELS[stim]}  (n={Rd.shape[1]} cells)")

    fname = FIG_DIR / f"fig_perisaccadic_{subject}_{stim}"
    fig.savefig(f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}.pdf  ({Rd.shape[1]} cells)")


for subject in ["Allen", "Logan"]:
    for stim in STIM_TYPES:
        make_perisaccadic_figure(subject, stim)

print("Done.")
