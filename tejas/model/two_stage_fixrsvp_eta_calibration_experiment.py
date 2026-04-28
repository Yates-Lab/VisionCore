from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import torch

from two_stage_core import TwoStage
from two_stage_fixrsvp_psth_pdf import load_fixrsvp, compute_group_fixrsvp_outputs, DEVICE, RUNS_ROOT


SESSION_NAME = os.environ.get("SESSION_NAME", "Allen_2022-04-13")
CELL_IDS = [16, 76]
DURATION_S = 0.5
SUMMARY_CSV = RUNS_ROOT / SESSION_NAME / "fixrsvp_eta_calibration_summary.csv"


def load_checkpoint(cell_id):
    ckpt_path = RUNS_ROOT / SESSION_NAME / "checkpoints" / f"cell_{int(cell_id):03d}_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = TwoStage(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return ckpt, model


def compute_trial_outputs(model, fixrsvp, lag_index, image_shape, cell_id):
    out = compute_group_fixrsvp_outputs(
        {int(cell_id): model}, fixrsvp, int(lag_index), image_shape, include_eta=True
    )[int(cell_id)]
    return out


def fit_eta_calibration(robs_trials, z_trials):
    valid = np.isfinite(robs_trials) & np.isfinite(z_trials)
    y = np.asarray(robs_trials[valid], dtype=np.float64)
    z = np.asarray(z_trials[valid], dtype=np.float64)
    sum_y = float(np.sum(y))

    def bias_for_scale(scale):
        scaled_z = scale * z
        max_scaled = float(np.max(scaled_z))
        sum_exp = float(np.exp(scaled_z - max_scaled).sum() * np.exp(max_scaled))
        return float(np.log(sum_y / sum_exp))

    def objective(log_scale):
        scale = float(np.exp(log_scale))
        bias = bias_for_scale(scale)
        eta = bias + scale * z
        return float(np.mean(np.exp(eta) - y * eta))

    result = minimize_scalar(objective, bounds=(-6.0, 6.0), method="bounded", options={"xatol": 1e-6, "maxiter": 500})
    scale = float(np.exp(result.x))
    bias = bias_for_scale(scale)
    rhat = np.exp(bias + scale * np.asarray(z_trials, dtype=np.float64)).astype(np.float32)
    return scale, bias, rhat


def compute_psth(trials):
    trials = np.asarray(trials, dtype=np.float32)
    valid = np.isfinite(trials)
    counts = valid.sum(axis=0)
    psth = np.full((trials.shape[1],), np.nan, dtype=np.float32)
    good = counts > 0
    psth[good] = np.nanmean(trials[:, good], axis=0)
    return psth


def corr_and_rmse(target, pred):
    mask = np.isfinite(target) & np.isfinite(pred)
    if np.sum(mask) < 3:
        return np.nan, np.nan
    corr = np.nan if np.std(target[mask]) == 0.0 or np.std(pred[mask]) == 0.0 else float(np.corrcoef(target[mask], pred[mask])[0, 1])
    rmse = float(np.sqrt(np.mean((target[mask] - pred[mask]) ** 2)))
    return corr, rmse


def save_plot(cell_id, time_s, robs_psth, base_psth, calib_psth, scale, bias):
    out_path = RUNS_ROOT / SESSION_NAME / f"cell_{int(cell_id):03d}_fixrsvp_eta_calibration.png"
    valid = np.isfinite(robs_psth) & np.isfinite(base_psth) & np.isfinite(calib_psth)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time_s[valid], robs_psth[valid], color="black", linewidth=2, label="robs")
    ax.plot(time_s[valid], base_psth[valid], color="red", linewidth=2, alpha=0.8, label="baseline rhat")
    ax.plot(time_s[valid], calib_psth[valid], color="dodgerblue", linewidth=2, alpha=0.9, label="eta-calibrated rhat")
    ax.set_title(f"{SESSION_NAME} cell {int(cell_id):03d} FixRSVP eta calibration")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean response")
    ax.legend(frameon=False, fontsize=9, title=f"scale={scale:.2f}, bias={bias:.3f}", title_fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    fixrsvp = load_fixrsvp()
    rows = []
    for cell_id in CELL_IDS:
        ckpt, model = load_checkpoint(cell_id)
        image_shape = tuple(int(x) for x in ckpt["model_kwargs"]["image_shape"])
        out = compute_trial_outputs(model, fixrsvp, lag_index=int(ckpt["lag_index"]), image_shape=image_shape, cell_id=cell_id)
        scale, bias_value, rhat_calib_trials = fit_eta_calibration(
            robs_trials=out["robs_trials"],
            z_trials=out["z_trials"],
        )
        robs_psth = compute_psth(out["robs_trials"])
        base_psth = compute_psth(out["rhat_trials"])
        calib_psth = compute_psth(rhat_calib_trials)
        corr_base, rmse_base = corr_and_rmse(robs_psth, base_psth)
        corr_calib, rmse_calib = corr_and_rmse(robs_psth, calib_psth)
        plot_path = save_plot(
            cell_id=cell_id,
            time_s=out["time_s"],
            robs_psth=robs_psth,
            base_psth=base_psth,
            calib_psth=calib_psth,
            scale=scale,
            bias=bias_value,
        )
        rows.append(
            {
                "cell_id": int(cell_id),
                "lag_index": int(ckpt["lag_index"]),
                "beta_base": float(out["beta_value"]),
                "scale_fit": float(scale),
                "bias_fit": float(bias_value),
                "robs_psth_min": float(np.nanmin(robs_psth)),
                "base_psth_min": float(np.nanmin(base_psth)),
                "calib_psth_min": float(np.nanmin(calib_psth)),
                "robs_psth_max": float(np.nanmax(robs_psth)),
                "base_psth_max": float(np.nanmax(base_psth)),
                "calib_psth_max": float(np.nanmax(calib_psth)),
                "corr_base": float(corr_base),
                "corr_calib": float(corr_calib),
                "rmse_base": float(rmse_base),
                "rmse_calib": float(rmse_calib),
                "plot_path": str(plot_path),
            }
        )
        print(
            f"cell {cell_id:03d}: scale={scale:.4f}, bias={bias_value:.5f}, "
            f"corr {corr_base:.4f}->{corr_calib:.4f}, rmse {rmse_base:.4f}->{rmse_calib:.4f}, plot={plot_path}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"saved summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
