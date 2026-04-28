from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from models.config_loader import load_dataset_configs
from models.data import prepare_data
from models.losses import PoissonBPSAggregator
from tejas.rsvp_util import get_fixrsvp_data
from two_stage_core import TwoStage
from two_stage_trainer import eval_step


SESSION_NAME = "Allen_2022-04-13"
CELL_ID = 14
RUNS_ROOT = Path("/home/tejas/VisionCore/tejas/model/final_runs_4levels")
GABORIUM_CONFIG_PATH = Path("/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml")
FIXRSVP_CONFIG_PATH = Path("/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_60_rsvp.yaml")
BATCH_SIZE = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_PATH = RUNS_ROOT / SESSION_NAME / f"cell_{CELL_ID:03d}_fixrsvp_playground.npz"
PSTH_PNG_PATH = RUNS_ROOT / SESSION_NAME / f"cell_{CELL_ID:03d}_fixrsvp_psth_overlay.png"
PSTH_SCALED_PNG_PATH = RUNS_ROOT / SESSION_NAME / f"cell_{CELL_ID:03d}_fixrsvp_psth_overlay_scaled.png"


def crop_slice(crop_size):
    return slice(int(crop_size), -int(crop_size)) if int(crop_size) > 0 else slice(None)


def prepare_batch_one_lag(batch, lag_index, crop_size, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    ys = crop_slice(crop_size)
    xs = crop_slice(crop_size)
    batch["stim"] = batch["stim"][:, :, [int(lag_index)], ys, xs]
    return batch


def get_session_config(parent_config_path, session_name):
    configs = load_dataset_configs(parent_config_path)
    for cfg in configs:
        if cfg["session"] == session_name:
            return deepcopy(cfg)
    raise KeyError(f"Could not find config for {session_name}")


def get_crop_size(stim_hw, image_shape):
    return max((int(stim_hw) - int(image_shape[-1])) // 2, 0)


def get_target_rate_hz():
    with open(FIXRSVP_CONFIG_PATH, "r") as f:
        return float(yaml.safe_load(f).get("sampling", {}).get("target_rate", 60))


def load_checkpoint():
    ckpt_path = RUNS_ROOT / SESSION_NAME / "checkpoints" / f"cell_{CELL_ID:03d}_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = TwoStage(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return ckpt_path, ckpt, model


def compute_gaborium_val_bps(model, lag_index, image_shape):
    cfg = get_session_config(GABORIUM_CONFIG_PATH, SESSION_NAME)
    _, val_dset, _ = prepare_data(cfg, strict=False)
    stim_hw = int(val_dset.dsets[int(val_dset.inds[0, 0])]["stim"].shape[-1])
    crop_size = get_crop_size(stim_hw, image_shape)
    loader = DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    agg = PoissonBPSAggregator(device="cpu")
    with torch.no_grad():
        for batch in loader:
            batch = prepare_batch_one_lag(batch, lag_index=lag_index, crop_size=crop_size, device=DEVICE)
            out = eval_step(model=model, batch=batch, cell_ids=[CELL_ID], use_resolver=True)
            agg(out)
    return float(agg.closure().cpu().numpy().reshape(-1)[0]), crop_size


def compute_fixrsvp_trial_predictions(model, lag_index, image_shape):
    subject, date = SESSION_NAME.split("_", 1)
    fixrsvp = get_fixrsvp_data(
        subject,
        date,
        str(FIXRSVP_CONFIG_PATH),
        use_cached_data=True,
        verbose=False,
    )
    robs_trials = np.asarray(fixrsvp["robs"][:, :, CELL_ID], dtype=np.float32)
    dfs_trials = np.asarray(fixrsvp["dfs"][:, :, CELL_ID], dtype=np.float32)
    eyepos_trials = np.asarray(fixrsvp["eyepos"], dtype=np.float32)
    stim_trials = np.asarray(fixrsvp["stim"], dtype=np.float32)

    target_rate_hz = get_target_rate_hz()
    crop_size = get_crop_size(stim_trials.shape[-1], image_shape)
    ys = crop_slice(crop_size)
    xs = crop_slice(crop_size)

    flat_stim = stim_trials[:, :, :, [int(lag_index)], ys, xs].reshape(
        -1,
        stim_trials.shape[2],
        1,
        stim_trials.shape[-2] - 2 * int(crop_size),
        stim_trials.shape[-1] - 2 * int(crop_size),
    )
    flat_robs = robs_trials.reshape(-1, 1)
    flat_dfs = dfs_trials.reshape(-1, 1)
    valid_flat = np.isfinite(flat_stim).reshape(flat_stim.shape[0], -1).all(axis=1)

    rhat_flat = np.full((flat_stim.shape[0],), np.nan, dtype=np.float32)
    with torch.no_grad():
        valid_idx = np.where(valid_flat)[0]
        for start in range(0, len(valid_idx), BATCH_SIZE):
            batch_idx = valid_idx[start:start + BATCH_SIZE]
            batch = {
                "stim": torch.from_numpy(flat_stim[batch_idx]).to(DEVICE),
                "robs": torch.from_numpy(flat_robs[batch_idx]).to(DEVICE),
                "dfs": torch.from_numpy(flat_dfs[batch_idx]).to(DEVICE),
            }
            out = eval_step(model=model, batch=batch, cell_ids=[0], use_resolver=False)
            rhat_flat[batch_idx] = out["rhat"].detach().cpu().numpy().reshape(-1)

    n_trials, n_time = robs_trials.shape
    return {
        "t_trial_s": np.arange(n_time, dtype=np.float32) / float(target_rate_hz),
        "robs_trials": robs_trials,
        "rhat_trials": rhat_flat.reshape(n_trials, n_time),
        "dfs_trials": dfs_trials,
        "eyepos_trials": eyepos_trials,
        "fix_dur": np.asarray(fixrsvp["fix_dur"], dtype=np.int64),
        "lag_index": int(lag_index),
        "crop_size": int(crop_size),
    }


def fit_affine_curve(robs_psth, rhat_psth):
    mask = np.isfinite(robs_psth) & np.isfinite(rhat_psth)
    if not np.any(mask):
        return 1.0, 0.0
    x = np.asarray(rhat_psth[mask], dtype=np.float64)
    y = np.asarray(robs_psth[mask], dtype=np.float64)
    denom = float(np.dot(x, x))
    scale = 1.0 if denom <= 0 else float(np.dot(y, x) / denom)
    offset = float(np.mean(y - scale * x))
    return scale, offset


def save_fixrsvp_psth_plot(fixrsvp):
    robs_masked = np.where(np.isfinite(fixrsvp["robs_trials"]), fixrsvp["robs_trials"], np.nan)
    rhat_masked = np.where(np.isfinite(fixrsvp["rhat_trials"]), fixrsvp["rhat_trials"], np.nan)
    valid_counts = np.sum(np.isfinite(robs_masked) & np.isfinite(rhat_masked), axis=0)
    robs_psth = np.full((robs_masked.shape[1],), np.nan, dtype=np.float32)
    rhat_psth = np.full((rhat_masked.shape[1],), np.nan, dtype=np.float32)
    valid_bins = valid_counts > 0
    robs_psth[valid_bins] = np.nanmean(robs_masked[:, valid_bins], axis=0)
    rhat_psth[valid_bins] = np.nanmean(rhat_masked[:, valid_bins], axis=0)
    valid = np.isfinite(robs_psth) & np.isfinite(rhat_psth)
    x = fixrsvp["t_trial_s"][valid]
    scale, offset = fit_affine_curve(robs_psth, rhat_psth)
    scaled_rhat_psth = scale * rhat_psth + offset

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, robs_psth[valid], label="robs", color="black", linewidth=2)
    ax.plot(x, rhat_psth[valid], label="rhat", color="red", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean response")
    ax.set_title(f"{SESSION_NAME} cell {CELL_ID} FixRSVP PSTH")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PSTH_PNG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, robs_psth[valid], label="robs", color="black", linewidth=2)
    ax.plot(x, scaled_rhat_psth[valid], label=f"scaled rhat (a={scale:.2f})", color="red", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean response")
    ax.set_title(f"{SESSION_NAME} cell {CELL_ID} FixRSVP PSTH (scaled)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PSTH_SCALED_PNG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ckpt_path, ckpt, model = load_checkpoint()
    lag_index = int(ckpt["lag_index"])
    image_shape = tuple(int(x) for x in ckpt["model_kwargs"]["image_shape"])

    recorded_val_bps = float(ckpt["best_val_bps"])
    gaborium_val_bps, gaborium_crop_size = compute_gaborium_val_bps(model, lag_index=lag_index, image_shape=image_shape)
    fixrsvp = compute_fixrsvp_trial_predictions(model, lag_index=lag_index, image_shape=image_shape)
    save_fixrsvp_psth_plot(fixrsvp)

    best_df = pd.read_csv(RUNS_ROOT / SESSION_NAME / "best_val_bps.csv")
    recorded_csv_bps = float(best_df.loc[best_df["cell_id"] == CELL_ID, "best_val_bps"].iloc[0])

    np.savez_compressed(
        OUT_PATH,
        session_name=SESSION_NAME,
        cell_id=np.int64(CELL_ID),
        checkpoint_path=str(ckpt_path),
        recorded_val_bps=np.float64(recorded_val_bps),
        recorded_csv_bps=np.float64(recorded_csv_bps),
        recomputed_val_bps=np.float64(gaborium_val_bps),
        gaborium_crop_size=np.int64(gaborium_crop_size),
        **fixrsvp,
    )

    print(f"checkpoint: {ckpt_path}")
    print(f"recorded checkpoint val bps: {recorded_val_bps:.9f}")
    print(f"recorded csv val bps:        {recorded_csv_bps:.9f}")
    print(f"recomputed val bps:          {gaborium_val_bps:.9f}")
    print(f"saved: {OUT_PATH}")
    print(f"psth plot: {PSTH_PNG_PATH}")
    print(f"fixrsvp rhat shape: {fixrsvp['rhat_trials'].shape}")


if __name__ == "__main__":
    main()
