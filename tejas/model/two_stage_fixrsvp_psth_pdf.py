from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from tejas.rsvp_util import get_fixrsvp_data
from two_stage_core import TwoStage


SESSION_NAME = os.environ.get("SESSION_NAME", "Allen_2022-04-13")
RUNS_ROOT = Path("/home/tejas/VisionCore/tejas/model/final_runs_4levels")
FIXRSVP_CONFIG_PATH = Path("/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_60_rsvp.yaml")
FIXRSVP_RASTER_CONFIG_PATH = Path("/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_rsvp_all_cells.yaml")
BATCH_SIZE = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PDF_PATH = RUNS_ROOT / SESSION_NAME / f"{SESSION_NAME}_fixrsvp_psths.pdf"
SCALED_PDF_PATH = RUNS_ROOT / SESSION_NAME / f"{SESSION_NAME}_fixrsvp_psths_scaled.pdf"
SUMMARY_CSV_PATH = RUNS_ROOT / SESSION_NAME / f"{SESSION_NAME}_fixrsvp_psth_scale_summary.csv"
DURATION_S = 0.5
RASTER_DT_S = 1.0 / 240.0


def crop_slice(crop_size):
    return slice(int(crop_size), -int(crop_size)) if int(crop_size) > 0 else slice(None)


def get_crop_size(stim_hw, image_shape):
    return max((int(stim_hw) - int(image_shape[-1])) // 2, 0)


def get_target_rate_hz():
    with open(FIXRSVP_CONFIG_PATH, "r") as f:
        return float(yaml.safe_load(f).get("sampling", {}).get("target_rate", 60))




def fit_affine_curve(robs_psth, rhat_psth):
    mask = np.isfinite(robs_psth) & np.isfinite(rhat_psth)
    if not np.any(mask):
        return 1.0, 0.0, np.nan, np.nan, np.nan
    x = np.asarray(rhat_psth[mask], dtype=np.float64)
    y = np.asarray(robs_psth[mask], dtype=np.float64)
    denom = float(np.dot(x, x))
    scale = 1.0 if denom <= 0 else float(np.dot(y, x) / denom)
    offset = float(np.mean(y - scale * x))
    rmse_raw = float(np.sqrt(np.mean((y - x) ** 2)))
    rmse_affine = float(np.sqrt(np.mean((y - (scale * x + offset)) ** 2)))
    corr = np.nan if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0 else float(np.corrcoef(y, x)[0, 1])
    return scale, offset, corr, rmse_raw, rmse_affine


def load_model_for_cell(cell_id):
    ckpt_path = RUNS_ROOT / SESSION_NAME / "checkpoints" / f"cell_{int(cell_id):03d}_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = TwoStage(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return ckpt, model


def load_fixrsvp():
    subject, date = SESSION_NAME.split("_", 1)
    return get_fixrsvp_data(subject, date, str(FIXRSVP_CONFIG_PATH), use_cached_data=True, verbose=False)


def load_fixrsvp_raster():
    subject, date = SESSION_NAME.split("_", 1)
    return get_fixrsvp_data(subject, date, str(FIXRSVP_RASTER_CONFIG_PATH), use_cached_data=True, verbose=False)


def compute_group_fixrsvp_outputs(models_by_cell, fixrsvp, lag_index, image_shape, include_eta=False):
    cell_ids = [int(cell_id) for cell_id in models_by_cell]
    target_rate_hz = get_target_rate_hz()
    robs_trials_all = np.asarray(fixrsvp["robs"][:, :, cell_ids], dtype=np.float32)
    stim_trials_all = np.asarray(fixrsvp["stim"], dtype=np.float32)
    time_s_full = np.arange(robs_trials_all.shape[1], dtype=np.float32) / float(target_rate_hz)
    valid_t = time_s_full <= DURATION_S
    robs_trials = robs_trials_all[:, valid_t, :]
    stim_trials = stim_trials_all[:, valid_t]
    time_s = time_s_full[valid_t]

    stim_hw = int(stim_trials.shape[-1])
    crop_size = get_crop_size(stim_hw, image_shape)
    ys = crop_slice(crop_size)
    xs = crop_slice(crop_size)
    flat_stim = stim_trials[:, :, :, [int(lag_index)], ys, xs].reshape(
        -1,
        stim_trials.shape[2],
        1,
        stim_trials.shape[-2] - 2 * int(crop_size),
        stim_trials.shape[-1] - 2 * int(crop_size),
    )
    valid_flat = np.isfinite(flat_stim).reshape(flat_stim.shape[0], -1).all(axis=1)
    valid_idx = np.where(valid_flat)[0]

    flat_shape = (robs_trials.shape[0], robs_trials.shape[1])
    rhat_trials_by_cell = {}
    eta_trials_by_cell = {} if include_eta else None
    beta_values = {}
    for cell_id in cell_ids:
        model = models_by_cell[int(cell_id)]
        rhat_flat = np.full((flat_stim.shape[0],), np.nan, dtype=np.float32)
        eta_flat = np.full((flat_stim.shape[0],), np.nan, dtype=np.float32) if include_eta else None
        beta_values[int(cell_id)] = float(model.beta.detach().reshape(-1)[0].cpu().item())
        with torch.no_grad():
            for start in range(0, len(valid_idx), BATCH_SIZE):
                batch_idx = valid_idx[start:start + BATCH_SIZE]
                out = model({"stim": torch.from_numpy(flat_stim[batch_idx]).to(DEVICE)})
                rhat_flat[batch_idx] = out["rhat"].detach().cpu().numpy().reshape(-1)
                if include_eta:
                    eta_flat[batch_idx] = out["eta"].detach().cpu().numpy().reshape(-1)
        rhat_trials_by_cell[int(cell_id)] = rhat_flat.reshape(flat_shape)
        if include_eta:
            eta_trials_by_cell[int(cell_id)] = eta_flat.reshape(flat_shape)

    results = {}
    for cell_idx, cell_id in enumerate(cell_ids):
        cell_robs_trials = robs_trials[:, :, cell_idx]
        cell_rhat_trials = rhat_trials_by_cell[int(cell_id)]
        result = {
            "time_s": time_s,
            "robs_trials": np.where(np.isfinite(cell_robs_trials), cell_robs_trials, np.nan),
            "rhat_trials": np.where(np.isfinite(cell_rhat_trials), cell_rhat_trials, np.nan),
            "n_trials_used": int(np.sum(np.any(np.isfinite(cell_rhat_trials), axis=1))),
            "window_start_s": 0.0,
            "beta_value": beta_values[int(cell_id)],
        }
        if include_eta:
            eta_trials = eta_trials_by_cell[int(cell_id)]
            result["eta_trials"] = np.where(np.isfinite(eta_trials), eta_trials, np.nan)
            result["z_trials"] = result["eta_trials"] - float(beta_values[int(cell_id)])
        results[int(cell_id)] = result
    return results


def extract_spike_raster_trials(fixrsvp_60, fixrsvp_240, cell_id, duration_s=DURATION_S, dt=RASTER_DT_S, window_start_s=0.0):
    trial_count_60 = int(fixrsvp_60["robs"].shape[0])
    trial_count_240 = int(fixrsvp_240["robs"].shape[0])
    if trial_count_60 != trial_count_240:
        import warnings
        warnings.warn(f"Trial count mismatch between 60 Hz ({trial_count_60}) and 240 Hz ({trial_count_240}) — skipping raster for cell {cell_id}")
        cids_60 = np.asarray(fixrsvp_60["cids"], dtype=int)
        return None, int(cids_60[int(cell_id)])

    cids_60 = np.asarray(fixrsvp_60["cids"], dtype=int)
    cids_240 = np.asarray(fixrsvp_240["cids"], dtype=int)
    true_cid = int(cids_60[int(cell_id)])
    match = np.where(cids_240 == true_cid)[0]
    if match.size == 0:
        return None, true_cid

    raster_cell_idx = int(match[0])
    spike_times_trials = fixrsvp_240.get("spike_times_trials", None)
    trial_t_bins = fixrsvp_240.get("trial_t_bins", None)
    if spike_times_trials is None or trial_t_bins is None:
        return None, true_cid

    raster_trials = []
    for trial_idx in range(trial_count_60):
        trial_bins = np.asarray(trial_t_bins[trial_idx], dtype=np.float32)
        valid_bins = np.isfinite(trial_bins)
        if not np.any(valid_bins):
            raster_trials.append(np.array([], dtype=np.float32))
            continue
        window_start = float(trial_bins[valid_bins][0] - dt / 2.0 + float(window_start_s))
        window_end = window_start + float(duration_s)
        spikes_abs = np.asarray(spike_times_trials[trial_idx][raster_cell_idx], dtype=np.float32)
        keep = (spikes_abs >= window_start) & (spikes_abs < window_end)
        raster_trials.append(spikes_abs[keep] - window_start)
    return raster_trials, true_cid


def plot_spike_raster(ax, spike_times_trials, duration_s=DURATION_S):
    if not spike_times_trials:
        ax.axis("off")
        return

    x_list = []
    y_list = []
    for trial_idx, spikes in enumerate(spike_times_trials):
        spikes = np.atleast_1d(np.asarray(spikes, dtype=np.float32))
        if spikes.size == 0:
            continue
        for spike_t in spikes:
            x_list.extend([spike_t, spike_t, np.nan])
            y_list.extend([trial_idx, trial_idx + 0.8, np.nan])

    if x_list:
        ax.plot(x_list, y_list, color="black", linewidth=0.35, rasterized=True)
    ax.set_xlim(0.0, float(duration_s))
    ax.set_ylim(len(spike_times_trials), 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def style_overlay_axes(raster_ax, rate_ax, n_trials, duration_s=DURATION_S):
    raster_ax.set_xlim(0.0, float(duration_s))
    raster_ax.set_ylim(n_trials, 0)
    raster_ax.set_xticks([])
    raster_ax.set_yticks([])
    raster_ax.spines["top"].set_visible(False)
    raster_ax.spines["right"].set_visible(False)
    raster_ax.spines["left"].set_visible(False)

    rate_ax.set_xlim(0.0, float(duration_s))
    rate_ax.set_xticks([])
    rate_ax.set_yticks([])
    rate_ax.spines["top"].set_visible(False)
    rate_ax.spines["left"].set_visible(False)
    rate_ax.spines["right"].set_visible(False)


def save_pdf(cell_ids, best_df):
    fixrsvp = load_fixrsvp()
    fixrsvp_raster = load_fixrsvp_raster()
    best_val_bps_map = {int(row.cell_id): float(row.best_val_bps) for row in best_df.itertuples(index=False)}

    lag_groups = {}
    for cell_id in cell_ids:
        ckpt_path = RUNS_ROOT / SESSION_NAME / "checkpoints" / f"cell_{int(cell_id):03d}_best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        lag_groups.setdefault(int(ckpt["lag_index"]), []).append((int(cell_id), ckpt))

    plot_rows = []
    for lag_index, group in sorted(lag_groups.items()):
        image_shape = tuple(int(x) for x in group[0][1]["model_kwargs"]["image_shape"])
        models_by_cell = {}
        for cell_id, ckpt in group:
            model = TwoStage(**ckpt["model_kwargs"])
            model.load_state_dict(ckpt["model_state"])
            model.to(DEVICE)
            model.eval()
            models_by_cell[int(cell_id)] = model

        group_outputs = compute_group_fixrsvp_outputs(
            models_by_cell=models_by_cell,
            fixrsvp=fixrsvp,
            lag_index=int(lag_index),
            image_shape=image_shape,
            include_eta=False,
        )

        for cell_id, ckpt in group:
            out = group_outputs[int(cell_id)]
            robs_psth = np.nanmean(out["robs_trials"], axis=0)
            rhat_psth = np.nanmean(out["rhat_trials"], axis=0)
            scale, offset, corr, rmse_raw, rmse_affine = fit_affine_curve(robs_psth, rhat_psth)
            raster_trials, true_cid = extract_spike_raster_trials(
                fixrsvp,
                fixrsvp_raster,
                int(cell_id),
                window_start_s=float(out["window_start_s"]),
            )
            plot_rows.append(
                {
                    "cell_id": int(cell_id),
                    "true_cid": int(true_cid),
                    "lag_index": int(lag_index),
                    "time_s": out["time_s"],
                    "robs_psth": robs_psth,
                    "rhat_psth": rhat_psth,
                    "scaled_rhat_psth": scale * rhat_psth + offset,
                    "raster_trials": raster_trials,
                    "n_trials_used": int(out["n_trials_used"]),
                    "window_start_s": float(out["window_start_s"]),
                    "scale_a": float(scale),
                    "offset_b": float(offset),
                    "psth_corr": float(corr),
                    "rmse_raw": float(rmse_raw),
                    "rmse_affine": float(rmse_affine),
                    "best_val_bps": float(best_val_bps_map[int(cell_id)]),
                }
            )

        for model in models_by_cell.values():
            del model
        torch.cuda.empty_cache()

    plot_rows.sort(
        key=lambda row: (
            np.isfinite(row["psth_corr"]),
            row["psth_corr"] if np.isfinite(row["psth_corr"]) else -np.inf,
        ),
        reverse=True,
    )

    per_page = 25
    ncols = 5
    nrows = 5
    plt.rcParams["pdf.compression"] = 9
    summary_rows = []

    with PdfPages(PDF_PATH) as pdf, PdfPages(SCALED_PDF_PATH) as scaled_pdf:
        fig = axes = scaled_fig = scaled_axes = None
        for idx, row in enumerate(plot_rows):
            page_pos = idx % per_page
            if page_pos == 0:
                if fig is not None:
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight", dpi=80)
                    plt.close(fig)
                    scaled_fig.tight_layout()
                    scaled_pdf.savefig(scaled_fig, bbox_inches="tight", dpi=80)
                    plt.close(scaled_fig)
                fig, axes = plt.subplots(nrows, ncols, figsize=(14, 14), sharex=False, sharey=False)
                axes = axes.reshape(-1)
                scaled_fig, scaled_axes = plt.subplots(nrows, ncols, figsize=(14, 14), sharex=False, sharey=False)
                scaled_axes = scaled_axes.reshape(-1)

            ax = axes[page_pos]
            scaled_ax = scaled_axes[page_pos]
            ax.set_axis_off()
            scaled_ax.set_axis_off()
            raster_ax = ax.inset_axes([0.08, 0.12, 0.88, 0.72])
            psth_ax = raster_ax.twinx()
            scaled_raster_ax = scaled_ax.inset_axes([0.08, 0.12, 0.88, 0.72])
            scaled_psth_ax = scaled_raster_ax.twinx()
            plot_spike_raster(raster_ax, row["raster_trials"], duration_s=DURATION_S)
            plot_spike_raster(scaled_raster_ax, row["raster_trials"], duration_s=DURATION_S)
            time_s = row["time_s"]
            robs_psth = np.asarray(row["robs_psth"], dtype=np.float32)
            rhat_psth = np.asarray(row["rhat_psth"], dtype=np.float32)
            scaled_psth = np.asarray(row["scaled_rhat_psth"], dtype=np.float32)
            valid_robs = np.isfinite(robs_psth)
            valid_rhat = np.isfinite(rhat_psth)
            valid_scaled = np.isfinite(scaled_psth)
            if np.any(valid_robs):
                psth_ax.plot(time_s[valid_robs], robs_psth[valid_robs], color="black", linewidth=0.8, rasterized=True)
                scaled_psth_ax.plot(time_s[valid_robs], robs_psth[valid_robs], color="black", linewidth=0.8, rasterized=True)
            if np.any(valid_rhat):
                psth_ax.plot(time_s[valid_rhat], rhat_psth[valid_rhat], color="red", linewidth=0.8, rasterized=True)
            if np.any(valid_scaled):
                scaled_psth_ax.plot(time_s[valid_scaled], scaled_psth[valid_scaled], color="red", linewidth=0.8, rasterized=True)

            ax.set_title(f"cell {int(row['cell_id']):03d} | unit {int(row['true_cid'])} | bps={row['best_val_bps']:.3f}", fontsize=8, pad=2)
            scaled_ax.set_title(f"cell {int(row['cell_id']):03d} | unit {int(row['true_cid'])} | bps={row['best_val_bps']:.3f} | a={row['scale_a']:.2f}", fontsize=7, pad=2)
            style_overlay_axes(raster_ax, psth_ax, len(row["raster_trials"]) if row["raster_trials"] is not None else 0, duration_s=DURATION_S)
            style_overlay_axes(scaled_raster_ax, scaled_psth_ax, len(row["raster_trials"]) if row["raster_trials"] is not None else 0, duration_s=DURATION_S)

            summary_rows.append(
                {
                    "cell_id": int(row["cell_id"]),
                    "true_cid": int(row["true_cid"]),
                    "best_val_bps": float(row["best_val_bps"]),
                    "lag_index": int(row["lag_index"]),
                    "n_trials_used": int(row["n_trials_used"]),
                    "t_start_s": float(row["time_s"][0]) if row["time_s"].size else np.nan,
                    "t_end_s": float(row["time_s"][-1]) if row["time_s"].size else np.nan,
                    "scale_a": float(row["scale_a"]),
                    "offset_b": float(row["offset_b"]),
                    "psth_corr": float(row["psth_corr"]),
                    "rmse_raw": float(row["rmse_raw"]),
                    "rmse_affine": float(row["rmse_affine"]),
                }
            )

        if fig is not None:
            used = (len(plot_rows) - 1) % per_page + 1
            for ax in axes[used:]:
                ax.axis("off")
            for ax in scaled_axes[used:]:
                ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight", dpi=80)
            plt.close(fig)
            scaled_fig.tight_layout()
            scaled_pdf.savefig(scaled_fig, bbox_inches="tight", dpi=80)
            plt.close(scaled_fig)

    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV_PATH, index=False)


def main():
    best_df = pd.read_csv(RUNS_ROOT / SESSION_NAME / "best_val_bps.csv").sort_values("cell_id").reset_index(drop=True)
    cell_ids = best_df["cell_id"].astype(int).tolist()
    save_pdf(cell_ids, best_df)
    print(f"saved pdf: {PDF_PATH}")
    print(f"saved scaled pdf: {SCALED_PDF_PATH}")
    print(f"saved summary csv: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
