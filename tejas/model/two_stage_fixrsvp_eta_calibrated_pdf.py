from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import torch

from util import get_dataset_info
from two_stage_fixrsvp_eta_calibration_experiment import (
    DURATION_S,
    SESSION_NAME,
    compute_psth,
    corr_and_rmse,
    fit_eta_calibration,
    load_checkpoint,
)
from two_stage_fixrsvp_psth_pdf import (
    RUNS_ROOT,
    compute_group_fixrsvp_outputs,
    extract_spike_raster_trials,
    load_fixrsvp,
    load_fixrsvp_raster,
    plot_spike_raster,
    style_overlay_axes,
)


SESSION_NAME = os.environ.get("SESSION_NAME", SESSION_NAME)
GABORIUM_CONFIG_PATH = Path("/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml")
PDF_PATH = RUNS_ROOT / SESSION_NAME / f"{SESSION_NAME}_fixrsvp_psths_eta_calibrated.pdf"
SUMMARY_CSV_PATH = RUNS_ROOT / SESSION_NAME / f"{SESSION_NAME}_fixrsvp_eta_calibrated_summary.csv"


def load_sta_ste_bank(image_shape):
    subject, date = SESSION_NAME.split("_", 1)
    info = get_dataset_info(str(GABORIUM_CONFIG_PATH), subject, date, image_shape)
    return info["stas"], info["stes"]


def build_plot_rows(cell_ids, best_df):
    fixrsvp = load_fixrsvp()
    fixrsvp_raster = load_fixrsvp_raster()
    first_ckpt, _ = load_checkpoint(int(cell_ids[0]))
    first_image_shape = tuple(int(x) for x in first_ckpt["model_kwargs"]["image_shape"])
    stas, stes = load_sta_ste_bank(first_image_shape)
    best_val_bps_map = {int(row.cell_id): float(row.best_val_bps) for row in best_df.itertuples(index=False)}
    lag_groups = {}
    for cell_id in cell_ids:
        ckpt, _ = load_checkpoint(int(cell_id))
        lag_groups.setdefault(int(ckpt["lag_index"]), []).append((int(cell_id), ckpt))
    plot_rows = []
    for lag_index, group in sorted(lag_groups.items()):
        image_shape = tuple(int(x) for x in group[0][1]["model_kwargs"]["image_shape"])
        models_by_cell = {}
        for cell_id, ckpt in group:
            _, model = load_checkpoint(int(cell_id))
            models_by_cell[int(cell_id)] = model

        group_outputs = compute_group_fixrsvp_outputs(
            models_by_cell=models_by_cell,
            fixrsvp=fixrsvp,
            lag_index=int(lag_index),
            image_shape=image_shape,
            include_eta=True,
        )

        for cell_id, ckpt in group:
            out = group_outputs[int(cell_id)]
            scale, bias, calib_rhat_trials = fit_eta_calibration(out["robs_trials"], out["z_trials"])
            robs_psth = compute_psth(out["robs_trials"])
            base_psth = compute_psth(out["rhat_trials"])
            calib_psth = compute_psth(calib_rhat_trials)
            corr_base, rmse_base = corr_and_rmse(robs_psth, base_psth)
            corr_calib, rmse_calib = corr_and_rmse(robs_psth, calib_psth)
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
                    "base_psth": base_psth,
                    "calib_psth": calib_psth,
                    "raster_trials": raster_trials,
                    "window_start_s": float(out["window_start_s"]),
                    "sta_img": np.asarray(stas[int(cell_id), int(lag_index)], dtype=np.float32),
                    "ste_img": np.asarray(stes[int(cell_id), int(lag_index)], dtype=np.float32),
                    "best_val_bps": float(best_val_bps_map[int(cell_id)]),
                    "eta_scale": float(scale),
                    "eta_bias": float(bias),
                    "corr_base": float(corr_base),
                    "corr_calib": float(corr_calib),
                    "rmse_base": float(rmse_base),
                    "rmse_calib": float(rmse_calib),
                }
            )

        for model in models_by_cell.values():
            del model
        torch.cuda.empty_cache()
    plot_rows.sort(
        key=lambda row: (
            np.isfinite(row["corr_calib"]),
            row["corr_calib"] if np.isfinite(row["corr_calib"]) else -np.inf,
        ),
        reverse=True,
    )
    return plot_rows


def save_pdf(plot_rows):
    per_page = 25
    ncols = 5
    nrows = 5
    plt.rcParams["pdf.compression"] = 9
    summary_rows = []

    with PdfPages(PDF_PATH) as pdf:
        fig = axes = None
        for idx, row in enumerate(plot_rows):
            page_pos = idx % per_page
            if page_pos == 0:
                if fig is not None:
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight", dpi=80)
                    plt.close(fig)
                fig, axes = plt.subplots(nrows, ncols, figsize=(14, 14), sharex=False, sharey=False)
                axes = axes.reshape(-1)

            ax = axes[page_pos]
            ax.set_axis_off()
            raster_ax = ax.inset_axes([0.08, 0.40, 0.88, 0.45])
            psth_ax = raster_ax.twinx()
            sta_ax = ax.inset_axes([0.10, 0.07, 0.34, 0.20])
            ste_ax = ax.inset_axes([0.56, 0.07, 0.34, 0.20])

            plot_spike_raster(raster_ax, row["raster_trials"], duration_s=DURATION_S)
            time_s = row["time_s"]
            robs_psth = np.asarray(row["robs_psth"], dtype=np.float32)
            calib_psth = np.asarray(row["calib_psth"], dtype=np.float32)
            valid_robs = np.isfinite(robs_psth)
            valid_calib = np.isfinite(calib_psth)
            if np.any(valid_robs):
                psth_ax.plot(time_s[valid_robs], robs_psth[valid_robs], color="black", linewidth=0.8, rasterized=True)
            if np.any(valid_calib):
                psth_ax.plot(time_s[valid_calib], calib_psth[valid_calib], color="dodgerblue", linewidth=0.8, rasterized=True)

            sta_ax.imshow(row["sta_img"], cmap="coolwarm_r")
            sta_ax.set_title(f"STA lag {int(row['lag_index'])}", fontsize=6, pad=1)
            sta_ax.axis("off")
            ste_ax.imshow(row["ste_img"], cmap="magma")
            ste_ax.set_title(f"STE lag {int(row['lag_index'])}", fontsize=6, pad=1)
            ste_ax.axis("off")

            ax.set_title(
                f"cell {int(row['cell_id']):03d} | unit {int(row['true_cid'])} | bps={row['best_val_bps']:.3f} | corr={row['corr_calib']:.2f}",
                fontsize=8,
                pad=2,
            )
            style_overlay_axes(
                raster_ax,
                psth_ax,
                len(row["raster_trials"]) if row["raster_trials"] is not None else 0,
                duration_s=DURATION_S,
            )
            summary_rows.append(
                {
                    "cell_id": int(row["cell_id"]),
                    "true_cid": int(row["true_cid"]),
                    "best_val_bps": float(row["best_val_bps"]),
                    "lag_index": int(row["lag_index"]),
                    "eta_scale": float(row["eta_scale"]),
                    "eta_bias": float(row["eta_bias"]),
                    "corr_base": float(row["corr_base"]),
                    "corr_calib": float(row["corr_calib"]),
                    "rmse_base": float(row["rmse_base"]),
                    "rmse_calib": float(row["rmse_calib"]),
                }
            )

        if fig is not None:
            used = (len(plot_rows) - 1) % per_page + 1
            for ax in axes[used:]:
                ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight", dpi=80)
            plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV_PATH, index=False)


def main():
    best_df = pd.read_csv(RUNS_ROOT / SESSION_NAME / "best_val_bps.csv").sort_values("cell_id").reset_index(drop=True)
    cell_ids = best_df["cell_id"].astype(int).tolist()
    plot_rows = build_plot_rows(cell_ids, best_df)
    save_pdf(plot_rows)
    print(f"saved pdf: {PDF_PATH}")
    print(f"saved summary csv: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
