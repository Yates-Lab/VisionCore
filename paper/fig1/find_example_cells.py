"""
Browse gaze-sorted rasters for orientation-tuned cells across all rsvp
sessions, to pick example cells for fig1d.

Flat #%% cell script for interactive use (ipython / VSCode cells).
"""

#%% Imports and config
import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt

FIG1_DIR = Path("/home/ryanress/v1-fovea/VisionCore/ryan/fig1")
if str(FIG1_DIR) not in sys.path:
    sys.path.insert(0, str(FIG1_DIR))

from eval.fixrsvp import get_fixrsvp_data
from eval.sta_ste import compute_sta_ste, peak_lag_from_ste, population_peak_lag
from generate_fig1d import (
    DATASET_CONFIGS_PATH,
    TOTAL_WINDOW_BINS,
    USE_UNIVERSAL_PEAK_LAG,
    _compute_gratings_for_session,
    _compute_segments,
    plot_raster_axis,
)

ORI_SNR_THRESH = 0.5
GRID_ROWS, GRID_COLS = 4, 4
PER_FIG = GRID_ROWS * GRID_COLS

with open(DATASET_CONFIGS_PATH) as f:
    SESSIONS = list(yaml.safe_load(f)["sessions"])
print(f"{len(SESSIONS)} sessions to scan")

#%% Loop over sessions: load data once per session, plot all tuned cells
for session_idx, session in enumerate(SESSIONS):
    subject, date = session.split("_", 1)
    print(f"\n=== Session: {session} ===")

    # Gratings + tuned-cell selection
    try:
        g = _compute_gratings_for_session(subject, date)
    except (RuntimeError, FileNotFoundError, AssertionError, ValueError) as e:
        print(f"skipping {session}: gratings failed ({e})")
        continue
    g_cids = np.asarray(g["cids"])
    snr = np.asarray(g["ori_snr"])
    peak = np.asarray(g["peak_ori"])
    mask = np.isfinite(snr) & (snr >= ORI_SNR_THRESH)
    order = np.argsort(-snr[mask])
    g_rows = np.where(mask)[0][order]
    cells = [(int(g_cids[r]), float(snr[r]), float(peak[r])) for r in g_rows]
    print(f"{len(cells)} tuned cells (ori_snr >= {ORI_SNR_THRESH})")
    if not cells:
        continue

    # Load fixrsvp data ONCE for this session; regenerate if cache is truncated
    try:
        data = get_fixrsvp_data(
            subject, date, DATASET_CONFIGS_PATH,
            use_cached_data=True,
            salvageable_mismatch_time_threshold=25,
            verbose=False,
        )
    except EOFError:
        print(f"{session}: cached fixrsvp pkl is truncated; regenerating")
        try:
            data = get_fixrsvp_data(
                subject, date, DATASET_CONFIGS_PATH,
                use_cached_data=False,
                salvageable_mismatch_time_threshold=25,
                verbose=False,
            )
        except Exception as e:
            print(f"skipping {session}: regenerate failed ({e})")
            continue
    except (FileNotFoundError, AssertionError, ValueError) as e:
        print(f"skipping {session}: get_fixrsvp_data failed ({e})")
        continue
    data_cids = list(data["cids"])
    eyepos = data["eyepos"]
    robs_all = data["robs"]
    spike_times_trials = data["spike_times_trials"]
    trial_t_bins = data["trial_t_bins"]

    # Session-wide peak lag from cached STEs (matches generate_fig1d)
    arrs = compute_sta_ste(session)
    if arrs is None:
        session_peak_lag = None  # fall back per-cell below
        stes_all = None
    else:
        stes_all = arrs["stes"]
        session_peak_lag = population_peak_lag(stes_all) if USE_UNIVERSAL_PEAK_LAG else None

    # Plot 4x4 grids
    for fig_i in range(int(np.ceil(len(cells) / PER_FIG))):
        chunk = cells[fig_i * PER_FIG:(fig_i + 1) * PER_FIG]
        fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(20, 14),
                                 squeeze=False)
        fig.suptitle(
            f"{session}  (ori_snr >= {ORI_SNR_THRESH})  page {fig_i + 1}",
            fontsize=12,
        )
        for slot, ax in enumerate(axes.flat):
            if slot >= len(chunk):
                ax.axis("off")
                continue
            cell, s, p = chunk[slot]
            try:
                if cell not in data_cids:
                    raise ValueError(f"cell {cell} not in fixrsvp cids")
                cell_col = data_cids.index(cell)
                robs_cell = robs_all[:, :, cell_col]
                spike_times = [
                    np.asarray(spike_times_trials[t][cell_col])
                    for t in range(len(spike_times_trials))
                ]

                if session_peak_lag is not None:
                    peak_lag = session_peak_lag
                elif stes_all is not None:
                    peak_lag = peak_lag_from_ste(stes_all[cell_col])
                else:
                    peak_lag = int(np.nanargmax(np.nanmean(robs_cell, axis=0)))

                segments = _compute_segments(eyepos, p, peak_lag)
                seg_means = [np.nanmean(robs_cell[:, sg["start"]:sg["end"]])
                             for sg in segments]
                example_idx = int(np.nanargmax(seg_means)) if seg_means else 0

                payload = {
                    "cell": cell,
                    "session": session,
                    "max_orientation": float(p),
                    "peak_lag": int(peak_lag),
                    "total_window": np.asarray(TOTAL_WINDOW_BINS, dtype=int),
                    "segments": segments,
                    "example_segment_idx": example_idx,
                    "eyepos_all": eyepos,
                    "spike_times_all": spike_times,
                    "trial_t_bins_all": trial_t_bins,
                    "robs_cell_all": robs_cell,
                }

                plot_raster_axis(ax, payload, n_psth=3, tick_height=0.7,
                                 tick_lw=0.4, show_segment_dividers=False)
                ax.set_title(f"cell {cell}  snr={s:.2f}  peak_ori={p:.0f}°",
                             fontsize=8)
                ax.set_xlabel(""); ax.set_ylabel("")
                ax.tick_params(labelsize=6)
            except Exception as e:
                ax.set_title(f"cell {cell}: error", fontsize=8)
                ax.text(0.5, 0.5, str(e)[:80], ha="center", va="center",
                        fontsize=6, transform=ax.transAxes)
                print(f"cell {cell} failed: {e}")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
