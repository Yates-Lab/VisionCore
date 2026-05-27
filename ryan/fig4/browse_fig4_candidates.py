"""Interactive browser: top-N candidate neurons by ccnorm.

Pops up one figure per candidate (PSTH overlay + observed|twin raster) so a
new example neuron can be picked for Panel B/C. Not invoked during normal
figure generation — run this standalone when reselecting an example.

Usage:
    uv run ryan/fig4/browse_fig4_candidates.py [--n-show 50]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from _fig4_data import DT, configure_matplotlib, load_fig4_data
from _fig4_helpers import (
    PANEL_B_WINDOW_S, N_BINS_B,
    draw_raster_pair, order_single_neuron_by_seriation,
)


def browse(n_show=50):
    data = load_fig4_data()
    session_results = data["session_results"]
    ccnorm = data["ccnorm"]
    good = data["good"]
    valid_indices = data["valid_indices"]
    all_trace_neuron_session = data["all_trace_neuron_session"]
    all_robs_mean = data["all_robs_mean"]
    all_rhat_mean = data["all_rhat_mean"]

    mask_all_cand = good & np.isfinite(ccnorm)
    candidates_ranked = np.where(mask_all_cand)[0]
    candidates_ranked = candidates_ranked[np.argsort(ccnorm[candidates_ranked])[::-1]]
    n_show = min(n_show, len(candidates_ranked))

    for rank in range(n_show):
        idx_local = candidates_ranked[rank]
        idx_global = valid_indices[idx_local]
        si_c, ni_c = all_trace_neuron_session[idx_global]
        sr_c = session_results[si_c]

        robs_t_full = sr_c["robs_used"][:, :, ni_c]
        rhat_t_full = sr_c["rhat_used"][:, :, ni_c]
        dfs_t_full = sr_c["dfs_used"][:, :, ni_c]

        robs_sorted, rhat_sorted, _, first_bin_c = order_single_neuron_by_seriation(
            robs_t_full, rhat_t_full, dfs_t_full
        )
        if robs_sorted.shape[0] < 2:
            continue
        robs_rate = (robs_sorted / DT)[:, :N_BINS_B]
        rhat_rate = (rhat_sorted / DT)[:, :N_BINS_B]

        robs_tr = all_robs_mean[idx_global] / DT
        rhat_tr = all_rhat_mean[idx_global] / DT
        tt = (np.arange(len(robs_tr)) - first_bin_c) * DT
        window_c = (np.isfinite(robs_tr) & np.isfinite(rhat_tr)
                    & (tt >= 0) & (tt <= PANEL_B_WINDOW_S))

        vm = 0
        vx = np.nanpercentile(
            np.concatenate([robs_rate.ravel(), rhat_rate.ravel()]), 97
        )

        fig = plt.figure(figsize=(8, 3))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.35)
        ax_psth = fig.add_subplot(gs[0, 0])
        ax_rast = fig.add_subplot(gs[0, 1])

        fig.suptitle(
            f"Rank {rank+1}: {sr_c['session']} neuron {sr_c['neuron_mask'][ni_c]} "
            f"({sr_c['subject']}) — ccnorm={ccnorm[idx_local]:.3f}, "
            f"N_trials={robs_sorted.shape[0]}",
            fontsize=9,
        )

        ax_psth.plot(tt[window_c], robs_tr[window_c], 'k', linewidth=1,
                     label="Observed")
        ax_psth.plot(tt[window_c], rhat_tr[window_c], 'tab:red', linewidth=1,
                     label="Twin")
        ax_psth.set_xlim(0, PANEL_B_WINDOW_S)
        ax_psth.set_xlabel("Time (s)")
        ax_psth.set_ylabel("Rate (sp/s)")
        ax_psth.legend(frameon=False, fontsize=7)
        ax_psth.spines["top"].set_visible(False)
        ax_psth.spines["right"].set_visible(False)

        im = draw_raster_pair(
            ax_rast, robs_rate, rhat_rate,
            window_s=PANEL_B_WINDOW_S, vmin=vm, vmax=vx,
        )
        fig.colorbar(im, ax=ax_rast, shrink=0.8, pad=0.02, label="sp/s")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-show", type=int, default=50,
                   help="Number of top candidates to display.")
    args = p.parse_args()
    configure_matplotlib()
    browse(n_show=args.n_show)
