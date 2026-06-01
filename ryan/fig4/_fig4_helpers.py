"""Shared plotting helpers for figure 4 panels B and C.

Holds the seriation routine that sorts trials for the single-neuron raster,
the concatenated observed|twin raster drawer, and the example-neuron
selection logic (pinned via PANEL_B_SESSION / PANEL_B_NEURON_ID, or
auto-picked as the highest-ccnorm reliable neuron across all sessions).
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import pdist
from scipy.ndimage import gaussian_filter1d

from _fig4_data import DT, VALID_TIME_BINS


# Panel B/C window and seriation parameters
PANEL_B_MIN_DUR_S = 0.5      # minimum trial length for seriation + raster (sec)
PANEL_B_SERIATION = "olo"    # "olo" or "pc1"
PANEL_B_SMOOTH_SIGMA_S = 0.015
PANEL_B_WINDOW_S = 0.5

# Pinned example neuron (set both to None to auto-pick the highest-ccnorm
# reliable neuron across all sessions).
PANEL_B_SESSION = "Allen_2022-04-08"
PANEL_B_NEURON_ID = 62

PANEL_B_MIN_BINS = int(round(PANEL_B_MIN_DUR_S / DT))
N_BINS_B = int(round(PANEL_B_WINDOW_S / DT))


def order_single_neuron_by_seriation(robs_trials, rhat_trials, dfs_trials,
                                     method=PANEL_B_SERIATION,
                                     min_bins=PANEL_B_MIN_BINS,
                                     smooth_sigma_s=PANEL_B_SMOOTH_SIGMA_S):
    """Filter trials with ≥ min_bins valid bins in the plotting window,
    truncate to the window, and seriate so adjacent raster rows are similar.

    "olo": hierarchical clustering + optimal leaf ordering on correlation
    distance between Gaussian-smoothed trial rates.
    "pc1": sort by PC1 score of the observed trial-by-time matrix.

    Returns (robs_sorted, rhat_sorted, order, first_bin).
    """
    any_valid = (dfs_trials > 0).any(axis=0)
    if not any_valid.any():
        empty = np.empty((0, min_bins))
        return empty, empty, np.arange(0), 0
    first_bin = int(np.argmax(any_valid))
    end_bin = first_bin + min_bins
    if end_bin > dfs_trials.shape[1]:
        empty = np.empty((0, min_bins))
        return empty, empty, np.arange(0), first_bin

    window_valid = dfs_trials[:, first_bin:end_bin] > 0
    trial_valid_count = window_valid.sum(axis=1)
    keep = trial_valid_count >= min_bins

    robs_k = robs_trials[keep, first_bin:end_bin].astype(float).copy()
    rhat_k = rhat_trials[keep, first_bin:end_bin].astype(float).copy()
    dfs_k = dfs_trials[keep, first_bin:end_bin]
    valid_k = dfs_k > 0

    if robs_k.shape[0] < 2:
        robs_k[~valid_k] = np.nan
        rhat_k[~valid_k] = np.nan
        return robs_k, rhat_k, np.arange(robs_k.shape[0]), first_bin

    obs_masked = np.where(valid_k, robs_k, np.nan)
    col_mean = np.nanmean(obs_masked, axis=0, keepdims=True)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    obs_filled = np.where(valid_k, robs_k, np.broadcast_to(col_mean, robs_k.shape))

    if method == "pc1":
        obs_centered = obs_filled - obs_filled.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(obs_centered, full_matrices=False)
        pc1_scores = U[:, 0] * S[0]
        order = np.argsort(pc1_scores)
    elif method == "olo":
        sigma_bins = max(smooth_sigma_s / DT, 1e-6)
        obs_smooth = gaussian_filter1d(obs_filled, sigma=sigma_bins, axis=1,
                                       mode="nearest")
        row_std = obs_smooth.std(axis=1)
        flat = row_std < 1e-12
        if flat.any():
            rng = np.random.default_rng(0)
            obs_smooth = obs_smooth.copy()
            obs_smooth[flat] += rng.normal(scale=1e-9, size=(flat.sum(),
                                                              obs_smooth.shape[1]))
        dists = pdist(obs_smooth, metric="correlation")
        Z = linkage(dists, method="average")
        Z = optimal_leaf_ordering(Z, dists)
        order = np.asarray(leaves_list(Z), dtype=int)
    else:
        raise ValueError(f"Unknown seriation method: {method!r} "
                         f"(expected 'olo' or 'pc1')")

    robs_k[~valid_k] = np.nan
    rhat_k[~valid_k] = np.nan
    return robs_k[order], rhat_k[order], order, first_bin


def add_raster_colorbar_below(fig, ax, im, label, *, y0=-0.05, height=0.035,
                              width=0.675, x0=None, fontsize=6):
    """Skinny horizontal colorbar floating beneath a raster axes.

    Drawn on an `inset_axes` anchored to `ax` (axes-fraction coords; y0 < 0
    places it below the axes). Inset axes are excluded from constrained_layout,
    so the colorbar floats and does NOT steal gridspec space — the raster keeps
    its full size and row alignment is preserved. Right-aligned to the panel by
    default (x0=None); the x-scalebar lives in the lower-left, so the right side
    is clear and the bar can sit just below the panel.
    """
    if x0 is None:
        x0 = 1.0 - width
    cax = ax.inset_axes([x0, y0, width, height])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=fontsize, length=2, pad=1)
    cb.set_label(label, fontsize=fontsize, labelpad=2)
    cb.outline.set_linewidth(0.5)
    return cb


def draw_raster_scalebars(ax, *, n_trials, scale_len_s=0.1, n_trials_scale=10,
                          scale_fontsize=8, show_text=True):
    """Draw the x (time) and y (trials) scale bars for a raster axes.

    The bars sit just outside the axes (x-bar below, y-bar to the left) using
    the axis-blended transforms, so they track the data extent regardless of
    aspect. With `show_text=False` only the bars are drawn (used by panel H,
    which inherits panel E's scale).
    """
    trans_x = ax.get_xaxis_transform()
    ax.plot([0.0, scale_len_s], [-0.06, -0.06], "k-", linewidth=2,
            transform=trans_x, clip_on=False)
    if show_text:
        ax.text(scale_len_s / 2, -0.11, f"{int(round(scale_len_s * 1000))} ms",
                transform=trans_x, ha="center", va="top",
                fontsize=scale_fontsize, clip_on=False)
    trans_y = ax.get_yaxis_transform()
    n_scale = min(n_trials_scale, n_trials)
    y0, y1 = n_trials, n_trials - n_scale
    ax.plot([-0.02, -0.02], [y0, y1], "k-", linewidth=2,
            transform=trans_y, clip_on=False)
    if show_text:
        ax.text(-0.04, (y0 + y1) / 2, f"{n_scale} trials",
                transform=trans_y, ha="right", va="center", rotation=90,
                fontsize=scale_fontsize, clip_on=False)


def draw_raster_pair(ax, robs_rate, rhat_rate, *, window_s, vmin, vmax,
                     scale_len_s=0.1, n_trials_scale=10,
                     label_fontsize=9, scale_fontsize=8):
    """Concatenated observed|twin raster on one axes, single imshow so the
    two halves share aspect/pixel size exactly. Returns the AxesImage.
    """
    combined = np.concatenate([robs_rate, rhat_rate], axis=1)
    n_trials_local = combined.shape[0]
    im = ax.imshow(
        combined, aspect="auto", origin="upper",
        extent=[0, 2 * window_s, n_trials_local, 0],
        vmin=vmin, vmax=vmax, cmap="binary", interpolation="none",
    )
    ax.axvline(window_s, color="k", linewidth=0.8)
    ax.text(0.25, 1.02, "Observed", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=label_fontsize)
    ax.text(0.75, 1.02, "Twin", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=label_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    draw_raster_scalebars(ax, n_trials=n_trials_local, scale_len_s=scale_len_s,
                          n_trials_scale=n_trials_scale,
                          scale_fontsize=scale_fontsize, show_text=True)
    return im


def select_example_neuron(data,
                          session=PANEL_B_SESSION,
                          neuron_id=PANEL_B_NEURON_ID):
    """Build the example-neuron payload used by panels B and C.

    If session/neuron_id are both set, the matching neuron is used; otherwise
    the highest-ccnorm reliable neuron across all sessions is auto-picked.
    Returns a dict with sorted rasters (rate), trial-averaged traces, the
    time axis, window mask, and shared color limits.
    """
    session_results = data["session_results"]
    ccnorm = data["ccnorm"]
    good = data["good"]
    valid_indices = data["valid_indices"]
    all_trace_neuron_session = data["all_trace_neuron_session"]
    all_robs_mean = data["all_robs_mean"]
    all_rhat_mean = data["all_rhat_mean"]

    if session is not None and neuron_id is not None:
        si = next(
            (i for i, sr in enumerate(session_results) if sr["session"] == session),
            None,
        )
        if si is None:
            raise ValueError(
                f"PANEL_B_SESSION={session!r} not found in session_results"
            )
        nmask = session_results[si]["neuron_mask"]
        matches = np.where(np.asarray(nmask) == neuron_id)[0]
        if len(matches) == 0:
            raise ValueError(
                f"PANEL_B_NEURON_ID={neuron_id} not in session {session} "
                f"(neurons passing spike threshold: "
                f"{sorted(int(x) for x in nmask)})"
            )
        ni = int(matches[0])
        best_global = all_trace_neuron_session.index((si, ni))
        loc = np.where(valid_indices == best_global)[0]
        if len(loc) == 0:
            raise ValueError(
                f"Pinned neuron (session={session}, id={neuron_id}) "
                "was filtered out (non-finite rho)"
            )
        best_local = int(loc[0])
    else:
        mask_all = good & np.isfinite(ccnorm)
        candidates_all = np.where(mask_all)[0]
        best_local = candidates_all[np.nanargmax(ccnorm[candidates_all])]
        best_global = valid_indices[best_local]
        si, ni = all_trace_neuron_session[best_global]

    best_sr = session_results[si]

    print(f"Panel B — example neuron: {best_sr['session']}, "
          f"neuron {best_sr['neuron_mask'][ni]}, "
          f"ccnorm={ccnorm[best_local]:.2f}")

    robs_trials = best_sr["robs_used"][:, :, ni]
    rhat_trials = best_sr["rhat_used"][:, :, ni]
    dfs_trials = best_sr["dfs_used"][:, :, ni]

    robs_sorted_b, rhat_sorted_b, _, first_bin_b = order_single_neuron_by_seriation(
        robs_trials, rhat_trials, dfs_trials
    )
    robs_trials_rate = (robs_sorted_b / DT)[:, :N_BINS_B]
    rhat_trials_rate = (rhat_sorted_b / DT)[:, :N_BINS_B]

    tbins = np.arange(VALID_TIME_BINS) * DT
    print(f"  Panel B raster: {robs_sorted_b.shape[0]} trials "
          f"(≥ {PANEL_B_WINDOW_S:.1f}s) ordered by '{PANEL_B_SERIATION}' seriation "
          f"of observed, starting at bin {first_bin_b} "
          f"({tbins[first_bin_b]*1000:.0f} ms → t=0)")

    robs_trace = all_robs_mean[best_global] / DT
    rhat_trace = all_rhat_mean[best_global] / DT
    t_valid = np.isfinite(robs_trace) & np.isfinite(rhat_trace)
    t = (np.arange(len(robs_trace)) - first_bin_b) * DT
    psth_window = t_valid & (t >= 0) & (t <= PANEL_B_WINDOW_S)

    vmin = 0
    vmax = np.nanpercentile(
        np.concatenate([robs_trials_rate.ravel(), rhat_trials_rate.ravel()]), 97
    )

    return {
        "si": si, "ni": ni,
        "best_local": best_local, "best_global": best_global,
        "session": best_sr["session"], "subject": best_sr["subject"],
        "neuron_id": int(best_sr["neuron_mask"][ni]),
        "ccnorm": float(ccnorm[best_local]),
        "robs_trials_rate": robs_trials_rate,
        "rhat_trials_rate": rhat_trials_rate,
        "first_bin": first_bin_b,
        "t": t, "psth_window": psth_window,
        "robs_trace": robs_trace, "rhat_trace": rhat_trace,
        "vmin": vmin, "vmax": vmax,
        "window_s": PANEL_B_WINDOW_S,
    }
