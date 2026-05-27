"""
Figure 2 panel A: lead-in demonstrating the conditioning we apply to the
neural responses.

Layout (2 x 2):
    Top-left      Eye horizontal position vs time for a pair of trials
                  (Allen_2022-03-04, trials 49 and 68; first 750 ms).
    Bottom-left   Per-bin spike counts for unit 151 (orig 151) on those
                  two trials, plotted as offset step traces with a scale
                  bar. X axis shared with top-left (time).
    Top-right     Distribution of Δ eye trajectory distance (pooled over
                  this session's fixrsvp pairs and bins).
    Bottom-right  Rate variance Ceye[k, j, j] vs Δ eye trajectory bin
                  centers, connected line. X axis shared with top-right
                  (Δ eye trajectory, deg).

Two time windows are highlighted (gray): W1 = 100–150 ms (a "close-eye"
period) and W2 = 350–400 ms (a "far-eye" period). Arrows go from the
midpoint of each window in the left column to the corresponding Δ bin in
the right column, showing how the conditioning sorts time bins into
distance bins where the within-bin variance differs systematically.

Composer usage:
    from generate_fig2a import plot_panel_a, make_axes
    axes = make_axes(fig, subplot_spec)   # 4 axes in a 2x2 sub-gridspec
    plot_panel_a(axes=axes)
"""
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch

from VisionCore.paths import CACHE_DIR
from VisionCore.covariance import (
    extract_valid_segments, extract_windows, estimate_rate_covariance,
)
from _panel_common import standalone_save

SESSION = "Allen_2022-03-04"
UNIT_ORIG = 151
TRIAL_A = 49
TRIAL_B = 68
WINDOW_BINS = 72              # 600 ms @ 120 Hz
DT = 1.0 / 120.0

W1 = (12, 18)                 # 100–150 ms (bin slice)
W2 = (42, 48)                 # 350–400 ms

# Recompute Ceye / histogram with uniform bins for this panel only.
# Bin width = INTERCEPT_THRESHOLD so the lowest bin is exactly the
# "below threshold" pool used by the decomposition's intercept.
from compute_fig2_data import INTERCEPT_THRESHOLD
HIST_D_MAX = 1.0              # deg
HIST_N_BINS = int(round(HIST_D_MAX / INTERCEPT_THRESHOLD))

# Decomposition params (mirror visualize_units.py)
DECOMP_WINDOW_BINS = 2
DECOMP_T_HIST = 1
DECOMP_SEG_MIN = 36

UNIT_EXPLORE_CACHE = CACHE_DIR / "fig2_unit_explore.pkl"
PAIR_SCAN_CACHE = CACHE_DIR / f"fig2_lead_pair_scan_{SESSION}.pkl"
UNIFORM_CACHE = CACHE_DIR / f"fig2a_uniform_bins_{SESSION}.pkl"


def _load_unit_payload():
    with open(UNIT_EXPLORE_CACHE, "rb") as f:
        payloads = pickle.load(f)
    p = next((q for q in payloads if q["session"] == SESSION), None)
    if p is None:
        avail = [q["session"] for q in payloads]
        raise RuntimeError(f"{SESSION} not in unit_explore cache. "
                           f"Available: {avail}")
    return p


def _load_trial_pair():
    with open(PAIR_SCAN_CACHE, "rb") as f:
        return pickle.load(f)


def _compute_uniform_bins():
    """Recompute the Δ-trajectory histogram and per-bin rate covariance
    with uniform bins (HIST_N_BINS edges 0..HIST_D_MAX), so the figure's
    histogram and rate-variance panel share a clean x-axis."""
    if UNIFORM_CACHE.exists():
        with open(UNIFORM_CACHE, "rb") as f:
            return pickle.load(f)

    pair = _load_trial_pair()
    robs = pair["robs"]
    eyepos = pair["eyepos"]
    valid_mask = pair["valid_mask"]

    robs_clean = np.nan_to_num(robs, nan=0.0)
    eyepos_clean = np.nan_to_num(eyepos, nan=0.0)
    robs_t = torch.tensor(robs_clean, dtype=torch.float32)
    eyepos_t = torch.tensor(eyepos_clean, dtype=torch.float32)

    segments = extract_valid_segments(valid_mask, min_len_bins=DECOMP_SEG_MIN)
    SpikeCounts, EyeTraj, T_idx, _ = extract_windows(
        robs_t, eyepos_t, segments,
        t_count=DECOMP_WINDOW_BINS,
        t_hist=max(DECOMP_T_HIST, DECOMP_WINDOW_BINS),
        device="cpu",
    )

    bin_edges = np.linspace(0.0, HIST_D_MAX, HIST_N_BINS + 1)
    _, _, Ceye_u, bin_centers_u, count_e_u, _ = estimate_rate_covariance(
        SpikeCounts, EyeTraj, T_idx, n_bins=bin_edges,
        Ctotal=None, intercept_mode="below_threshold",
        intercept_kwargs={"threshold": INTERCEPT_THRESHOLD},
    )

    out = {
        "bin_edges": np.asarray(bin_edges),
        "bin_centers": np.asarray(bin_centers_u),
        "count_e": np.asarray(count_e_u),
        "Ceye": np.asarray(Ceye_u),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(UNIFORM_CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def make_axes(fig, subplot_spec=None):
    """Create the 2x2 sub-gridspec for panel A inside `fig` (or inside
    the given `subplot_spec` of fig). Returns a dict of 4 axes:
    {"eye", "spk", "hist", "cov"}.
    """
    kwargs = dict(width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0],
                  hspace=0.12, wspace=0.28)
    if subplot_spec is None:
        gs = fig.add_gridspec(2, 2, **kwargs)
    else:
        gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=subplot_spec, **kwargs)
    ax_eye = fig.add_subplot(gs[0, 0])
    ax_spk = fig.add_subplot(gs[1, 0], sharex=ax_eye)
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_cov = fig.add_subplot(gs[1, 1], sharex=ax_hist)
    return {"eye": ax_eye, "spk": ax_spk, "hist": ax_hist, "cov": ax_cov}


def plot_panel_a(axes=None, fig=None, refresh=False, data=None):
    """Render the 4-axis lead-in panel.

    Either pass `axes` (dict from make_axes) or pass `fig` and we'll
    build the axes ourselves on that figure.
    """
    if axes is None:
        if fig is None:
            fig = plt.figure(figsize=(11, 6.5))
        axes = make_axes(fig)
    ax_eye = axes["eye"]
    ax_spk = axes["spk"]
    ax_hist = axes["hist"]
    ax_cov = axes["cov"]
    fig = ax_eye.figure

    p = _load_unit_payload()
    pair = _load_trial_pair()
    uniform = _compute_uniform_bins()

    neuron_mask = np.asarray(p["neuron_mask"])
    j = int(np.where(neuron_mask == UNIT_ORIG)[0][0])
    Ceye = uniform["Ceye"]
    bin_centers = uniform["bin_centers"]
    bin_edges = uniform["bin_edges"]
    count_e = uniform["count_e"]
    bin_width = float(bin_edges[1] - bin_edges[0])

    robs = pair["robs"]
    eyepos = pair["eyepos"]
    j_pair = int(np.where(np.asarray(pair["neuron_mask"]) == UNIT_ORIG)[0][0])

    W = WINDOW_BINS
    t_ms = np.arange(W) * DT * 1000.0

    e_a = eyepos[TRIAL_A, :W, 0]
    e_b = eyepos[TRIAL_B, :W, 0]
    r_a = robs[TRIAL_A, :W, j_pair] / DT
    r_b = robs[TRIAL_B, :W, j_pair] / DT

    def _window_d(slice_):
        d = np.abs(e_a[slice_[0]:slice_[1]] - e_b[slice_[0]:slice_[1]])
        return float(np.nanmean(d))

    d_w1 = _window_d(W1)
    d_w2 = _window_d(W2)

    color_a, color_b = "tab:cyan", "tab:red"
    trace_lw = 2.0

    # ---------- top-left: eye traces ----------
    ax_eye.set_title("A", loc="left", fontweight="bold")
    ax_eye.plot(t_ms, e_a, color=color_a, lw=trace_lw)
    ax_eye.plot(t_ms, e_b, color=color_b, lw=trace_lw)
    ax_eye.set_ylabel("Eye position (°)")
    ax_eye.spines["top"].set_visible(False)
    ax_eye.spines["right"].set_visible(False)
    plt.setp(ax_eye.get_xticklabels(), visible=False)

    # ---------- bottom-left: spike rate (raw, jagged, offset) ----------
    ymax = float(max(r_a.max(), r_b.max(), 1.0))
    offset = 1.25 * ymax
    ax_spk.step(t_ms, r_b + offset, color=color_b, lw=trace_lw, where="mid")
    ax_spk.step(t_ms, r_a, color=color_a, lw=trace_lw, where="mid")
    ax_spk.axhline(0, color="0.7", lw=0.5, zorder=-1)
    ax_spk.axhline(offset, color="0.7", lw=0.5, zorder=-1)
    ax_spk.set_xlabel("Time from fixation onset (ms)")
    ax_spk.set_yticks([])
    ax_spk.spines["left"].set_visible(False)
    ax_spk.spines["top"].set_visible(False)
    ax_spk.spines["right"].set_visible(False)
    sb_len = max(10.0, round(ymax / 2 / 10) * 10)
    sb_x = t_ms[-1] + 12
    ax_spk.plot([sb_x, sb_x], [0, sb_len], color="k", lw=2, clip_on=False)
    ax_spk.text(sb_x + 5, sb_len / 2, f"{int(sb_len)} spk/s",
                rotation=90, va="center", ha="left", fontsize=8,
                clip_on=False)
    ax_spk.set_xlim(0, t_ms[-1] + 4)

    # Gray highlight rectangles spanning eye + spike axes
    def _shade(ax, slice_):
        t0 = slice_[0] * DT * 1000.0
        t1 = slice_[1] * DT * 1000.0
        ax.axvspan(t0, t1, color="0.85", zorder=-1)

    for ax in (ax_eye, ax_spk):
        _shade(ax, W1)
        _shade(ax, W2)


    # ---------- top-right: distribution of Δ eye trajectory ----------
    ax_hist.set_title("B", loc="left", fontweight="bold")
    ax_hist.bar(bin_centers, count_e, width=bin_width * 0.9,
                color="0.55", edgecolor="white", linewidth=0.4,
                align="center")
    ax_hist.set_ylabel("# samples")
    ax_hist.set_ylim(bottom=0)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    plt.setp(ax_hist.get_xticklabels(), visible=False)

    # ---------- bottom-right: rate variance vs Δ eye ----------
    var_by_bin = Ceye[:, j, j]
    ok = np.isfinite(var_by_bin) & (count_e > 0)
    ax_cov.axhline(0, color="0.7", lw=0.5, zorder=-1)
    ax_cov.plot(bin_centers[ok], var_by_bin[ok],
                color="tab:purple", lw=1.5, marker="o", ms=4,
                markeredgecolor="k", markeredgewidth=0.4, zorder=3)

    # Var(PSTH) horizontal reference line
    var_psth = float(p["Cpsth"][j, j])
    ax_cov.axhline(var_psth, color="k", ls="--", lw=1.0, zorder=2)
    ax_cov.text(HIST_D_MAX, var_psth, "Var(PSTH)  ",
                ha="right", va="bottom", fontsize=12)

    # Vertical guide on the lowest bin labeled Var(Δe < 0.1°)
    lowest_x = float(bin_centers[0])
    lowest_y = float(var_by_bin[0]) if np.isfinite(var_by_bin[0]) else 0.0
    ymax_cov = max(np.nanmax(var_by_bin[ok]), var_psth) if np.any(ok) else lowest_y
    ax_cov.annotate(
        rf"Var($\Delta e < {INTERCEPT_THRESHOLD:g}\degree$)",
        xy=(lowest_x, lowest_y),
        xytext=(lowest_x + 0.20, lowest_y - 0.4 * (ymax_cov - lowest_y)
                if ymax_cov > lowest_y else lowest_y - 0.2),
        fontsize=12, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="k", lw=0.8,
                        shrinkA=2, shrinkB=4),
    )

    ax_cov.set_xlabel("Δ eye trajectory (°)")
    ax_cov.set_ylabel("Rate variance (spk²)")
    ax_cov.spines["top"].set_visible(False)
    ax_cov.spines["right"].set_visible(False)

    # ---------- distance-line markers (dark grey + ball at each end) ----------
    # Drawn at the midpoint time of each window, vertically spanning the
    # two eye traces so the reader can read the Δ at a glance.
    def _distance_marker(t_bin_mid, color):
        t_ms_mid = t_bin_mid * DT * 1000.0
        y0 = float(e_a[t_bin_mid])
        y1 = float(e_b[t_bin_mid])
        ax_eye.plot([t_ms_mid, t_ms_mid], [y0, y1],
                    color="0.25", lw=1.5, zorder=4)
        ax_eye.plot([t_ms_mid, t_ms_mid], [y0, y1],
                    color="0.25", marker="o", ms=5,
                    markeredgecolor="0.25", markerfacecolor="0.25",
                    lw=0, zorder=5)

    t_w1_mid = (W1[0] + W1[1]) // 2
    t_w2_mid = (W2[0] + W2[1]) // 2

    # ---------- bin assignments + colored highlights on right column ----------
    def _bin_for(d_value):
        if not np.any(ok):
            return float(d_value)
        return float(bin_centers[ok][np.argmin(np.abs(bin_centers[ok] - d_value))])

    x_w1 = _bin_for(d_w1)
    x_w2 = _bin_for(d_w2)
    c_w1 = "tab:green"
    c_w2 = "tab:red"

    for x, c in [(x_w1, c_w1), (x_w2, c_w2)]:
        half = bin_width / 2.0
        for ax in (ax_hist, ax_cov):
            ax.axvspan(x - half, x + half, color=c, alpha=0.25, zorder=-1)

    _distance_marker(t_w1_mid, c_w1)
    _distance_marker(t_w2_mid, c_w2)

    # Force a draw so transforms have correct positions before placing arrows
    fig.canvas.draw()

    t_w1_ms = 0.5 * (W1[0] + W1[1]) * DT * 1000.0
    t_w2_ms = 0.5 * (W2[0] + W2[1]) * DT * 1000.0

    def _arrow(ax_src, ax_dst, x_src_ms, x_dst_data, color):
        # Source: top of the source axes at x_src_ms.
        # Destination: top of the destination axes at x_dst_data.
        # Both anchored at the top of their axes so the arrow rides above
        # the data and never crosses traces or legends.
        src_disp = ax_src.transAxes.transform((0.0, 1.02))
        # X in data, Y in axes fraction = (x_src_ms_data, top of axes)
        src_disp = ax_src.transData.transform((x_src_ms, ax_src.get_ylim()[1])) \
            + np.array([0.0, 18.0])
        dst_disp = ax_dst.transData.transform((x_dst_data, ax_dst.get_ylim()[1])) \
            + np.array([0.0, 18.0])
        inv = fig.transFigure.inverted()
        p_src = inv.transform(src_disp)
        p_dst = inv.transform(dst_disp)
        arrow = FancyArrowPatch(
            p_src, p_dst,
            transform=fig.transFigure,
            arrowstyle="->", mutation_scale=14, lw=1.3,
            color=color, alpha=0.9,
            connectionstyle="arc3,rad=-0.12",
            shrinkA=2, shrinkB=2,
        )
        fig.patches.append(arrow)

    _arrow(ax_eye, ax_hist, t_w1_ms, x_w1, c_w1)
    _arrow(ax_eye, ax_hist, t_w2_ms, x_w2, c_w2)


if __name__ == "__main__":
    fig = plt.figure(figsize=(11, 6.5))
    axes = make_axes(fig)
    plot_panel_a(axes=axes)
    standalone_save(fig, "panel_a_lead_demo")
