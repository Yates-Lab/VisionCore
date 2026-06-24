"""
Figure 2 example panels — the lead-in for the covariance decomposition.

Provides (used by generate_figure2.py):
    plot_eye_rate_example(ax_eye, ax_spk)
        Two-trial eye-position traces + offset spike-rate step traces for the
        example unit (Allen_2022-03-04, trials 49/68, unit 151), with matched
        (blue) and divergent (red) Δe arrows. Rendered as Panel A.
    _load_unit_payload / _compute_uniform_bins
        Per-bin rate-covariance helpers behind the covariance-mismatch curve
        (Panel B).

Also retains the original standalone 2x2 lead-in (plot_panel_a / make_axes,
with the Δ-eye histogram) for reference; the main figure no longer uses it.

Standalone:
    uv run paper/fig2/generate_panel_example.py
"""
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from matplotlib.transforms import offset_copy

from VisionCore.paths import CACHE_DIR
from VisionCore.covariance import (
    extract_valid_segments, extract_windows, rate_variance_by_distance,
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


def make_axes(fig, subplot_spec=None):
    """Create the sub-gridspec for panel A inside `fig` (or inside the given
    `subplot_spec`). Returns a dict of 5 axes:
    {"eye", "delta", "spk", "hist", "cov"}.

    Left column has 3 rows (eye, skinny delta-eye trajectory, spike rates);
    right column has 2 rows (histogram, rate-variance). The two columns are
    built as independent nested gridspecs so the skinny delta row on the left
    does not force a thin slot in the right column.
    """
    if subplot_spec is None:
        outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28)
    else:
        outer = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=subplot_spec,
            width_ratios=[1.0, 1.0], wspace=0.28,
        )
    gs_left = GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[0, 0],
        height_ratios=[1.0, 0.25, 1.0], hspace=0.12,
    )
    gs_right = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0, 1],
        height_ratios=[2.0, 3.0], hspace=0.12,
    )
    ax_eye = fig.add_subplot(gs_left[0, 0])
    ax_delta = fig.add_subplot(gs_left[1, 0], sharex=ax_eye)
    ax_spk = fig.add_subplot(gs_left[2, 0], sharex=ax_eye)
    ax_hist = fig.add_subplot(gs_right[0, 0])
    ax_cov = fig.add_subplot(gs_right[1, 0], sharex=ax_hist)
    return {"eye": ax_eye, "delta": ax_delta, "spk": ax_spk,
            "hist": ax_hist, "cov": ax_cov}


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

    segments = extract_valid_segments(valid_mask, min_len_bins=DECOMP_SEG_MIN)
    counts, traj, T_idx = extract_windows(
        robs_clean, eyepos_clean, segments,
        t_count=DECOMP_WINDOW_BINS,
        t_hist=max(DECOMP_T_HIST, DECOMP_WINDOW_BINS),
    )

    bin_edges = np.linspace(0.0, HIST_D_MAX, HIST_N_BINS + 1)
    res = rate_variance_by_distance(counts, traj, T_idx, bin_edges)

    out = {
        "bin_edges": np.asarray(bin_edges),
        "bin_centers": np.asarray(res["bin_centers"]),
        "count_e": np.asarray(res["count_e"]),
        "Ceye": np.asarray(res["Ceye"]),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(UNIFORM_CACHE, "wb") as f:
        pickle.dump(out, f)
    return out






def plot_eye_rate_example(ax_eye, ax_spk, arrow_color="tab:blue"):
    """Render the two-trial lead-in example into two provided axes:

        ax_eye  horizontal eye position vs time for the trial pair, with a
                vertical double-arrow between the two traces at a *matched*
                (W1) and a *divergent* (W2) time window — mirroring the
                stimulus-vs-total (FEM) variance arrow in the mismatch panel.
        ax_spk  per-bin spike rates for both trials as offset step traces.

    Used by the combined figure's composite Panel A (histogram dropped).
    """
    pair = _load_trial_pair()
    robs = pair["robs"]
    eyepos = pair["eyepos"]
    j_pair = int(np.where(np.asarray(pair["neuron_mask"]) == UNIT_ORIG)[0][0])

    W = WINDOW_BINS
    t_ms = np.arange(W) * DT * 1000.0
    e_a = eyepos[TRIAL_A, :W, 0]
    e_b = eyepos[TRIAL_B, :W, 0]
    r_a = robs[TRIAL_A, :W, j_pair] / DT
    r_b = robs[TRIAL_B, :W, j_pair] / DT

    color_a, color_b = "tab:cyan", "tab:red"
    trace_lw = 2.0

    # ---------- eye traces ----------
    ax_eye.plot(t_ms, e_a, color=color_a, lw=trace_lw)
    ax_eye.plot(t_ms, e_b, color=color_b, lw=trace_lw)
    ax_eye.set_ylabel("Eye position (°)")
    ax_eye.spines["top"].set_visible(False)
    ax_eye.spines["right"].set_visible(False)
    ax_eye.spines["left"].set_visible(False)
    ax_eye.set_yticks([])
    plt.setp(ax_eye.get_xticklabels(), visible=False)
    ymin, ymax = ax_eye.get_ylim()
    ax_eye.set_ylim(min(ymin, -0.55), ymax)
    sb_x_eye = t_ms[-1] + 12
    ax_eye.plot([sb_x_eye, sb_x_eye], [-0.5, -0.3], color="k", lw=2,
                clip_on=False)
    ax_eye.text(sb_x_eye + 5, -0.4, "0.2°", rotation=90, va="center",
                ha="left", fontsize=8, clip_on=False)

    # ---------- spike rates (offset step traces) ----------
    rmax = float(max(r_a.max(), r_b.max(), 1.0))
    offset = 1.25 * rmax
    ax_spk.step(t_ms, r_b + offset, color=color_b, lw=trace_lw, where="mid")
    ax_spk.step(t_ms, r_a, color=color_a, lw=trace_lw, where="mid")
    ax_spk.axhline(0, color="0.7", lw=0.5, zorder=-1)
    ax_spk.axhline(offset, color="0.7", lw=0.5, zorder=-1)
    ax_spk.set_xlabel("Time from fixation onset (ms)")
    ax_spk.set_ylabel("Spike rates")
    ax_spk.set_yticks([])
    ax_spk.spines["left"].set_visible(False)
    ax_spk.spines["top"].set_visible(False)
    ax_spk.spines["right"].set_visible(False)
    sb_len = max(10.0, round(rmax / 2 / 10) * 10)
    sb_x = t_ms[-1] + 12
    ax_spk.plot([sb_x, sb_x], [0, sb_len], color="k", lw=2, clip_on=False)
    ax_spk.text(sb_x + 5, sb_len / 2, f"{int(sb_len)} spk/s", rotation=90,
                va="center", ha="left", fontsize=8, clip_on=False)
    ax_spk.set_xlim(0, t_ms[-1] + 4)

    # ---------- window shading (both axes) ----------
    def _shade(ax, slice_):
        ax.axvspan(slice_[0] * DT * 1000.0, slice_[1] * DT * 1000.0,
                   color="0.85", zorder=-1)

    for ax in (ax_eye, ax_spk):
        _shade(ax, W1)
        _shade(ax, W2)

    # ---------- Δe connector arrows between the two eye traces ----------
    # For this trial pair W2 is the matched window (small Δe, eye trajectories
    # aligned -> variance ~ total) and W1 is the divergent window (large Δe).
    # These mirror the variance arrows in the covariance-mismatch panel:
    # matched = blue, divergent = red.
    d_vals = {}
    for slice_, color, key in ((W2, arrow_color, "matched"),
                               (W1, "crimson", "divergent")):
        t_mid = (slice_[0] + slice_[1]) // 2
        t_mid_ms = t_mid * DT * 1000.0
        ax_eye.annotate(
            "", xy=(t_mid_ms, float(e_b[t_mid])),
            xytext=(t_mid_ms, float(e_a[t_mid])),
            arrowprops=dict(arrowstyle="<->", color=color, lw=2.0),
            zorder=6,
        )
        d_vals[key] = float(np.nanmean(
            np.abs(e_a[slice_[0]:slice_[1]] - e_b[slice_[0]:slice_[1]])))
    return d_vals


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
    ax_delta = axes["delta"]
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
    ax_eye.set_title("A", loc="left", fontweight="bold", pad=30)
    ax_eye.set_ylabel("Eye position (°)")
    ax_eye.plot(t_ms, e_a, color=color_a, lw=trace_lw)
    ax_eye.plot(t_ms, e_b, color=color_b, lw=trace_lw)
    ax_eye.spines["top"].set_visible(False)
    ax_eye.spines["right"].set_visible(False)
    ax_eye.spines["left"].set_visible(False)
    ax_eye.set_yticks([])
    plt.setp(ax_eye.get_xticklabels(), visible=False)
    # Ensure the scale bar fits and add a 0.2° vertical scale bar at the right.
    ymin, ymax = ax_eye.get_ylim()
    ax_eye.set_ylim(min(ymin, -0.55), ymax)
    sb_x_eye = t_ms[-1] + 12
    ax_eye.plot([sb_x_eye, sb_x_eye], [-0.5, -0.3], color="k", lw=2, clip_on=False)
    ax_eye.text(sb_x_eye + 5, -0.4, "0.2°",
                rotation=90, va="center", ha="left", fontsize=8,
                clip_on=False)

    # ---------- middle-left: Δ eye trajectory over time ----------
    delta_e = np.abs(e_a - e_b)
    ax_delta.fill_between(t_ms, 0.0, delta_e, color="0.55", alpha=0.55, lw=0)
    ax_delta.plot(t_ms, delta_e, color="0.2", lw=1.0)
    ax_delta.set_ylim(0.0, 1.3)
    ax_delta.set_ylabel("Δ eye (°)")
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)
    ax_delta.spines["left"].set_visible(False)
    ax_delta.set_yticks([])
    plt.setp(ax_delta.get_xticklabels(), visible=False)
    ax_delta.plot([sb_x_eye, sb_x_eye], [0.0, 1.0], color="k", lw=2,
                  clip_on=False)
    ax_delta.text(sb_x_eye + 5, 0.5, "1°",
                  rotation=90, va="center", ha="left", fontsize=8,
                  clip_on=False)

    # ---------- bottom-left: spike rate (raw, jagged, offset) ----------
    ymax = float(max(r_a.max(), r_b.max(), 1.0))
    offset = 1.25 * ymax
    ax_spk.step(t_ms, r_b + offset, color=color_b, lw=trace_lw, where="mid")
    ax_spk.step(t_ms, r_a, color=color_a, lw=trace_lw, where="mid")
    ax_spk.axhline(0, color="0.7", lw=0.5, zorder=-1)
    ax_spk.axhline(offset, color="0.7", lw=0.5, zorder=-1)
    ax_spk.set_xlabel("Time from fixation onset (ms)")
    ax_spk.set_ylabel("Spike rates")
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

    for ax in (ax_eye, ax_delta, ax_spk):
        _shade(ax, W1)
        _shade(ax, W2)


    # ---------- top-right: distribution of Δ eye trajectory ----------
    ax_hist.set_title("B", loc="left", fontweight="bold", pad=30)
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
                color="k", lw=1.5, marker="o", ms=4,
                markeredgecolor="k", markeredgewidth=0.4, zorder=3)

    # Var(PSTH) horizontal reference line. Right edge sits just left of the
    # green (close-eye) highlight bar so the label doesn't get covered.
    var_psth = float(p["Cpsth"][j, j])
    ax_cov.axhline(var_psth, color="k", ls="--", lw=1.0, zorder=2)

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
    band_color = "0.85"

    for x in (x_w1, x_w2):
        half = bin_width / 2.0
        for ax in (ax_hist, ax_cov):
            ax.axvspan(x - half, x + half, color=band_color, zorder=-1)

    _distance_marker(t_w1_mid, band_color)
    _distance_marker(t_w2_mid, band_color)

    # Var(PSTH) label — right edge just left of the green (W1) highlight bar.
    psth_label_x = x_w1 - 0.6 * bin_width
    ax_cov.text(psth_label_x, var_psth, "Var(PSTH)",
                ha="right", va="bottom", fontsize=12)

    # Var(Δe < threshold) label — sits down and to the right of the lowest-bin
    # marker, left/center justified, with the arrow emerging just to its left.
    lowest_x = float(bin_centers[0])
    lowest_y = float(var_by_bin[0]) if np.isfinite(var_by_bin[0]) else 0.0
    ymax_cov = np.nanmax(var_by_bin[ok]) if np.any(ok) else lowest_y
    # Shift the text down by one ~12pt text-height (~14 pts) via an offset
    # transform so the nudge is independent of the y-axis data range.
    text_trans = offset_copy(ax_cov.transData, fig=fig, y=-14, units="points")
    ax_cov.annotate(
        rf"Var($\Delta e < {INTERCEPT_THRESHOLD:g}\degree$)",
        xy=(lowest_x, lowest_y),
        xytext=(lowest_x + 4.5 * bin_width, lowest_y),
        textcoords=text_trans,
        fontsize=12, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="k", lw=0.8,
                        shrinkA=6, shrinkB=4),
    )

    # Force a draw so transforms have correct positions before placing arrows
    fig.canvas.draw()

    t_w1_ms = 0.5 * (W1[0] + W1[1]) * DT * 1000.0
    t_w2_ms = 0.5 * (W2[0] + W2[1]) * DT * 1000.0

    def _orthogonal_arrow(ax_src, ax_dst, x_src_data, y_src_data,
                          x_dst_data, y_dst_data, color="0.3",
                          corridor_offset_pts=14.0):
        """Draw an orthogonal connector (up → across → down) with rounded
        corners from a data point in ax_src to a data point in ax_dst, with
        the arrow head landing on the destination point.
        """
        inv = fig.transFigure.inverted()
        # Tail starts a few pixels above the source marker so it visibly
        # detaches from the black distance-marker dot.
        src_disp = ax_src.transData.transform((x_src_data, y_src_data)) \
            + np.array([0.0, 12.0])
        dst_disp = ax_dst.transData.transform((x_dst_data, y_dst_data))
        # Corridor sits comfortably above both axes' top edges.
        top_src = ax_src.transAxes.transform((0.0, 1.0))[1]
        top_dst = ax_dst.transAxes.transform((0.0, 1.0))[1]
        y_corr_disp = max(top_src, top_dst) + corridor_offset_pts
        p_src = inv.transform(src_disp)
        p_dst = inv.transform(dst_disp)
        cy = inv.transform((0.0, y_corr_disp))[1]
        sx, sy = p_src
        ex, ey = p_dst
        # Corner radius in figure-fraction; cap to avoid overshoot.
        r = min(0.008, abs(ex - sx) / 2.5, abs(cy - sy) / 2.0,
                abs(cy - ey) / 2.0)
        sign = 1.0 if ex >= sx else -1.0
        verts = [
            (sx, sy),
            (sx, cy - r),
            (sx, cy),                   # CURVE3 control
            (sx + sign * r, cy),        # CURVE3 end
            (ex - sign * r, cy),
            (ex, cy),                   # CURVE3 control
            (ex, cy - r),               # CURVE3 end
            (ex, ey),
        ]
        codes = [
            Path.MOVETO, Path.LINETO,
            Path.CURVE3, Path.CURVE3,
            Path.LINETO,
            Path.CURVE3, Path.CURVE3,
            Path.LINETO,
        ]
        path = Path(verts, codes)
        arrow = FancyArrowPatch(
            path=path, transform=fig.transFigure,
            arrowstyle="->", mutation_scale=12, lw=1.3,
            color=color,
        )
        fig.patches.append(arrow)

    # Source y = just above the topmost endpoint of the black distance marker.
    y_src_w1 = max(float(e_a[t_w1_mid]), float(e_b[t_w1_mid]))
    y_src_w2 = max(float(e_a[t_w2_mid]), float(e_b[t_w2_mid]))
    # Destination y = top of the (now neutral) shaded band on the histogram.
    y_dst_hist = ax_hist.get_ylim()[1]
    _orthogonal_arrow(ax_eye, ax_hist, t_w1_ms, y_src_w1, x_w1, y_dst_hist,
                      corridor_offset_pts=24.0)
    _orthogonal_arrow(ax_eye, ax_hist, t_w2_ms, y_src_w2, x_w2, y_dst_hist,
                      corridor_offset_pts=16.0)


if __name__ == "__main__":
    fig = plt.figure(figsize=(11, 6.5))
    axes = make_axes(fig)
    plot_panel_a(axes=axes)
    standalone_save(fig, "panel_a_lead_demo")
