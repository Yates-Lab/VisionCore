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
EXAMPLE_UNIT = 110           # Figure-2 panel A/B example cell
TRIAL_A = 41                 # "Trial 1" (solid black)
TRIAL_B = 55                 # "Trial 2" (solid black)
EYE_AXIS = 1                 # 0 = horizontal, 1 = vertical
WINDOW_BINS = 72             # 600 ms @ 120 Hz
DT = 1.0 / 120.0

W1 = (21, 27)                # divergent window, centered at 200 ms (bin 24)
W2 = (51, 57)                # matched window, centered at 450 ms (bin 54)

# Panel A/B styling: matched arrow = blue, divergent = crimson (mirrors panel B).
MATCHED_COLOR = "tab:blue"
DIVERGENT_COLOR = "crimson"
WIN_SHADE = "0.85"

# Font rc-context matching the size panels A/B occupy in the composite figure
# (also used by the per-unit survey pages).
_FIG2_RC = {
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
}

# Recompute Ceye / histogram with uniform bins for this panel only.
# Bin width = INTERCEPT_THRESHOLD so the lowest bin is exactly the
# "below threshold" pool used by the decomposition's intercept.
from compute_fig2_data import INTERCEPT_THRESHOLD
HIST_D_MAX = 1.0              # deg
HIST_N_BINS = int(round(HIST_D_MAX / INTERCEPT_THRESHOLD))

# Decomposition params (mirror visualize_units.py)
DECOMP_WINDOW_BINS = 3       # 3 bins @ 120 Hz = 25 ms counting window (fig2 standard)
DECOMP_T_HIST = 1            # t_hist = max(DECOMP_T_HIST, t_count); matches T_HIST_MS=10
DECOMP_SEG_MIN = 36
# Spike-rate display bin = the counting window (25 ms), so Panel A matches Panel B.
RATE_BIN_FACTOR = DECOMP_WINDOW_BINS

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






# Production analysis radius (mirror covariance_decomposition.data_loading.FIXATION_RADIUS).
PANEL_FIXATION_RADIUS = 0.5
ALLUNITS_CACHE = CACHE_DIR / f"fig2b_unaccounted_allunits_{SESSION}.pkl"


def _decompose_all_units(radius=PANEL_FIXATION_RADIUS, d_max=1.0, n_bins=20,
                         refresh=False):
    """0.5°-radius 'unaccounted-for variability' decomposition for *every* unit
    in the lead example session, computed in one pass from the lead-pair-scan
    cache.

    Re-filters the aligned session to an eye-position radius < ``radius`` (the
    production ``FIXATION_RADIUS``) and forms the same close-pair windows the
    decomposition uses. Because ``rate_variance_by_distance`` already builds the
    full C×C conditional covariance, every unit costs the same as one. Returns
    per-unit arrays (indexed by the session's neuron column) plus the shared Δe
    bin centers:

        bin_centers (n_ok,)        Δe bin centers (deg, RMS-trajectory units)
        cum_crate   (n_ok, C)      cumulative close-pair rate covariance Crate(Δe)
        Ctotal, Crate, Cpsth, sigma_int, rate_hz   (C,)
        neuron_mask (C,)           original unit ids

    Panel B plots U(Δe) = Ctotal - cum_crate, rising from the internal floor
    sigma_int (perfect matching) to the eye-blind asymptote Ctotal - Cpsth.
    """
    if ALLUNITS_CACHE.exists() and not refresh:
        with open(ALLUNITS_CACHE, "rb") as f:
            return pickle.load(f)

    pair = _load_trial_pair()
    eyepos_raw = np.asarray(pair["eyepos"], float)
    robs = np.nan_to_num(pair["robs"], nan=0.0)
    eyepos = np.nan_to_num(eyepos_raw, nan=0.0)
    neuron_mask = np.asarray(pair["neuron_mask"])

    # Tighten the fixation criterion to the production radius: keep only bins
    # whose eye position lies within `radius` of the origin (the same per-bin
    # fixation test align_fixrsvp_trials applies, just at a smaller radius).
    rad = np.hypot(eyepos_raw[..., 0], eyepos_raw[..., 1])
    fin = np.isfinite(eyepos_raw).all(-1)
    valid = np.asarray(pair["valid_mask"], bool) & fin & (rad < radius)

    segments = extract_valid_segments(valid, min_len_bins=DECOMP_SEG_MIN)
    counts, traj, T = extract_windows(
        robs, eyepos, segments, t_count=DECOMP_WINDOW_BINS,
        t_hist=max(DECOMP_T_HIST, DECOMP_WINDOW_BINS),
    )
    C = counts.shape[1]
    di = np.arange(C)

    # Pooled total spike-count variance per unit (matches visualize_units' Ctotal).
    fin_w = np.isfinite(counts.sum(1))
    Ctotal = np.diag(np.cov(counts[fin_w].T, ddof=1)).copy()

    # Cumulative close-pair rate covariance Crate(Δe), per unit (diagonal).
    edges = np.linspace(0.0, d_max, n_bins + 1)
    res = rate_variance_by_distance(counts, traj, T, edges)
    ceye = np.asarray(res["Ceye"])[:, di, di]            # (n_bins, C)
    count_e = np.asarray(res["count_e"], float)
    cum_num = np.cumsum(np.nan_to_num(ceye) * count_e[:, None], axis=0)
    cum_den = np.cumsum(count_e)
    ok = cum_den > 0
    cum_crate = cum_num[ok] / cum_den[ok][:, None]       # (n_ok, C)
    bin_centers = np.asarray(res["bin_centers"], float)[ok]

    # All-pairs second moment == Cpsth (cumulative over the full Δe support).
    allp = rate_variance_by_distance(counts, traj, T, np.array([0.0, 1e9]))
    Cpsth = np.asarray(allp["Ceye"])[0][di, di].copy()
    Crate = cum_crate[0].copy()
    sigma_int = Ctotal - Crate
    rate_hz = counts[fin_w].mean(0) / (DECOMP_WINDOW_BINS * DT)

    out = {
        "bin_centers": bin_centers,
        "cum_crate": cum_crate,
        "Ctotal": Ctotal,
        "Crate": Crate,
        "Cpsth": Cpsth,
        "sigma_int": sigma_int,
        "rate_hz": rate_hz,
        "neuron_mask": neuron_mask,
        "radius": radius,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(ALLUNITS_CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def _compute_unaccounted_curve(unit_orig=EXAMPLE_UNIT, **kw):
    """Per-unit slice of :func:`_decompose_all_units` for one unit (default: the
    example unit). ``kw`` (radius/d_max/n_bins/refresh) are forwarded."""
    allu = _decompose_all_units(**kw)
    j = int(np.where(np.asarray(allu["neuron_mask"]) == unit_orig)[0][0])
    return {
        "bin_centers": allu["bin_centers"],
        "cum_crate": allu["cum_crate"][:, j],
        "Ctotal": float(allu["Ctotal"][j]),
        "Crate": float(allu["Crate"][j]),
        "Cpsth": float(allu["Cpsth"][j]),
        "sigma_int": float(allu["sigma_int"][j]),
        "radius": allu["radius"],
    }


def _binned_rate(counts_1d, factor=RATE_BIN_FACTOR):
    """Aggregate per-bin spike counts into ``factor``-bin (25 ms) windows.

    Returns (bin-center times in ms, rate in spk/s) for a step trace, so the
    displayed spike rate uses the same counting window as the Panel B analysis.
    """
    counts_1d = np.asarray(counts_1d, dtype=float)
    n = (len(counts_1d) // factor) * factor
    c = counts_1d[:n].reshape(-1, factor).sum(1)
    rate = c / (factor * DT)
    t = (np.arange(len(c)) + 0.5) * factor * DT * 1000.0
    return t, rate


def _window_delta(e_a, e_b, win):
    return float(np.nanmean(np.abs(e_a[win[0]:win[1]] - e_b[win[0]:win[1]])))


def plot_eye_rate_example(ax_eye, ax_spk, unit_orig=EXAMPLE_UNIT):
    """Panel A: two-trial eye-position traces above per-bin spike rates for the
    example unit. Both trials solid black with "Trial 1"/"Trial 2" text labels;
    gray matched/divergent windows shaded, each marked with a Δe callout arrow
    (matched = blue, divergent = crimson) mirroring panel B. The eye
    traces/windows are unit-independent; only the spike-rate traces depend on
    ``unit_orig``. Returns the matched/divergent Δe values."""
    pair = _load_trial_pair()
    robs = pair["robs"]
    eyepos = pair["eyepos"]
    j = int(np.where(np.asarray(pair["neuron_mask"]) == unit_orig)[0][0])

    W = WINDOW_BINS
    t_ms = np.arange(W) * DT * 1000.0
    e_a = eyepos[TRIAL_A, :W, EYE_AXIS]
    e_b = eyepos[TRIAL_B, :W, EYE_AXIS]
    # Spike rates in 25 ms bins (matches the Panel B counting window).
    t_spk, r_a = _binned_rate(robs[TRIAL_A, :W, j])
    _, r_b = _binned_rate(robs[TRIAL_B, :W, j])
    trace_lw = 1.8

    # ---------- window shading (drawn first, behind everything) ----------
    def _shade(ax, win):
        ax.axvspan(win[0] * DT * 1000.0, win[1] * DT * 1000.0,
                   color=WIN_SHADE, zorder=-1)

    for ax in (ax_eye, ax_spk):
        _shade(ax, W1)
        _shade(ax, W2)

    # ---------- eye traces (both solid black) ----------
    ax_eye.plot(t_ms, e_a, color="k", lw=trace_lw)
    ax_eye.plot(t_ms, e_b, color="k", lw=trace_lw)
    ax_eye.set_ylabel("Eye position (°)")
    for s in ("top", "right", "left"):
        ax_eye.spines[s].set_visible(False)
    ax_eye.set_yticks([])
    plt.setp(ax_eye.get_xticklabels(), visible=False)
    ymin, ymax = ax_eye.get_ylim()
    ax_eye.set_ylim(min(ymin, -0.55), ymax)
    # Trial labels just inside the left edge (x in axes fraction so they clear
    # the y-axis title, y in data so they sit next to each trace).
    _lbl_bbox = dict(fc="white", ec="none", pad=0.4)
    eye_trans = ax_eye.get_yaxis_transform()
    t_div = (W1[0] + W1[1]) // 2
    ax_eye.text(0.015, float(e_a[t_div]), "Trial 1", transform=eye_trans,
                ha="left", va="center", fontsize=8, color="k",
                bbox=_lbl_bbox, zorder=7)
    ax_eye.text(0.015, float(e_b[t_div]), "Trial 2", transform=eye_trans,
                ha="left", va="center", fontsize=8, color="k",
                bbox=_lbl_bbox, zorder=7)
    # vertical scale bar
    sb_x_eye = t_ms[-1] + 12
    ax_eye.plot([sb_x_eye, sb_x_eye], [-0.5, -0.3], color="k", lw=2,
                clip_on=False)
    ax_eye.text(sb_x_eye + 5, -0.4, "0.2°", rotation=90, va="center",
                ha="left", fontsize=8, clip_on=False)

    # ---------- Δe connectors between the two eye traces ----------
    # Divergent window: double-headed arrow between the (far-apart) traces.
    # Matched window: the traces coincide, so a thick dot + a "Δe = X" callout.
    # Both windows carry a Δe label so the two read consistently.
    d_vals = {}
    for win, key, color in ((W2, "matched", MATCHED_COLOR),
                            (W1, "divergent", DIVERGENT_COLOR)):
        d_vals[key] = _window_delta(e_a, e_b, win)
        t_mid = (win[0] + win[1]) // 2
        t_mid_ms = t_mid * DT * 1000.0
        if key == "divergent":
            ax_eye.annotate(
                "", xy=(t_mid_ms, float(e_b[t_mid])),
                xytext=(t_mid_ms, float(e_a[t_mid])),
                arrowprops=dict(arrowstyle="<->", color=color, lw=2.0),
                zorder=6,
            )
            # Δe callout beside the divergent arrow (consistent with matched).
            ax_eye.text(
                t_mid_ms + 10,
                0.5 * (float(e_a[t_mid]) + float(e_b[t_mid])),
                rf"$\Delta e = {d_vals['divergent']:.2f}\degree$",
                fontsize=8, color=color, ha="left", va="center", zorder=7,
            )
        else:
            y_conv = 0.5 * (float(e_a[t_mid]) + float(e_b[t_mid]))
            # Thick solid segment between the two (near-coincident) eye
            # positions: reads as a dot that the Δe arrow calls out.
            ax_eye.plot([t_mid_ms, t_mid_ms],
                        [float(e_a[t_mid]), float(e_b[t_mid])],
                        color=color, lw=3.5, solid_capstyle="round", zorder=6)
            ax_eye.annotate(
                rf"$\Delta e = {d_vals['matched']:.2f}\degree$",
                xy=(t_mid_ms, y_conv - 0.02),
                xytext=(t_mid_ms + 12, y_conv - 0.36),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                fontsize=8, color=color, ha="left", va="center", zorder=7,
            )

    # ---------- spike rates (offset step traces, both solid black) ----------
    rmax = float(max(r_a.max(), r_b.max(), 1.0))
    offset = 1.25 * rmax
    ax_spk.step(t_spk, r_b + offset, color="k", lw=trace_lw, where="mid")
    ax_spk.step(t_spk, r_a, color="k", lw=trace_lw, where="mid")
    ax_spk.axhline(0, color="0.7", lw=0.5, zorder=-1)
    ax_spk.axhline(offset, color="0.7", lw=0.5, zorder=-1)
    ax_spk.set_xlabel("Time from fixation onset (ms)")
    ax_spk.set_ylabel("Spike rates")
    ax_spk.set_yticks([])
    for s in ("top", "right", "left"):
        ax_spk.spines[s].set_visible(False)
    spk_trans = ax_spk.get_yaxis_transform()
    ax_spk.text(0.015, 0.0, "Trial 1", transform=spk_trans, ha="left",
                va="bottom", fontsize=8, color="k", bbox=_lbl_bbox, zorder=7)
    ax_spk.text(0.015, offset, "Trial 2", transform=spk_trans, ha="left",
                va="bottom", fontsize=8, color="k", bbox=_lbl_bbox, zorder=7)
    sb_len = max(10.0, round(rmax / 2 / 10) * 10)
    sb_x = t_ms[-1] + 12
    ax_spk.plot([sb_x, sb_x], [0, sb_len], color="k", lw=2, clip_on=False)
    ax_spk.text(sb_x + 5, sb_len / 2, f"{int(sb_len)} spk/s", rotation=90,
                va="center", ha="left", fontsize=8, clip_on=False)
    ax_spk.set_xlim(0, t_ms[-1] + 4)
    return d_vals


def plot_unaccounted_variance_panel(ax, decomp=None, caption=True):
    """Panel B: 'unaccounted-for variability' curve, Ctotal − Crate(Δe) — the
    spike-count variability left unaccounted-for when eye trajectories are
    matched only to within Δe. Rises from the internal-noise floor (perfect
    matching) to the eye-blind PSTH asymptote, with total variability marked on
    top. ``decomp`` may be a precomputed per-unit decomposition dict (keys
    bin_centers/cum_crate/Ctotal/Cpsth/sigma_int); if None, the example unit's
    curve is computed. ``caption`` toggles the take-home text (off for the
    per-unit survey pages, where it would be redundant)."""
    d = decomp if decomp is not None else _compute_unaccounted_curve()
    x = np.asarray(d["bin_centers"], float)
    Ctotal = float(d["Ctotal"])
    Cpsth = float(d["Cpsth"])
    sigma_int = float(d["sigma_int"])
    U = Ctotal - np.asarray(d["cum_crate"], float)   # unaccounted-for variability
    U_naive = Ctotal - Cpsth                         # eye-blind (PSTH) asymptote
    y_hi = Ctotal * 1.12

    # Reference levels: total / eye-blind / internal.
    ax.axhline(Ctotal, color="k", ls="-", lw=1.0, zorder=1)
    ax.axhline(U_naive, color="k", ls="--", lw=1.0, zorder=1)
    ax.axhline(sigma_int, color="0.55", ls=":", lw=1.0, zorder=1)
    ax.plot(x, U, color="k", lw=1.5, marker="o", ms=4,
            markeredgecolor="k", markeredgewidth=0.4, zorder=3)

    # Total variability label (top-left, above its line).
    ax.text(0.04, Ctotal + 0.015 * y_hi, "Total variability",
            fontsize=7.5, ha="left", va="bottom")
    # Internal-noise floor label (just above its line at mid-x).
    ax.text(0.30, sigma_int + 0.012 * y_hi, "Internal variability",
            color="0.35", fontsize=7.5, ha="left", va="bottom")

    # Right-side decomposition bar at x = xa: FEM (internal floor -> eye-blind
    # level) in blue, stimulus/PSTH (eye-blind level -> total) in red.
    xa = 0.92
    ax.annotate("", xy=(xa, U_naive), xytext=(xa, sigma_int),
                arrowprops=dict(arrowstyle="<->", color=MATCHED_COLOR, lw=2.0),
                zorder=4)
    ax.annotate("", xy=(xa, Ctotal), xytext=(xa, U_naive),
                arrowprops=dict(arrowstyle="<->", color=DIVERGENT_COLOR, lw=2.0),
                zorder=4)
    ax.text(xa - 0.02, 0.5 * (sigma_int + U_naive),
            "FEM variability\n$(\\sigma^2_{\\mathrm{FEM}})$",
            color="k", fontsize=7.5, ha="right", va="center")
    ax.text(xa - 0.02, 0.5 * (U_naive + Ctotal),
            "Eye position ignored $(\\sigma^2_{\\mathrm{PSTH}})$",
            color="k", fontsize=7.5, ha="right", va="center")

    # Matched end sits on the internal floor (eye position fully accounted for).
    ax.annotate("Trajectories matched",
                xy=(x[0], U[0]), xytext=(0.16, 0.30), textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color="k", lw=0.9),
                fontsize=7.5, ha="left", va="center")

    if caption:
        # Take-home: descriptive phrase bottom-left, fraction equation
        # bottom-right (larger so it reads clearly).
        ax.text(0.02, 0.04,
                "Fraction of consistent\nvariability obscured\nby eye movements",
                transform=ax.transAxes, fontsize=7.0, ha="left", va="bottom")
        ax.text(0.99, 0.05,
                r"$f_{\mathrm{FEM}} = \frac{\sigma^2_{\mathrm{FEM}}}"
                r"{\sigma^2_{\mathrm{FEM}}+\sigma^2_{\mathrm{PSTH}}}$",
                transform=ax.transAxes, fontsize=11.5, ha="right", va="bottom")

    ax.set_xlim(-0.05, 1.03)
    ax.set_ylim(0.0, y_hi)
    ax.set_xlabel("Eye-trajectory mismatch threshold, Δe < x (°)")
    ax.set_ylabel("Unaccounted-for variability (spk²)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return float(U[-1]), Ctotal


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
    j = int(np.where(neuron_mask == EXAMPLE_UNIT)[0][0])
    Ceye = uniform["Ceye"]
    bin_centers = uniform["bin_centers"]
    bin_edges = uniform["bin_edges"]
    count_e = uniform["count_e"]
    bin_width = float(bin_edges[1] - bin_edges[0])

    robs = pair["robs"]
    eyepos = pair["eyepos"]
    j_pair = int(np.where(np.asarray(pair["neuron_mask"]) == EXAMPLE_UNIT)[0][0])

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
