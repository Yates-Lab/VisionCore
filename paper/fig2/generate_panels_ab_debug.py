"""
Debug harness for Figure 2 panels A and B (the covariance-decomposition lead-in).

Renders two stand-alone files so the current and iterated designs can be
compared by flipping between them:

    panels_ab_v1.png   the committed design (calls the production functions):
                       A = two-trial eye + spike traces (cyan/red, matched/
                       divergent Delta-e arrows); B = per-bin cross-trial rate
                       variance vs Delta-e.

    panels_ab_v2.png   the iteration under discussion:
                       A = eye + spike traces in BLACK with "Trial 1"/"Trial 2"
                           text labels (no reused red); gray matched/divergent
                           window shading kept, Delta-e arrows recolored neutral.
                       B = CUMULATIVE cross-trial rate variance for all pairs
                           with Delta-e < x. Starts at the matched-trajectory
                           (stimulus+FEM) variance and converges to the PSTH
                           variance as the threshold grows to include all pairs.
                           The matched/divergent window distances from A are
                           marked: a close trajectory sits far above PSTH
                           variance (large FEM gap), a far trajectory sits just
                           above it (small gap) -- the symmetric reading.

Trial pair / axis are config constants at the top so a pick from
``pick_lead_trial_pair.py`` can be dropped straight in.

Run:
    uv run paper/fig2/generate_panels_ab_debug.py              # both versions
    uv run paper/fig2/generate_panels_ab_debug.py --version 1
    uv run paper/fig2/generate_panels_ab_debug.py --version 2
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

from _panel_common import FIG_DIR
from generate_panel_example import (
    SESSION, UNIT_ORIG, WINDOW_BINS, DT,
    _load_trial_pair, _load_unit_payload, _compute_uniform_bins,
    _compute_unaccounted_curve, _binned_rate,
    plot_eye_rate_example,
)
from generate_figure2 import _plot_covariance_mismatch_panel

# --- v2 example-pair config (from pick_lead_trial_pair.py) ----------------
# v1 keeps the committed (49,68) example baked into plot_eye_rate_example;
# v2 uses the picker's cleaner pair below.
TRIAL_A = 41                 # "Trial 1" (solid)
TRIAL_B = 55                 # "Trial 2" (solid)
EYE_AXIS = 1                 # 0 = horizontal, 1 = vertical
W1 = (21, 27)                # divergent window, centered at 200 ms (bin 24)
W2 = (51, 57)                # matched window, centered at 450 ms (bin 54)
# v3 example cell, chosen from survey_lead_cells.py (high rate, high Cfem/Ctot,
# moderate internal floor). Kept separate from the global UNIT_ORIG (=151) so the
# production generate_figure2.py example is unaffected.
V3_UNIT_ORIG = 110

MATCHED_COLOR = "tab:blue"   # matched-trajectory arrow (mirrors panel B)
DIVERGENT_COLOR = "crimson"  # divergent-trajectory arrow
WIN_SHADE = "0.85"


# ===========================================================================
# Version 1: reproduce the committed panels by calling production code.
# ===========================================================================

def render_v1():
    fig = plt.figure(figsize=(10.5, 4.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.32)
    gs_a = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0, 0], height_ratios=[1.0, 1.0], hspace=0.18,
    )
    ax_eye = fig.add_subplot(gs_a[0, 0])
    ax_spk = fig.add_subplot(gs_a[1, 0], sharex=ax_eye)
    d_vals = plot_eye_rate_example(ax_eye, ax_spk, arrow_color="tab:blue")
    ax_eye.text(-0.16, 1.18, "A", transform=ax_eye.transAxes,
                fontweight="bold", fontsize=12, va="top", ha="left")

    ax_b = fig.add_subplot(outer[0, 1])
    _plot_covariance_mismatch_panel(ax_b, divergent_de=d_vals["divergent"])
    ax_b.text(-0.16, 1.18, "B", transform=ax_b.transAxes,
              fontweight="bold", fontsize=12, va="top", ha="left")

    _save(fig, "panels_ab_v1")


# ===========================================================================
# Version 2: black-trace example + cumulative variance curve.
# ===========================================================================

def _window_delta(e_a, e_b, win):
    return float(np.nanmean(np.abs(e_a[win[0]:win[1]] - e_b[win[0]:win[1]])))


def plot_eye_rate_example_v2(ax_eye, ax_spk, unit_orig=UNIT_ORIG):
    """Panel A, black-trace redesign. Both trials solid black, text-labelled
    on the left (no reused red on the traces); gray matched/divergent windows
    kept, Delta-e arrows colored matched=blue / divergent=crimson to mirror
    panel B. The eye traces/windows are unit-independent; only the spike-rate
    traces depend on ``unit_orig``. Returns matched/divergent Delta-e."""
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
    # trial labels just inside the left edge (blended transform: x in axes
    # fraction so they clear the rotated y-axis title, y in data so they sit
    # next to each trace's start).
    _lbl_bbox = dict(fc="white", ec="none", pad=0.4)
    eye_trans = ax_eye.get_yaxis_transform()
    # anchor label heights to the divergent window, where the traces separate.
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

    # ---------- Delta-e connectors between the two eye traces ----------
    # Divergent window: double-headed arrow between the (far-apart) traces.
    # Matched window: the traces coincide so a connector is invisible -- instead
    # drop a dashed locator at that timepoint and a "Delta e = X" callout arrow.
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
        else:
            y_conv = 0.5 * (float(e_a[t_mid]) + float(e_b[t_mid]))
            # Thick solid segment between the two (matched -> near-coincident)
            # eye positions: reads as a dot that the Δe arrow calls out.
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


def _cumulative_binned(j):
    """Cumulative cross-trial rate variance over the same uniform Delta-e bins
    as the committed panel B (0..1.0deg, 0.05deg wide). For each bin's upper
    edge x_k the value is the variance over all pairs with Delta-e < x_k, i.e.
    the count-weighted running mean of the per-bin Ceye -- algebraically the
    cumulative second moment minus Erate^2. The Delta-e -> 0 end is the
    matched-trajectory variance; the all-pairs end approaches the PSTH variance.

    Returns (bin_centers, cum_var) over bins that have any pairs.
    """
    uniform = _compute_uniform_bins()
    bin_centers = np.asarray(uniform["bin_centers"], float)
    count_e = np.asarray(uniform["count_e"], float)
    var_by_bin = np.asarray(uniform["Ceye"], float)[:, j, j]

    contrib = np.nan_to_num(var_by_bin, nan=0.0) * count_e
    cum_num = np.cumsum(contrib)
    cum_den = np.cumsum(count_e)
    ok = cum_den > 0
    cum_var = np.full_like(bin_centers, np.nan)
    cum_var[ok] = cum_num[ok] / cum_den[ok]
    return bin_centers[ok], cum_var[ok]


def plot_cumulative_variance_panel(ax, d_vals=None):
    """Panel B, cumulative redesign (same bins/markers as the committed B)."""
    payload = _load_unit_payload()
    j = int(np.where(np.asarray(payload["neuron_mask"]) == UNIT_ORIG)[0][0])
    var_psth = float(payload["Cpsth"][j, j])

    x, y = _cumulative_binned(j)

    ax.axhline(var_psth, color="k", ls="--", lw=1.0, zorder=1)
    ax.plot(x, y, color="k", lw=1.5, marker="o", ms=4,
            markeredgecolor="k", markeredgewidth=0.4, zorder=3)

    ax.set_xlim(-0.05, 1.03)
    # Variance cannot be negative: floor the y-axis at 0.
    y_hi = float(np.nanmax(y)) * 1.12
    ax.set_ylim(0.0, y_hi)

    ax.text(0.45, var_psth + 0.012 * y_hi, "PSTH variance", ha="left",
            va="bottom", fontsize=7.5)
    ax.annotate("Matched-trajectory variance",
                xy=(x[0], y[0]),
                xytext=(0.30, 0.90), textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color="k", lw=0.9),
                fontsize=7.5, ha="left", va="center")

    # Matched threshold: the FEM-variance double-arrow (large, self-evident gap
    # to the PSTH line) at the left -- no dashed locator needed. Divergent
    # threshold: the curve has already reached PSTH variance, so instead of an
    # (invisible) connector use a dashed locator + a "variance decreases" callout
    # arrow that marks where on the Delta-e axis the divergence sets in.
    if d_vals is not None:
        for key, color in (("matched", MATCHED_COLOR),
                           ("divergent", DIVERGENT_COLOR)):
            # Clamp to the plotted range so a window Delta-e below the first bin
            # (matched ~0) or above the last still marks the nearest point.
            xv = float(np.clip(d_vals[key], x[0], x[-1]))
            yv = float(np.interp(xv, x, y))
            if key == "matched":
                ax.annotate("", xy=(xv, yv), xytext=(xv, var_psth),
                            arrowprops=dict(arrowstyle="<->", color=color,
                                            lw=2.0),
                            zorder=4)
            else:
                ax.axvline(xv, color=color, ls="--", lw=1.0, zorder=2)
                ax.annotate(
                    "Variance decreases as\neye trajectories diverge",
                    xy=(xv, yv), xycoords="data",
                    xytext=(0.66, 0.54), textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    fontsize=7.5, color="k", ha="center", va="center",
                    zorder=4)

    ax.set_xlabel("Eye-trajectory mismatch threshold, Δe < x (°)")
    ax.set_ylabel("Cumulative cross-trial\nrate variance (spk²)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return float(y[-1]), var_psth


def render_v2():
    fig = plt.figure(figsize=(10.5, 4.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.38)
    gs_a = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0, 0], height_ratios=[1.0, 1.0], hspace=0.18,
    )
    ax_eye = fig.add_subplot(gs_a[0, 0])
    ax_spk = fig.add_subplot(gs_a[1, 0], sharex=ax_eye)
    d_vals = plot_eye_rate_example_v2(ax_eye, ax_spk)
    ax_eye.text(-0.16, 1.18, "A", transform=ax_eye.transAxes,
                fontweight="bold", fontsize=12, va="top", ha="left")

    ax_b = fig.add_subplot(outer[0, 1])
    asymptote, var_psth = plot_cumulative_variance_panel(ax_b, d_vals=d_vals)
    ax_b.text(-0.16, 1.18, "B", transform=ax_b.transAxes,
              fontweight="bold", fontsize=12, va="top", ha="left")

    _save(fig, "panels_ab_v2")
    print(f"  cumulative all-pairs asymptote = {asymptote:.4f} spk²  "
          f"(PSTH variance = {var_psth:.4f} spk²)")


# ===========================================================================
# Version 3: black-trace example + "unaccounted-for variability" curve.
#   B = U(Δe) = Ctotal - Crate(Δe): the spike-count variability left
#   unaccounted-for when eye trajectories are matched only to within Δe.
#   Rises from the internal-noise floor (Σ_int = Ctotal - Crate, perfect
#   matching) to the eye-blind PSTH asymptote (Ctotal - Cpsth = Σ_int + Cfem),
#   with the total variability (Ctotal) marked on top. Computed at the
#   production 0.5° fixation radius; divergent reference pinned at Δe = 1°.
# ===========================================================================

def plot_unaccounted_variance_panel(ax, d_vals=None, decomp=None, caption=True):
    """Panel B, 'unaccounted-for variability' redesign (Ctotal − Crate(Δe)).

    ``decomp`` may be a precomputed per-unit decomposition dict (keys
    bin_centers/cum_crate/Ctotal/Crate/Cpsth/sigma_int); if None, the example
    unit's curve is computed. ``caption`` toggles the take-home text (off for the
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
    # Internal-noise floor label (just above its line at mid-x: empty for any
    # floor height since the curve has risen well above the floor there).
    ax.text(0.30, sigma_int + 0.012 * y_hi, "Internal variability",
            color="0.35", fontsize=7.5, ha="left", va="bottom")

    # Right-side decomposition bar at x = xa: FEM (internal floor -> eye-blind
    # level) in blue, stimulus/PSTH (eye-blind level -> total) in red. Each is
    # labelled right-justified next to its arrow.
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
        # Take-home split across the bottom: descriptive phrase bottom-left
        # (left-justified), the fraction equation bottom-right (right-justified,
        # larger so it reads clearly).
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


# Production Figure 2 places panels A and B at ~2.36" wide x ~2.17" tall inside
# an 8.5"-wide manuscript figure, under this font rc-context. Match both so the
# annotation/tick fonts here are representative of the final figure.
_FIG2_RC = {
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
}


def render_v3():
    with plt.rc_context(_FIG2_RC):
        # figsize + margins chosen so each panel axis is ~2.36" x ~2.17", the
        # size panels A/B actually occupy in the composite figure.
        fig = plt.figure(figsize=(6.3, 3.0))
        outer = fig.add_gridspec(
            1, 2, width_ratios=[1.0, 1.0], wspace=0.32,
            left=0.10, right=0.975, top=0.88, bottom=0.16,
        )
        gs_a = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[0, 0], height_ratios=[1.0, 1.0], hspace=0.18,
        )
        ax_eye = fig.add_subplot(gs_a[0, 0])
        ax_spk = fig.add_subplot(gs_a[1, 0], sharex=ax_eye)
        d_vals = plot_eye_rate_example_v2(ax_eye, ax_spk, unit_orig=V3_UNIT_ORIG)
        ax_eye.text(-0.20, 1.12, "A", transform=ax_eye.transAxes,
                    fontweight="bold", fontsize=10, va="top", ha="left")

        ax_b = fig.add_subplot(outer[0, 1])
        asymptote, ctotal = plot_unaccounted_variance_panel(
            ax_b, d_vals=d_vals,
            decomp=_compute_unaccounted_curve(unit_orig=V3_UNIT_ORIG))
        ax_b.xaxis.label.set_size(8)
        ax_b.yaxis.label.set_size(8)
        ax_b.tick_params(labelsize=7)
        ax_b.text(-0.20, 1.12, "B", transform=ax_b.transAxes,
                  fontweight="bold", fontsize=10, va="top", ha="left")

        _save(fig, "panels_ab_v3")
    print(f"  unaccounted-for asymptote (eye ignored) = {asymptote:.4f} spk²  "
          f"(total variability = {ctotal:.4f} spk²)")


def _save(fig, name):
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(description="Debug fig2 panels A/B.")
    p.add_argument("--version", choices=["1", "2", "3", "both", "all"],
                   default="all")
    args, _ = p.parse_known_args()
    print(f"Example: session={SESSION} unit={UNIT_ORIG} "
          f"trials=({TRIAL_A},{TRIAL_B}) axis={'h' if EYE_AXIS == 0 else 'v'}")
    if args.version in ("1", "both", "all"):
        render_v1()
    if args.version in ("2", "both", "all"):
        render_v2()
    if args.version in ("3", "all"):
        render_v3()


if __name__ == "__main__":
    main()
