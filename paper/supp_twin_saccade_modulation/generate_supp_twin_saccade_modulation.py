r"""Supplement: the twin's extraretinal signal is a saccade-locked modulation.

The ``zeroed`` ablation removes only the behavior (eye-velocity / eye-position)
input, so ``residual = twin(intact) - twin(zeroed)`` isolates the twin's
extraretinal contribution. In fig4 this residual looked sparse and biphasic. Here
we show it is a stereotyped, (micro)saccade-locked modulation, consistent across
the population.

Row 1 - five aligned single-trial strips for one example unit (same trial
seriation): Observed | Twin | Twin(ablated) | Residual | Eye speed. Detected
(micro)saccade onsets are marked on the eye-speed strip.

Row 2 - saccade-triggered averages over a -100/+200 ms window:
  (A) example-unit residual STA, with the eye-speed STA overlaid for timing;
  (B) population-mean residual and observed STA (+/- SEM) over reliable units;
  (C) residual STA for every reliable unit, sorted by peak time.

Usage:
    uv run python paper/supp_twin_saccade_modulation/generate_supp_twin_saccade_modulation.py
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR

sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "fig3"))
sys.path.insert(0, str(VISIONCORE_ROOT / "paper" / "supp_twin_saccade_modulation"))

from _fig3_data import configure_matplotlib  # noqa: E402
from _supp_saccade_data import (  # noqa: E402
    compute_sta_bundle, baseline_subtract, RELIABLE_THRESHOLD, DT,
)

FIG_DIR = FIGURES_DIR / "supp_twin_saccade_modulation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESID_COLOR = "#d62728"
OBS_COLOR = "k"
SPEED_COLOR = "#1f77b4"


# --- row 1: raster strips --------------------------------------------------
def _draw_strip(ax, arr, *, cmap, vmin, vmax, window_s, title, markers=None):
    n = arr.shape[0]
    im = ax.imshow(arr, aspect="auto", origin="upper", extent=[0, window_s, n, 0],
                   vmin=vmin, vmax=vmax, cmap=cmap, interpolation="none")
    if markers is not None and len(markers):
        ax.scatter(markers[:, 1] * DT, markers[:, 0] + 0.5, s=3.5,
                   facecolor="#39ff14", edgecolor="k", linewidth=0.15, zorder=3)
    ax.set_xlim(0, window_s)
    ax.set_ylim(n, 0)
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    return im


def _add_hbar(fig, ax, im, label):
    cax = ax.inset_axes([0.0, -0.06, 1.0, 0.045])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5.5, length=2, pad=1)
    cb.set_label(label, fontsize=6, labelpad=2)
    cb.outline.set_linewidth(0.5)


def draw_raster_row(fig, axes, ex):
    w = ex["window_s"]
    rate_max = np.nanpercentile(
        np.concatenate([ex["observed"].ravel(), ex["intact"].ravel(),
                        ex["zeroed"].ravel()]), 97)
    rmax = np.nanpercentile(np.abs(ex["residual"]), 97)
    smax = np.nanpercentile(ex["speed"], 97)

    im0 = _draw_strip(axes[0], ex["observed"], cmap="binary", vmin=0, vmax=rate_max,
                      window_s=w, title="Observed")
    im1 = _draw_strip(axes[1], ex["intact"], cmap="binary", vmin=0, vmax=rate_max,
                      window_s=w, title="Twin")
    _draw_strip(axes[2], ex["zeroed"], cmap="binary", vmin=0, vmax=rate_max,
                window_s=w, title="Twin (ablated)")
    im3 = _draw_strip(axes[3], ex["residual"], cmap="RdBu_r", vmin=-rmax, vmax=rmax,
                      window_s=w, title="Residual")
    im4 = _draw_strip(axes[4], ex["speed"], cmap="magma", vmin=0, vmax=smax,
                      window_s=w, title="Eye speed", markers=ex["sacc_markers"])

    _add_hbar(fig, axes[1], im1, "rate (sp/s)")
    _add_hbar(fig, axes[3], im3, "Δ rate (sp/s)")
    _add_hbar(fig, axes[4], im4, "deg/s")

    # scale bars on the first strip: 100 ms + 10 trials
    n = ex["observed"].shape[0]
    tr = axes[0].get_xaxis_transform()
    axes[0].plot([0, 0.1], [-0.04, -0.04], "k-", lw=2, transform=tr, clip_on=False)
    axes[0].text(0.05, -0.075, "100 ms", transform=tr, ha="center", va="top",
                 fontsize=6)
    ty = axes[0].get_yaxis_transform()
    axes[0].plot([-0.03, -0.03], [n, n - min(10, n)], "k-", lw=2, transform=ty,
                 clip_on=False)
    axes[0].text(-0.06, n - min(10, n) / 2, f"{min(10, n)} trials", transform=ty,
                 ha="right", va="center", rotation=90, fontsize=6)
    axes[0].text(0.5, 1.16, f"{ex['session']}  n{ex['neuron_id']}  "
                 f"({ex['n_sacc']} saccades)", transform=axes[0].transAxes,
                 ha="center", va="bottom", fontsize=8)


# --- row 2: saccade-triggered averages -------------------------------------
def _overlay_speed(ax, t, sp):
    """Overlay a normalized eye-speed trace (timing reference, no second axis)."""
    y0, y1 = ax.get_ylim()
    spn = (sp - np.nanmin(sp)) / (np.nanmax(sp) - np.nanmin(sp) + 1e-9)
    y = y0 + spn * (y1 - y0) * 0.9
    ax.plot(t, y, color=SPEED_COLOR, lw=1.0, alpha=0.45, zorder=0)
    ax.text(t[np.nanargmax(sp)] + 4, y0 + 0.96 * (y1 - y0), "eye speed",
            color=SPEED_COLOR, fontsize=6, ha="left", va="top")
    ax.set_ylim(y0, y1)


def draw_example_sta(ax, ex, bundle):
    t = bundle["lags_ms"]
    bm = bundle["baseline_mask"]
    for k, c, lab in [("observed", OBS_COLOR, "observed"),
                      ("residual", RESID_COLOR, "residual (intact−ablated)")]:
        st = ex["sta"][k]
        st = st - np.nanmean(st[bm])
        ax.plot(t, st, color=c, lw=1.4, label=lab)
    ax.axvline(0, ls=":", c="gray", lw=0.8)
    ax.axhline(0, lw=0.5, c="gray")
    ax.set_xlabel("time from saccade (ms)", fontsize=8)
    ax.set_ylabel("Δ rate (sp/s)", fontsize=8)
    ax.set_title("Example unit STA", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    _clean(ax)
    _overlay_speed(ax, t, ex["sta"]["speed"])


def draw_population_sta(ax, bundle, good):
    t = bundle["lags_ms"]
    n = int(good.sum())
    for key, c, lab in [("observed", OBS_COLOR, "observed"),
                        ("residual", RESID_COLOR, "twin residual")]:
        arr = baseline_subtract(bundle["sta"][key])[good]
        m = np.nanmean(arr, axis=0)
        s = np.nanstd(arr, axis=0) / np.sqrt(n)
        ax.plot(t, m, color=c, lw=1.6, label=lab)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.25, lw=0)
    ax.axvline(0, ls=":", c="gray", lw=0.8)
    ax.axhline(0, lw=0.5, c="gray")
    ax.set_xlabel("time from saccade (ms)", fontsize=8)
    ax.set_ylabel("Δ rate (sp/s)", fontsize=8)
    ax.set_title(f"Population mean STA (n={n} reliable)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    _clean(ax)
    _overlay_speed(ax, t, np.nanmean(bundle["speed_sta_sessions"], axis=0))


def draw_unit_heatmap(fig, ax, bundle, good):
    t = bundle["lags_ms"]
    R = baseline_subtract(bundle["sta"]["residual"])[good]
    peakt = np.array([t[np.nanargmax(np.abs(row))] if np.isfinite(row).any()
                      else 0.0 for row in R])
    order = np.argsort(peakt)
    vmax = np.nanpercentile(np.abs(R), 98)
    im = ax.imshow(R[order], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[t[0], t[-1], R.shape[0], 0], interpolation="none")
    ax.axvline(0, ls=":", c="k", lw=0.8)
    ax.set_xlabel("time from saccade (ms)", fontsize=8)
    ax.set_ylabel("reliable unit (sorted by peak)", fontsize=8)
    ax.set_title(f"Residual STA, all reliable units", fontsize=8)
    ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Δ rate (sp/s)", fontsize=7)
    cb.ax.tick_params(labelsize=6)


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.tick_params(labelsize=7)


def main():
    configure_matplotlib()
    bundle = compute_sta_bundle()
    ex = bundle["example"]
    if ex is None:
        raise RuntimeError("Example neuron payload missing; check EXAMPLE_* in "
                           "_supp_saccade_data.py")
    rel = bundle["reliability"]
    good = np.isfinite(rel) & (rel > RELIABLE_THRESHOLD)

    fig = plt.figure(figsize=(11, 6.2))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1.0, 0.95],
                  hspace=0.5, wspace=0.28)
    raster_axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    draw_raster_row(fig, raster_axes, ex)

    ax_a = fig.add_subplot(gs[1, 0:2])
    ax_b = fig.add_subplot(gs[1, 2:4])
    ax_c = fig.add_subplot(gs[1, 4])
    draw_example_sta(ax_a, ex, bundle)
    draw_population_sta(ax_b, bundle, good)
    draw_unit_heatmap(fig, ax_c, bundle, good)

    for ax, lab in [(raster_axes[0], "A"), (ax_a, "B"), (ax_b, "C"), (ax_c, "D")]:
        ax.text(-0.08, 1.22 if lab == "A" else 1.06, lab, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom", ha="right")

    out_pdf = FIG_DIR / "supp_twin_saccade_modulation.pdf"
    out_png = FIG_DIR / "supp_twin_saccade_modulation.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
