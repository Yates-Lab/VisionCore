"""Combined Figure 3: digital twin mechanism plus compact reafferent geometry.

This compositor collapses the current digital-twin validation figure and the
compact retinal-translation geometry figure into one mechanism figure:

  A  Digital twin schematic
  B  Example neuron PSTH overlay
  C  Held-out response validation
  D  Extraretinal-pathway zeroing control
  E  Empirical vs model FEM modulation
  F  Image-specific local translation directions
  G  Compact, image-generalizing translation-tangent subspace
  H  Translation-predicted recorded FEM covariance

The older full figures remain useful as source/supplemental figures. This file
only selects the panels needed for the combined main-text story.

Usage:
    uv run declan/fig3/generate_figure3_combined.py
"""
from __future__ import annotations

import argparse
import dill
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from VisionCore.paths import VISIONCORE_ROOT

from _fig3_data import FIG_DIR, configure_matplotlib, load_fig3_data
from _fig3_ablation_data import CACHE_PATH as ABLATION_CACHE_PATH
from _fig3_ablation_data import load_ablation_data
from _fig3_helpers import select_example_neuron
from generate_fig3b import plot_panel_b as plot_example_psth
from generate_fig3d import compute_model_one_minus_alpha


GEOM_DIR = VISIONCORE_ROOT / "declan" / "fig4_cov_TFTS"
if str(GEOM_DIR) not in sys.path:
    sys.path.insert(0, str(GEOM_DIR))

import generate_covTFTS_figure as geom  # noqa: E402


POOLED_COLOR = "0.25"
POOLED_FILL = "0.55"
BEHAVIOR_COLOR = "#d62728"
ZEROED_COLOR = "#1f77b4"
WITHIN_MODEL_CACHE = VISIONCORE_ROOT / "outputs" / "cache" / "behavior_vs_vision_within_model.pkl"
PANEL_LETTER_SIZE = 11
PANEL_TITLE_SIZE = 9.0
SUBJECT_COLORS = {"Allen": "tab:blue", "Logan": "tab:green"}


def _panel_title(ax, letter: str):
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=11, pad=4)


def _clear_panel_heading(ax):
    """Remove source-panel headings so the combined figure can place them uniformly."""
    ax.set_title("", loc="left")
    ax.set_title("", loc="center")
    ax.set_title("", loc="right")
    for txt in list(ax.texts):
        if txt.get_transform() == ax.transAxes:
            x, y = txt.get_position()
            if y >= 0.98 and x <= 0.28:
                txt.remove()


def _standard_panel_heading(ax, letter: str, title: str):
    """Place a consistent panel letter/title just above the axes."""
    _clear_panel_heading(ax)
    ax.text(
        -0.035,
        1.045,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LETTER_SIZE,
        fontweight="bold",
        color="#202124",
        clip_on=False,
    )
    ax.text(
        0.08,
        1.045,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
        color="#202124",
        linespacing=0.9,
        clip_on=False,
    )


def _box(ax, xy, wh, text, *, fc="#f7f7f7", ec="#444444", color="#202124"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=0.9,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=8.2, color=color, fontweight="bold", linespacing=1.0)
    return patch


def _arrow(ax, start, end, *, color="#444444", lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    ))


def _plot_twin_mechanism_schematic(ax):
    """Small mechanism schematic for the merged figure."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    _box(ax, (0.06, 0.35), (0.18, 0.30),
         "Gaze-contingent\nretinal movie",
         fc="#f2f6fb", ec=geom.MODEL, color=geom.MODEL)
    _box(ax, (0.35, 0.35), (0.20, 0.30),
         "Digital twin\nResNet + GRU core",
         fc="#f7f7f7", ec="#333333")
    _box(ax, (0.68, 0.35), (0.16, 0.30),
         "V1 readouts\n(rate)",
         fc="#f4eef8", ec=geom.BRIDGE, color=geom.BRIDGE)

    _arrow(ax, (0.24, 0.50), (0.35, 0.50), color=geom.MODEL, lw=1.4)
    _arrow(ax, (0.55, 0.50), (0.68, 0.50), color="#333333", lw=1.4)
    _arrow(ax, (0.84, 0.50), (0.94, 0.50), color=geom.BRIDGE, lw=1.2)

    _box(ax, (0.34, 0.02), (0.22, 0.14),
         "optional eye-state\nbehavior pathway",
         fc="#fbf8fd", ec=geom.BRIDGE, color=geom.BRIDGE)
    _arrow(ax, (0.45, 0.16), (0.45, 0.35), color=geom.BRIDGE, lw=1.1)

    ax.text(0.06, 0.83, "A", fontweight="bold", fontsize=11,
            ha="left", va="top")
    ax.text(0.10, 0.83,
            "retinal-input twin links response prediction to reafferent geometry",
            fontsize=9.2, fontweight="bold", ha="left", va="top",
            color="#202124")
    ax.text(0.94, 0.50, "recorded\nspikes", fontsize=7.1, color="#555555",
            ha="left", va="center")
    ax.text(0.06, 0.25,
            "FEMs move the image on the retina; the behavior pathway is tested as a separate route.",
            fontsize=7.1, color="#555555", ha="left", va="center")


def _plot_existing_schematic(ax, schematic_image: Path | None):
    """Use the existing draft schematic when available."""
    if schematic_image is None or not schematic_image.exists():
        _plot_twin_mechanism_schematic(ax)
        return False

    img = plt.imread(schematic_image)
    rgb = img[..., :3]
    nonwhite = np.any(rgb < 0.985, axis=2)
    ys, xs = np.where(nonwhite)
    if len(xs) and len(ys):
        pad_y = max(5, int(0.03 * (ys.max() - ys.min() + 1)))
        pad_x = max(5, int(0.03 * (xs.max() - xs.min() + 1)))
        y0 = max(0, int(ys.min()) - pad_y)
        y1 = min(img.shape[0], int(ys.max()) + pad_y + 1)
        x0 = max(0, int(xs.min()) - pad_x)
        x1 = min(img.shape[1], int(xs.max()) + pad_x + 1)
        img = img[y0:y1, x0:x1]
    img = _clean_schematic_heading_and_lift_right_half(img)
    ax.imshow(img)
    ax.set_anchor("N")
    ax.set_axis_off()
    ax.text(
        0.01,
        0.98,
        "A",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#202124",
    )
    return True


def _clean_schematic_heading_and_lift_right_half(img: np.ndarray) -> np.ndarray:
    """Remove the embedded right-half heading and nudge that half upward."""
    out = np.ones_like(img)
    h, w = img.shape[:2]
    split_x = int(0.43 * w)
    clear_y = int(0.105 * h)
    shift_y = int(0.065 * h)
    work = img.copy()
    work[:clear_y, split_x:, :3] = 1.0
    if work.shape[-1] == 4:
        work[:clear_y, split_x:, 3] = 1.0
    out[:, :split_x] = work[:, :split_x]
    out[: h - shift_y, split_x:] = work[shift_y:, split_x:]
    return out


def _plot_validation_panel_pooled(ax, data, *, letter: str = "C"):
    """Validation panel: pooled ccnorm histogram across Allen and Logan."""
    ccnorm = data["ccnorm"]
    good = data["good"]
    vals = ccnorm[good & np.isfinite(ccnorm)]
    bins = np.linspace(0, 1, 21)
    med = float(np.nanmedian(vals))
    q25, q75 = np.nanpercentile(vals, [25, 75])
    ax.hist(vals, bins=bins, color=POOLED_FILL, edgecolor="white", alpha=0.55)
    ax.axvline(
        med,
        color=POOLED_COLOR,
        linewidth=2,
        ls=(0, (1, 1)),
    )
    ax.text(0.05, 0.92, f"median {med:.2f} [{q25:.2f}, {q75:.2f}]",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=5.8, color=POOLED_COLOR)
    ax.set_xlabel("Normalized correlation (ccnorm)")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{letter}  Held-out responses", loc="left",
                 fontweight="bold", fontsize=10, pad=4)
    print(
        f"Panel C — pooled (N={len(vals)}): median ccnorm={med:.2f}, "
        f"IQR=[{q25:.2f}, {q75:.2f}]"
    )


def _plot_example_psth_intact_vs_zeroed(ax, abl_data, *, fallback_data, fallback_example):
    """Example PSTH with behavior-input and zeroed-behavior predictions."""
    ex = None if abl_data is None else abl_data.get("example")
    if ex is None:
        plot_example_psth(
            ax=ax,
            data=fallback_data,
            example=fallback_example,
            legend_fontsize=6.0,
            show_ccnorm_title=False,
        )
        if len(ax.lines) >= 2:
            ax.lines[1].set_label("Intact")
            ax.lines[1].set_color(BEHAVIOR_COLOR)
            ax.legend(frameon=False, fontsize=6.0)
        return

    obs = np.nanmean(ex["obs_rate"], axis=0)
    intact = np.nanmean(ex["rate"]["intact"], axis=0)
    zeroed = np.nanmean(ex["rate"]["zeroed"], axis=0)
    t = np.linspace(0, float(ex["window_s"]), obs.size, endpoint=False)
    ax.plot(t, obs, color="k", lw=1.0, label="Observed")
    ax.plot(t, intact, color=BEHAVIOR_COLOR, lw=1.0, label="Intact")
    ax.plot(t, zeroed, color=ZEROED_COLOR, lw=1.0, label="Zeroed")
    ax.set_xlim(0, float(ex["window_s"]))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (sp/s)")
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_within_model_ccnorm(data):
    """Load matched intact-vs-zeroed ccnorm arrays from the within-model cache."""
    if not WITHIN_MODEL_CACHE.exists():
        return None
    with open(WITHIN_MODEL_CACHE, "rb") as f:
        rows = dill.load(f)

    intact = np.concatenate([np.asarray(r["cc_norm"]["beh_intact"], dtype=float) for r in rows])
    zeroed = np.concatenate([np.asarray(r["cc_norm"]["beh_zeroed"], dtype=float) for r in rows])
    ccmax = np.concatenate([np.asarray(r["cc_max"]["beh_intact"], dtype=float) for r in rows])
    good = ccmax > 0.85
    return {"intact": intact, "zeroed": zeroed, "good": good}


def _plot_ccnorm_hist_intact_vs_zeroed(ax, data, *, letter: str = "C"):
    """Overlaid normalized-correlation histograms for intact and zeroed inputs."""
    matched = _load_within_model_ccnorm(data)
    if matched is None:
        intact = data["ccnorm"]
        zeroed = None
        good = np.asarray(data["good"], dtype=bool)
    else:
        intact = matched["intact"]
        zeroed = matched["zeroed"]
        good = matched["good"]

    m_intact = good & np.isfinite(intact)
    bins = np.linspace(0, 1, 21)
    ax.hist(intact[m_intact], bins=bins, color=BEHAVIOR_COLOR, alpha=0.32,
            edgecolor="none", label="Intact")
    ax.axvline(float(np.nanmedian(intact[m_intact])), color=BEHAVIOR_COLOR, lw=1.4)

    if zeroed is not None and len(zeroed) == len(intact):
        both = good & np.isfinite(intact) & np.isfinite(zeroed)
        ax.hist(zeroed[both], bins=bins, color=ZEROED_COLOR, alpha=0.32,
                edgecolor="none", label="Zeroed")
        ax.axvline(float(np.nanmedian(zeroed[both])), color=ZEROED_COLOR, lw=1.4)
        note = f"median Δ={np.nanmedian(zeroed[both] - intact[both]):+.3f}"
        print(f"Panel {letter} — intact/zeroed ccnorm (N={both.sum()}): {note}")
    else:
        note = "zeroed ccnorm\nnot cached"
        ax.text(0.97, 0.92, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=6.0, color=ZEROED_COLOR)
        print(f"Panel {letter} — intact ccnorm only; zeroed ccnorm not cached")

    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized correlation (ccnorm)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    ax.text(0.97, 0.08,
            f"intact median {np.nanmedian(intact[m_intact]):.2f}"
            if zeroed is None else note,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.0, color="0.25")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{letter}  Held-out responses", loc="left",
                 fontweight="bold", fontsize=10, pad=4)


def _plot_prediction_similarity_panel(ax, abl_data, *, cond: str = "zeroed", letter: str = "C"):
    """Held-out prediction: intact vs zeroed single-trial performance."""
    if abl_data is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "ablation cache\nnot found", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=geom.ACCENT)
        return
    x = abl_data["ve"]["intact"]
    y = abl_data["ve"][cond]
    good = abl_data["good"]
    m = good & np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[m], y[m], s=5, alpha=0.38, color=POOLED_COLOR)
    lims = [0, 0.35]
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel("Single-trial $r^2$\n(intact)")
    ax.set_ylabel("Single-trial $r^2$\n(zeroed)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    d = y[m] - x[m]
    med_delta = float(np.nanmedian(d))
    pct = 100.0 * med_delta / float(np.nanmedian(x[m]))
    ax.text(0.97, 0.08,
            f"{pct:+.0f}% of intact median\nmedian Δ$r^2$={med_delta:+.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.4, color="0.25")
    ax.set_title(f"{letter}  Held-out prediction", loc="left",
                 fontweight="bold", fontsize=10, pad=4)
    print(f"Panel {letter} — intact vs zeroed r² (N={m.sum()}): "
          f"median Δr²={med_delta:+.4f}")


def _plot_fem_prediction_by_route(
    ax,
    abl_data,
    fig3_data,
    *,
    cond: str = "zeroed",
    letter: str = "D",
):
    """Prediction performance across empirical FEM modulation for intact and zeroed inputs."""
    if abl_data is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "ablation cache\nnot found", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=geom.ACCENT)
        return
    alpha = fig3_data["alpha"]
    fem = 1.0 - alpha
    y_intact = abl_data["ve"]["intact"]
    y_zeroed = abl_data["ve"][cond]
    good = abl_data["good"] & fig3_data["good"]
    base = good & np.isfinite(fem) & np.isfinite(y_intact) & np.isfinite(y_zeroed)
    x = fem[base]
    intact = y_intact[base]
    zeroed = y_zeroed[base]

    ax.scatter(x, intact, s=4, alpha=0.16, color=BEHAVIOR_COLOR, linewidths=0)
    ax.scatter(x, zeroed, s=4, alpha=0.16, color=ZEROED_COLOR, linewidths=0)
    if x.size >= 20:
        edges = np.quantile(x, np.linspace(0, 1, 7))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        bx, b_intact, b_zeroed = [], [], []
        bin_id = np.digitize(x, edges) - 1
        for b in range(6):
            sel = bin_id == b
            if sel.sum() >= 5:
                bx.append(float(np.nanmedian(x[sel])))
                b_intact.append(float(np.nanmedian(intact[sel])))
                b_zeroed.append(float(np.nanmedian(zeroed[sel])))
        ax.plot(bx, b_intact, "o-", color=BEHAVIOR_COLOR, lw=1.4, ms=3.6,
                label="Intact")
        ax.plot(bx, b_zeroed, "o-", color=ZEROED_COLOR, lw=1.4, ms=3.6,
                label="Zeroed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.35)
    ax.set_xlabel(r"Empirical FEM modulation ($1-\alpha$)")
    ax.set_ylabel("Single-trial $r^2$")
    if len(ax.lines) >= 2:
        ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    med = float(np.nanmedian(zeroed - intact))
    ax.text(0.97, 0.08, f"median Δ$r^2$={med:+.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.4, color="0.25")
    ax.set_title(f"{letter}  FEM modulation", loc="left",
                 fontweight="bold", fontsize=10, pad=4)
    print(f"Panel {letter} — intact/zeroed prediction across FEM modulation "
          f"(N={x.size}): median Δr²={med:+.4f}")


def _plot_fem_modulation_pooled(ax, data, *, letter: str = "D"):
    """Pooled model-vs-empirical 1-alpha scatter."""
    comp = compute_model_one_minus_alpha(data)
    x = comp["emp"]
    y = comp["model"]
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    rho = spearmanr(x, y).correlation if len(x) >= 3 else np.nan
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
    ax.scatter(x, y, s=5, alpha=0.38, color=POOLED_COLOR)
    ax.text(0.06, 0.94, f"ρ={rho:.2f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=5.8, color=POOLED_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel(r"Empirical $1-\alpha$")
    ax.set_ylabel(r"Model $1-\alpha$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{letter}  FEM modulation", loc="left",
                 fontweight="bold", fontsize=10, pad=4)
    print(f"Panel {letter} — pooled (N={len(x)}): Spearman ρ={rho:.3f}")


def _load_within_model_fem_covariance_metrics():
    """Return session-level FEM covariance metrics for intact and zeroed inputs."""
    if not WITHIN_MODEL_CACHE.exists():
        return None
    with open(WITHIN_MODEL_CACHE, "rb") as f:
        rows = dill.load(f)

    out = []
    for r in rows:
        em = r.get("empirical", {})
        metrics = em.get("metrics", {})
        mi = metrics.get("beh_intact")
        mz = metrics.get("beh_zeroed")
        if mi is None or mz is None:
            continue
        out.append({
            "session": r.get("session"),
            "subject": r.get("subject"),
            "n_common_cells": r.get("n_common_cells"),
            "intact_trace_ratio": float(mi.get("tr_ratio", np.nan)),
            "zeroed_trace_ratio": float(mz.get("tr_ratio", np.nan)),
            "intact_overlap_k": float(mi.get("overlap_k", np.nan)),
            "zeroed_overlap_k": float(mz.get("overlap_k", np.nan)),
        })
    return out


def _plot_fem_modulation_intact_vs_zeroed(ax, fig3_data, abl_data=None, *, letter: str = "D"):
    """Paired per-cell model 1-alpha clouds for intact and zeroed inputs."""
    model_oma = None if abl_data is None else abl_data.get("model_one_minus_alpha")
    if not model_oma:
        _plot_fem_modulation_pooled(ax, fig3_data, letter=letter)
        return

    emp = 1.0 - np.asarray(abl_data["alpha"], dtype=float)
    intact = np.asarray(model_oma["intact"], dtype=float)
    zeroed = np.asarray(model_oma["zeroed"], dtype=float)
    good = np.asarray(abl_data["good"], dtype=bool)
    ok = good & np.isfinite(emp) & np.isfinite(intact) & np.isfinite(zeroed)
    emp = emp[ok]
    intact = intact[ok]
    zeroed = zeroed[ok]

    if emp.size == 0:
        _plot_fem_modulation_pooled(ax, fig3_data, letter=letter)
        return

    order = np.argsort(emp)
    for i in order[::2]:
        ax.plot([emp[i], emp[i]], [intact[i], zeroed[i]],
                color="0.72", lw=0.38, alpha=0.32, zorder=1)
    ax.scatter(
        emp,
        intact,
        s=5,
        color=BEHAVIOR_COLOR,
        alpha=0.32,
        linewidths=0,
        zorder=2,
        label="Intact",
    )
    ax.scatter(
        emp,
        zeroed,
        s=5,
        color=ZEROED_COLOR,
        alpha=0.30,
        linewidths=0,
        zorder=2,
        label="Zeroed",
    )
    delta = zeroed - intact
    med_delta = float(np.nanmedian(delta))
    rho_i = spearmanr(emp, intact).correlation
    rho_z = spearmanr(emp, zeroed).correlation
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.5, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel(r"Empirical $1-\alpha$")
    ax.set_ylabel(r"Model $1-\alpha$")
    ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    ax.text(
        0.97,
        0.08,
        f"Δ={med_delta:+.3f}\nρ {rho_i:.2f}/{rho_z:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color="0.25",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{letter}  FEM modulation", loc="left",
                 fontweight="bold", fontsize=10, pad=4)
    print(
        f"Panel {letter} — intact/zeroed model 1-alpha (N={emp.size}): "
        f"median Δ={med_delta:+.4f}, "
        f"Spearman intact={rho_i:.3f}, zeroed={rho_z:.3f}"
    )


def _plot_improvement_vs_fem_modulation(ax, data, *, legend_fontsize: float = 5.8):
    """Ryan Figure 4G: model/PSTH single-trial r2 improvement vs FEM modulation."""
    ve_model = np.asarray(data["ve_model"], dtype=float)
    ve_psth = np.asarray(data["ve_psth"], dtype=float)
    alpha = np.asarray(data["alpha"], dtype=float)
    subjects = np.asarray(data["subjects"])
    good = np.asarray(data["good"], dtype=bool)
    has_alpha = good & np.isfinite(alpha) & np.isfinite(ve_model) & np.isfinite(ve_psth) & (ve_psth > 0)

    plotted = False
    for subj, color in SUBJECT_COLORS.items():
        mask = has_alpha & (subjects == subj)
        if not mask.any():
            continue
        fem_mod = 1.0 - alpha[mask]
        improvement = ve_model[mask] / ve_psth[mask]
        ax.scatter(
            fem_mod,
            improvement,
            s=5,
            alpha=0.5,
            color=color,
            linewidths=0,
            label=subj,
        )
        plotted = True

    ax.axhline(1, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r"FEM modulation ($1-\alpha$)")
    ax.set_ylabel("$r^2$ improvement\n(Model / PSTH)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    if plotted:
        ax.legend(frameon=False, fontsize=legend_fontsize, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fem_all = 1.0 - alpha[has_alpha]
    improvement_all = ve_model[has_alpha] / ve_psth[has_alpha]
    ok = np.isfinite(fem_all) & np.isfinite(improvement_all)
    rho = spearmanr(fem_all[ok], improvement_all[ok]).correlation if ok.sum() >= 3 else np.nan
    ax.text(0.97, 0.92, f"ρ={rho:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.8, color="0.25")
    print(
        f"Panel E — Ryan Fig 4G improvement vs FEM modulation "
        f"(N={ok.sum()}): Spearman ρ={rho:.3f}"
    )


def _plot_ablation_r2_pooled(ax, data, *, cond: str = "zeroed", letter: str = "E"):
    """Pooled intact-vs-zeroed single-trial r2 scatter."""
    x = data["ve"]["intact"]
    y = data["ve"][cond]
    good = data["good"]
    m = good & np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[m], y[m], s=5, alpha=0.38, color=POOLED_COLOR)
    lims = [0, 0.35]
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Single-trial $r^2$ (intact)")
    ax.set_ylabel("Single-trial $r^2$ (zeroed)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    d = y[m] - x[m]
    med_delta = float(np.nanmedian(d))
    pct = 100.0 * med_delta / float(np.nanmedian(x[m]))
    ax.text(0.97, 0.08,
            f"{pct:+.0f}% of intact median\nmedian Δ$r^2$={med_delta:+.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="0.25")
    ax.set_title(f"{letter}  Eye-state zeroing", loc="left",
                 fontweight="bold", fontsize=10, pad=4)
    print(f"Panel {letter} — pooled (N={m.sum()}): median Δr²={med_delta:+.4f}")


def _replace_panel_label_text(ax, old_prefix: str, new_text: str, fontsize: float = 7.3):
    for txt in ax.texts:
        if txt.get_text().startswith(old_prefix):
            txt.set_text(new_text)
            txt.set_fontsize(fontsize)
            txt.set_linespacing(0.9)
            return


def _load_geometry_data(paths: geom.DataPaths):
    """Load the geometry datasets used by selected panels."""
    union_df = geom.load_union(paths)
    basis_df = geom.load_basis(paths)
    spec_df = geom.load_union_spectrum(paths, delta=0.25, n_show=32)
    null_spec_df = geom.load_null_spectrum_summary(paths, delta=0.25, n_show=32)
    tangent_data = geom.load_tangent_family(paths.tangent_maps) if paths.tangent_maps else None
    closure_summary_df, closure_metrics_df, closure_audit = geom.load_panel_f_closure(paths)
    return {
        "union_df": union_df,
        "basis_df": basis_df,
        "spec_df": spec_df,
        "null_spec_df": null_spec_df,
        "tangent_data": tangent_data,
        "closure_summary_df": closure_summary_df,
        "closure_metrics_df": closure_metrics_df,
        "closure_audit": closure_audit,
    }


def _load_ablation_cache():
    """Load ablation data without triggering a heavy inference run."""
    if not ABLATION_CACHE_PATH.exists():
        return None
    return load_ablation_data(recompute=False)


def _plot_ablation_placeholder(ax):
    ax.set_axis_off()
    ax.set_title("D  Extraretinal-pathway zeroing control", loc="left",
                 fontweight="bold", fontsize=10, pad=4)
    ax.text(0.5, 0.58, "ablation cache not found",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, color=geom.ACCENT, fontweight="bold")
    ax.text(0.5, 0.42, f"Missing: {ABLATION_CACHE_PATH.name}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=7.0, color="0.45")


def _plot_covariance_closure_subset(
    ax,
    closure_summary: pd.DataFrame | None,
    closure_metrics: pd.DataFrame | None,
    *,
    letter: str = "I",
):
    """Two-control version of geometry Figure 4E."""
    geom.panel_label(ax, letter, "Translation-predicted\nFEM covariance")
    ax.set_ylabel("Excess capture\nover unit-shuffle null")

    if closure_summary is None or closure_metrics is None or len(closure_summary) == 0:
        ax.text(0.5, 0.52, "finite-difference closure\nnot found",
                transform=ax.transAxes, ha="center", va="center",
                color=geom.ACCENT, fontsize=8)
        geom.clean_axes(ax, grid=True)
        return

    source = "fd_sample_eye_trace_cov"
    target = "psd"
    k = 2
    controls = ["none", "global_rate+target_pc1"]
    labels = ["none", "global\n+ PC1"]

    s = closure_summary[
        (closure_summary["target_variant"].astype(str) == target)
        & (closure_summary["basis_source"].astype(str) == source)
        & (closure_summary["k"].astype(int) == k)
    ].copy()
    m = closure_metrics[
        (closure_metrics["target_variant"].astype(str) == target)
        & (closure_metrics["basis_source"].astype(str) == source)
        & (closure_metrics["k"].astype(int) == k)
        & (closure_metrics["row_status"].astype(str) == "ok")
    ].copy()

    rows = []
    finite_vals: list[float] = []
    rng = np.random.default_rng(4)
    for i, control in enumerate(controls):
        sr = s[s["projection_control"].astype(str) == control]
        mr = m[m["projection_control"].astype(str) == control]
        if len(sr) == 0 or len(mr) == 0:
            continue

        capture = float(sr["capture_mean"].iloc[0])
        mean = float(sr["effect_unit_mean"].iloc[0])
        lo = float(sr["effect_unit_boot_ci_low"].iloc[0])
        hi = float(sr["effect_unit_boot_ci_high"].iloc[0])
        vals = pd.to_numeric(
            mr["effect_minus_unit_shuffle_median"], errors="coerce"
        ).dropna().to_numpy(float)
        finite_vals.extend([mean, lo, hi, *vals])

        jitter = rng.uniform(-0.08, 0.08, size=vals.size)
        ax.scatter(np.full(vals.size, i) + jitter, vals,
                   s=13, color="0.25", alpha=0.24, linewidths=0, zorder=2)
        ax.errorbar(i, mean,
                    yerr=[[max(mean - lo, 0.0)], [max(hi - mean, 0.0)]],
                    fmt="o", color=geom.BRIDGE, ecolor=geom.BRIDGE,
                    elinewidth=1.5, capsize=3.4, markersize=6.0,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=4)
        rows.append((control, capture, mean, lo, hi, vals.size,
                     int(sr["n_effect_positive"].iloc[0])))

    ax.axhline(0, color="0.48", lw=0.75, ls=":", zorder=1)
    ax.set_xticks(np.arange(len(controls)))
    ax.set_xticklabels(labels)
    finite = np.asarray(finite_vals, dtype=float)
    finite = finite[np.isfinite(finite)]
    ymax = max(0.46, float(np.nanmax(finite)) + 0.045) if finite.size else 0.46
    ax.set_ylim(-0.05, ymax)
    geom.clean_axes(ax, grid=True)

    full = next((r for r in rows if r[0] == "none"), None)
    if full is not None:
        _, capture, mean, lo, hi, _n, _n_pos = full
        ax.text(0.04, 0.96,
                "Translation subspace predicts\n"
                "recorded FEM covariance\n"
                f"{100 * capture:.0f}% total; excess\n"
                f"+{mean:.3f} [{lo:.3f}, {hi:.3f}]",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.15,
                color=geom.BRIDGE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.23", fc="white",
                          ec=geom.BRIDGE_L, lw=0.7, alpha=0.96))


def _shift_axes_y(axes, dy: float):
    """Translate axes vertically in figure coordinates without touching row peers."""
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])


def _pad_axis_limits(ax, *, xfrac: float = 0.05, yfrac: float = 0.05):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = (x1 - x0) * xfrac
    dy = (y1 - y0) * yfrac
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + dy)


def _move_first_inset_axes(ax, bounds):
    """Move the first inset axes using parent-axes-relative bounds."""
    if not getattr(ax, "child_axes", None):
        return
    inset = ax.child_axes[0]
    parent = ax.get_position()
    x, y, w, h = bounds
    inset.set_axes_locator(None)
    inset.set_position([
        parent.x0 + x * parent.width,
        parent.y0 + y * parent.height,
        w * parent.width,
        h * parent.height,
    ])


def _write_sidecars(out_dir: Path, paths: geom.DataPaths, manifest: dict):
    caption = """Figure 3. A retinal-input digital twin reveals a compact reafferent geometry underlying FEM-linked V1 variability.

(A) Gaze-contingent digital twin architecture. The model receives the retinal stimulus history and optional extraretinal behavior inputs, then predicts simultaneously recorded V1 responses. (B) Observed PSTH for an example reliable neuron, overlaid with predictions from the intact behavior-input twin and the same twin with the separate behavior input zeroed. (C) Held-out stimulus-locked response prediction across pooled Allen and Logan cells, shown by normalized correlation distributions for intact and behavior-zeroed predictions from the same twin. (D) Single-trial prediction is nearly unchanged when the separate extraretinal eye-state pathway is zeroed, pooled across Allen and Logan, supporting a retinal-input route for FEM-linked variability. (E) The twin's single-trial improvement over a PSTH baseline is largest for cells with stronger empirical FEM modulation, measured as \\(1-\\alpha\\). (F) Small retinal translations induce image-specific local response directions rather than a universal signed x/y population axis. (G) Pooling those local tangents reveals a compact translation subspace. (H) An image-disjoint basis captures held-out translation tangent variance above unit-shuffled controls. (I) Finite-difference fitted-twin translation covariances capture the recorded FEM covariance component above a unit-shuffled source-basis null, shown for no projection control and after removing both global-rate and target-PC1 components.
"""
    (out_dir / "fig3_combined_mechanism_caption.md").write_text(caption, encoding="utf-8")
    (out_dir / "fig3_combined_mechanism_caption.txt").write_text(caption, encoding="utf-8")

    readme = f"""# Combined Figure 3

Generated by `declan/fig3/generate_figure3_combined.py`.

This is the merged digital-twin/mechanism figure. It keeps the old Figure 3 and
covTFTS Figure 4 scripts as source and supplemental material, but promotes only
the panels needed for the main-text mechanism chain:

digital twin schematic -> intact-vs-zeroed example PSTH -> held-out response
validation -> extraretinal route control -> FEM-linked prediction gain ->
image-specific translations -> compact
image-generalizing tangent geometry -> translation-predicted recorded FEM
covariance closure.

## Outputs
- `fig3_combined_mechanism.png`
- `fig3_combined_mechanism.pdf`
- `fig3_combined_mechanism.svg`
- `fig3_combined_mechanism_caption.md`
- `fig3_combined_mechanism_manifest.json`

## Geometry Source
- Tangent maps: `{paths.tangent_maps}`
- Union spectrum: `{paths.spec_file}`
- Union summary: `{paths.union_file}`
- Basis file: `{paths.basis_file}`
- Ablation cache: `{ABLATION_CACHE_PATH}`
- Finite-difference closure summary: `{paths.panel_f_closure_summary_file}`
- Finite-difference closure metrics: `{paths.panel_f_closure_metrics_file}`
- Finite-difference closure audit: `{paths.panel_f_closure_audit_file}`

## Warnings
{chr(10).join("- " + w for w in paths.warnings) or "- none"}
"""
    (out_dir / "fig3_combined_mechanism_README.md").write_text(readme, encoding="utf-8")

    with open(out_dir / "fig3_combined_mechanism_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)


def compose(
    *,
    recompute: bool = False,
    tfts_root: Path | None = None,
    out_dir: Path = FIG_DIR,
    schematic_image: Path | None = None,
    dpi: int = 300,
):
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_fig3_data(recompute=recompute)
    example = select_example_neuron(data)
    abl = _load_ablation_cache()

    if tfts_root is None:
        tfts_root = VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure_prod_v2"
    paths = geom.resolve_paths(tfts_root)
    g = _load_geometry_data(paths)

    if schematic_image is None:
        schematic_image = out_dir / "panel_a_schematic_draft.png"

    fig = plt.figure(figsize=(13.2, 10.6), constrained_layout=False)
    gs = GridSpec(
        3, 1,
        figure=fig,
        left=0.055,
        right=0.985,
        bottom=0.06,
        top=0.92,
        # Panel A is 2.659:1. Give its row the same vertical budget as the
        # combined two-row block below, so that A sets the horizontal frame.
        height_ratios=[2.25, 1.0, 1.35],
        hspace=0.24,
    )

    # Row 1. Model schematic.
    ax_a = fig.add_subplot(gs[0, 0])
    used_existing_schematic = _plot_existing_schematic(ax_a, schematic_image)

    # Row 2. Digital-twin example, validation, ablation control, and FEM modulation.
    gs_mid = gs[1, 0].subgridspec(
        1,
        6,
        width_ratios=[0.15, 1.0, 1.0, 1.0, 1.0, 0.15],
        wspace=0.36,
    )
    ax_b = fig.add_subplot(gs_mid[0, 1])
    _plot_example_psth_intact_vs_zeroed(
        ax_b,
        abl,
        fallback_data=data,
        fallback_example=example,
    )
    _standard_panel_heading(ax_b, "B", "Example PSTH")

    ax_c = fig.add_subplot(gs_mid[0, 2])
    _plot_ccnorm_hist_intact_vs_zeroed(ax_c, data, letter="C")
    _standard_panel_heading(ax_c, "C", "Held-out responses")

    ax_d = fig.add_subplot(gs_mid[0, 3])
    if abl is not None:
        _plot_ablation_r2_pooled(ax_d, abl, cond="zeroed", letter="D")
    else:
        _plot_ablation_placeholder(ax_d)
    _standard_panel_heading(ax_d, "D", "Eye-state zeroing")

    ax_e = fig.add_subplot(gs_mid[0, 4])
    _plot_improvement_vs_fem_modulation(ax_e, data)
    _standard_panel_heading(ax_e, "E", "FEM-linked model gain")
    _shift_axes_y([ax_b, ax_c, ax_d, ax_e], 0.035)

    # Row 3. Reafferent geometry.
    gs_bottom = gs[2, 0].subgridspec(
        1,
        6,
        width_ratios=[0.14, 1.45, 1.35, 1.35, 0.70, 0.46],
        wspace=0.58,
    )
    ax_f = fig.add_subplot(gs_bottom[0, 1])
    if g["tangent_data"] is not None:
        geom.plot_panel_b(ax_f, g["tangent_data"], n_show=18, letter="F")
        for txt in list(ax_f.texts):
            if "charts in response PCA" in txt.get_text() or "local charts shown" in txt.get_text():
                txt.remove()
        _pad_axis_limits(ax_f, xfrac=0.04, yfrac=0.06)
        _move_first_inset_axes(ax_f, [0.055, 0.735, 0.25, 0.25])
    else:
        ax_f.text(0.5, 0.5, "tangent maps\nnot found", transform=ax_f.transAxes,
                  ha="center", va="center", color=geom.ACCENT, fontsize=8)
        geom.panel_label(ax_f, "F", "Local translation directions\nare image-specific")
        geom.clean_axes(ax_f)
    _standard_panel_heading(ax_f, "F", "Image-specific directions")

    ax_g1 = fig.add_subplot(gs_bottom[0, 2])
    geom.plot_panel_c(
        ax_g1,
        g["spec_df"],
        g["union_df"],
        null_spec_df=g["null_spec_df"],
        letter="G",
    )
    _replace_panel_label_text(ax_g1, "Pooled translation", "Compact\nsubspace")
    leg = ax_g1.get_legend()
    if leg is not None:
        leg.remove()
    _standard_panel_heading(ax_g1, "G", "Compact subspace")
    ax_g2 = fig.add_subplot(gs_bottom[0, 3])
    geom.plot_panel_e(ax_g2, g["basis_df"], paths.basis_source_label, letter="H")
    _replace_panel_label_text(ax_g2, "Compactness", "Cross-image\ngeneralization")
    for txt in list(ax_g2.texts):
        if "0% image-ID leakage" in txt.get_text():
            txt.set_text(txt.get_text().replace("\n0% image-ID leakage", ""))
    leg = ax_g2.get_legend()
    if leg is not None:
        leg.remove()
    _standard_panel_heading(ax_g2, "H", "Cross-image\ngeneralization")

    ax_h = fig.add_subplot(gs_bottom[0, 4])
    _plot_covariance_closure_subset(
        ax_h,
        g["closure_summary_df"],
        g["closure_metrics_df"],
        letter="I",
    )
    leg = ax_h.get_legend()
    if leg is not None:
        leg.remove()
    _standard_panel_heading(ax_h, "I", "Predicted FEM\ncovariance")

    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"fig3_combined_mechanism.{ext}",
                    dpi=dpi, bbox_inches="tight")

    manifest = {
        "figure": "fig3_combined_mechanism",
        "digital_twin_cache": str(VISIONCORE_ROOT / "outputs" / "cache" / "fig3_digitaltwin.pkl"),
        "ablation_cache": str(ABLATION_CACHE_PATH),
        "ablation_cache_present": abl is not None,
        "geometry_root": str(tfts_root),
        "schematic_image": str(schematic_image),
        "used_existing_schematic": used_existing_schematic,
        "geometry_warnings": paths.warnings,
        "source_files": {
            "combined_script": str(Path(__file__).resolve()),
            "digital_twin_compositor": str(VISIONCORE_ROOT / "declan" / "fig3" / "generate_figure3.py"),
            "geometry_compositor": str(VISIONCORE_ROOT / "declan" / "fig4_cov_TFTS" / "generate_covTFTS_figure.py"),
            "tangent_maps": str(paths.tangent_maps),
            "union_file": str(paths.union_file),
            "spec_file": str(paths.spec_file),
            "basis_file": str(paths.basis_file),
            "panel_f_closure_summary_file": str(paths.panel_f_closure_summary_file),
            "panel_f_closure_metrics_file": str(paths.panel_f_closure_metrics_file),
            "panel_f_closure_audit_file": str(paths.panel_f_closure_audit_file),
        },
        "panel_mapping": {
            "A": "current Figure 3A schematic",
            "B": "example reliable-neuron PSTH with behavior-input and zeroed-behavior predictions",
            "C": "intact and behavior-zeroed ccnorm histograms from behavior_vs_vision_within_model cache",
            "D": "current Figure 3I zeroed extraretinal input vs intact single-trial r2",
            "E": "current Figure 3D empirical vs model 1-alpha",
            "F": "current geometry Figure 4A local translation charts",
            "G": "current geometry Figure 4B compact translation-tangent subspace",
            "H": "current geometry Figure 4C image-disjoint generalization",
            "I": "current geometry Figure 4E covariance closure subset (none and global_rate+target_pc1)",
        },
    }
    _write_sidecars(out_dir, paths, manifest)
    return fig, manifest


def parse_args():
    p = argparse.ArgumentParser(description="Generate merged digital-twin/mechanism Figure 3.")
    p.add_argument("--recompute", action="store_true",
                   help="Force digital-twin recomputation instead of cached results.")
    p.add_argument("--tfts-root", type=Path,
                   default=VISIONCORE_ROOT / "outputs" / "twin_feature_tangent_structure_prod_v2",
                   help="Root containing twin feature tangent structure outputs.")
    p.add_argument("--out-dir", type=Path, default=FIG_DIR,
                   help="Directory for figure outputs.")
    p.add_argument("--schematic-image", type=Path, default=None,
                   help="Existing draft schematic image to use for Panel A.")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    fig, _manifest = compose(
        recompute=args.recompute,
        tfts_root=args.tfts_root,
        out_dir=args.out_dir,
        schematic_image=args.schematic_image,
        dpi=args.dpi,
    )
    plt.close(fig)
    print(f"Saved combined Figure 3 to: {args.out_dir}")


if __name__ == "__main__":
    main()
