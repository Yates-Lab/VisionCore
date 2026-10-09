#!/usr/bin/env python3
"""Draw the stabilization-control supplement from retained scores and one example.

Rendering needs only the compact per-unit scores and the panel-A example saved
by example.py, so the figure can be revised without model inference:

    uv run python paper/supp_stabilization_control/figure.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D

# Full, Retinal, and Global match paper/fig3/generate_figure3.py.
COLORS = {"psth": ".55", "full": "#1f77b4", "retinal": "#d62728",
          "global": "#b97800", "history": "#009e73", "trial": "#7b3294"}
LABELS = {"psth": "PSTH", "full": "Full", "retinal": "Retinal-only",
          "global": "Global", "history": "History-local", "trial": "Trial"}
SUBTITLES = {"full": "follows gaze", "history": "frozen per history",
             "trial": "frozen at trial mean", "global": "frozen at session center"}
# Retinal stabilization widens from left to right.
STABILIZATION = ("full", "history", "trial", "global")
ABLATIONS = ("retinal", "history", "trial", "global")
PANEL_CONDITIONS = {
    "ccnorm": ("full", *ABLATIONS),
    "fraction": ("psth", "full", *ABLATIONS),
}
# Projected time and horizontal-position axes of the stimulus cubes. Time runs
# left to right, as in the Figure 3A lag cube, and the latest frame faces right.
CUBE_TIME, CUBE_DEPTH = (1.3, -.26), (.74, .27)


def _window_path(example, condition, axis, history):
    """Retinal-window position over one input history, in arcmin."""
    if condition == "full":
        return example["gaze_arcmin"][history, axis]
    return np.full(history.stop - history.start, example["frozen_arcmin_" + condition][axis])


def _draw_gaze(ax, example, axis, history, view, limits):
    time, gaze = example["time_ms"], example["gaze_arcmin"]
    anchor = example["frozen_arcmin_trial"][axis]
    ax.plot(time[[0, -1]], (0, 0), color=COLORS["global"], lw=1, ls=(0, (4, 2)))
    ax.plot(time[[0, -1]], (anchor, anchor), color=COLORS["trial"], lw=1, ls=(0, (4, 2)))
    ax.plot(time, gaze[:, axis], color=".15", lw=.9)
    for condition in reversed(STABILIZATION):
        path = _window_path(example, condition, axis, history)
        # Full is a translucent band under the trace; frozen windows are opaque lines.
        style = dict(lw=5, alpha=.4) if condition == "full" else dict(lw=2.2)
        ax.plot(time[history], path, color=COLORS[condition], solid_capstyle="butt", **style)
        ax.plot(time[history][-1], path[-1], "o", color=COLORS[condition],
                ms=6 if condition == "history" else 3.8, mec="white", mew=.6, zorder=5)
    ax.axvline(time[history][-1], color=".3", lw=.7, ls=(0, (2, 2)), zorder=1)
    ax.set_ylim(*limits)
    ax.set_xlim(time[view][0], time[view][-1])
    ax.set_ylabel(f"{('Horizontal', 'Vertical')[axis]} gaze\n(arcmin)")


def _draw_cube(ax, movie, color, limits):
    """Oblique view of one (lag, row, column) retinal movie."""
    kw = dict(cmap="gray", vmin=limits[0], vmax=limits[1], interpolation="nearest")
    middle = movie.shape[1] // 2

    def face(image, origin, u, v):
        im = ax.imshow(image, extent=(0, 1, 0, 1), origin="lower", **kw)
        im.set_transform(Affine2D.from_values(*u, *v, *origin) + ax.transData)

    (a0, a1), (b0, b1) = CUBE_TIME, CUBE_DEPTH
    face(movie[:, ::-1, middle].T, (0, 0), CUBE_TIME, (0, 1))       # vertical position x time
    face(movie[:, middle, :].T, (0, 1), CUBE_TIME, CUBE_DEPTH)      # horizontal position x time
    face(movie[-1][::-1], CUBE_TIME, CUBE_DEPTH, (0, 1))            # latest frame
    for xs, ys in (((0, a0, a0, 0, 0), (0, a1, a1 + 1, 1, 0)),
                   ((0, b0, a0 + b0, a0), (1, 1 + b1, 1 + a1 + b1, 1 + a1)),
                   ((a0, a0 + b0, a0 + b0), (a1, a1 + b1, 1 + a1 + b1))):
        ax.plot(xs, ys, color=color, lw=1.6, solid_joinstyle="round", clip_on=False)
    ax.set_xlim(-.04, a0 + b0 + .04)
    ax.set_ylim(a1 - .04, 1 + b1 + .04)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_example(fig, spec, example):
    grid = spec.subgridspec(2, 3, width_ratios=(1.3, 1, 1), wspace=.12, hspace=.3)
    time, gaze, endpoint = example["time_ms"], example["gaze_arcmin"], int(example["endpoint"])
    n_lags = len(example["movie_full"])
    history = slice(endpoint - n_lags + 1, endpoint + 1)
    end = time[endpoint]
    view = (time >= end - 410) & (time <= end + 160)
    trial = example["frozen_arcmin_trial"]
    span = time[history]
    axes = []
    for axis in range(2):
        seen = gaze[view, axis]
        limits = min(seen.min(), trial[axis], 0) - 5, max(seen.max(), trial[axis], 0) + 4
        ax = fig.add_subplot(grid[axis, 0])
        _draw_gaze(ax, example, axis, history, view, limits)
        # The history bar is the only time scale.
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)
        axes.append(ax)
    low = axes[0].get_ylim()[0]
    axes[0].plot(span[[0, -1]], (low + 1.2,) * 2, color=".35", lw=.8)
    axes[0].text(span.mean(), low + 1.6, "250-ms history", ha="center", va="bottom",
                 fontsize=6.5, color=".35")
    # Name the two reference lines on the axis that separates them, each on the side away from gaze.
    axis = int(np.argmax(np.abs(trial)))
    before = np.median(gaze[view][time[view] < span[0], axis])
    for level, text, color in ((0., "session center", COLORS["global"]),
                               (trial[axis], "trial mean", COLORS["trial"])):
        above = level > before
        axes[axis].text(time[view][0] + 6, level + (.8 if above else -.8), text, color=color,
                        va="bottom" if above else "top", fontsize=6.5)
    gray = np.percentile(example["movie_full"], [1, 99])
    for index, condition in enumerate(STABILIZATION):
        ax = fig.add_subplot(grid[index // 2, 1 + index % 2])
        _draw_cube(ax, example["movie_" + condition].astype(float), COLORS[condition], gray)
        title = ax.text(0, 1.33, LABELS[condition], color=COLORS[condition], fontsize=8,
                        weight="bold", va="bottom")
        ax.annotate(SUBTITLES[condition], (1, 0), xycoords=title, xytext=(4, 0),
                    textcoords="offset points", fontsize=6.2, color=".3", va="bottom")
        if index == 2:
            (a0, a1), (b0, b1) = CUBE_TIME, CUBE_DEPTH
            ax.annotate("", (a0, a1 - .1), (0, -.1), annotation_clip=False,
                        arrowprops=dict(arrowstyle="->", color=".35", lw=.7))
            ax.text(a0 / 2, a1 / 2 - .16, "250 ms", ha="center", va="top", fontsize=6.5,
                    color=".35", rotation=-11)
            ax.text(a0 + b0 / 2 + .05, a1 + b1 / 2 - .08, "0.9°", ha="center", va="top",
                    fontsize=6.5, color=".35", rotation=20)
    return axes[0]


def _boxes(ax, groups, keys, number_y, number_format="{:.3f}"):
    positions = np.arange(len(keys), dtype=float)
    bp = ax.boxplot(groups, positions=positions, widths=.56, whis=(10, 90),
                    showfliers=False, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.1})
    for patch, key in zip(bp["boxes"], keys):
        patch.set_facecolor(COLORS[key]); patch.set_alpha(.55)
    ax.set_xticks(positions, [LABELS[key] for key in keys], rotation=35, ha="right")
    for position, group in zip(positions, groups):
        ax.text(position, number_y, number_format.format(np.median(group)), ha="center",
                va="bottom", fontsize=5.8, color=".25")


def _check_whiskers(ax, groups, name, ceiling):
    """Guard against cropping a displayed whisker or overprinting a median label."""
    low = ax.get_ylim()[0]
    if any(np.percentile(v, 10) < low or np.percentile(v, 90) > ceiling for v in groups):
        raise AssertionError(f"Displayed {name} whisker outside plotting bounds")


def plot(arrays, example):
    """Return the figure for per-unit scores keyed ``<metric>_<condition>``."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(7.1, 6.0))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.05, 1), left=.095, right=.975,
                             bottom=.11, top=.95, hspace=.36)
    ax = _draw_example(fig, outer[0], example)
    ax.text(-.24, 1.12, "A", transform=ax.transAxes, fontsize=11, weight="bold")

    bottom = outer[1].subgridspec(1, 3, width_ratios=(5, 6, 4.8), wspace=.5)
    scores = {name: [np.asarray(arrays[name+"_"+key])[np.asarray(arrays[name+"_mask"])]
                     for key in keys] for name, keys in PANEL_CONDITIONS.items()}

    ax = fig.add_subplot(bottom[0])
    _boxes(ax, scores["ccnorm"], PANEL_CONDITIONS["ccnorm"], .9)
    ax.set_ylim(.2, 1.)
    ax.set_ylabel("Normalized correlation")
    ax.set_title("Trial-averaged prediction", fontsize=8.5)
    ax.text(-.42, 1.07, "B", transform=ax.transAxes, fontsize=11, weight="bold")
    _check_whiskers(ax, scores["ccnorm"], "ccnorm", .9)

    ax = fig.add_subplot(bottom[1])
    keys = PANEL_CONDITIONS["fraction"]
    _boxes(ax, scores["fraction"], keys, .72)
    ax.axhline(0, color=".4", lw=.7)
    ax.axhline(float(np.median(scores["fraction"][keys.index("psth")])), color=".55", lw=.7, ls="--")
    ax.set_ylim(-.3, .82)
    ax.set_ylabel("Fraction of conditional rate\nvariance explained")
    ax.set_title("Single-trial prediction", fontsize=8.5)
    ax.text(-.36, 1.07, "C", transform=ax.transAxes, fontsize=11, weight="bold")
    _check_whiskers(ax, scores["fraction"], "fraction", .72)

    # Each unit's ablated score minus its own Full score.
    ax = fig.add_subplot(bottom[2])
    mask = np.asarray(arrays["fraction_mask"])
    costs = [(np.asarray(arrays["fraction_"+key]) - np.asarray(arrays["fraction_full"]))[mask]
             for key in ABLATIONS]
    top = max(np.percentile(cost, 90) for cost in costs)
    _boxes(ax, costs, ABLATIONS, top + .03, "{:+.3f}")
    ax.axhline(0, color=".4", lw=.7)
    ax.set_ylim(min(np.percentile(cost, 10) for cost in costs) - .09, top + .12)
    ax.set_ylabel("Δ fraction explained\n(condition − Full, paired)")
    ax.set_title("Single-trial ablation cost", fontsize=8.5)
    ax.text(-.5, 1.07, "D", transform=ax.transAxes, fontsize=11, weight="bold")
    return fig


def save(fig, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"stabilization_control.{suffix}", dpi=220)


def main():
    from VisionCore.paths import FIGURES_DIR, VISIONCORE_ROOT

    retained = VISIONCORE_ROOT / "manuscript/analysis/stabilization_control"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=retained / "paired_scores.npz")
    parser.add_argument("--example", type=Path, default=retained / "panel_a_example.npz")
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR / "supp_stabilization_control")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    fig = plot(np.load(args.scores), np.load(args.example))
    save(fig, args.out_dir)
    plt.close(fig)
    print(args.out_dir / "stabilization_control.pdf")


if __name__ == "__main__":
    main()
