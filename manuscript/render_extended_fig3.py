"""Render ED3 from frozen, audited 725-unit plot inputs; never rerun inference.

From VisionCore root: uv run --project .. --no-sync python -m manuscript.render_extended_fig3
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from VisionCore.paths import VISIONCORE_ROOT
from paper.supp_twin_saccade_modulation import figure as drawing
from paper.supp_twin_saccade_modulation.analysis import RATE

HERE = VISIONCORE_ROOT / 'manuscript'
DATA = HERE / 'analysis/extended_fig3'
# Digest checks distinguish this frozen plotting replay from source-cache inference.
HASHES = {'plot_inputs.npz': '4f46683bd717f6a0563d5b11473bf1d0fedcd37d96aa31f30a6a67dc2b160049',
          'psth_three_bin.npz': '1362efd48970ff2612bad739916beb26eb0cd2fd85faab86bb13d628db6b1fde',
          'complete_725_units.csv': '4183f24ab616bb53fbf5e2ff7adb5ec56ee6f35ebf5b2d5b364df97cfa01949c',
          'complete_725_summary.json': 'dd331096fdaacd261bd2eda31be8d46766237fdb671197bba114bf44ec23b823',
          'full_A_to_H_B_three_bin_725_metadata.json': '9fe0f07b4462977e1017b6a895387e004d787c42a930528377d9fc9b143f62dc'}


def load_inputs():
    for name, expected in HASHES.items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'ED3 frozen input changed: {name}')
    with np.load(DATA / 'plot_inputs.npz', allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    with np.load(DATA / 'psth_three_bin.npz', allow_pickle=False) as archive:
        psth = {key: archive[key] for key in archive.files}
    maps = dict(trials=saved['example_trial_ids'], time_s=saved['example_time_s'],
                mask=saved['example_map_mask'],
                markers=[(int(r), float(t), False) for r, t in zip(saved['example_marker_rows'],
                                                                   saved['example_marker_times_s'])],
                **{key: saved[f'example_map_{key}'] for key in ('observed', 'full', 'gain', 'additive')})
    population = {key: saved[key] for key in ('gain_hz', 'additive_hz', 'lags_ms', 'sessions', 'cids', 'supported_events')}
    areas = drawing.integrated_effects(population['lags_ms'], population['gain_hz'], population['additive_hz'])
    for key, value in areas.items():
        np.testing.assert_allclose(value, saved[key], equal_nan=True)
    means = {name: saved[f'example_{name}_mean_hz'] for name in ('gain', 'additive')}
    intervals = {name: saved[f'example_{name}_bootstrap_95_hz'] for name in ('gain', 'additive')}
    if (len(maps['trials']) != 56 or len(saved['example_event_ids']) != 53
            or len(population['cids']) != 725 or len(set(zip(population['sessions'], population['cids']))) != 725
            or len(set(population['sessions'])) != 15):
        raise ValueError('ED3 frozen example or population changed')
    return saved, maps, psth, population, areas, means, intervals


def statistics_tex(saved, population, areas, metadata):
    bins = metadata['psth_smoothing_bins']
    if (bins != 3 or not np.isclose(metadata['psth_smoothing_ms'], 1000 * bins / RATE)
            or metadata['example_displayed_trials'] != len(saved['example_trial_ids'])
            or metadata['cohort_units'] != len(population['cids'])
            or metadata['example_complete_supported_events'] != len(saved['example_event_ids'])
            or metadata['event_smoothing_bins'] != 0 or metadata['population_smoothing_bins'] != 0
            or metadata['example_bootstrap_draws'] != 400):
        raise ValueError('ED3 frozen smoothing, bootstrap, or cohort metadata differs from plotted arrays')
    values = {'Trials': str(len(saved['example_trial_ids'])),
              'Units': str(len(population['cids'])),
              'SmoothingBins': 'three',
              'SmoothingMs': f"{metadata['psth_smoothing_ms']:g}",
              'BootstrapDraws': str(metadata['example_bootstrap_draws'])}
    for name in ('gain', 'additive'):
        prefix = name.capitalize()
        values[prefix + 'Balance'] = f"{np.median(abs(areas[f'{name}_signed_balance'])):.3f}"
        values[prefix + 'Total'] = f"{np.median(areas[f'{name}_absolute_area_spikes']):.3f}"
    return ('% Generated from frozen ED3 plot arrays and pinned three-bin metadata.\n' +
            ''.join(f'\\newcommand{{\\EDThree{key}}}{{{value}}}\n'
                    for key, value in sorted(values.items())))


def sync_statistics(path=HERE / 'extended_fig3_stats.tex', check=False):
    saved, maps, psth, population, areas, _, _ = load_inputs()
    metadata = json.loads((DATA / 'full_A_to_H_B_three_bin_725_metadata.json').read_text())
    for key, expected in drawing.matched_psth(maps, bins=metadata['psth_smoothing_bins']).items():
        np.testing.assert_allclose(psth[key], expected, equal_nan=True)
    text = statistics_tex(saved, population, areas, metadata)
    path = Path(path)
    if check:
        if not path.exists() or path.read_text() != text:
            raise ValueError(f'stale ED3 statistics: {path}')
    else:
        path.write_text(text)
    return text


def render(output=HERE / 'figures/extended_fig3.pdf'):
    sync_statistics()
    _, maps, psth, population, areas, means, intervals = load_inputs()
    lag = population['lags_ms']
    limit = max(.1, float(np.percentile(abs(np.r_[population['gain_hz'].ravel(),
                                              population['additive_hz'].ravel()]), 98)))
    effects = np.r_[maps['gain'][maps['mask']], maps['additive'][maps['mask']]]
    effect_limit = max(.1, float(np.percentile(abs(effects), 99)))
    fig, grid = drawing.make_figure()
    top_grid = drawing.make_top_grid(grid)
    color = plt.get_cmap('RdBu_r').copy(); color.set_bad('#e9e9e9')
    grey = plt.get_cmap('Greys').copy(); grey.set_bad('#e9e9e9')
    norm = drawing.trial_rate_norm(maps)
    lo, hi, n = maps['time_s'][0], maps['time_s'][-1], len(maps['trials'])
    extent = (lo-.5/RATE, hi+.5/RATE, n-.5, -.5)
    top_axes, images = [], []
    for j, (key, title) in enumerate((('observed', 'Observed spikes'), ('full', 'Full prediction'),
                                      ('gain', 'Gain component'), ('additive', 'Additive component'))):
        ax = fig.add_subplot(top_grid[0, 3*j:3*j+3]); top_axes.append(ax)
        if key in ('observed', 'full'):
            image = drawing.draw_trial_rate(ax, maps, key, extent, norm, grey)
        else:
            image = ax.imshow(np.ma.masked_invalid(maps[key]), origin='upper', aspect='auto',
                              interpolation='nearest', extent=extent, cmap=color,
                              norm=TwoSlopeNorm(vmin=-effect_limit, vcenter=0, vmax=effect_limit))
        images.append(image)
        drawing.draw_saccade_markers(ax, maps['markers'])
        ax.set(title=title, xlabel='Trial time (s)', xlim=(lo, .8), ylim=(n-.5, -.5))
        ax.tick_params(axis='x', labelsize=8)
        if j == 0:
            ticks = np.arange(0, n, 8)
            ax.set_yticks(ticks, [str(maps['trials'][int(t)]) for t in ticks])
            ax.set_ylabel('Trial ID (chronological)')
            ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=.9,
                      fontsize=8, markerscale=1, borderpad=.3)
            ax.text(-.18, 1.05, 'A', transform=ax.transAxes, fontweight='bold', fontsize=11)
        else:
            ax.set_yticks([])
    b, c, d = drawing.make_example_axes(fig, grid)
    drawing.draw_psth(b, maps['time_s'], psth)
    b.set_xlim(lo, .8)
    for ax, name in ((c, 'gain'), (d, 'additive')):
        drawing.draw_example_pathway(ax, lag, name, means, intervals)
        ax.set(title=f'Saccade-aligned\n{name} component', ylabel='')
    drawing.share_example_pathway_ylim((c, d), lag, means, intervals)
    for letter, ax in (('B', b), ('C', c), ('D', d)):
        ax.text(-.09, 1.09, letter, transform=ax.transAxes, fontweight='bold', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
    drawing.add_trial_colorbars(fig, top_axes, images[0], images[3])
    panel_axes = drawing.draw_population_summary(fig, grid, population, areas, limit)
    panel_axes[0].set_title('Gain component')
    panel_axes[2].set_title('Additive component')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=210 if output.suffix == '.png' else None)
    plt.close(fig)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--stats-only', action='store_true')
    mode.add_argument('--check-stats', action='store_true')
    args = parser.parse_args()
    if args.check_stats:
        sync_statistics(check=True)
    elif args.stats_only:
        sync_statistics()
    else:
        render()
