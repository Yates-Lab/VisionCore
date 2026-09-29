"""Cached-only FIXRSVP figure; no model execution or calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

from paper.supp_twin_saccade_modulation.analysis import (
    CACHE, FIGURES, STATS, RATE, read_json, session_inputs, sha256,
    provenance_paths, resolved_inputs, validate_session_cache,
)
from paper.supp_twin_saccade_modulation.plot import event_summary, unit_maps

SESSION = 'Allen_2022-04-06'
CID = 98


def matched_psth(maps, bins=5):
    """Pool shared-support native counts; smooth by supported sample weight."""
    mask = np.asarray(maps['mask'], bool)
    obs = np.asarray(maps['observed'])
    pred = np.asarray(maps['full']) / RATE
    if mask.shape != obs.shape or mask.shape != pred.shape or not np.isfinite(obs[mask]).all() or not np.isfinite(pred[mask]).all():
        raise ValueError('PSTH support differs from the displayed maps')
    if bins < 1 or bins % 2 != 1:
        raise ValueError('smoothing width must be a positive odd number')
    count = mask.sum(0)
    kernel = np.ones(bins)
    pad = bins // 2
    result = dict(mask=mask.copy(), valid_trials=count)
    for name, values in [('observed', obs), ('predicted', pred)]:
        sums = np.where(mask, values, 0.).sum(0) * RATE
        raw = np.divide(sums, count, out=np.full(len(count), np.nan), where=count > 0)
        smooth_num = np.convolve(np.pad(sums, (pad, pad), mode='edge'), kernel, mode='valid')
        smooth_den = np.convolve(np.pad(count, (pad, pad), mode='edge'), kernel, mode='valid')
        smooth = np.divide(smooth_num, smooth_den, out=np.full(len(count), np.nan), where=smooth_den > 0)
        smooth[count == 0] = np.nan
        result[f'{name}_raw_hz'] = raw
        result[f'{name}_smooth_hz'] = smooth
        # Smooth each trial on its own valid support; center bins without support stay missing.
        trials = []
        for values_row, mask_row in zip(values, mask):
            numerator = np.convolve(np.pad(np.where(mask_row, values_row, 0.) * RATE,
                                            (pad, pad), mode='edge'), kernel, mode='valid')
            denominator = np.convolve(np.pad(mask_row.astype(float), (pad, pad), mode='edge'),
                                      kernel, mode='valid')
            trial = np.divide(numerator, denominator, out=np.full(len(count), np.nan),
                              where=denominator > 0)
            trial[~mask_row] = np.nan
            trials.append(trial)
        trial_rates = np.stack(trials)
        supported = np.isfinite(trial_rates)
        n = supported.sum(0)
        trial_mean = np.divide(np.where(supported, trial_rates, 0.).sum(0), n,
                               out=np.full(len(n), np.nan), where=n > 0)
        squared = np.where(supported, (trial_rates - trial_mean) ** 2, 0.).sum(0)
        result[f'{name}_std_hz'] = np.sqrt(np.divide(squared, n - 1,
                                      out=np.full(len(n), np.nan), where=n > 1))
    return result


def draw_population(axes, gain, additive, lags, limit):
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#e9e9e9')
    for ax, matrix, title in zip(axes, (gain, additive), ('Gain effect', 'Additive effect')):
        image = ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap=cmap,
                          norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                          extent=(lags[0]-500/RATE, lags[-1]+500/RATE, len(matrix)-.5, -.5))
        ax.axvline(0, color='black', linewidth=.7)
        ax.set(title=title, xlim=(-100, 200), xticks=[-100, 0, 100, 200],
               xlabel='Time from saccade onset (ms)')
        ax.tick_params(axis='y', length=0)
    axes[0].set_ylabel('Units (gain-peak order)')
    axes[1].set_yticks([])
    return image


def integrated_effects(lags, gain, additive):
    """Trapezoids of rectified smoothed rows at inclusive 0–200 ms bin centers."""
    inside = (lags >= -1e-8) & (lags <= 200 + 1e-8)
    times_ms = lags[inside]
    if len(times_ms) < 2 or not np.isclose(times_ms[[0, -1]], [0, 200]).all():
        raise ValueError('missing integration endpoints')
    edges = np.linspace(-1, 1, 11)
    result = dict(integration_lags_ms=times_ms, balance_bin_edges=edges)
    for name, matrix in (('gain', gain), ('additive', additive)):
        values = matrix[:, inside]
        if not np.isfinite(values).all():
            raise ValueError('nonfinite population effect')
        enhance = np.trapz(np.maximum(values, 0), times_ms / 1000, axis=1)
        suppress = np.trapz(np.maximum(-values, 0), times_ms / 1000, axis=1)
        absolute = enhance + suppress
        result[f'{name}_enhancing_area_spikes'] = enhance
        result[f'{name}_suppressive_area_spikes'] = suppress
        result[f'{name}_absolute_area_spikes'] = absolute
        balance = np.divide(enhance - suppress, absolute,
                            out=np.full_like(absolute, np.nan), where=absolute > 0)
        result[f'{name}_signed_balance'] = balance
        counts = np.histogram(balance[np.isfinite(balance)], bins=edges)[0]
        result[f'{name}_balance_bin_counts'] = counts
        result[f'{name}_balance_bin_fractions'] = np.divide(
            counts, counts.sum(), out=np.zeros(10), where=counts.sum() > 0)
    max_strength = max(np.max(result[f'{name}_absolute_area_spikes'])
                       for name in ('gain', 'additive'))
    strength_edges = np.arange(max(1, int(np.ceil(max_strength / .05))) + 1) * .05
    result['strength_bin_edges_spikes'] = strength_edges
    for name in ('gain', 'additive'):
        strength = result[f'{name}_absolute_area_spikes']
        counts = np.histogram(strength, bins=strength_edges)[0]
        result[f'{name}_strength_bin_counts'] = counts
        result[f'{name}_strength_bin_fractions'] = counts / len(strength)
    return result


def trial_rate_norm(maps):
    """Common 0–500 spikes/s display scale; retain raw image arrays."""
    mask = np.asarray(maps['mask'], bool)
    counts = np.asarray(maps['observed'])
    full = np.asarray(maps['full'])
    if (not mask.any() or not np.isfinite(counts[mask]).all()
            or np.any(counts[mask] < 0) or not np.all(counts[mask] == np.rint(counts[mask]))
            or not np.isfinite(full[mask]).all()):
        raise ValueError('trial maps need finite nonnegative integer counts and finite rates on support')
    return Normalize(vmin=0, vmax=500, clip=True)


def draw_trial_rate(ax, maps, key, extent, norm, cmap=None):
    """Display shared-support raw native-bin rates without smoothing."""
    if cmap is None:
        cmap = plt.get_cmap('Greys').copy()
        cmap.set_bad('#e9e9e9')
    values = np.asarray(maps[key]) * (RATE if key == 'observed' else 1)
    return ax.imshow(np.ma.array(values, mask=~np.asarray(maps['mask'], bool)),
                     cmap=cmap, norm=norm, aspect='auto', interpolation='nearest', extent=extent)


def add_trial_colorbars(fig, axes, rate_image, effect_image):
    """One compact horizontal bar beneath each top-row map pair."""
    bars = []
    for pair, image, label in ((axes[:2], rate_image, 'Rate (spikes/s)'),
                               (axes[2:], effect_image, 'Signed effect (spikes/s)')):
        left, right = pair
        bounds = left.get_position()
        cax = fig.add_axes([bounds.x0, bounds.y0 - .056,
                            right.get_position().x1 - bounds.x0, .007])
        bar = fig.colorbar(image, cax=cax, orientation='horizontal', label=label)
        if image is rate_image:
            bar.set_ticks([0, 100, 200, 300, 400, 500], labels=['0', '100', '200', '300', '400', '500+'])
        bar.ax.tick_params(labelsize=7)
        bars.append(bar)
    return bars


def draw_saccade_markers(ax, markers):
    """Every displayed detection has the same labeled hollow marker."""
    ax.scatter([t for _, t, _ in markers], [r for r, _, _ in markers],
               marker='>', s=10, facecolors='none', edgecolors='#15803d', linewidths=.75,
               label='Saccade onset', zorder=5, rasterized=True)


def draw_example_pathway(ax, lags, name, means, intervals):
    color = {'gain': '#b74640', 'additive': '#276bab'}[name]
    ax.plot(lags, means[name], color=color, label=f'{name.capitalize()} effect')
    ax.fill_between(lags, *intervals[name], color=color, alpha=.16, linewidth=0)
    ax.axvline(0, color='.4', linestyle=':', linewidth=.8)
    ax.axhline(0, color='.65', linewidth=.8)
    ax.set(xlim=(-100, 200), xticks=[-100, 0, 100, 200],
           xlabel='Time from saccade onset (ms)', ylabel='Rate effect (spikes/s)',
           title=f'Saccade-aligned\n{name} pathway')
    ax.spines[['top', 'right']].set_visible(False)


def share_example_pathway_ylim(axes, lags, means, intervals):
    """Give C and D one zero-centered range containing both displayed bootstrap bands."""
    shown = (np.asarray(lags) >= -100) & (np.asarray(lags) <= 200)
    bound = max(np.max(np.abs(values[..., shown])) for name in ('gain', 'additive')
                for values in (np.asarray(means[name]), np.asarray(intervals[name]))) * 1.08
    for ax in axes:
        ax.set_ylim(-bound, bound)


def draw_area_scatter(axes, histograms, strength_histograms, areas):
    bound = max(np.max(areas[f'{name}_absolute_area_spikes'])
                for name in ('gain', 'additive')) * 1.1
    fraction_bound = max(np.max(areas[f'{name}_balance_bin_fractions'])
                         for name in ('gain', 'additive')) * 1.08
    strength_fraction_bound = max(np.max(areas[f'{name}_strength_bin_fractions'])
                                  for name in ('gain', 'additive')) * 1.08
    edges = areas['balance_bin_edges']
    strength_edges = areas['strength_bin_edges_spikes']
    for ax, hist, right, name, color in zip(axes, histograms, strength_histograms,
                                            ('gain', 'additive'), ('#b74640', '#276bab')):
        x = areas[f'{name}_signed_balance']
        y = areas[f'{name}_absolute_area_spikes']
        ax.scatter(x, y, s=9, color=color, alpha=.35, linewidths=0,
                   rasterized=True, clip_on=False)
        ax.axvline(0, color='.7', linewidth=.8, zorder=0)
        ax.set(xlim=(-1, 1), ylim=(0, bound), ylabel='Total (spikes/saccade)',
               title=f'{name.capitalize()} balance and strength')
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.spines[['top', 'right']].set_visible(False)
        hist.bar(edges[:-1], areas[f'{name}_balance_bin_fractions'], width=np.diff(edges),
                 align='edge', color=color, alpha=.65, linewidth=0)
        hist.set(xlim=(-1, 1), ylim=(0, fraction_bound), xlabel='Signed balance')
        hist.set_xticks([-1, 0, 1], ['−1\nNegative only', '0\nBalanced', '+1\nPositive only'])
        hist.tick_params(axis='x', labelsize=8)
        hist.tick_params(axis='y', labelsize=8)
        hist.yaxis.set_major_locator(MaxNLocator(3))
        hist.spines[['top', 'right']].set_visible(False)
        right.barh(strength_edges[:-1], areas[f'{name}_strength_bin_fractions'],
                   height=np.diff(strength_edges), align='edge', color=color, alpha=.65, linewidth=0)
        right.set(xlim=(0, strength_fraction_bound), ylim=(0, bound), xlabel='Fraction')
        right.tick_params(axis='y', left=False, labelleft=False)
        right.tick_params(axis='x', labelsize=8)
        right.set_xticks([0, strength_fraction_bound], ['0', f'{strength_fraction_bound:.2f}'])
        right.spines[['top', 'right']].set_visible(False)


def draw_population_summary(fig, grid, population, areas, limit):
    """Pair each wide population heatmap with its shared-scale balance scatter."""
    gain_map = fig.add_subplot(grid[2, :7])
    gain_pair = grid[2, 7:].subgridspec(2, 3, height_ratios=[4, 1],
                                         width_ratios=[.4, 3.4, 1], hspace=.06, wspace=.12)
    gain_scatter = fig.add_subplot(gain_pair[0, 1])
    gain_hist = fig.add_subplot(gain_pair[1, 1])
    gain_strength_hist = fig.add_subplot(gain_pair[0, 2])
    additive_map = fig.add_subplot(grid[3, :7])
    additive_pair = grid[3, 7:].subgridspec(2, 3, height_ratios=[4, 1],
                                             width_ratios=[.4, 3.4, 1], hspace=.06, wspace=.12)
    additive_scatter = fig.add_subplot(additive_pair[0, 1])
    additive_hist = fig.add_subplot(additive_pair[1, 1])
    additive_strength_hist = fig.add_subplot(additive_pair[0, 2])
    image = draw_population((gain_map, additive_map), population['gain_hz'],
                            population['additive_hz'], population['lags_ms'], limit)
    additive_map.set_ylabel('Units (gain-peak order)')
    additive_map.set_yticks([tick for tick in gain_map.get_yticks()
                            if 0 <= tick < len(population['gain_hz'])])
    draw_area_scatter((gain_scatter, additive_scatter), (gain_hist, additive_hist),
                      (gain_strength_hist, additive_strength_hist), areas)
    axes = (gain_map, gain_scatter, additive_map, additive_scatter)
    for letter, axis in zip('EFGH', axes):
        axis.text(-.22 if letter in 'FH' else -.09, 1.04, letter, transform=axis.transAxes,
                  fontweight='bold', fontsize=11)
    colorbar_axis = additive_map.inset_axes([0, -.25, 1, .045])
    bar = fig.colorbar(image, cax=colorbar_axis, orientation='horizontal',
                       label='Signed rate effect (spikes/s)')
    bar.ax.tick_params(labelsize=8)
    return axes + (gain_hist, additive_hist, gain_strength_hist, additive_strength_hist)


def check_trial_indices(a, maps):
    """Confirm every plotted trial uses the cached stimulus-aligned native bin index."""
    for trial in maps['trials']:
        idx = np.flatnonzero(a['trial_ids'] == trial)
        expected = np.rint((a['times'][idx]-a['times'][idx[0]]) * RATE).astype(int)
        if not np.array_equal(a['psth_indices'][idx], expected):
            raise AssertionError(f'trial {trial}: cached PSTH index is not native trial time')
    if not np.allclose(maps['time_s'] * RATE, np.rint(maps['time_s'] * RATE), rtol=0, atol=1e-10):
        raise AssertionError('map times are not native-bin coordinates')


def select_example(session, cid):
    """Select only an empirically/event-eligible unit in the validated session cohort."""
    cohort = read_json(CACHE / 'sessions' / session / 'results.json')['cohort']
    row = next((r for r in cohort if r['cid'] == cid), None)
    if row is None:
        raise ValueError(f'{session} CID {cid} not in cached cohort')
    if not row['example_eligible'] or row['supported_events'] < 20:
        raise ValueError(f'{session} CID {cid} ineligible for example figure')
    return row


def make_top_grid(grid):
    return grid[0, :].subgridspec(2, 12, height_ratios=[1, .58], hspace=0, wspace=.85)


def make_example_axes(fig, grid):
    """Add horizontal padding only to the B–D row."""
    row = grid[1, :].subgridspec(1, 3, wspace=.3)
    return [fig.add_subplot(row[0, j]) for j in range(3)]


def make_figure():
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(8, 10.5))
    grid = fig.add_gridspec(4, 12, left=.11, right=.96, top=.96, bottom=.09,
                            height_ratios=[1.9, 1.15, 1.65, 1.65], hspace=.4, wspace=.85)
    return fig, grid


def draw_psth(ax, time, psth):
    for key, color, label in (('observed', '#222222', 'Observed'),
                              ('predicted', '#a4473c', 'Full prediction')):
        mean = psth[f'{key}_smooth_hz']
        std = psth[f'{key}_std_hz']
        ax.fill_between(time, mean-std, mean+std, color=color, alpha=.16, linewidth=0)
        ax.plot(time, mean, color=color, label=label, linewidth=1)
    low = min(np.nanmin(psth[f'{key}_smooth_hz'] - psth[f'{key}_std_hz'])
              for key in ('observed', 'predicted'))
    ax.set_ylim(min(0, low), None)
    ax.set(xlabel='Trial time (s)', ylabel='Rate (spikes/s)', title='Stimulus-aligned PSTH')
    ax.legend(frameon=False, ncol=1, fontsize=7)


def assemble(paths, session=SESSION, cid=CID):
    selection_path = STATS / 'selection.json'
    pop_path = STATS / 'population.npz'
    rows_path = STATS / 'population_rows.csv'
    component_path = CACHE / 'sessions' / session / 'components.npz'
    detection_path = session_inputs(paths, session)['saccades']
    inputs = {str(path): sha256(path) for path in provenance_paths(paths)}
    sources = {key: sha256(path) for key, path in session_inputs(paths, session).items()}
    validate_session_cache(component_path.parent, inputs, sources)
    selected = read_json(selection_path)
    row = select_example(session, cid)
    old_row = next((x for x in selected['examples'] if x['session'] == session and x['cid'] == cid), None)
    with np.load(component_path) as archive:
        a = {key: archive[key] for key in archive.files}
    with detection_path.open() as stream:
        detections = [(float(x['start_time']), j) for j, x in enumerate(json.load(stream))
                      if np.isfinite(x.get('start_time', np.nan))]
    with np.load(pop_path) as archive:
        population = {key: archive[key] for key in archive.files}
    found = np.flatnonzero(a['cids'] == cid)
    if len(found) != 1 or int(found[0]) != row['ordinal']:
        raise AssertionError('cached CID/ordinal differs from cohort')
    unit = int(found[0])
    maps = unit_maps(a, unit, detections)
    if not maps['trials']:
        raise AssertionError('eligible example has no plottable trials')
    check_trial_indices(a, maps)
    if old_row is not None and (maps['trials'] != old_row['trial_ids']
            or not np.isclose(maps['time_s'][0], old_row['map_range_s'][0])
            or not np.isclose(maps['time_s'][-1], old_row['map_range_s'][-1])):
        raise AssertionError('cached example selection/crop changed')
    if session == SESSION and cid == CID and not np.array_equal(maps['time_s'], np.arange(59, 193) / RATE):
        raise AssertionError('default example native crop changed')
    psth = matched_psth(maps)
    n_events, means, intervals = event_summary(a, unit, 20260928 + sum(map(ord, session)) * 1000 + cid)
    if n_events != row['supported_events']:
        raise AssertionError('example event count changed')
    if len(population['gain_hz']) != selected['population_units'] or not np.array_equal(population['gain_hz'].shape, population['additive_hz'].shape):
        raise AssertionError('population matrices differ from selection')
    limit = selected['pooled_abs_effect_98th_limit_hz']
    areas = integrated_effects(population['lags_ms'], population['gain_hz'], population['additive_hz'])

    effects = np.r_[maps['gain'][maps['mask']], maps['additive'][maps['mask']]]
    effect_limit = max(.1, float(np.percentile(abs(effects), 99)))
    predicted_mean = float(np.mean(maps['full'][maps['mask']]))
    observed_mean = float(np.mean(maps['observed'][maps['mask']]))
    ratio = predicted_mean / (observed_mean * RATE) if observed_mean > 0 else None
    fig, grid = make_figure()
    # Reserve room for trial-map colorbars without enlarging the lower row gaps.
    top_grid = make_top_grid(grid)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#e9e9e9')
    grey = plt.get_cmap('Greys').copy()
    grey.set_bad('#e9e9e9')
    rate_norm = trial_rate_norm(maps)
    lo, hi, n = maps['time_s'][0], maps['time_s'][-1], len(maps['trials'])
    extent = (lo-.5/RATE, hi+.5/RATE, n-.5, -.5)
    top_axes, images = [], []
    for j, (key, title) in enumerate((('observed', 'Observed spikes'), ('full', 'Full prediction'),
                                      ('gain', 'Gain pathway'), ('additive', 'Additive pathway'))):
        ax = fig.add_subplot(top_grid[0, 3*j:3*j+3])
        top_axes.append(ax)
        if key in ('observed', 'full'):
            im = draw_trial_rate(ax, maps, key, extent, rate_norm, grey)
        else:
            im = ax.imshow(np.ma.masked_invalid(maps[key]), origin='upper', aspect='auto', interpolation='nearest',
                           extent=extent, cmap=cmap,
                           norm=TwoSlopeNorm(vmin=-effect_limit, vcenter=0, vmax=effect_limit))
        images.append(im)
        draw_saccade_markers(ax, maps['markers'])
        ax.set(title=title, xlabel='Trial time (s)', xlim=(lo, .8), ylim=(n-.5, -.5))
        ax.tick_params(axis='x', labelsize=8)
        if j == 0:
            ticks = np.arange(0, n, 8)
            ax.set_yticks(ticks, [str(maps['trials'][int(t)]) for t in ticks])
            ax.set_ylabel('Trial ID (chronological)')
        else:
            ax.set_yticks([])
        if j == 0:
            ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=.9,
                      fontsize=8, markerscale=1, borderpad=.3)
            ax.text(-.18, 1.05, 'A', transform=ax.transAxes, fontweight='bold', fontsize=11)

    b, c, d = make_example_axes(fig, grid)
    draw_psth(b, maps['time_s'], psth)
    b.set_xlim(lo, .8)
    for axis, name in ((c, 'gain'), (d, 'additive')):
        draw_example_pathway(axis, a['event_lags_ms'], name, means, intervals)
        axis.set_ylabel('')  # Units are shared with B; keep compact axes separate.
    share_example_pathway_ylim((c, d), a['event_lags_ms'], means, intervals)
    for letter, axis in [('B', b), ('C', c), ('D', d)]:
        axis.text(-.09, 1.09, letter, transform=axis.transAxes, fontweight='bold', fontsize=11)
        axis.spines[['top', 'right']].set_visible(False)

    add_trial_colorbars(fig, top_axes, images[0], images[3])
    draw_population_summary(fig, grid, population, areas, limit)
    dest = FIGURES / 'assembled'
    dest.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(dest / f'fixrsvp_full.{ext}', dpi=190 if ext == 'png' else None)
    plt.close(fig)
    np.savez_compressed(STATS / 'integrated_areas.npz',
                        sessions=population['sessions'], cids=population['cids'],
                        supported_events=population['supported_events'],
                        integration_bounds_ms=np.array([0, 200]),
                        integration_method='trapezoidal quadrature of rectified sampled 5-bin-smoothed rows; inclusive native lag centers; ms converted to seconds',
                        area_units='spikes/saccade', **areas)
    provenance = {str(path): sha256(path) for path in (
        selection_path, pop_path, rows_path, component_path,
        component_path.parent / 'results.json', detection_path,
    )}
    np.savez_compressed(STATS / 'verification.npz', trial_ids=np.asarray(maps['trials']),
                        time_s=maps['time_s'], **psth, gain_hz=population['gain_hz'],
                        additive_hz=population['additive_hz'], lags_ms=population['lags_ms'],
                        sessions=population['sessions'], cids=population['cids'],
                        supported_events=population['supported_events'])
    metadata = dict(session=session, cid=cid, displayed_trials=n, map_range_s=[float(lo), float(hi)],
                    supported_events=n_events, smoothing_bins=5, native_rate_hz=RATE,
                    population_rows=len(population['cids']), population_no_positive_rows=selected['no_positive_gain_rows'],
                    population_limit_hz=limit, ccnorm_figure3=row['ccnorm'], ccmax_figure3=row['ccmax'],
                    raw_mean_predicted_observed_ratio=ratio,
                    signed_map_limit_hz=effect_limit,
                    page_size_inches=[8, 10.5],
                    psth_band='mean ± 1 sample SD (ddof=1) across displayed trial rates, each smoothed within its own supported bins; observed includes spike-count noise, prediction is model-rate variation; undefined for <2 supported trials',
                    population_order='identical to output/population.npz; late positive gain peak first',
                    plotted_areas='integrated_areas.npz; enhancing and nonnegative suppressive trapezoidal areas (spikes/saccade) of event-mean smoothed rows at inclusive native lag centers 0–200 ms; signed balance (P-N)/(P+N) is undefined when total area is zero; F/H plot balance vs total area',
                    balance_histogram='integrated_areas.npz; 10 shared equal-width bins from -1 to +1, rightmost bin includes +1; counts and fractions of units with defined balance (all 773 currently); shared linear fraction scale',
                    strength_histogram='integrated_areas.npz; shared edges every 0.05 spikes/saccade from zero through maximum P+N, rightmost endpoint included; counts and fractions of all 773 units per pathway; shared linear fraction scale',
                    event_display_limits_ms=[-100, 200],
                    source_sha256=provenance)
    (STATS / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'assembled {n} trials, {n_events} events, {len(population["cids"])} population units in {dest}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--session', default=SESSION, help='session with validated cached cohort')
    parser.add_argument('--cid', type=int, default=CID, help='eligible cohort CID')
    args = parser.parse_args()
    assemble(resolved_inputs(args.data_root), args.session, args.cid)


if __name__ == '__main__':
    main()
