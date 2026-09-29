"""Cached-only examples and descriptive population panels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import numpy as np

from paper.supp_twin_saccade_modulation.analysis import (
    CACHE, FIGURES, STATS, RATE, crop_maps, load_cohort, provenance_paths,
    read_json, resolved_inputs, save_json, session_inputs, sha256, smooth_rows,
    sort_population, supported_effects, validate_session_cache,
)


def event_summary(a, unit, seed):
    effect = supported_effects(a, a['event_windows'], RATE, unit)
    if effect['n_events'] == 0:
        return None
    windows = a['event_windows'][effect['event_indices']]
    trials = a['trial_ids'][windows[:, 24]]
    means, intervals = {}, {}
    rng = np.random.default_rng(seed)
    unique = np.unique(trials)
    for key in ('gain', 'additive'):
        values = smooth_rows(effect[key])
        means[key] = values.mean(0)
        draws = []
        for _ in range(400):
            sampled = rng.choice(unique, len(unique), replace=True)
            chosen = np.concatenate([np.flatnonzero(trials == t) for t in sampled])
            draws.append(values[chosen].mean(0))
        intervals[key] = np.percentile(draws, [2.5, 97.5], axis=0)
    return effect['n_events'], means, intervals


def unit_maps(a, unit, detections):
    from paper.supp_twin_saccade_modulation.decomposition import shared_condition_mask
    zero, full, gain, add = [a[key][:, unit] for key in ('r0', 'rfull', 'r_gain', 'r_additive')]
    valid = shared_condition_mask(a['robs'][:, unit], a['dfs'][:, unit], zero, full, gain, add)
    maps = crop_maps(a['times'], a['trial_ids'], valid,
                     dict(observed=a['robs'][:, unit], full=full * RATE,
                          gain=(gain-zero)*RATE, additive=(add-zero)*RATE))
    if not maps['trials']:
        return maps
    # Exact per-trial recorded bin mapping. Markers never invent a bin.
    markers = []
    retained = set(map(int, a['event_ids']))
    for row, trial in enumerate(maps['trials']):
        idx = np.flatnonzero(a['trial_ids'] == trial)
        origin = a['times'][idx[0]]
        for onset, eid in detections:
            offset = onset-origin
            if not (maps['time_s'][0]-.5/RATE <= offset <= .8+.5/RATE):
                continue
            nearest = idx[np.argmin(abs(a['times'][idx]-onset))]
            if abs(a['times'][nearest]-onset) <= .5/RATE+1e-9:
                markers.append((row, float(a['times'][nearest]-origin), eid in retained))
    maps['markers'] = markers
    return maps


def render_unit(a, row, folder, detections):
    unit = int(np.flatnonzero(a['cids'] == row['cid'])[0])
    maps = unit_maps(a, unit, detections)
    if not maps['trials']:
        return dict(row, skipped='no recorded trials reaching 0.8 s with any shared valid bins')
    summary = event_summary(a, unit, 20260928 + sum(map(ord, row['session'])) * 1000 + row['cid'])
    if summary is None:
        return dict(row, skipped='no complete supported STA events')
    count, mean, interval = summary
    effects = np.r_[maps['gain'][maps['mask']], maps['additive'][maps['mask']]]
    effect_limit = max(.1, float(np.percentile(abs(effects), 99)))
    full_limit = max(1., float(np.percentile(maps['full'][maps['mask']], 99)))
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#e9e9e9')
    sequential = plt.get_cmap('Greys').copy()
    sequential.set_bad('#e9e9e9')
    n = len(maps['trials'])
    lo = maps['time_s'][0]
    hi = maps['time_s'][-1]
    extent = (lo-.5/RATE, hi+.5/RATE, n-.5, -.5)
    fig = plt.figure(figsize=(16, 8.8), layout='constrained')
    grid = fig.add_gridspec(2, 4, height_ratios=[2.35, 1])
    names = [('observed', 'Observed spikes (counts/bin)'), ('full', 'Full prediction'),
             ('gain', 'Gain rate effect'), ('additive', 'Additive rate effect')]
    for j, (key, title) in enumerate(names):
        ax = fig.add_subplot(grid[0, j])
        if key == 'observed':
            ax.imshow(maps['mask'].astype(float), cmap=ListedColormap(['#e9e9e9', 'white']),
                      vmin=0, vmax=1, origin='upper', aspect='auto', interpolation='nearest', extent=extent)
            for count_spikes, marker in [(1, '|'), (2, 'o')]:
                hit = np.argwhere(maps['observed'] == 1 if count_spikes == 1 else maps['observed'] > 1)
                if len(hit):
                    ax.scatter(maps['time_s'][hit[:, 1]], hit[:, 0], marker=marker,
                               s=11 if count_spikes == 1 else 5, c='#111111', linewidths=.4)
        else:
            signed = key != 'full'
            im = ax.imshow(np.ma.masked_invalid(maps[key]), origin='upper', aspect='auto',
                           interpolation='nearest', extent=extent, cmap=cmap if signed else sequential,
                           norm=TwoSlopeNorm(vmin=-effect_limit, vcenter=0, vmax=effect_limit) if signed else None,
                           vmin=None if signed else 0, vmax=None if signed else full_limit)
        for r, t, retained in maps['markers']:
            ax.plot(t, r, 'o' if retained else '|', color='#cf7500', markerfacecolor='none' if retained else '#cf7500',
                    markersize=3.8, linestyle='none')
        ax.set(title=title, xlabel='Time since first recorded trial bin (s)', xlim=(lo, .8), ylim=(n-.5, -.5))
        if j == 0:
            ticks = np.arange(0, n, 5)
            ax.set_yticks(ticks, [str(maps['trials'][int(t)]) for t in ticks])
            ax.set_ylabel('Trial ID (chronological)')
        else:
            ax.set_yticks([])
        if key in ('full', 'additive'):
            fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.065, pad=.13,
                         label='Rate (spikes/s)' if key == 'full' else 'Signed effect (spikes/s)')
    ax = fig.add_subplot(grid[1, :])
    plot_traces(ax, a['event_lags_ms'], mean, interval)
    ax.set(xlabel='Time from retained onset (ms)', ylabel='Rate effect (spikes/s)')
    ax.legend(frameon=False, ncol=2)
    fig.suptitle(f"{row['session']} · CID {row['cid']} · CCnorm {row['ccnorm']:.3f} · CCmax {row['ccmax']:.3f} · {n} displayed trials · {count} complete events\n"
                 f"STA uses all supported events (not just visible map markers); raw predictions, no calibration", fontsize=12)
    stem = f"{row['session']}_cid{row['cid']}"
    for ext in ('png', 'pdf'):
        fig.savefig(folder / f'{stem}.{ext}', dpi=170 if ext == 'png' else None)
    plt.close(fig)
    supported = maps['mask']
    return dict(row, displayed_trials=n, trial_ids=maps['trials'], map_range_s=[float(lo), .8],
                excluded_short_or_empty=maps['excluded_short'],
                invalid_endpoint_trial_ids=maps['invalid_endpoint_trials'], supported_events=count,
                retained_visible=sum(keep for _, _, keep in maps['markers']),
                all_visible_detections=len(maps['markers']),
                shared_effect_limit_hz=effect_limit,
                effect_saturated_bins=int((abs(effects) > effect_limit).sum()),
                raw_mean_predicted_observed_ratio=float(np.nanmean(maps['full'][supported]) /
                                                       (np.nanmean(maps['observed'][supported]) * RATE))
                if np.nanmean(maps['observed'][supported]) > 0 else None)


def plot_traces(ax, lags, mean, interval):
    for key, color in [('gain', '#b74640'), ('additive', '#276bab')]:
        ax.plot(lags, mean[key], color=color, label=f'{key.capitalize()} effect')
        ax.fill_between(lags, *interval[key], color=color, alpha=.14, linewidth=0)
    ax.axvline(0, color='.4', linestyle=':', linewidth=.8)
    ax.axhline(0, color='.65', linewidth=.8)
    ax.set_xlim(float(lags[0]), float(lags[-1]))
    ax.spines[['top', 'right']].set_visible(False)


def rank_example_rows(rows):
    return sorted((r for r in rows if r['example_eligible'] and r['supported_events'] >= 20),
                  key=lambda r: (-r['ccnorm'], r['session'], r['cid']))


def render(rows, sessions, paths):
    complete = [session for session in sessions
                if (CACHE / 'sessions' / session / 'manifest.json').exists()]
    inputs = {str(path): sha256(path) for path in provenance_paths(paths)}
    for session in complete:
        sources = {key: sha256(path) for key, path in session_inputs(paths, session).items()}
        validate_session_cache(CACHE / 'sessions' / session, inputs, sources)
    FIGURES.mkdir(parents=True, exist_ok=True)
    STATS.mkdir(parents=True, exist_ok=True)
    folder = FIGURES / 'gallery'
    folder.mkdir(parents=True, exist_ok=True)
    reports = {session: read_json(CACHE / 'sessions' / session / 'results.json')
               for session in complete}
    candidates = rank_example_rows([r for s in complete for r in reports[s]['cohort']])
    skipped = []
    examples = []
    lags = np.arange(-24, 49) / RATE * 1000
    pop_rows, gain, additive = [], [], []
    for session in complete:
        with np.load(CACHE / 'sessions' / session / 'components.npz') as archive:
            a = {key: archive[key] for key in archive.files}
        for row in reports[session]['cohort']:
            if row['supported_events'] < 20:
                continue
            unit = int(np.flatnonzero(a['cids'] == row['cid'])[0])
            e = supported_effects(a, a['event_windows'], RATE, unit)
            if e['n_events'] != row['supported_events']:
                raise AssertionError('cached event count differs from summary')
            pop_rows.append(row)
            gain.append(smooth_rows(e['gain']).mean(0))
            additive.append(smooth_rows(e['additive']).mean(0))
        del a
    for row in candidates:
        if len(examples) >= 10:
            break
        session = row['session']
        with np.load(CACHE / 'sessions' / session / 'components.npz') as archive:
            a = {key: archive[key] for key in archive.files}
        with session_inputs(paths, session)['saccades'].open() as f:
            detections = [(float(x['start_time']), j) for j, x in enumerate(json.load(f))
                          if np.isfinite(x.get('start_time', np.nan))]
        rendered = render_unit(a, row, folder, detections)
        (skipped if rendered.get('skipped') else examples).append(rendered)
        del a
    if pop_rows:
        ordered, g, add = sort_population(pop_rows, np.stack(gain), np.stack(additive), lags)
        finite = np.r_[g[np.isfinite(g)], add[np.isfinite(add)]]
        limit = max(.1, float(np.percentile(abs(finite), 98)))
        saturated = int((abs(finite) > limit).sum())
        n_nonpositive = sum(r['positive_gain_peak_ms'] is None for r in ordered)
        np.savez_compressed(STATS / 'population.npz', gain_hz=g, additive_hz=add, lags_ms=lags,
                            sessions=np.asarray([r['session'] for r in ordered]),
                            cids=np.asarray([r['cid'] for r in ordered]),
                            supported_events=np.asarray([r['supported_events'] for r in ordered]))
        with (STATS / 'population_rows.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['session', 'cid', 'ordinal', 'rate_hz', 'psth_r2',
                         'ccnorm', 'ccmax', 'supported_events', 'positive_gain_peak_ms', 'positive_gain_peak_hz'])
            writer.writeheader()
            for row in ordered:
                writer.writerow({key: row[key] for key in writer.fieldnames})
        fig, axes = plt.subplots(1, 2, figsize=(12, max(6, min(21, len(ordered)*.018))), layout='constrained')
        cmap = plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('#e9e9e9')
        for ax, matrix, label in zip(axes, [g, add], ['Gain effect', 'Additive effect']):
            im = ax.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                           aspect='auto', interpolation='nearest', extent=[lags[0]-1000/(2*RATE),
                           lags[-1]+1000/(2*RATE), len(ordered)-.5, -.5])
            ax.axvline(0, color='black', linewidth=.6)
            if n_nonpositive:
                ax.axhline(len(ordered)-n_nonpositive-.5, color='#f2b03a', linewidth=1.5)
            ax.set(xlabel='Time from retained onset (ms)', ylabel='Units: later positive gain peak → earlier → no positive peak' if ax is axes[0] else '',
                   title=label)
        fig.colorbar(im, ax=axes, orientation='horizontal', fraction=.05, label='Signed effect relative to eye-input-zero (spikes/s)')
        contributing = len({r['session'] for r in ordered})
        fig.suptitle(f'FIXRSVP · {len(ordered)} units from {contributing} sessions · {len(complete)}/19 sessions processed\n'
                     f'{n_nonpositive} no-positive-gain rows in separated bottom block', fontsize=12)
        for ext in ('png', 'pdf'):
            fig.savefig(FIGURES / f'population.{ext}', dpi=170 if ext == 'png' else None)
        plt.close(fig)
    else:
        ordered, saturated, limit, n_nonpositive = [], 0, None, 0
    if examples:
        fig, axes = plt.subplots(5, 2, figsize=(13, 14), layout='constrained', sharex=True)
        for ax, row in zip(axes.flat, examples):
            if row.get('skipped'):
                ax.set_title(row['skipped'])
                continue
            with np.load(CACHE / 'sessions' / row['session'] / 'components.npz') as data:
                a = {key: data[key] for key in data.files}
            unit = int(np.flatnonzero(a['cids'] == row['cid'])[0])
            _, mean, interval = event_summary(a, unit, 20260928 + sum(map(ord, row['session'])) * 1000 + row['cid'])
            plot_traces(ax, a['event_lags_ms'], mean, interval)
            ax.set_title(f"{row['session']} CID {row['cid']} · CCnorm {row['ccnorm']:.3f}", fontsize=9)
        for j, ax in enumerate(axes.flat[:len(examples)]):
            if j % 2 == 0:
                ax.set_ylabel('Rate effect (spikes/s)')
            if j // 2 == 4:
                ax.set_xlabel('Time from saccade onset (ms)')
        axes.flat[0].legend(frameon=False, ncol=2, fontsize=8)
        fig.suptitle('CCnorm-ranked examples · raw model pathway effects', fontsize=13)
        for ax in axes.flat[len(examples):]:
            ax.axis('off')
        fig.savefig(FIGURES / 'example_traces.png', dpi=160)
        fig.savefig(FIGURES / 'example_traces.pdf')
        plt.close(fig)
    save_json(STATS / 'selection.json', dict(complete_sessions=complete, expected_sessions=sessions,
            population_units=len(ordered), population_sessions=sorted({r['session'] for r in ordered}),
            excluded_units_for_event_support=sum(len(reports[s]['cohort']) for s in complete)-len(ordered),
            no_positive_gain_rows=n_nonpositive,
            pooled_abs_effect_98th_limit_hz=limit, saturated_population_cells=saturated,
            examples=examples, skipped_examples=skipped, example_selection='CCnorm descending; empirical rate >2 Hz and PSTH R2 >.10; CCmax >.85; finite CCnorm <=1; >=20 wholly supported events; session/CID tie break',
            population_selection='all empirically eligible units with >=20 wholly supported windows; no CCnorm gate'))
    print(f'rendered {len(examples)} examples, {len(ordered)} population units from {len(complete)}/19 sessions', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='processed FIXRSVP session root')
    args = parser.parse_args()
    paths = resolved_inputs(args.data_root)
    rows, sessions, _ = load_cohort(paths)
    if len(rows) != 1022 or len(sessions) != 19:
        raise AssertionError('manuscript cohort changed')
    render(rows, sessions, paths)


if __name__ == '__main__':
    main()
