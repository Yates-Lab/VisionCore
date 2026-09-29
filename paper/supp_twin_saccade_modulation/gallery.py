"""Cached-only 50-page candidate review, one A–D example per ranked page."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from paper.supp_twin_saccade_modulation.analysis import (
    CACHE, FIGURES, STATS, RATE, provenance_paths, read_json, resolved_inputs,
    session_inputs, sha256, validate_session_cache,
)
from paper.supp_twin_saccade_modulation.figure import (
    SESSION, CID, add_trial_colorbars, check_trial_indices, draw_example_pathway,
    draw_trial_rate, draw_psth, draw_saccade_markers, make_example_axes,
    make_top_grid, matched_psth, share_example_pathway_ylim, trial_rate_norm,
)
from paper.supp_twin_saccade_modulation.plot import event_summary, rank_example_rows, unit_maps


def candidates():
    """Rank the cached cohort independently of the top-10 selection artifact."""
    folders = sorted((CACHE / 'sessions').glob('*/results.json'))
    if len(folders) != 19:
        raise AssertionError(f'expected 19 completed sessions, got {len(folders)}')
    rows = [r for file in folders for r in read_json(file)['cohort']]
    return rank_example_rows(rows)[:50]


def page(row, rank, maps, lags, means, intervals, n_events, effect_limit):
    """The composite's A–D content with a separate review heading and footer."""
    psth = matched_psth(maps)
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 12, left=.065, right=.97, top=.88, bottom=.13,
                            height_ratios=[2.3, 1], hspace=.34, wspace=.85)
    top = make_top_grid(grid)
    lo, hi, n = maps['time_s'][0], maps['time_s'][-1], len(maps['trials'])
    extent = (lo-.5/RATE, hi+.5/RATE, n-.5, -.5)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#e9e9e9')
    grey = plt.get_cmap('Greys').copy()
    grey.set_bad('#e9e9e9')
    rate_norm = trial_rate_norm(maps)
    top_axes, images = [], []
    for j, (key, title) in enumerate((('observed', 'Observed spikes'), ('full', 'Full prediction'),
                                      ('gain', 'Gain pathway'), ('additive', 'Additive pathway'))):
        ax = fig.add_subplot(top[0, 3*j:3*j+3])
        top_axes.append(ax)
        if key in ('observed', 'full'):
            im = draw_trial_rate(ax, maps, key, extent, rate_norm, grey)
        else:
            im = ax.imshow(np.ma.masked_invalid(maps[key]), origin='upper', aspect='auto',
                           interpolation='nearest', extent=extent, cmap=cmap,
                           norm=TwoSlopeNorm(vmin=-effect_limit, vcenter=0, vmax=effect_limit))
        images.append(im)
        draw_saccade_markers(ax, maps['markers'])
        ax.set(title=title, xlabel='Trial time (s)', xlim=(lo, .8), ylim=(n-.5, -.5))
        ax.tick_params(axis='x', labelsize=8)
        if j == 0:
            ticks = np.arange(0, n, 8)
            ax.set_yticks(ticks, [str(maps['trials'][int(t)]) for t in ticks])
            ax.set_ylabel('Trial ID (chronological)')
            ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=.9,
                      fontsize=8, markerscale=1, borderpad=.3)
            ax.text(-.18, 1.05, 'A', transform=ax.transAxes, fontweight='bold', fontsize=14)
        else:
            ax.set_yticks([])
    b, c, d = make_example_axes(fig, grid)
    draw_psth(b, maps['time_s'], psth)
    b.set_xlim(lo, .8)
    for axis, name in ((c, 'gain'), (d, 'additive')):
        draw_example_pathway(axis, lags, name, means, intervals)
        axis.set_ylabel('')
    share_example_pathway_ylim((c, d), lags, means, intervals)
    for letter, axis in [('B', b), ('C', c), ('D', d)]:
        axis.text(-.09, 1.09, letter, transform=axis.transAxes, fontweight='bold', fontsize=14)
        axis.spines[['top', 'right']].set_visible(False)
    add_trial_colorbars(fig, top_axes, images[0], images[3])
    tag = ' · Current example' if row['session'] == SESSION and row['cid'] == CID else ''
    fig.text(.065, .955, f"Rank {rank} / page {rank} · {row['session']} · CID {row['cid']} · "
             f"CCnorm {row['ccnorm']:.6f} · CCmax {row['ccmax']:.6f} · "
             f"{n} displayed trials · {n_events} supported events{tag}", fontsize=12)
    fig.text(.065, .045, 'Raw predictions; no calibration. Event means use all qualifying events, not only displayed trials. '
             'Observed/full map colors saturate at 500 spikes/s; signed effects retain per-unit limits.', fontsize=9)
    return fig


def render(paths):
    ranked = candidates()
    if len(ranked) != 50:
        raise AssertionError(f'expected 50 eligible examples, got {len(ranked)}')
    input_hashes = {str(path): sha256(path) for path in provenance_paths(paths)}
    payloads = {}
    sources = {}
    for session in sorted({r['session'] for r in ranked}):
        folder = CACHE / 'sessions' / session
        source = {key: sha256(path) for key, path in session_inputs(paths, session).items()}
        validate_session_cache(folder, input_hashes, source)
        manifest = read_json(folder / 'manifest.json')
        sources[session] = dict(source_sha256=source, components_sha256=manifest['components_sha256'],
                                results_sha256=manifest['results_sha256'])
        with np.load(folder / 'components.npz') as archive:
            a = {key: archive[key] for key in archive.files}
        with session_inputs(paths, session)['saccades'].open() as stream:
            detections = [(float(x['start_time']), j) for j, x in enumerate(json.load(stream))
                          if np.isfinite(x.get('start_time', np.nan))]
        for rank, row in enumerate(ranked, 1):
            if row['session'] != session:
                continue
            found = np.flatnonzero(a['cids'] == row['cid'])
            if len(found) != 1:
                raise AssertionError(f"missing/duplicate CID: {session} {row['cid']}")
            unit = int(found[0])
            if unit != row['ordinal']:
                raise AssertionError(f'ordinal mismatch: {session} {row["cid"]}')
            maps = unit_maps(a, unit, detections)
            if not maps['trials']:
                raise AssertionError(f'no plottable trials: {session} {row["cid"]}')
            check_trial_indices(a, maps)
            summary = event_summary(a, unit, 20260928 + sum(map(ord, session)) * 1000 + row['cid'])
            if summary is None or summary[0] != row['supported_events']:
                raise AssertionError(f'event count mismatch: {session} {row["cid"]}')
            count, means, intervals = summary
            effects = np.r_[maps['gain'][maps['mask']], maps['additive'][maps['mask']]]
            limit = max(.1, float(np.percentile(abs(effects), 99)))
            psth = matched_psth(maps)
            observed_mean = float(np.mean(maps['observed'][maps['mask']]))
            checks = dict(mask=maps['mask'], time_s=maps['time_s'],
                          observed_psth=psth['observed_smooth_hz'],
                          predicted_psth=psth['predicted_smooth_hz'],
                          gain_mean=means['gain'], additive_mean=means['additive'],
                          gain_interval=intervals['gain'], additive_interval=intervals['additive'])
            numeric_sha256 = {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                              for key, value in checks.items()}
            metadata = dict(page=rank, rank=rank, session=session, cid=row['cid'], ordinal=row['ordinal'],
                            ccnorm=row['ccnorm'], ccmax=row['ccmax'], rate_hz=row['rate_hz'],
                            psth_r2=row['psth_r2'], supported_events=count,
                            displayed_trials=len(maps['trials']), trial_ids=maps['trials'],
                            map_range_s=[float(maps['time_s'][0]), float(maps['time_s'][-1])],
                            excluded_short_or_empty=maps['excluded_short'],
                            invalid_endpoint_trial_ids=maps['invalid_endpoint_trials'],
                            supported_map_bins=int(maps['mask'].sum()), numeric_sha256=numeric_sha256,
                            all_visible_detections=len(maps['markers']),
                            retained_visible=sum(keep for _, _, keep in maps['markers']),
                            shared_effect_limit_hz=limit,
                            full_limit_hz=trial_rate_norm(maps).vmax,
                            effect_saturated_bins=int((abs(effects) > limit).sum()),
                            raw_mean_predicted_observed_ratio=float(np.mean(maps['full'][maps['mask']]) /
                                                                    (observed_mean * RATE)) if observed_mean > 0 else None,
                            current_example=(session == SESSION and row['cid'] == CID))
            payloads[rank] = (row, maps, a['event_lags_ms'].copy(), means, intervals, count, limit, psth, metadata)
        del a
    figure_dir = FIGURES / 'candidate_review'
    stats_dir = STATS / 'candidate_review'
    figure_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_dir / 'top50_examples_A-D.pdf'
    with PdfPages(pdf_path) as pdf:
        for rank in range(1, 51):
            row, maps, lags, means, intervals, count, limit, psth, metadata = payloads[rank]
            fig = page(row, rank, maps, lags, means, intervals, count, limit)
            pdf.savefig(fig, dpi=170)
            if rank == 1:
                fig.savefig(figure_dir / 'page1_preview.png', dpi=170)
            plt.close(fig)
    fields = ('page', 'rank', 'session', 'cid', 'ordinal', 'ccnorm', 'ccmax', 'rate_hz',
              'psth_r2', 'supported_events', 'displayed_trials', 'map_range_s',
              'shared_effect_limit_hz', 'full_limit_hz', 'current_example')
    with (stats_dir / 'candidate_index.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: payloads[i][-1][key] for key in fields} for i in range(1, 51))
    manifest = dict(pdf=str(pdf_path), rank='CCnorm descending; session/CID tie break; existing eligibility and >=20 supported events',
                    input_sha256=input_hashes, sessions=sources, native_rate_hz=RATE,
                    bootstrap_trials=400, bootstrap_seed='20260928 + sum(map(ord, session))*1000 + cid',
                    smoothing_bins=5, event_display_limits_ms=[-100, 200],
                    scales='Per-unit PSTH axes; observed native-bin count × 240 Hz and full-model rate share a 0–500 spikes/s saturated display scale without altering arrays; C/D share a zero-centered y range containing means and bootstrap intervals per unit; signed maps joint 99th absolute percentile per unit (floor 0.1 Hz)',
                    pages=[payloads[i][-1] for i in range(1, 51)])
    (stats_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(f'{len(payloads)} pages, {len(sources)} sessions: {pdf_path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    render(resolved_inputs(args.data_root))


if __name__ == '__main__':
    main()
