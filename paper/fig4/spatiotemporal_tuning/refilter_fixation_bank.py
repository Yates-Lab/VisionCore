#!/usr/bin/env python3
"""Reprocess fixed bank windows from continuous DDPI and audit event classes.

No original bank or saved detector file is modified. Multiple banks are read
in one session pass. Classification is independent of spectra and responses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from paper.fig4.spatiotemporal_tuning.eye_trace_filter import (
    anti_alias_eye_position, gaussian_filter_spec, load_ddpi,
)
from paper.fig4.spatiotemporal_tuning.build_real_fixation_bank import (
    ij_pixels_to_centered_xy_degrees, sha256, spectral_filter_qc, trace_metrics,
)


def audit_session_events(ddpi: pd.DataFrame, event_path: Path, ppd: float):
    """Check saved-event coordinates/time against raw DDPI and flag omissions.

    The event-list interior is a conservative coverage restriction, not proof
    of detector sensitivity. Fully valid native data and an independent rapid
    motion screen are also required. The screen never creates microsaccade
    labels: unmatched movements remain excluded.
    """
    time = ddpi.t_ephys.to_numpy(float)
    ij = ddpi[['dpi_i', 'dpi_j']].to_numpy(float)
    dt = np.diff(time)
    if not np.all(dt > 0):
        raise ValueError('DDPI timestamps must be strictly increasing')
    rate = float(1 / np.median(dt))
    raw_speed = np.linalg.norm(np.diff(ij, axis=0), axis=1) / ppd / dt
    source_rows = ddpi.index.to_numpy(dtype=int)
    source_to_clean = np.full(int(source_rows.max()) + 1, -1, dtype=int)
    source_to_clean[source_rows] = np.arange(len(ddpi))

    def event_indices(event):
        original = np.array([event['start_idx'], event['end_idx']], dtype=int)
        if np.any(original < 0) or np.any(original >= len(source_to_clean)):
            return -1, -1
        return tuple(source_to_clean[original])

    valid = ddpi.valid.to_numpy(bool).copy()
    invalid_edge = (dt > 2.5/rate) | (raw_speed >= 5000) | (np.diff(source_rows) != 1)
    valid[:-1] &= ~invalid_edge
    valid[1:] &= ~invalid_edge
    # Match the saved detector's 100-ms exclusion around invalid samples.
    valid = ~ndimage.maximum_filter1d(~valid, 2*int(np.ceil(.1*rate))+1,
                                     mode='constant', cval=1)
    position = ndimage.gaussian_filter1d(ij / ppd, .006*rate, axis=0, truncate=5)
    speed = np.r_[np.linalg.norm(np.diff(position, axis=0), axis=1) / dt, 0.0]
    median = float(np.median(speed[valid]))
    mad = float(np.median(np.abs(speed[valid] - median)))
    # Same 21-MAD multiplier as the saved detector; fixed floor avoids flagging
    # ordinary drift in exceptionally quiet records. This only excludes, never
    # labels, candidate microsaccades.
    threshold = max(5.0, median + 21*mad)
    saved_events = json.loads(event_path.read_text())
    raw_delta, saved_delta = [], []
    for event in saved_events:
        s, e = event_indices(event)
        if 0 <= s < e < len(time):
            raw_delta.append((ij[e]-ij[s])[::-1] * [1, -1] / ppd)
            saved_delta.append([event['end_x']-event['start_x'], event['end_y']-event['start_y']])
    # Saved event coordinates can predate a later affine DDPI calibration.
    # Establish the common coordinate transform, rather than mistaking its gain
    # difference for an event mismatch. Classify amplitude in CURRENT DDPI degrees.
    transform = np.linalg.lstsq(np.asarray(raw_delta), np.asarray(saved_delta), rcond=None)[0]
    events = []
    for event_id, event in enumerate(saved_events):
        start, stop = float(event['start_time']), float(event['end_time'])
        if not (np.isfinite(start+stop) and 0 < stop-start <= .25):
            continue
        saved_vector = np.array([event['end_x']-event['start_x'], event['end_y']-event['start_y']])
        saved_amp = float(np.linalg.norm(saved_vector))
        s, e = event_indices(event)
        index_valid = 0 <= s < e < len(time)
        amplitude_error = float('inf')
        timing_ok = False
        amp = float('nan')
        if index_valid:
            delta = (ij[e]-ij[s])[::-1] * [1, -1] / ppd
            amp = float(np.linalg.norm(delta))
            amplitude_error = float(np.linalg.norm(delta @ transform - saved_vector))
            timing_ok = time[s] <= float(event['mu']) <= time[e]
        verified = bool(np.isfinite(amp) and amplitude_error <= .02 and timing_ok)
        events.append({'event_id': event_id, 'start_time': start, 'end_time': stop,
                       'amplitude_deg': amp, 'saved_amplitude_deg': saved_amp,
                       'coordinate_transform_residual_deg': amplitude_error,
                       'verified': verified})
    events = pd.DataFrame(events).sort_values('start_time').reset_index(drop=True)
    verified = events.loc[events.verified]
    labeled = np.zeros(len(time), bool)
    for event in verified.itertuples():
        lo, hi = np.searchsorted(time, [event.start_time-.05, event.end_time+.05])
        labeled[lo:hi] = True
    unassigned = (speed > threshold) & ~labeled & valid
    return {'time': time, 'valid': valid, 'unassigned': unassigned,
            'coverage_start': float(verified.start_time.min()) if len(verified) >= 2 else float('inf'),
            'coverage_stop': float(verified.end_time.max()) if len(verified) >= 2 else -float('inf'),
            'screen_threshold_deg_s': threshold, 'source_rate_hz': rate,
            'events': events, 'speed': speed, 'current_to_saved_coordinate_transform': transform.tolist()}


def classify_window(audit, start: float, stop: float, guard: float = .05):
    events = audit['events']
    near = events[(events.start_time < stop+guard) & (events.end_time > start-guard)]
    inside = near[(near.start_time >= start+guard) & (near.end_time <= stop-guard)]
    micro = inside[(inside.amplitude_deg > 0) & (inside.amplitude_deg < 1) & inside.verified]
    lo, hi = np.searchsorted(audit['time'], [start-guard, stop+guard])
    covered = (start-guard >= audit['coverage_start'] and stop+guard <= audit['coverage_stop']
               and hi > lo and bool(np.all(audit['valid'][lo:hi])))
    unmatched = bool(np.any(audit['unassigned'][lo:hi]))
    reasons = []
    if not covered:
        reasons.append('uncertain_coverage_or_invalid_native_data')
    if unmatched:
        reasons.append('unmatched_rapid_motion')
    if np.any(~near.verified):
        reasons.append('unverified_saved_event')
    amplitudes = near.amplitude_deg.to_numpy(dtype=float)
    if np.any(~np.isfinite(amplitudes) | (amplitudes >= 1) | (amplitudes <= 0)):
        reasons.append('non_microsaccade_event')
    if len(near) != len(inside):
        reasons.append('event_near_window_boundary')
    group = 'excluded' if reasons else ('microsaccade' if len(micro) else 'drift')
    return {'event_class': group, 'event_class_exclusion': ';'.join(reasons),
            'event_coverage_pass': bool(covered), 'unmatched_rapid_motion': unmatched,
            'verified_microsaccade_count': int(len(micro)),
            'verified_microsaccade_max_amplitude_deg': float(micro.amplitude_deg.max()) if len(micro) else 0.0,
            'verified_event_ids': ';'.join(str(x) for x in micro.event_id),
            'rapid_motion_screen_threshold_deg_s': audit['screen_threshold_deg_s']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bank', type=Path, nargs='+', required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--processed-root', type=Path, default=Path('/mnt/ssd/YatesMarmoV1/processed'))
    parser.add_argument('--sigma-ms', type=float, default=6)
    parser.add_argument('--sensitivity-sigma-ms', type=float, nargs='*', default=[4, 8])
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    widths = sorted(set([args.sigma_ms, *args.sensitivity_sigma_ms]))
    cache_contract = {'version': 3, 'widths_ms': widths,
                      'source_manifests': [sha256(p/'manifest.json') for p in args.source_bank]}
    banks = []
    for source in args.source_bank:
        manifest = json.loads((source/'manifest.json').read_text())
        table = pd.read_csv(source/'trace_table.csv')
        shape = (len(table), manifest['history_samples']+manifest['analysis_samples'], 2)
        name = f"fixation_bank_gaussian_{args.sigma_ms:g}ms_{manifest['analysis_samples']}samples_n{len(table)}"
        destination = args.out_root/name
        destination.mkdir(exist_ok=True)
        banks.append({'source': source.resolve(), 'destination': destination.resolve(),
                      'manifest': manifest, 'table': table,
                      'filtered': {w: np.full(shape, np.nan, np.float32) for w in widths},
                      'raw': np.full(shape, np.nan, np.float32), 'reports': []})
    sessions = sorted(set().union(*(set(b['table'].session) for b in banks)))
    audit_dir = args.out_root/'event_audit'
    audit_dir.mkdir(exist_ok=True)
    for session_number, session in enumerate(sessions, 1):
        cache = audit_dir/f'{session}.npz'
        # Cache complete outputs per session, permitting an interrupted pass to
        # resume without rereading multi-million-sample DDPI recordings.
        cache_json = cache.with_suffix('.json')
        if (cache.exists() and cache_json.exists() and
                json.loads(cache_json.read_text()).get('cache_contract') == cache_contract):
            values = np.load(cache)
            cached = json.loads(cache_json.read_text())
            for n, bank in enumerate(banks):
                ids = bank['table'].index[bank['table'].session == session].to_numpy()
                for width in widths:
                    bank['filtered'][width][ids] = values[f'b{n}_sigma{width:g}']
                bank['raw'][ids] = values[f'b{n}_raw']
                bank['reports'].extend(cached['banks'][n])
            print(f'{session_number}/{len(sessions)} restored {session}', flush=True)
            continue
        path = args.processed_root/session/'dpi/ddpi.csv'
        print(f'{session_number}/{len(sessions)} reading {session}', flush=True)
        ddpi = load_ddpi(path)
        first = next(b for b in banks if session in set(b['table'].session))
        ppd = float(first['table'].loc[first['table'].session == session, 'ppd'].iloc[0])
        event_path = args.processed_root/session/'saccades/saccades.json'
        audit = audit_session_events(ddpi, event_path, ppd)
        audit['events'].to_csv(audit_dir/f'{session}_events.csv', index=False)
        saved = {}
        reports = []
        for n, bank in enumerate(banks):
            ids = bank['table'].index[bank['table'].session == session].to_numpy()
            history = bank['manifest']['history_samples']
            current_reports = []
            for index in ids:
                row = bank['table'].iloc[index]
                entry = {'trace_index': int(index), **classify_window(audit,
                    row.window_start_ephys+history/240, row.window_stop_ephys)}
                for width in widths:
                    try:
                        eye = anti_alias_eye_position(ddpi, start_ephys=row.window_start_ephys,
                            stop_ephys=row.window_stop_ephys, target_rate_hz=240,
                            passband_hz=20, stopband_hz=30, padding_seconds=.6,
                            filter_family='gaussian', gaussian_sigma_seconds=width/1000)
                    except ValueError as error:
                        if 'padding' not in str(error):
                            raise
                        entry.update(event_class='excluded', event_class_exclusion='insufficient_raw_filter_padding')
                        break
                    xy = ij_pixels_to_centered_xy_degrees(eye['filtered_position_px'],
                        ppd=row.ppd, center_from=slice(history, None))
                    bank['filtered'][width][index] = xy
                    if width == args.sigma_ms:
                        entry.update(trace_metrics(xy, history_samples=history))
                        entry['raw_eye_sample_rate_hz'] = eye['source_rate_hz']
                        bank['raw'][index] = ij_pixels_to_centered_xy_degrees(
                            eye['unfiltered_position_px'], ppd=row.ppd, center_from=slice(history, None))
                current_reports.append(entry)
            bank['reports'].extend(current_reports)
            reports.append(current_reports)
            saved.update({f'b{n}_sigma{w:g}': bank['filtered'][w][ids] for w in widths})
            saved[f'b{n}_raw'] = bank['raw'][ids]
        np.savez_compressed(cache, **saved)
        cache_json.write_text(json.dumps({'source_ddpi': str(path), 'source_ddpi_size': path.stat().st_size,
            'source_ddpi_sha256': sha256(path), 'cache_contract': cache_contract,
            'saved_events_sha256': sha256(event_path), 'source_rate_hz': audit['source_rate_hz'],
            'current_to_saved_coordinate_transform': audit['current_to_saved_coordinate_transform'],
            'rapid_motion_screen_threshold_deg_s': audit['screen_threshold_deg_s'],
            'verified_saved_events': int(audit['events'].verified.sum()),
            'total_saved_valid_duration_events': len(audit['events']), 'banks': reports}, indent=2)+'\n')
        print(f'{session}: verified {audit["events"].verified.sum()}/{len(audit["events"])} saved events', flush=True)
    for bank in banks:
        out = bank['destination']
        table = bank['table'].copy()
        reports = pd.DataFrame(bank['reports']).sort_values('trace_index').set_index('trace_index')
        assert np.array_equal(reports.index, table.trace_index)
        for column in reports:
            table[column] = reports[column].to_numpy()
        finite = np.isfinite(bank['raw']).all(axis=(1,2))
        for xy in bank['filtered'].values():
            finite &= np.isfinite(xy).all(axis=(1,2))
        table.loc[~finite].to_csv(out/'rejected_missing_padding.csv', index=False)
        table = table.loc[finite].copy().reset_index(drop=True)
        table['source_trace_index'] = table.trace_index
        table['trace_index'] = np.arange(len(table))
        bank['filtered'] = {w: xy[finite] for w,xy in bank['filtered'].items()}
        bank['raw'] = bank['raw'][finite]
        filtered, raw = bank['filtered'][args.sigma_ms], bank['raw']
        assert np.isfinite(filtered).all() and np.isfinite(raw).all()
        analysis = filtered[:, bank['manifest']['history_samples']:]
        maximum_displacement = np.linalg.norm(analysis[:, 12:]-analysis[:, :-12], axis=2).max(axis=1)
        table['maximum_50ms_displacement_deg'] = maximum_displacement
        # Conservative guard against a larger movement being split into several
        # subdegree fitted events. This marks it outside the requested condition,
        # not a tracking artifact. No movement is deleted from the stored trace.
        large = maximum_displacement >= 1.0
        table.loc[large, 'event_class'] = 'excluded'
        table.loc[large, 'event_class_exclusion'] = table.loc[large, 'event_class_exclusion'].fillna('') + ';displacement_at_least_1deg_within_50ms'
        table['filtered_vs_raw_rms_difference_deg'] = np.sqrt(np.mean((filtered-raw)**2, axis=(1,2)))
        table.to_csv(out/'trace_table.csv', index=False)
        np.save(out/'trace_xy_filtered.npy', filtered)
        np.save(out/'trace_xy_raw.npy', raw)
        for width, xy in bank['filtered'].items():
            np.save(out/f'trace_xy_gaussian_{width:g}ms.npy', xy)
        manifest = bank['manifest'].copy()
        manifest['analysis'] = 'fixed-window continuous-DDPI Gaussian refilter with conservative event classification'
        manifest['n_traces'] = len(table)
        manifest['n_sessions'] = int(table.session.nunique())
        manifest['session_counts'] = {str(k): int(v) for k,v in table.session.value_counts().items()}
        manifest['filter'] = gaussian_filter_spec(args.sigma_ms/1000)
        rate = float(table.raw_eye_sample_rate_hz.median())
        step = np.r_[np.zeros(1000), np.ones(1000)]
        response = ndimage.gaussian_filter1d(step, args.sigma_ms/1000*rate, truncate=5)
        manifest['filter_validation'] = {
            'positive_kernel': True, 'monotonic_step_response': bool(np.min(np.diff(response)) >= -1e-12),
            'continuous_raw_before_resampling': True, 'padding_covers_kernel': .6 >= 5*max(widths)/1000,
            'sensitivity_sigma_ms': widths,
            'gaussian_does_not_have_a_20_30_hz_brickwall_stopband': True}
        manifest['spectral_filter_qc'] = spectral_filter_qc(filtered, raw,
            history_samples=manifest['history_samples'], sample_rate_hz=240, stopband_hz=30)
        manifest['event_classification'] = {
            'definition': 'drift: no saved event within ±50 ms; microsaccade: fully contained verified 0<amplitude<1 deg event with 50-ms margins; both require coverage and no unmatched rapid motion',
            'coverage': 'interior bracketed by coordinate/time-verified saved events; native validity with 100-ms invalid guard and no >2.5-sample gaps',
            'unmatched_rapid_screen': '6-ms Gaussian native speed > max(5 deg/s, session median + 21 MAD), outside verified-event intervals expanded by 50 ms',
            'amplitude_max_deg_exclusive': 1.0, 'event_guard_seconds': .05,
            'maximum_50ms_displacement_deg_exclusive': 1.0,
            'amplitude_coordinates': 'current DDPI displacement at saved detector start_idx/end_idx; saved coordinates verified via a session affine transform',
            'selection_reads_spectra_or_responses': False,
            'counts': {str(k): int(v) for k,v in table.event_class.value_counts().items()},
            'audit_directory': str(audit_dir.resolve())}
        manifest['refilter_source'] = {'bank': str(bank['source']),
            'manifest_sha256': sha256(bank['source']/'manifest.json'),
            'trace_table_sha256': sha256(bank['source']/'trace_table.csv'),
            'same_windows_and_order_except_missing_raw_padding': True,
            'n_rejected_missing_padding': int((~finite).sum())}
        manifest['files'] = {key: str((out/name).resolve()) for key,name in
            [('filtered','trace_xy_filtered.npy'), ('raw','trace_xy_raw.npy'), ('table','trace_table.csv')]}
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(out, manifest['event_classification']['counts'], flush=True)


if __name__ == '__main__':
    main()
