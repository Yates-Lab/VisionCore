#!/usr/bin/env python3
"""Build response-replay inputs from the audited Gaussian fixation bank."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from paper.fig4.spatiotemporal_tuning.build_real_fixation_bank import evenly_spaced_indices, sha256
from paper.fig4.spatiotemporal_tuning.eye_trace_filter import filter_qc_passed
from paper.fig4.upstream.real_trace_matrix.core import (
    trace_scale_metrics, trace_covariance_shape, trace_covariance_anisotropy,
    trace_bank_metadata_row, trace_rms, path_length,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixation-bank', type=Path, required=True)
    parser.add_argument('--image-table', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--n-traces', type=int, default=200)
    args = parser.parse_args()
    manifest = json.loads((args.fixation_bank/'manifest.json').read_text())
    if not filter_qc_passed(manifest) or manifest['analysis_samples'] != 60:
        raise ValueError('Requires a validated 60-sample Gaussian analysis bank')
    table = pd.read_csv(args.fixation_bank/'trace_table.csv')
    traces = np.load(args.fixation_bank/'trace_xy_filtered.npy')[:, -60:]
    eligible = table.index[table.event_class.isin(['drift', 'microsaccade'])
        & (60*table.analysis_path_length_deg <= 350)].to_numpy()
    animal = table.session.str.split('_').str[0].to_numpy()
    animals = sorted(set(animal))
    if args.n_traces % len(animals):
        raise ValueError('n-traces must allow equal animal counts')
    count = args.n_traces // len(animals)
    selected = np.sort(np.concatenate([
        eligible[animal[eligible] == subject][evenly_spaced_indices(int(np.sum(animal[eligible] == subject)), count)]
        for subject in animals]))
    output_traces, rows = [], []
    for index, source in enumerate(selected):
        row = table.iloc[source]
        trace = traces[source].copy()
        trace -= trace.mean(axis=0, keepdims=True)
        item = {'source_row': int(source), 'session': row.session, 'trial_idx': row.trial_idx,
            'global_start': -1, 'global_stop': -1,
            'snippet_n_samples': 60, 'snippet_duration_s': 60/240,
            'trace': trace, 'observed_rms_deg': trace_rms(trace),
            'path_length_deg': path_length(trace),
            'covariance_shape': trace_covariance_shape(trace),
            'trace_cov_anisotropy': trace_covariance_anisotropy(trace),
            'source_trace_cov_anisotropy': trace_covariance_anisotropy(trace),
            **trace_scale_metrics(trace, dt=1/240, prefix='source_'),
            **trace_scale_metrics(trace, dt=1/240, prefix='rendered_')}
        for prefix in ('', 'source_', 'rendered_'):
            item[prefix+'n_microsaccade_events'] = int(row.verified_microsaccade_count)
        features = trace_bank_metadata_row(item, index, n_timepoints=60, scale_metric='path_length_arcmin')
        features.update(event_class=row.event_class, source_trace_index=int(row.source_trace_index),
            scored_start_ephys=float(row.window_start_ephys+manifest['history_samples']/240),
            scored_stop_ephys=float(row.window_stop_ephys))
        rows.append(features)
        output_traces.append(trace)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir/'trace_xy.npy', np.stack(output_traces))
    pd.DataFrame(rows).to_csv(args.out_dir/'trace_feature_table.csv', index=False)
    shutil.copy2(args.image_table, args.out_dir/'image_feature_table.csv')
    provenance = {'analysis': 'continuously Gaussian-filtered, event-audited real-fixation replay inputs',
        'filter': manifest['filter'], 'filter_validation': manifest['filter_validation'],
        'spectral_filter_qc': manifest['spectral_filter_qc'],
        'event_classification': manifest['event_classification'],
        'fixation_bank': str(args.fixation_bank.resolve()), 'sample_rate_hz': 240.0,
        'n_traces': len(rows), 'n_timepoints': 60, 'n_images': len(pd.read_csv(args.image_table)),
        'n_microsaccade_traces': sum(r['event_class'] == 'microsaccade' for r in rows),
        'native_global_indices': 'not applicable: uniform resampling at explicit ephys timestamps; -1 denotes unavailable dataset indices',
        'trace_selection': {'kind': 'equal animal counts; evenly spaced eligible source rows within animal; no response or spectral selection',
            'animal_counts': {a: count for a in animals},
            'eligible_n_traces': len(eligible), 'selected_source_indices': selected.tolist(),
            'maximum_path_length_arcmin': 350},
        'source_files': {name: {'path': str(path.resolve()), 'sha256': sha256(path)} for name,path in
            [('manifest',args.fixation_bank/'manifest.json'), ('table',args.fixation_bank/'trace_table.csv'),
             ('filtered',args.fixation_bank/'trace_xy_filtered.npy'), ('images',args.image_table)]}}
    (args.out_dir/'trace_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({'n_traces': len(rows), 'n_microsaccade_traces': provenance['n_microsaccade_traces'],
                      'eligible_n_traces': len(eligible)}, indent=2))


if __name__ == '__main__':
    main()
