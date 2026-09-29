"""FIXRSVP multisession population and example analysis; no calibration or fitting."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from VisionCore.paths import CACHE_DIR, FIGURES_DIR, STATS_DIR, VISIONCORE_ROOT
from manuscript.analysis_selection import selected_analysis, source_path

CACHE = CACHE_DIR / 'supp_twin_saccade_modulation'
FIGURES = FIGURES_DIR / 'supp_twin_saccade_modulation'
STATS = STATS_DIR / 'supp_twin_saccade_modulation'
CHECKPOINT_SHA256 = 'e70f287462155607a96c7d23d9d56e347111319aa48cafd3f54fe88dd4484cb9'
RATE = 240
PRE, POST = 24, 48


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def input_paths(source_root, empirical_cache_dir, data_root):
    """Locate the manuscript's selected inputs without using recorded machine paths."""
    source_root = Path(source_root)
    selection = read_json(VISIONCORE_ROOT / 'manuscript/analysis/selected_model_bundle.json')
    selected_bundle = source_root / selection['bundle']
    run_environment = read_json(selected_bundle / 'figure3/run_manifest.json')['environment']

    def relocated(environment_key):
        recorded_path = Path(run_environment[environment_key])
        for anchor in ('outputs', 'paper', 'scripts'):
            if anchor in recorded_path.parts:
                return source_root.joinpath(*recorded_path.parts[recorded_path.parts.index(anchor):])
        raise ValueError(f'cannot relocate selected input: {environment_key}')

    return dict(checkpoint=relocated('FIG3_MODEL_CHECKPOINT'),
                scores=relocated('FIG3_CACHE_PATH'),
                config=VISIONCORE_ROOT / 'paper/model_selection/configs/multi_240_long_split3_dekel35_allgratings.yaml',
                aligned=Path(empirical_cache_dir) / 'covdecomp_aligned_sessions.pkl',
                pointer=VISIONCORE_ROOT / 'manuscript/analysis/selected_model_bundle.json',
                spec=selected_bundle / 'model/no_phase_model.yaml', data=Path(data_root))


def resolved_inputs(data_root):
    """Validate the completed selection and resolve local empirical/data substitutes."""
    selection = selected_analysis()
    if selection['checkpoint_sha256'] != CHECKPOINT_SHA256:
        raise ValueError('manuscript selection changed')
    source_root = source_path(selection['bundle']).parents[2]
    paths = input_paths(source_root, os.environ.get('VISIONCORE_MANUSCRIPT_EMPIRICAL_CACHE_DIR', CACHE_DIR), data_root)
    if sha256(paths['checkpoint']) != CHECKPOINT_SHA256 or sha256(paths['spec']) != selection['model_spec_sha256']:
        raise ValueError('selected model inputs changed')
    run_manifest = read_json(source_root / selection['bundle'] / 'figure3/run_manifest.json')
    for input_key, expected in [('config', run_manifest['dataset_config_sha256']),
                                ('aligned', run_manifest['covdecomp_aligned_cache_sha256'])]:
        if sha256(paths[input_key]) != expected:
            raise ValueError(f'selected input changed: {input_key}')
    return paths


def session_inputs(paths, session):
    folder = paths['data'] / session
    return {'fixrsvp': folder / 'datasets/fixrsvp.dset', 'saccades': folder / 'saccades/saccades.json'}


def provenance_paths(paths):
    return tuple(paths[key] for key in ('checkpoint', 'config', 'aligned', 'scores', 'pointer', 'spec'))


def ensure_empty_session_cache(folder):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise AssertionError(f'{folder}: partial session cache without a valid manifest')


def validate_session_cache(folder, inputs, source_hashes):
    """Accept relocated manifests by digest, retaining their recorded provenance."""
    folder = Path(folder)
    manifest = read_json(folder / 'manifest.json')
    if (list(manifest['inputs'].values()) != list(inputs.values())
            or manifest['source_hashes'] != source_hashes
            or not (folder / 'components.npz').is_file()
            or not (folder / 'results.json').is_file()
            or sha256(folder / 'components.npz') != manifest['components_sha256']
            or sha256(folder / 'results.json') != manifest['results_sha256']):
        raise AssertionError(f'{folder}: incomplete or mismatched session cache')


def cohort_rows(aligned, scores, snapshot):
    """Join cache ordinals, never bare CIDs; do not require scores for population."""
    lookup = {}
    for sr in scores:
        for j, ordinal in enumerate(sr['neuron_mask']):
            key = sr['session'], int(ordinal)
            if key in lookup:
                raise ValueError(f'duplicate score: {key}')
            lookup[key] = (float(sr['ccnorm'][j]), float(sr['ccmax'][j]))
    rows = []
    for sr in aligned:
        session = sr['session']
        selected = np.flatnonzero(np.isfinite(sr['rate_hz']) & (np.asarray(sr['rate_hz']) > 2)
                                  & np.isfinite(sr['psth_r2']) & (np.asarray(sr['psth_r2']) > .10))
        if len(selected) < 10:
            continue
        cids = snapshot[session]
        for j in selected:
            ordinal = int(sr['neuron_mask'][j])
            if ordinal < 0 or ordinal >= len(cids):
                raise ValueError(f'ordinal missing from snapshot: {session} {ordinal}')
            cc, ccmax = lookup.get((session, ordinal), (float('nan'), float('nan')))
            rows.append(dict(session=session, ordinal=ordinal, cid=int(cids[ordinal]),
                             rate_hz=float(sr['rate_hz'][j]), psth_r2=float(sr['psth_r2'][j]),
                             ccnorm=cc if np.isfinite(cc) else None,
                             ccmax=ccmax if np.isfinite(ccmax) else None,
                             example_eligible=bool(np.isfinite(cc) and cc <= 1 and np.isfinite(ccmax) and ccmax > .85)))
    if len({(r['session'], r['cid']) for r in rows}) != len(rows):
        raise ValueError('duplicate cohort session/CID')
    return rows


def load_cohort(paths):
    import dill
    import torch
    with paths['aligned'].open('rb') as f:
        aligned = dill.load(f)
    with paths['scores'].open('rb') as f:
        scores = pickle.load(f)
    checkpoint = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    snapshot = checkpoint['hyper_parameters']['dataset_cids']
    rows = cohort_rows(aligned, scores, snapshot)
    sessions = [r['session'] for r in aligned if sum(x['session'] == r['session'] for x in rows) >= 10]
    return rows, sessions, snapshot


def crop_maps(times, trial_ids, valid, values, end_s=.8, rate=RATE):
    """Endpoint from recorded samples, preserve original trial-relative coordinates and holes."""
    n = int(round(end_s * rate)) + 1
    kept, excluded, invalid_endpoints = [], 0, []
    starts = np.r_[0, np.flatnonzero(np.diff(trial_ids) != 0) + 1]
    stops = np.r_[starts[1:], len(trial_ids)]
    mats = {k: [] for k in values}
    masks = []
    for lo, hi in zip(starts, stops):
        idx = np.arange(lo, hi)
        col = np.rint((times[idx] - times[lo]) * rate).astype(int)
        if len(set(col)) != len(col):
            raise ValueError('duplicate native trial-relative bin')
        if not np.any(col == n - 1):
            excluded += 1
            continue
        inside = (col >= 0) & (col < n)
        idx, col = idx[inside], col[inside]
        mask = np.zeros(n, bool)
        mask[col] = valid[idx]
        if not mask.any():
            excluded += 1
            continue
        kept.append(int(trial_ids[lo]))
        if not mask[-1]:
            invalid_endpoints.append(int(trial_ids[lo]))
        masks.append(mask)
        for key, arr in values.items():
            row = np.full(n, np.nan)
            row[col[valid[idx]]] = arr[idx[valid[idx]]]
            mats[key].append(row)
    if not kept:
        return dict(trials=[], excluded_short=excluded, invalid_endpoint_trials=[],
                    time_s=np.array([]), mask=np.zeros((0, 0), bool))
    mask = np.stack(masks)
    first = int(np.flatnonzero(mask.any(axis=0))[0])
    return dict(trials=kept, excluded_short=excluded, invalid_endpoint_trials=invalid_endpoints,
                time_s=np.arange(first, n) / rate,
                mask=mask[:, first:], **{key: np.stack(rows)[:, first:] for key, rows in mats.items()})


def supported_effects(a, windows, rate=RATE, unit=0):
    """Only events with complete shared unit support over the entire window."""
    width = windows.shape[1]
    if not len(windows):
        return dict(n_events=0, n_windows=0, gain=np.full(width, np.nan),
                    additive=np.full(width, np.nan), event_indices=np.array([], int))
    from paper.supp_twin_saccade_modulation.decomposition import shared_condition_mask
    arrays = [a[k][windows, unit] for k in ('r0', 'rfull', 'r_gain', 'r_additive')]
    support = shared_condition_mask(a['robs'][windows, unit], a['dfs'][windows, unit], *arrays)
    selected = np.flatnonzero(support.all(axis=1))
    return dict(n_events=len(selected), n_windows=len(windows), event_indices=selected,
                gain=(a['r_gain'][windows[selected], unit] - a['r0'][windows[selected], unit]) * rate,
                additive=(a['r_additive'][windows[selected], unit] - a['r0'][windows[selected], unit]) * rate)


def smooth_rows(values, bins=5):
    """Smooth each complete event separately; no crossing event boundaries."""
    pad = bins // 2
    return np.stack([np.convolve(np.pad(row, (pad, pad), mode='edge'),
                                  np.ones(bins) / bins, mode='valid') for row in values])


def sort_population(rows, gain, additive, lags, tolerance=1e-9):
    """Image top-to-bottom: late positive peaks, early positive peaks, then nonpositive block."""
    post = np.flatnonzero((lags >= 0) & (lags <= 200))
    annotated = []
    for j, row in enumerate(rows):
        values = gain[j, post]
        finite = np.isfinite(values)
        if finite.any() and np.max(values[finite]) > tolerance:
            peak = post[np.flatnonzero(finite)[np.argmax(values[finite])]]
            annotated.append(dict(row, positive_gain_peak_ms=float(lags[peak]),
                                  positive_gain_peak_hz=float(gain[j, peak])))
        else:
            annotated.append(dict(row, positive_gain_peak_ms=None, positive_gain_peak_hz=None))
    order = sorted(range(len(rows)), key=lambda j: (annotated[j]['positive_gain_peak_ms'] is None,
                  -(annotated[j]['positive_gain_peak_ms'] or 0), annotated[j]['session'], annotated[j]['cid']))
    return [annotated[j] for j in order], gain[order], additive[order]


def read_json(path):
    with Path(path).open() as f:
        return json.load(f)


def save_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    temp.replace(path)
