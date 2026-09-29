#!/usr/bin/env python3
"""Sequential, resumable (validated complete sessions only) FIXRSVP inference."""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np

from paper.supp_twin_saccade_modulation.analysis import (
    CACHE, POST, PRE, RATE, ensure_empty_session_cache, load_cohort, provenance_paths,
    resolved_inputs, save_json, session_inputs, sha256, smooth_rows,
    supported_effects, validate_session_cache,
)


def assert_registry_fixrsvp_path(registry_session, paths, session):
    """Ensure preprocessing and source audits read the same FIXRSVP file."""
    registered = Path(registry_session.sess_dir) / 'datasets/fixrsvp.dset'
    requested = session_inputs(paths, session)['fixrsvp']
    if not (registered.is_file() and requested.is_file() and registered.samefile(requested)):
        raise ValueError(
            f'{session}: --data-root FIXRSVP file {requested} differs from '
            f'the configured DataYatesV1 registry file {registered}'
        )


def extract(model, session, cids, paths):
    import torch
    from models.data import prepare_data
    from DataYatesV1.utils.io import get_session
    from paper.supp_twin_saccade_modulation.decomposition import (
        analytic_components, isolated_event_indices, valid_event_windows,
        valid_inference_indices,
    )

    i = model.names.index(session)
    cfg = copy.deepcopy(model.cfgs[i])
    cfg['types'] = ['fixrsvp']
    cfg.pop('train_all_stimulus_types', None)
    registry_session = get_session(*cfg['session'].split('_'))
    if registry_session is None:
        raise ValueError(f'{session}: not found in the configured DataYatesV1 registry')
    assert_registry_fixrsvp_path(registry_session, paths, session)
    train, val, test, cfg = prepare_data(cfg, strict=True, return_test=True)
    dset = train.dsets[0]
    if not np.array_equal(cfg['cids'], cids):
        raise AssertionError(f'{session}: resolved CIDs differ from checkpoint snapshot')
    raw = torch.load(
        session_inputs(paths, session)['fixrsvp'], weights_only=False,
        mmap=True, map_location='cpu',
    )['covariates']
    if not torch.equal(dset['robs'], raw['robs'][:, cids]):
        raise AssertionError(f'{session}: observations disagree with source CIDs')
    trial = dset['trial_inds'].cpu().numpy().astype(np.int64)
    times = dset['t_bins'].cpu().numpy()
    dfs = dset['dfs'].cpu().numpy()
    indices, inference_audit = valid_inference_indices(dfs, trial, times, 60, 1 / RATE)
    lags = torch.arange(60)
    history_audit = {}
    for split_name, split in [('train', train), ('validation', val), ('test', test)]:
        positions = np.flatnonzero(np.isin(split.inds[:, 1].cpu().numpy(), indices))[:8]
        if len(positions):
            canonical = split[torch.from_numpy(positions)]['stim']
            source = split.inds[torch.from_numpy(positions), 1]
            manual = dset['stim'][source[:, None] - lags[None]].permute(0, 2, 1, 3, 4)
            if not torch.equal(canonical, manual):
                raise AssertionError(f'{session}: {split_name} stimulus history mismatch')
        history_audit[split_name] = len(positions)
    if len(indices):
        rows = torch.from_numpy(indices[:8])
        stim = raw['stim'][rows]
        top, left = (stim.shape[-2] - 35) // 2, (stim.shape[-1] - 35) // 2
        expected = (stim[:, top:top+35, left:left+35].float() - 127) / 255
        if not torch.equal(expected, dset['stim'][rows, 0]):
            raise AssertionError(f'{session}: source stimulus crop mismatch')
    module = model.model.eval()
    device = next(module.parameters()).device
    head = module.readouts[i]
    if (module.phase_readouts is not None or module.baseline_enabled
            or not isinstance(module.recurrent, torch.nn.Identity)):
        raise AssertionError('wrong model architecture')
    fields = ('z0', 'zfull', 'delta_additive', 'delta_gain', 'r0', 'rfull', 'r_additive', 'r_gain', 'interaction')
    a = {k: np.full((len(dset), len(cids)), np.nan, np.float32) for k in fields}
    a.update(robs=dset['robs'].cpu().numpy().astype(np.float32, copy=True), dfs=dfs.astype(np.float32),
             times=times, trial_ids=trial, cids=np.asarray(cids, np.int64),
             psth_indices=dset['psth_inds'].cpu().numpy().astype(np.int64), inferred_indices=indices)
    errors = dict(full_logit=0., zero_logit=0., latent=0., rates=0., ordinary_full=0., ordinary_zero=0.)
    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            batch = indices[start:start+128]
            rows = torch.from_numpy(batch)
            stim = dset['stim'][rows[:, None] - lags[None]].permute(0, 2, 1, 3, 4).to(device)
            behavior = dset['behavior'][rows].to(device)
            scaffold = module.convnet(module.frontend(module.adapters[i](stim)))
            result = analytic_components(scaffold, behavior, module.modulator, head)
            full = head(module.modulator(scaffold, behavior))
            zero = head(module.modulator(scaffold, torch.zeros_like(behavior)))
            errors['full_logit'] = max(errors['full_logit'], float((result['zfull'] - full).abs().max()))
            errors['zero_logit'] = max(errors['zero_logit'], float((result['z0'] - zero).abs().max()))
            errors['latent'] = max(errors['latent'], float((result['z0'] + result['delta_additive'] + result['delta_gain'] - full).abs().max()))
            errors['rates'] = max(errors['rates'], float((result['rfull'] - result['r0'] - (result['r_gain'] - result['r0']) - (result['r_additive'] - result['r0']) - result['interaction']).abs().max()))
            if start == 0:
                n = min(32, len(batch))
                errors['ordinary_full'] = float((module(stim[:n], i, behavior[:n]) - result['rfull'][:n]).abs().max())
                errors['ordinary_zero'] = float((module(stim[:n], i, torch.zeros_like(behavior[:n])) - result['r0'][:n]).abs().max())
            for key in fields:
                a[key][batch] = result[key].cpu().numpy()
    if max(errors.values()) >= 2e-4:
        raise AssertionError(f'{session}: failed forward identity: {errors}')
    with session_inputs(paths, session)['saccades'].open() as f:
        events = json.load(f)
    selected, event_audit = isolated_event_indices(events)
    event_times = np.array([events[int(j)]['start_time'] for j in selected])
    predicted = np.zeros(len(times), bool)
    predicted[indices] = True
    eye_valid = np.isfinite(dset['eyepos'].cpu().numpy()).all(1) & (dset['dpi_valid'].cpu().numpy() != 0)
    windows, window_audit = valid_event_windows(times, trial, predicted & eye_valid,
                                                 event_times, PRE, POST)
    centers = windows[:, PRE]
    if len(centers):
        mapped = train.get_inds_from_times(torch.from_numpy(event_times))[:, 1].cpu().numpy()
        if not np.isin(centers, mapped).all():
            raise AssertionError(f'{session}: event center time mapping mismatch')
    event_ids = np.asarray([int(selected[np.argmin(abs(event_times - times[c]))]) for c in centers], np.int64)
    a.update(event_windows=windows, event_ids=event_ids,
             event_lags_ms=np.arange(-PRE, POST+1) / RATE * 1000)
    event_audit.update(window_mapping=window_audit, retained=len(windows))
    return a, dict(session=session, identity_errors=errors, inference=inference_audit,
                   history_samples=history_audit, events=event_audit,
                   n_units=len(cids), n_native_bins=len(times))


def summarize(a, rows):
    windows = a['event_windows']
    lag = a['event_lags_ms']
    gain, additive, audit = [], [], []
    for row in rows:
        unit = int(np.flatnonzero(a['cids'] == row['cid'])[0])
        effect = supported_effects(a, windows, RATE, unit)
        audit.append(dict(row, window_count=effect['n_windows'], supported_events=effect['n_events']))
        if effect['n_events'] >= 20:
            gain.append(smooth_rows(effect['gain']).mean(axis=0))
            additive.append(smooth_rows(effect['additive']).mean(axis=0))
        else:
            gain.append(np.full(len(lag), np.nan))
            additive.append(np.full(len(lag), np.nan))
    return np.asarray(gain), np.asarray(additive), audit


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--session', help='one manuscript session')
    modes.add_argument('--all', action='store_true', help='all 19 manuscript sessions')
    parser.add_argument('--data-root', type=Path, required=True, help='processed FIXRSVP session root')
    args = parser.parse_args()
    if os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
        parser.error('NVIDIA_TF32_OVERRIDE=0 required for inference')
    paths = resolved_inputs(args.data_root)
    rows, sessions, snapshot = load_cohort(paths)
    if len(sessions) != 19 or len(rows) != 1022:
        raise AssertionError(f'cohort unexpectedly changed: {len(sessions)} sessions, {len(rows)} units')
    if args.session and args.session not in sessions:
        parser.error(f'not a manuscript session: {args.session}')
    chosen = sessions if args.all else [args.session]
    inputs = {str(path): sha256(path) for path in provenance_paths(paths)}
    model = None
    for session in chosen:
        files = session_inputs(paths, session)
        source_hashes = {key: sha256(path) for key, path in files.items()}
        folder = CACHE / 'sessions' / session
        manifest_path = folder / 'manifest.json'
        cache = folder / 'components.npz'
        if manifest_path.exists():
            validate_session_cache(folder, inputs, source_hashes)
            print(f'{session}: validated cache', flush=True)
        else:
            ensure_empty_session_cache(folder)
            if model is None:
                from training import MultiDatasetModel
                import torch
                model = MultiDatasetModel.load_from_checkpoint(paths['checkpoint'], strict=True, map_location='cpu',
                            pretrained_checkpoint=None, cfg_dir=str(paths['config']))
                model.model.to('cuda:0' if torch.cuda.is_available() else 'cpu').eval()
            folder.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            a, report = extract(model, session, np.asarray(snapshot[session], int), paths)
            gain, additive, audit = summarize(a, [r for r in rows if r['session'] == session])
            temp = folder / 'components.npz.tmp'
            with temp.open('wb') as f:
                np.savez_compressed(f, **a)
            temp.replace(cache)
            save_json(folder / 'results.json', dict(report, cohort=audit,
                      elapsed_seconds=time.monotonic()-started))
            save_json(manifest_path, dict(session=session, inputs=inputs, source_hashes=source_hashes,
                      components_sha256=sha256(cache), results_sha256=sha256(folder / 'results.json'),
                      model='NO_PHASE_R1_s201', rate_hz=RATE, calibration='none'))
            print(f'{session}: {len(a["event_windows"])} windows; {sum(x["supported_events"] >= 20 for x in audit)} population units; identity {report["identity_errors"]}; {time.monotonic()-started:.1f}s', flush=True)
            del a


if __name__ == '__main__':
    main()
