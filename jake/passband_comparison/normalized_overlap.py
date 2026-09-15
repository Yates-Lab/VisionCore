"""Minimal power-distribution audit using the existing Figure 4 movies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jake.passband_comparison.data import ROOT, SEED, digest, write_json
from jake.passband_comparison.run import save_per_unit, strata_results
from jake.passband_comparison.statistics import (
    animal_weights, bootstrap_weights, fit_predict, make_features,
    paired_scores, trial_folds,
)


def rename_overlap(value):
    if isinstance(value, dict):
        return {k.replace('engagement', 'normalized_overlap'): rename_overlap(v)
                for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    src = parser.parse_args().input_dir.resolve()
    out = src / 'normalized_overlap'
    out.mkdir(parents=True, exist_ok=True)
    original = json.loads((src / 'design.json').read_text())
    selected = json.loads((ROOT / 'manuscript/analysis/selected_model_bundle.json').read_text())
    checks = {}

    def check(name, ok, detail=None):
        checks[name] = {'passed': bool(ok), 'detail': detail}
        if not ok:
            write_json(out / 'failed_audit.json', checks)
            raise ValueError(name)

    check('selected_checkpoint', original['checkpoint_sha256'] == selected['checkpoint_sha256'])
    old_audit = json.loads((src / 'audit.json').read_text())
    check('prior_audit_passed', old_audit['passed'])
    for name, expected in original['source_sha256'].items():
        check('source:' + name, digest(ROOT / name) == expected)
    input_names = ['analysis_inputs.npz', 'traces.csv', 'images.csv', 'units.csv',
                   'primary_predictions_lambda_0.01.npz']
    for name in input_names:
        check('cached_artifact:' + name, digest(src / name) == old_audit['artifact_sha256'][name])

    # The cached E and D subtract stabilized power. Verify that this
    # subtraction does not turn the ratio into a ratio of signed spectra.
    for name in original['source_sha256']:
        if name.endswith('/causal_chain_shard.npz'):
            with np.load(ROOT / name) as shard:
                scale = shard['motion_scales']
                check('motion_axis:' + name, np.array_equal(scale, [0., 1.]))
                for field in ['joint_passband_power', 'total_dynamic_power']:
                    p = shard[field]
                    maximum = float(np.max(np.abs(p[:, :, 0])))
                    check('stabilized_zero:' + name + ':' + field, maximum < 1e-12, maximum)

    with np.load(src / 'analysis_inputs.npz') as data:
        y = data['primary_y']
        e = data['movie_engagement']
        d = data['movie_dynamic']
        dynamic = data['primary_dynamic']
    check('axes', e.shape == (40, 200, 725) and d.shape == (40, 200)
          and y.shape == (200, 725, 2))
    check('positive_total_power', np.all(d > 0), float(np.min(d)))
    q = e / d[:, :, None]
    check('finite_nonnegative_overlap', np.isfinite(q).all() and np.all(q >= 0))
    # Vary each movie's total power independently while retaining its shape.
    rescale = np.exp(np.random.default_rng(SEED + 401).uniform(-3, 3, d.shape))
    recovered = (e * rescale[:, :, None]) / (d * rescale)[:, :, None]
    scale_error = float(np.max(np.abs(q - recovered)))
    check('movie_power_scale_invariance', scale_error < 1e-12, scale_error)
    overlap = np.median(q, axis=0)
    table = pd.read_csv(src / 'traces.csv')
    units = pd.read_csv(src / 'units.csv')
    strict = units.validated_for_figure4.to_numpy(dtype=bool)
    check('strict_subset', int(strict.sum()) == 145)
    animal = table.session.str.startswith('Logan').to_numpy(dtype=float)
    event = table.event_class.eq('microsaccade').to_numpy(dtype=float)
    path = table.rendered_path_length_arcmin.to_numpy(dtype=float)
    weights = animal_weights(animal)
    groups = (table.session + '|' + table.trial_idx.astype(str)).to_numpy()
    names = ['class_path', 'class_path_engagement', 'class_path_dynamic',
             'class_path_dynamic_engagement']
    code = {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
            Path(__file__).with_name('MINIMAL_AUDIT.md'),
            Path(__file__).with_name('statistics.py'), Path(__file__).with_name('run.py'),
            Path(__file__).with_name('data.py')]}
    design = {
        'checkpoint_sha256': original['checkpoint_sha256'],
        'parent_design_sha256': digest(src / 'design.json'),
        'parent_summary_sha256': digest(src / 'summary.json'),
        'source_sha256': {str((src / name).relative_to(ROOT)): digest(src / name)
                          for name in input_names},
        'source_code_sha256': code,
        'predictor': 'q_u(movie) = movie_engagement_u / movie_dynamic; then median over images',
        'primary_contrast': 'class_path_dynamic_normalized_overlap__over__class_path_dynamic',
        'secondary_contrast': 'class_path_normalized_overlap__over__class_path',
        'outcome_order': ['rate_percent', 'ssi_percent'],
        'n_images': 40, 'n_histories': 200, 'n_units': 725, 'strict_n_units': 145,
        'ridge_penalty': .01, 'folds': 5, 'repeats': 3,
        'bootstrap_repeats': 1000, 'bootstrap_seed': SEED + 100,
        'scope': 'Same fixed image ensemble, finite-window estimator, fitted twin and regression rules.'
    }
    write_json(out / 'design.json', design)
    predictions = {name: np.full((3, *y.shape), np.nan, np.float32) for name in names}
    folds = []
    max_baseline_error = 0.
    with np.load(src / 'primary_predictions_lambda_0.01.npz') as reference:
        for repeat in range(3):
            fold = trial_folds(table, SEED + repeat)
            check(f'identical_folds:{repeat}', np.array_equal(fold, reference['fold_assignments'][repeat]))
            folds.append(fold)
            for k in range(5):
                train = np.flatnonzero(fold != k)
                test = np.flatnonzero(fold == k)
                check(f'trial_disjoint:{repeat}:{k}', not set(groups[train]) & set(groups[test]))
                features, _ = make_features(path, overlap, dynamic, animal, event, train)
                for name in names:
                    pred = fit_predict(features[name], y, train, test, weights, penalty=.01)
                    predictions[name][repeat, test] = pred
                    if name in ['class_path', 'class_path_dynamic']:
                        expected = reference[name][repeat, test]
                        err = float(np.max(np.abs(pred - expected)))
                        max_baseline_error = max(max_baseline_error, err)
                        check(f'baseline_reproduction:{repeat}:{k}:{name}',
                              np.allclose(pred, expected, atol=5e-5, rtol=2e-6), err)
                print(f'Normalized overlap: repeat {repeat+1}/3, fold {k+1}/5', flush=True)
    for name, pred in predictions.items():
        check('complete_finite_predictions:' + name, np.isfinite(pred).all())
    errors = {name: np.mean((pred.astype(float) - y[None])**2, axis=0)
              for name, pred in predictions.items()}
    boot = bootstrap_weights(table, np.array(['fixed_image_ensemble']), 1000, SEED + 100)
    result, r2, contrasts = paired_scores(y, errors, weights, boot, strict)
    result['strata'] = strata_results(table, y, errors, strict)
    result = rename_overlap(result)
    save_per_unit(out, 'normalized_overlap', units, rename_overlap(r2), rename_overlap(contrasts))
    np.savez_compressed(out / 'predictions.npz', **rename_overlap(predictions),
                        fold_assignments=np.stack(folds), primary_normalized_overlap=overlap)
    result.update({
        'passed': True, 'checkpoint_sha256': original['checkpoint_sha256'],
        'design_sha256': digest(out / 'design.json'), 'checks': checks,
        'source_code_sha256': code,
        'artifact_sha256': {p.name: digest(p) for p in out.iterdir() if p.suffix in ['.npz', '.csv']},
        'outcome_order': design['outcome_order'],
        'maximum_baseline_reproduction_error': max_baseline_error,
        'predictor_definition': design['predictor'],
        'uncertainty': '1,000 paired source-trial bootstraps within animal, conditional on the fixed image ensemble and fitted models.',
        'limitations': [
            'Normalized overlap removes overall estimated movie power, not estimator spectral blurring.',
            'The additive regression tests a scalar descriptor, not all nonlinear effects of retinal structure.',
            'A conditional gain does not rule out other eye-movement descriptors or establish neuron-specific tuning.',
            'Fixed image-ensemble results do not establish scene-by-trajectory generalization.'
        ]
    })
    write_json(out / 'summary.json', result)
    prior = json.loads((src / 'primary_summary.json').read_text())
    broad = json.loads((src / 'secondary_summary.json').read_text())
    raw = prior['contrasts']['class_path_engagement__over__class_path']
    raw_power = prior['contrasts']['class_path_dynamic_engagement__over__class_path_dynamic']
    shape = result['contrasts']['class_path_dynamic_normalized_overlap__over__class_path_dynamic']
    shape_only = result['contrasts']['class_path_normalized_overlap__over__class_path']
    movie = broad['contrasts']['class_path_engagement__over__class_path']

    def cell(record, index, strict=False):
        prefix = 'strict_' if strict else ''
        point = 100 * record[prefix + 'median_error_reduction'][index]
        low, high = np.array(record[prefix + 'error_reduction_ci95'][index]) * 100
        return f'{point:.1f}% [{low:.1f}, {high:.1f}]'

    findings = [
        '# Minimal Figure 4 passband audit', '',
        'Question: does the retinal-input descriptor add predictive information beyond movement class and distance, and does its spectral distribution contribute beyond total dynamic power?', '',
        'The audit reuses the selected Figure 4 movies and responses. The only new descriptor is per-movie normalized overlap q = passband power / total dynamic power, followed by the same median over 40 images. This summarizes spectral distribution after removing scalar power changes from each movie spectrum.', '',
        '| Added predictor | Baseline | Rate error reduction (95% CI) | SSI error reduction (95% CI) |',
        '|---|---|---:|---:|',
        f'| Raw engagement (existing) | Class + flexible path length | {cell(raw, 0)} | {cell(raw, 1)} |',
        f'| Normalized overlap | Class + flexible path length | {cell(shape_only, 0)} | {cell(shape_only, 1)} |',
        f'| Raw engagement (existing) | Class + path + total power | {cell(raw_power, 0)} | {cell(raw_power, 1)} |',
        f'| Normalized overlap: primary structure check | Class + path + total power | {cell(shape, 0)} | {cell(shape, 1)} |', '',
        'Values are median paired reductions in residual squared prediction error across the fixed 725 model units, not percentage points of total explained variance. Folds group source eye trials; each comparison uses the same held-out observations and 1,000 paired source-trial bootstrap resamples within animal. The 40-image ensemble and fitted models are fixed.', '',
        f'The corresponding primary structure contrast in the strict 145-unit subset is {cell(shape, 0, True)} for rate and {cell(shape, 1, True)} for SSI.', '',
        'Interpretation for the manuscript: engagement adds prediction beyond movement class and distance; the normalized spectral distribution adds modest predictive value for SSI beyond those descriptors and overall dynamic power. There is no resolved rate increment in the primary structure contrast. This is a conditional predictive result for the model and image ensemble.', '',
        'The result does not identify unique neuron-to-passband routing or rule out all other eye-movement descriptors. Normalization does not undo finite-window spectral blurring, and the additive regression does not test all nonlinear structure effects. The previous tuning-shuffle result remains inconclusive about tuning specificity.', '',
        f'The previous separate raw-engagement test on held-out source canvases and eye trials had reductions of {cell(movie, 0)} for rate and {cell(movie, 1)} for SSI; both intervals include zero. The new normalized-overlap audit does not establish generalization across scenes.', '',
        f'Validation: {len(checks)} checks passed, including selected checkpoint/source hashes, zero stabilized power, positive denominators, invariance to movie power scaling, identical source-trial folds, and independent reproduction of every baseline fold. Maximum baseline prediction discrepancy: {max_baseline_error:.3g} percentage points.', '',
        'Reproduction: `jake/passband_comparison/MINIMAL_AUDIT.md`. Full new results: `summary.json`; per-unit scores: `normalized_overlap_per_unit.csv`; predictions and folds: `predictions.npz`.', ''
    ]
    (out / 'FINDINGS.md').write_text('\n'.join(findings))
    print(json.dumps({'passed': True, 'checks': len(checks), 'contrasts': result['contrasts']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
