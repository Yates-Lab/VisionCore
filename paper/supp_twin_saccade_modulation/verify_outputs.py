"""Independent checks of saved multisession numerical artifacts."""
import csv
import json
import numpy as np

from paper.supp_twin_saccade_modulation.analysis import CACHE, STATS

root = STATS
selection = json.loads((root / 'selection.json').read_text())
with (root / 'population_rows.csv').open() as stream:
    rows = list(csv.DictReader(stream))
with np.load(root / 'population.npz') as archive:
    population = {key: archive[key] for key in archive.files}
assert len(selection['complete_sessions']) == 19
assert len(rows) == selection['population_units'] == 773
assert len(set(population['sessions'])) == 15
assert selection['population_sessions'] == sorted(set(population['sessions']))
keys = list(zip(population['sessions'], population['cids'].astype(int)))
assert keys == [(r['session'], int(r['cid'])) for r in rows]
assert len(set(keys)) == len(keys)
post = (population['lags_ms'] >= 0) & (population['lags_ms'] <= 200)
positive = np.max(population['gain_hz'][:, post], axis=1) > 1e-9
assert positive.sum() == 757 and not positive[757:].any()
peaks = population['lags_ms'][post][np.argmax(population['gain_hz'][:757, post], axis=1)]
assert np.all(np.diff(peaks) <= 0)
np.testing.assert_allclose(peaks, [float(r['positive_gain_peak_ms']) for r in rows[:757]])
index = {key: i for i, key in enumerate(keys)}
cohort = []
max_error = 0.0
for session in selection['complete_sessions']:
    folder = CACHE / 'sessions' / session
    report = json.loads((folder / 'results.json').read_text())
    cohort.extend(report['cohort'])
    max_error = max(max_error, *report['identity_errors'].values())
    assert max(report['identity_errors'].values()) < 2e-4
    with np.load(folder / 'components.npz') as archive:
        a = {key: archive[key] for key in ('robs', 'dfs', 'r0', 'rfull', 'r_gain', 'r_additive', 'event_windows', 'cids', 'trial_ids', 'times')}
    w = a['event_windows']
    assert np.all(a['trial_ids'][w] == a['trial_ids'][w[:, 0], None])
    np.testing.assert_allclose(np.diff(a['times'][w], axis=1), 1/240, atol=1e-6, rtol=0)
    for row in report['cohort']:
        unit = int(np.flatnonzero(a['cids'] == row['cid'])[0])
        valid = np.isfinite(a['dfs'][w, unit]) & (a['dfs'][w, unit] != 0)
        for key in ('robs', 'r0', 'rfull', 'r_gain', 'r_additive'):
            valid &= np.isfinite(a[key][w, unit])
        events = w[valid.all(axis=1)]
        assert len(events) == row['supported_events']
        key = (session, row['cid'])
        assert (key in index) == (len(events) >= 20)
        if key not in index:
            continue
        i = index[key]
        for source, target in [('r_gain', 'gain_hz'), ('r_additive', 'additive_hz')]:
            delta = (a[source][events, unit] - a['r0'][events, unit]) * 240
            mean = delta.mean(axis=0, dtype=np.float64)
            expected = np.convolve(np.pad(mean, (2, 2), mode='edge'), np.ones(5)/5, mode='valid')
            np.testing.assert_allclose(population[target][i], expected, atol=1e-5, rtol=1e-5)
    for ex in (x for x in selection['examples'] if x['session'] == session):
        assert ex['map_range_s'][-1] == .8 and ex['map_range_s'][0] > 0
        for trial in ex['trial_ids']:
            t = a['times'][a['trial_ids'] == trial]
            assert np.any(np.rint((t-t[0])*240).astype(int) == 192)
    del a
assert len(cohort) == 1022
assert sum(r['supported_events'] < 20 for r in cohort) == selection['excluded_units_for_event_support'] == 249
ranked = sorted((r for r in cohort if r['example_eligible'] and r['supported_events'] >= 20), key=lambda r: (-r['ccnorm'], r['session'], r['cid']))
assert [(r['session'], r['cid']) for r in ranked[:10]] == [(r['session'], r['cid']) for r in selection['examples']]
print(f'VERIFIED: 19 sessions, {len(cohort)} cohort units, 773 retained from 15 sessions; 249 event-support exclusions')
print('VERIFIED: every saved population row independently reconstructed; paired IDs, positive-peak order, event continuity, top-10 ranking and 0.8-s trial eligibility')
print(f'Max recorded forward/latent identity error: {max_error:.8g}')
