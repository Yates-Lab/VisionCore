"""Check identities, count matching, and decoder bounds for the prototype."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jake.eye_movement_routing.analyze import sha256
from jake.eye_movement_routing.decode import check_decoder


def main():
    out = ROOT/'outputs/eye_movement_routing_20260914'
    design = np.load(out/'routing_analysis.npz')
    selected = json.loads((ROOT/'manuscript/analysis/selected_model_bundle.json').read_text())
    groups = design['groups'].reshape(-1)
    assert len(np.unique(groups)) == len(groups) == 96
    assert not set(design['selection_images']) & set(design['evaluation_images'])
    assert set(design['decode_images']).issubset(design['evaluation_images'])
    assert np.array_equal(np.sort(design['bins'].reshape(-1)),np.arange(200))
    errors = []
    maps = []
    for image in design['decode_images']:
        for trace in np.r_[-1,design['decode_traces']]:
            path = out/'rate_maps'/f'image_{image:03d}_trace_{trace:03d}.npz'
            z = np.load(path)
            assert np.array_equal(z['unit_indices'],groups)
            assert str(z['checkpoint_sha256']) == selected['checkpoint_sha256']
            assert int(z['image_row']) == image and int(z['trace_row']) == trace
            assert z['integrated_counts'].shape == (96,30,30)
            assert z['temporal_counts'].shape == (60,96,7,7)
            assert np.allclose(z['temporal_counts'].sum(axis=0),z['integrated_counts'][:,12:19,12:19],rtol=2e-6,atol=1e-6)
            assert np.isfinite(z['integrated_counts']).all()
            assert (z['integrated_counts']>=0).all()
            errors.append(float(z['max_rate_error_hz']))
            maps.append(path)
    table = pd.read_csv(out/'decoding_trials_summary.csv')
    assert len(table)==len(maps)*4
    assert not table[['image_row','trace_row','group','mode']].duplicated().any()
    assert np.isfinite(table.select_dtypes('number')).all().all()
    assert table.accuracy.between(0,1).all()
    assert (table.information_bits<=np.log2(49)+1e-10).all()
    assert np.allclose(table.loc[table['mode'].eq('matched_10_spikes'),'expected_total_spikes'],10)
    files = [out/'routing_analysis.npz',out/'summary.json',out/'decoding_summary.json',
             out/'count_only_decoding_summary.json',
             out/'decoding_trials_summary.csv',*sorted(Path(__file__).parent.glob('*.py'))]
    audit = {'passed':True,'checkpoint_sha256':selected['checkpoint_sha256'],
        'movie_replays':len(maps),'decoder_rows':len(table),
        'maximum_replay_vs_cache_rate_error_hz':max(errors),
        'decoder_checks':check_decoder(),
        'hashes':{str(p.relative_to(ROOT)):sha256(p) for p in files},
        'interpretation':'Numerical/identity audit; not a validation of biological decoding or attention.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='hashes'},indent=2))


if __name__=='__main__':
    main()
