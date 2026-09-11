#!/usr/bin/env python3
"""Export the exact statistics plotted by the current Figure 2 code."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'paper/fig2'), str(ROOT / 'paper/covariance_decomposition'),
                str(ROOT / 'paper/fig1')]
import derive
SOURCE = ROOT / 'outputs/dekel240_paper/m77_epoch279/production_figure3/cache/covdecomp_derived.pkl'
EMPIRICAL_SHA256 = 'b9c058c9c3d99bc18826c1b4177c1f680af3eae6a4c815c70f546dea7b3a5fd2'
derive.DERIVED_CACHE = SOURCE
from generate_figure2 import load_prepared_data, compute_alignment_aggregate
from recording_unit_counts import _session_counts, _summarize
from scipy.stats import binomtest


def main():
    source_digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_digest != EMPIRICAL_SHA256:
        raise ValueError('The empirical cache differs from the selected Figure 3 input')
    d = load_prepared_data()
    a, f, nc = (d[k][25.0] for k in ('alpha_stats', 'fano_stats', 'nc_stats'))
    alignment = compute_alignment_aggregate(d)
    out = {'source': str(SOURCE), 'sha256': source_digest,
           'window_ms': 25, 'n_sessions': d['n_sessions'],
           'alpha': a, 'fano': {}, 'noise_correlation': {}, 'participation_ratio': {},
           'alignment': {}, 'unit_counts': {}}
    for key in ('unc', 'cor'):
        v = np.asarray(f['sess_slope_' + key])
        out['fano'][key] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1))}
    out['fano']['p_emp'] = f['p_emp_mean_sess']
    out['fano']['null_ci'] = f['mean_sess_cor_null_ci']
    for key in ('r_u_mean','r_u_ci','r_c_mean','r_c_ci','dr_mean','dr_ci','null_dr_ci','p_emp_dr'):
        out['noise_correlation'][key] = nc[key]
    for key in ('fem','psth','resid'):
        v = np.asarray(d['pr_' + key + '_list'],float)
        out['participation_ratio'][key] = {'mean':float(v.mean()), 'sd':float(v.std(ddof=1))}
    delta=np.asarray(d['pr_psth_list'])-np.asarray(d['pr_fem_list'])
    out['participation_ratio']['sign_p']=float(binomtest(int((delta>0).sum()),int((delta!=0).sum())).pvalue)
    for key in ('x','y'):
        v=alignment[key]
        out['alignment'][key]={k:v[k] for k in ('mean','sd','p')}
        out['alignment'][key]['null_ci']=np.percentile(v['null_mean'],[2.5,97.5]).tolist()
    counts=_session_counts(d['session_results'])
    for subject in ('Allen','Logan','all'):
        out['unit_counts'][subject]=_summarize([r['analyzed'] for r in counts if subject=='all' or r['subject']==subject])
    out['alpha_null_note']='Existing alpha p-values compare the pooled observed median with session-level shuffle medians; not reported as a pooled-population hypothesis test in the manuscript.'
    def default(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    target=ROOT/'manuscript/analysis/empirical_stats.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,indent=2,default=default)+'\n')
    print(target)

if __name__=='__main__':main()
