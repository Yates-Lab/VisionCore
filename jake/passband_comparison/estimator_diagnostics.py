"""Finite-window resolution and predictor similarity, without neural replay.

This diagnostic cannot identify the cause of a failed specificity control.
It measures resolution on known temporal carriers using the production
estimator and documents collinearity of the actual engagement predictors.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from paper.fig4.spatiotemporal_tuning.spectral_power import folded_dpss_mode_power
from jake.passband_comparison.data import write_json,digest,ROOT


def cosine(a,b): return float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',required=True)
    from pathlib import Path
    out=Path(p.parse_args().out_dir)
    tested=np.array([1.,2.,4.,8.,16.,32.,64.])
    report={};arrays={}
    for n in [60,240]:
        t=np.arange(n)/240.
        carriers=np.exp(2j*np.pi*tested[:,None]*t[None])
        hz,power=folded_dpss_mode_power(carriers,240.)
        power/=power.sum(axis=1,keepdims=True)
        report[str(n)]={'duration_seconds':n/240,'fourier_bin_spacing_hz':240/n,
            'dpss_nominal_half_bandwidth_hz':1.5/(n/240),
            'known_carrier_pairs':[{ 'frequencies_hz':tested[[j,j+1]],
                'estimated_spectrum_cosine':cosine(power[j],power[j+1])} for j in range(len(tested)-1)]}
        arrays[f'frequencies_{n}']=hz;arrays[f'normalized_power_{n}']=power
    z=np.load(out/'analysis_inputs.npz');ep=z['primary_engagement']
    er=rankdata(ep,axis=0);er-=er.mean(axis=0);er/=np.sqrt(np.sum(er*er,axis=0))
    correlation=er.T@er
    off=correlation[np.triu_indices(len(correlation),1)]
    total=rankdata(z['primary_dynamic']);total-=total.mean();total/=np.linalg.norm(total)
    vs_total=total@er
    units=pd.read_csv(out/'units.csv')
    report['actual_predictors']={'median_pairwise_rank_correlation':float(np.median(off)),
        'pairwise_rank_correlation_10_90':np.quantile(off,[.1,.9]),
        'median_engagement_vs_dynamic_power_rank_correlation':float(np.median(vs_total)),
        'n_units_with_preferred_tf_below_first_bin':int((units.exact_twin_yu_preferred_tf_hz<4).sum()),
        'n_strict_units_with_preferred_tf_below_first_bin':int(((units.exact_twin_yu_preferred_tf_hz<4)&units.validated_for_figure4).sum())}
    report['interpretation']='Noise-free calibration demonstrates spectral ambiguity induced by finite duration and the selected tapers; it does not establish that this caused the observed shuffle result. Actual engagement predictors are also compared for collinearity. A negative shuffle comparison is conditional on this estimator and cannot rule out true tuning-specific effects.'
    report['sources']={str(p.relative_to(ROOT)):digest(p) for p in [
        ROOT/'paper/fig4/spatiotemporal_tuning/spectral_power.py',out.resolve()/'analysis_inputs.npz',out.resolve()/'units.csv']}
    np.savez_compressed(out/'estimator_resolution_calibration.npz',tested_hz=tested,**arrays)
    write_json(out/'estimator_diagnostics.json',report)
    print(report,flush=True)


if __name__=='__main__':main()
