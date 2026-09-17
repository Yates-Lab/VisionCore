"""Measure scene × FEM × neuron tuning interactions and their gain controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from jake.fem_band_tuning.common import (
    SEED, STRENGTHS, sweep_indices, gain_statistics, three_way_residual,
    sha256, write_json)


def condition_weights(table, rng=None, allowed=None):
    """Equal animal weights within event class; resample source trials.

    Histories from the same trial share a bootstrap multiplicity, including
    when that trial contributes both event classes.
    """
    animal = table.session.str.split('_').str[0].to_numpy()
    event_array=table.event_class.to_numpy()
    cluster=(table.session.astype(str)+'|'+table.trial_idx.astype(str)).to_numpy() if 'trial_idx' in table else np.arange(len(table)).astype(str)
    multiplicity=np.ones(len(table))
    if rng is not None:
        for subject in np.unique(animal):
            member=np.flatnonzero(animal==subject)
            unique=np.unique(cluster[member])
            for _ in range(100):
                chosen=rng.choice(unique,len(unique),replace=True)
                multiplicity[member]=[(chosen==key).sum() for key in cluster[member]]
                if all(multiplicity[member[event_array[member]==event]].sum()>0
                       for event in ['drift','microsaccade']):
                    break
            else:
                raise ValueError('Bootstrap could not retain both event classes')
    weights = []
    for event in ['drift', 'microsaccade']:
        w = np.zeros(len(table))
        for subject in np.unique(animal):
            idx = np.flatnonzero((event_array == event) & (animal == subject))
            if allowed is not None:
                idx = np.intersect1d(idx, allowed)
            if not len(idx):
                raise ValueError('Empty animal × event stratum')
            if rng is not None:
                w[idx]=multiplicity[idx]/multiplicity[idx].sum()/len(np.unique(animal))
            else:
                w[idx] = 1/len(idx)/len(np.unique(animal))
        weights.append(w)
    return np.stack(weights)


def bootstrap_ratio(metric, table, clusters, *, repeats=1000):
    """Crossed scene-cluster/trace-source-trial bootstrap of condition ratio.

    metric[scene,eye,band] is already reduced across the fixed 725 neurons.
    Clusters are full-screen source hashes; patches are not treated as fully
    independent when their complete source canvas is identical.
    """
    rng = np.random.default_rng(SEED+101)
    unique = np.unique(clusters)
    means = np.einsum('ce,sek->sck', condition_weights(table), metric)
    mean = means.mean(axis=0)
    ratios = []
    for _ in range(repeats):
        chosen = rng.choice(unique, len(unique), replace=True)
        scene_w = np.array([(chosen == c).sum() for c in clusters], dtype=float)
        scene_w /= scene_w.sum()
        w = condition_weights(table, rng)
        value = np.einsum('s,ce,sek->ck', scene_w, w, metric, optimize=True)
        ratios.append(np.log2(np.maximum(value[1],1e-30)/np.maximum(value[0],1e-30)))
    return {'drift': mean[0].tolist(), 'microsaccade': mean[1].tolist(),
        'log2_ratio': np.log2(np.maximum(mean[1],1e-30)/np.maximum(mean[0],1e-30)).tolist(),
        'log2_ratio_ci95': np.percentile(ratios,[2.5,97.5],axis=0).T.tolist(),
        'bootstrap_repeats': repeats, 'scene_clusters': len(unique)}


def split_reliability(gain, table, *, repetitions=30):
    """Disjoint source trials test reproducibility of scene × condition × neuron effects."""
    rng = np.random.default_rng(SEED+55)
    animal=table.session.str.split('_').str[0].to_numpy()
    cluster=(table.session.astype(str)+'|'+table.trial_idx.astype(str)).to_numpy()
    cosine, cross_energy = [], []
    for _ in range(repetitions):
        halves = [[],[]]
        for subject in np.unique(animal):
            member=np.flatnonzero(animal==subject)
            order=rng.permutation(np.unique(cluster[member]))
            halves[0].extend(member[np.isin(cluster[member],order[::2])])
            halves[1].extend(member[np.isin(cluster[member],order[1::2])])
        effects = []
        for allowed in halves:
            w = condition_weights(table, allowed=allowed)
            d = np.einsum('e,sekn->skn', w[1]-w[0], gain, optimize=True)
            # Remove effects shared across scenes and effects shared across
            # neurons, leaving the scene × condition × neuron interaction.
            d = d - d.mean(axis=0, keepdims=True)
            d = d - d.mean(axis=2, keepdims=True)
            effects.append(d)
        a,b = effects
        cross = np.sum(a*b,axis=(0,2))
        norm = np.sqrt(np.sum(a*a,axis=(0,2))*np.sum(b*b,axis=(0,2)))
        cosine.append(cross/np.maximum(norm,1e-30))
        cross_energy.append(cross / np.prod([gain.shape[0],gain.shape[-1]]))
    return {'cosine_median': np.median(cosine,axis=0).tolist(),
        'cosine_range_10_90': np.percentile(cosine,[10,90],axis=0).T.tolist(),
        'cross_energy_median_hz2': np.median(cross_energy,axis=0).tolist(),
        'repeated_trace_splits': repetitions,
        'meaning': 'Positive agreement across disjoint source-trial halves supports a reproducible scene-specific redistribution across neurons; intervals describe splits, not confidence intervals.'}


def scalar_gain_residual(gain, reference):
    """Energy left after best global or per-neuron scalar gain of static tuning."""
    ref = reference[:,None,:,:]
    global_scale = np.maximum(0, np.sum(gain*ref,axis=(-2,-1))/np.maximum(np.sum(ref*ref,axis=(-2,-1)),1e-30))
    global_resid = gain-global_scale[...,None,None]*ref
    unit_scale = np.maximum(0, np.sum(gain*ref,axis=-2)/np.maximum(np.sum(ref*ref,axis=-2),1e-30))
    unit_resid = gain-unit_scale[...,None,:]*ref
    energy = np.sum(gain*gain,axis=(-2,-1))
    return global_resid, unit_resid, global_scale, {
        'global_scalar_unexplained_energy_fraction': float(np.sum(global_resid**2)/np.sum(energy)),
        'per_neuron_scalar_unexplained_energy_fraction': float(np.sum(unit_resid**2)/np.sum(energy)),
        'reference': 'stationary, same scene; optimal nonnegative scaling of band-gain vector',
        'global_scale_median': float(np.median(global_scale)),
        'per_neuron_scale_median': float(np.median(unit_scale))}


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=1000)
    args = p.parse_args()
    out = args.out_dir
    design = json.loads((out/'design.json').read_text())
    digest = sha256(out/'design.json')
    sources = sorted((out/'responses').glob('scene_*.npz'))
    if len(sources) != design['n_scenes']:
        raise ValueError(f'Incomplete replay: {len(sources)}/{design["n_scenes"]} scenes')
    rates = []
    for j,path in enumerate(sources):
        with np.load(path) as z:
            if int(z['scene']) != j or str(z['design_sha256']) != digest:
                raise ValueError('Scene/design identity mismatch')
            if not np.array_equal(z['trace_rows'], np.arange(-1,design['n_traces'])):
                raise ValueError('Missing or reordered eye histories')
            rates.append(z['rates_hz'])
    rates = np.stack(rates)  # scene, stationary+eye, intervention, neuron
    if not np.isfinite(rates).all() or rates.min() < 0:
        raise ValueError('Invalid model rates')
    table = pd.read_csv(out/'traces.csv')
    images = pd.read_csv(out/'images.csv')
    traces = np.load(out/'traces_endpoint_aligned.npy')
    speed = np.linalg.norm(np.diff(traces,axis=1),axis=-1)*240
    table['peak_speed_deg_s'] = speed.max(axis=1)
    table['peak_speed_ms_before_output'] = (58-np.argmax(speed,axis=1)+0.5)/240*1000
    # Join the independently audited event amplitude, not path length mislabeled as amplitude.
    prov = json.loads(Path(design['source_files']['trace_provenance']['path']).read_text())
    original_table = pd.read_csv(Path(prov['fixation_bank'])/'trace_table.csv')
    selected = original_table.iloc[table.source_row.to_numpy(dtype=int)]
    if not np.array_equal(selected.source_trace_index.to_numpy(),table.source_trace_index.to_numpy()):
        raise ValueError('Event metadata source identity mismatch')
    table['verified_microsaccade_max_amplitude_deg'] = selected.verified_microsaccade_max_amplitude_deg.to_numpy()
    table.to_csv(out/'trace_analysis_metadata.csv',index=False)
    indices = sweep_indices(design['interventions'])
    sweeps = rates[:,:,indices,:]
    gain, wide, curvature = gain_statistics(sweeps)
    static = gain[:,0]
    gain = gain[:,1:]
    baseline = rates[:,1:,0]
    w = condition_weights(table)
    gain_condition = np.einsum('ce,sekn->sckn',w,gain,optimize=True)
    gain_energy = np.mean(gain.astype(np.float64)**2,axis=-1)
    # Independent-Poisson local information for one native 1/240-second output bin.
    poisson = gain.astype(np.float64)**2/(np.maximum(baseline[:,:,None,:],1e-12)*240)
    poisson_population = poisson.sum(axis=-1)
    # Exact distance between the two finite-intervention distributions, without
    # linearizing the model or estimating a derivative. For independent Poisson
    # counts, -log Bhattacharyya affinity = 1/2 sum (sqrt(mu+) - sqrt(mu-))^2.
    pair_distance = .5*np.sum((np.sqrt(sweeps[:,1:,:,3,:].astype(float)/240)
        - np.sqrt(sweeps[:,1:,:,1,:].astype(float)/240))**2,axis=-1)
    # Expected spike budget control: retain only coefficient-dependent changes
    # in the allocation of spikes across neurons. Total count itself carries
    # no evidence under this conditional multinomial-like rate comparison.
    count_sweeps = sweeps[:,1:].astype(np.float64)
    count_sweeps /= np.maximum(count_sweeps.sum(axis=-1,keepdims=True),1e-30)
    conditional_gain,_,_ = gain_statistics(count_sweeps)
    allocation_info = (conditional_gain**2/np.maximum(count_sweeps[:,:,:,2,:],1e-30)).sum(axis=-1)
    global_resid,unit_resid,global_scale,gain_controls = scalar_gain_residual(gain,static)
    # Exact balanced decomposition, treating neurons as the fixed analyzed population.
    residual = three_way_residual(gain.transpose(0,1,3,2))
    centered = gain.transpose(0,1,3,2).astype(np.float64)
    centered -= centered.mean(axis=(0,1,2),keepdims=True)
    triple_fraction = (residual**2).sum(axis=(0,1,2))/np.maximum((centered**2).sum(axis=(0,1,2)),1e-30)
    joint_idx = [i for i,r in enumerate(design['interventions']) if r['kind']=='joint_validation']
    offsets = np.load(out/'coefficient_offsets.npy')
    observed = rates[:,1:,joint_idx,:]-baseline[:,:,None,:]
    predicted = np.einsum('pk,sekn->sepn',offsets[joint_idx],gain,optimize=True)
    joint_error = observed-predicted
    joint_explained = 1-float(np.sum(joint_error.astype(float)**2)/np.maximum(np.sum(observed.astype(float)**2),1e-30))
    outer_observed=sweeps[:,1:,:,[0,4],:] - baseline[:,:,None,None,:]
    outer_predicted=gain[:,:,:,None,:]*np.array([-.5,.5])[None,None,None,:,None]
    outer_error=outer_observed-outer_predicted
    outer_explained=1-float(np.sum(outer_error.astype(float)**2)/np.maximum(np.sum(outer_observed.astype(float)**2),1e-30))
    clusters = np.array([row['source_canvas_sha256'] for row in design['scenes']])
    geometry=json.loads((out/'input_geometry_audit.json').read_text())
    interior=np.setdiff1d(np.arange(len(images)),geometry['scenes_whose_swept_field_may_cross_screen_edge'])
    _,_,_,interior_gain_control=scalar_gain_residual(gain[interior],static[interior])
    stats = {
        'analysis': design['analysis'], 'design_sha256': digest,
        'n_scenes': len(images), 'n_source_canvas_clusters': len(set(clusters)),
        'n_traces': len(table), 'n_units': design['n_units'],
        'n_eye_source_trials':len(table[['session','trial_idx']].drop_duplicates()),
        'event_counts': table.event_class.value_counts().to_dict(),
        'event_by_animal': table.groupby([table.session.str.split('_').str[0], 'event_class']).size().to_string(),
        'estimand': 'One endpoint rate vector after 60 native frames; d rate / d coefficient at a=1 from a=0.75 and 1.25',
        'band_order': 'finest to coarsest',
        'population_gain_energy': bootstrap_ratio(gain_energy,table,clusters,repeats=args.bootstrap),
        'poisson_information_per_endpoint_bin': bootstrap_ratio(poisson_population,table,clusters,repeats=args.bootstrap),
        'exact_poisson_pair_distance': bootstrap_ratio(pair_distance,table,clusters,repeats=args.bootstrap),
        'information_per_fixed_expected_spike': bootstrap_ratio(allocation_info,table,clusters,repeats=args.bootstrap),
        'global_and_neuron_gain_controls': gain_controls,
        'interior_scene_control': {'scene_rows':interior.tolist(),
            'excluded_screen_edge_scene_rows':geometry['scenes_whose_swept_field_may_cross_screen_edge'],
            'gain_energy':bootstrap_ratio(gain_energy[interior],table,clusters[interior],repeats=args.bootstrap),
            'gain_controls':interior_gain_control},
        'three_way_interaction': {'raw_gain_variance_fraction_by_band': triple_fraction.tolist(),
            'trace_split_reliability': split_reliability(gain,table),
            'after_per_neuron_scalar_gain_removal': split_reliability(unit_resid,table)},
        'local_curve_checks': {
            'relative_inner_outer_slope_rmse': float(np.sqrt(np.sum((gain-wide[:,1:])**2)/np.sum(gain**2))),
            'joint_validation_change_energy_explained_by_local_additive_gains': joint_explained,
            'outer_single_band_change_energy_explained': outer_explained,
            'outer_single_band_prediction_rmse_hz': float(np.sqrt(np.mean(outer_error**2))),
            'joint_validation_delta_rate_rmse_hz': float(np.sqrt(np.mean(joint_error**2))),
            'joint_validation_observed_delta_rate_rms_hz': float(np.sqrt(np.mean(observed**2)))},
        'interpretation': [
            'Rate sensitivity to the strength of existing spatial structures, conditional on scene and FEM history.',
            'Pyramid bands overlap; this is not a reconstruction of arbitrary image phase or unseen features.',
            'Poisson information and fixed-spike analyses are model assumptions, not measured joint neural noise.',
            'Local information uses the finite-difference slope at coefficients 0.75/1.25; exact Poisson pair distances compare those finite edits without a local linear approximation.',
            'An attention-like change in represented content is a functional analogy; no attentional state was manipulated.',
            'Effects of event class also include differences in amplitude, direction, speed, and event timing; they are not an isolated causal effect of the class label.',
            'Using the nonlinear network does not by itself establish that a measured interaction requires nonlinear computation.'],
        'response_files': {str(path.relative_to(out)):sha256(path) for path in sources}}
    write_json(out/'summary.json',stats)
    # Store the precise response-derived quantities needed to remake every plot.
    np.savez_compressed(out/'analysis_arrays.npz', gain_hz=gain, static_gain_hz=static,
        gain_condition_hz=gain_condition, baseline_hz=baseline, sweeps_hz=sweeps,
        gain_energy=gain_energy, poisson_information=poisson_population,
        poisson_pair_distance=pair_distance,
        allocation_information=allocation_info, curvature_hz=curvature[:,1:],
        unit_scalar_residual_hz=unit_resid.astype(np.float32),
        global_scale=global_scale, condition_weights=w)
    rows=[]
    for name, metric in [('gain_energy_hz2',gain_energy), ('poisson_information',poisson_population),
                         ('allocation_information',allocation_info)]:
        for k in range(metric.shape[-1]):
            for s in range(metric.shape[0]):
                for e in range(metric.shape[1]):
                    rows.append({'scene':s,'trace':e,'band':k,'metric':name,'value':metric[s,e,k]})
    pd.DataFrame(rows).to_csv(out/'scene_trace_band_metrics.csv',index=False)
    units = pd.read_csv(design['tuning_table']).sort_values('unit_index').reset_index(drop=True)
    if len(units)!=gain.shape[-1]:
        raise ValueError('Tuning table must retain all exact units')
    if not np.array_equal(units.unit_index,np.arange(gain.shape[-1])):
        raise ValueError('Unit tuning indices are not the exact population order')
    unit_rows=[]
    for k in range(gain.shape[-2]):
        for n in range(gain.shape[-1]):
            energy = np.einsum('ce,se->sc',w,gain[:,:,k,n].astype(float)**2).mean(axis=0)
            unit_rows.append({'unit':n,'band':k,'drift_gain_rms_hz':np.sqrt(energy[0]),
                'microsaccade_gain_rms_hz':np.sqrt(energy[1]),
                'log2_gain_energy_ratio':np.log2(max(energy[1],1e-30)/max(energy[0],1e-30))})
    unit_result = pd.DataFrame(unit_rows).merge(units,left_on='unit',right_on='unit_index',validate='many_to_one')
    unit_result.to_csv(out/'neuron_band_tuning.csv',index=False)
    # Link contextual band sensitivity to independently assayed neural tuning.
    tuning_associations=[]
    for k in range(gain.shape[-2]):
        frame=unit_result.loc[unit_result.band.eq(k)]
        for population,keep in [('all_exact_units',np.ones(len(frame),bool)),
            ('strict_validation_subset',frame.validated_for_figure4.to_numpy(dtype=bool))]:
            for feature in ['exact_twin_yu_preferred_sf_cpd','exact_twin_yu_preferred_tf_hz']:
                x=frame[feature].to_numpy(float)
                y=frame.log2_gain_energy_ratio.to_numpy(float)
                good=keep & np.isfinite(x) & (x>0) & np.isfinite(y)
                rho=spearmanr(np.log2(x[good]),y[good]).statistic
                tuning_associations.append({'band':k,'population':population,'n_units':int(good.sum()),
                    'feature':feature,'spearman_with_movement_gain_modulation':finite_or_none(rho)})
    write_json(out/'neural_tuning_associations.json',tuning_associations)
    sf=units.exact_twin_yu_preferred_sf_cpd.to_numpy(float)
    order=np.argsort(sf); size=len(order)//3
    groups={'lower_SF_third':order[:size], 'higher_SF_third':order[-size:]}
    grouped={}
    for name,idx in groups.items():
        grouped[name]={'unit_indices':idx.tolist(),'n_units':len(idx),
            'median_model_sf_cpd':float(np.median(sf[idx])),
            'gain_energy':bootstrap_ratio(np.mean(gain[:,:,:,idx].astype(float)**2,axis=-1),
                table,clusters,repeats=args.bootstrap)}
    write_json(out/'population_groups.json',grouped)
    # Descriptive associations with independently measured tuning and movement magnitude.
    assoc=[]
    for k in range(gain.shape[-2]):
        metric=gain_energy[:,:,k].mean(axis=0)
        for event in ['drift','microsaccade']:
            keep=table.event_class.eq(event).to_numpy()
            rho,pvalue=spearmanr(metric[keep],table.rendered_path_length_arcmin.to_numpy()[keep])
            assoc.append({'band':k,'event':event,'n_traces':int(keep.sum()),
                'path_length_spearman':finite_or_none(rho),'descriptive_p':finite_or_none(pvalue)})
    write_json(out/'descriptive_associations.json',assoc)
    print(json.dumps({'gain_log2_ratios':stats['population_gain_energy']['log2_ratio'],
        'three_way':stats['three_way_interaction'], 'gain_controls':gain_controls},indent=2),flush=True)


if __name__ == '__main__':
    main()
