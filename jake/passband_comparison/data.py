"""Read and verify selected Figure 4 caches without importing the twin."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[2]
SEED=20260914


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()


def write_json(path,value):
    def convert(x):
        if isinstance(x,Path): return str(x)
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=convert)+'\n')


def load_data(out):
    sources={}; checks={}
    def bind(path):
        p=Path(path); sources[str(p.resolve().relative_to(ROOT))]=digest(p); return p
    def read(path): return json.loads(bind(path).read_text())
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition: raise ValueError('Input audit failed: '+name)
    selection=read(ROOT/'manuscript/analysis/selected_model_bundle.json')
    base=ROOT/selection['bundle']/'figure4'
    release=read(base/'production_figure4/audit/production_audit.json')
    check('selected_figure4_released',release['release_ready'] and release['checkpoint_sha256']==selection['checkpoint_sha256'])
    src=read(base/'top_passband_stage_trajectory_10img_x_10fix/summary.json')['inputs']
    traces=pd.read_csv(bind(src['trace_table'])).sort_values('trace_bank_index').reset_index(drop=True)
    images=pd.read_csv(bind(src['image_table'])).sort_values('image_index').reset_index(drop=True)
    provenance=read(src['trace_provenance'])
    tuning_contract=read(base/'all_available_yu_tuning/summary.json')
    tuning=pd.read_csv(bind(tuning_contract['files']['tuning_summary'])).sort_values('unit_index').reset_index(drop=True)
    check('all_725_exact_readouts',len(tuning)==725 and tuning.unit_index.is_unique)
    check('strict_tuning_subset',int(tuning.validated_for_figure4.sum())==145)
    keys=['mean_rate','expected_spikes','map_ssi','joint_passband_power']
    data={k:[] for k in keys}; dynamic=[]; axis=None; scene_ids=[]
    for path in src['spectral_shards']:
        path=bind(path); report=read(path.parent/'summary.json')
        check('checkpoint_'+path.parent.name,report['checkpoint_sha256']==selection['checkpoint_sha256'])
        check('filter_'+path.parent.name,report['trace_filter']['family']=='gaussian' and report['trace_filter']['sigma_seconds']==.006)
        check('tuning_source_'+path.parent.name,digest(bind(report['source_tuning_table']))==report['source_tuning_table_sha256'])
        check('native_window_'+path.parent.name,report['frame_rate_hz']==240 and report['n_timepoints']==60)
        with np.load(path,allow_pickle=False) as z:
            current={k:z[k] for k in ['unit_indices','trace_indices','motion_scales','spatial_cpd','temporal_hz','orientation_deg']}
            if axis is not None:
                check('fixed_axes_'+path.parent.name,all(np.array_equal(axis[k],current[k]) for k in axis))
            axis=current; scene_ids.extend(z['image_indices'].tolist())
            for k in keys:
                v=z[k]; check('finite_'+path.parent.name+'_'+k,np.isfinite(v).all() and (v>=0).all())
                data[k].append(v)
            v=z['total_dynamic_power']
            check('dynamic_power_is_unit_independent_'+path.parent.name,np.array_equal(v,np.broadcast_to(v[...,:1],v.shape)))
            dynamic.append(v[...,0])
    order=np.argsort(scene_ids)
    check('complete_scene_axes',np.array_equal(np.sort(scene_ids),images.image_index.to_numpy()))
    check('complete_trace_axes',np.array_equal(axis['trace_indices'],traces.trace_bank_index.to_numpy()))
    check('complete_unit_axes',np.array_equal(axis['unit_indices'],tuning.unit_index.to_numpy()))
    check('motion_and_stabilized_axes',np.array_equal(axis['motion_scales'],[0.,1.]))
    check('event_counts',traces.event_class.value_counts().to_dict()=={'drift':146,'microsaccade':54})
    for k in keys: data[k]=np.concatenate(data[k],axis=0)[order].astype(np.float64)
    dynamic=np.concatenate(dynamic,axis=0)[order].astype(np.float64)
    rate=data.pop('mean_rate'); spikes=data.pop('expected_spikes'); ssi=data.pop('map_ssi'); power=data.pop('joint_passband_power')
    for k,v in [('rate',rate),('ssi',ssi)]:
        check('stabilized_'+k+'_broadcast',np.allclose(v[:,:,0],v[:,0:1,0],rtol=1e-6,atol=1e-6))
    check('response_rates_match_counts',np.allclose(spikes,.25*rate,rtol=2e-5,atol=1e-5))
    check('positive_stabilized_denominators',(rate[:,:,0]>0).all() and (ssi[:,:,0]>0).all())
    primary_rate=100*(rate[:,:,1].mean(axis=0)/rate[:,:,0].mean(axis=0)-1)
    pooled_ssi=(spikes*ssi).sum(axis=0)/spikes.sum(axis=0)
    primary_ssi=100*(pooled_ssi[:,1]/pooled_ssi[:,0]-1)
    primary_y=np.stack([primary_rate,primary_ssi],axis=-1)
    movie_y=np.stack([100*(rate[:,:,1]/rate[:,:,0]-1),100*(ssi[:,:,1]/ssi[:,:,0]-1)],axis=-1)
    engagement=power[:,:,1]-power[:,:,0]; dynamic=dynamic[:,:,1]-dynamic[:,:,0]
    check('finite_outcomes',np.isfinite(primary_y).all() and np.isfinite(movie_y).all())
    # Existing scene audit supplies only full-canvas identity, never FEM intervention responses.
    canvas_source=read(ROOT/'outputs/fem_band_tuning_20260914/design.json')
    check('canvas_metadata_same_images',canvas_source['source_files']['image_table']['sha256']==digest(src['image_table']))
    check('canvas_metadata_scene_order',[s['scene'] for s in canvas_source['scenes']]==images.image_index.tolist())
    canvas=np.array([s['source_canvas_sha256'] for s in canvas_source['scenes']])
    images['source_canvas_sha256']=canvas
    check('distinct_source_canvases',len(np.unique(canvas))==13)
    groups=(traces.session+'|'+traces.trial_idx.astype(str)).to_numpy()
    check('source_eye_trial_clusters',len(np.unique(groups))==187)
    # Reproduce the released marginal correlations independently.
    existing=read(base/'passband_vs_path_length/summary.json')
    ep=np.median(engagement,axis=0)
    er=rankdata(ep,axis=0); pr=rankdata(traces.rendered_path_length_arcmin.to_numpy())
    corr=lambda a,b:np.sum((a-a.mean(axis=0))*(b-b.mean(axis=0)),axis=0)/np.sqrt(np.sum((a-a.mean(axis=0))**2,axis=0)*np.sum((b-b.mean(axis=0))**2,axis=0))
    reproduction={}
    for o,name in enumerate(['rate_percent','ssi_percent']):
        yr=rankdata(primary_y[...,o],axis=0)
        value=[float(np.median(corr(er,yr))),float(np.median(corr(pr[:,None],yr)))]
        expected=existing['outcomes'][name]
        check('released_correlations_'+name,np.allclose(value,[expected['median_within_unit_passband_spearman'],expected['median_within_unit_path_length_spearman']],atol=1e-10,rtol=0))
        reproduction[name]=value
    write_json(out/'input_audit.json',{'passed':True,'checks':checks,'released_correlation_reproduction':reproduction})
    traces.to_csv(out/'traces.csv',index=False); images.to_csv(out/'images.csv',index=False); tuning.to_csv(out/'units.csv',index=False)
    np.savez_compressed(out/'analysis_inputs.npz',primary_y=primary_y,movie_y=movie_y,
        primary_engagement=ep,movie_engagement=engagement,
        primary_dynamic=np.median(dynamic,axis=0),movie_dynamic=dynamic)
    design={'checkpoint_sha256':selection['checkpoint_sha256'],'source_sha256':sources,
        'n_scenes':40,'n_traces':200,'n_units':725,'n_canvas_clusters':13,'n_eye_trial_clusters':187,
        'strict_n_units':145,'outcome_order':['rate_percent','ssi_percent'],
        'primary_repeats':3,'folds':5,'ridge_penalty':.01,'ridge_sensitivity':[.001,.1],
        'shuffle_assignments':32,'bootstrap_repeats':1000,'seed':SEED,
        'readme_sha256':digest(Path(__file__).with_name('README.md')),
        'primary_estimand':'Released image-reduced modulation; held-out eye trials; intervals conditional on fixed 40-image ensemble.',
        'secondary_estimand':'Per-movie modulation; both source canvas and source eye trial held out; crossed stimulus/history intervals.',
        'spectrum':'Exact Figure 4 rendered movies; 6-ms Gaussian eye filter, 250-ms window, temporal demeaning and DPSS NW=1.5 K=2; no resimulation.',
        'population':'Fixed all-725 model population, with strict-145 sensitivity subset; no unit bootstrap.',
        'class_path_baseline':'animal + event class + five spline terms for path + five event-by-path terms',
        'inference_scope':'Conditional on fitted twin; no measured neural noise or animal-population inference.'}
    write_json(out/'design.json',design)
    return design,traces,images,tuning,primary_y,movie_y,ep,engagement,np.median(dynamic,axis=0),dynamic
