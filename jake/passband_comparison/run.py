"""Run the predeclared stronger passband comparison on existing Figure 4 data."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from jake.passband_comparison.data import ROOT,SEED,digest,write_json,load_data
from jake.passband_comparison.statistics import (
    trial_folds,make_features,fit_predict,animal_weights,bootstrap_weights,
    paired_scores,partial_rank_correlations)


def shuffled_assignments(n,repeats):
    rng=np.random.default_rng(SEED+200)
    result=[]
    for _ in range(repeats):
        for _ in range(10000):
            p=rng.permutation(n)
            if np.all(p!=np.arange(n)): break
        else: raise RuntimeError('Failed to sample derangement')
        result.append(p)
    return np.stack(result)


def primary_predictions(out,table,y,engagement,dynamic,penalty=.01,shuffles=32,repeats=3):
    n,u,o=y.shape
    animal=(table.session.str.startswith('Logan')).to_numpy(dtype=float)
    event=table.event_class.eq('microsaccade').to_numpy(dtype=float)
    path=table.rendered_path_length_arcmin.to_numpy()
    weights=animal_weights(animal)
    permutations=shuffled_assignments(u,shuffles) if shuffles else np.empty((0,u),int)
    predictions={}; shuffle_errors=np.zeros((shuffles,n,u,o),dtype=np.float32)
    assignments=[]
    for repeat in range(repeats):
        folds=trial_folds(table,SEED+repeat)
        assignments.append(folds)
        for k in range(5):
            train=np.flatnonzero(folds!=k); test=np.flatnonzero(folds==k)
            features,e=make_features(path,engagement,dynamic,animal,event,train)
            for name,x in features.items():
                if name not in predictions: predictions[name]=np.empty((repeats,n,u,o),np.float32)
                predictions[name][repeat,test]=fit_predict(x,y,train,test,weights,penalty)
            base=features['class_path_dynamic']
            for j,p in enumerate(permutations):
                x=np.concatenate([np.broadcast_to(base,(u,*base.shape[1:])),e[p]],axis=-1)
                pred=fit_predict(x,y,train,test,weights,penalty)
                shuffle_errors[j,test]+=(pred-y[test])**2/repeats
            print(f'Primary penalty={penalty}: repeat {repeat+1}/{repeats}, fold {k+1}/5',flush=True)
    error={name:np.mean((p.astype(float)-y[None])**2,axis=0) for name,p in predictions.items()}
    if shuffles: error['class_path_dynamic_shuffled']=shuffle_errors.mean(axis=0,dtype=np.float64)
    np.savez_compressed(out/f'primary_predictions_lambda_{penalty:g}.npz',**predictions,
        fold_assignments=np.stack(assignments),shuffle_assignments=permutations,
        shuffle_squared_errors=shuffle_errors)
    return error,shuffle_errors,np.stack(assignments)


def secondary_predictions(out,table,images,y,engagement,dynamic):
    s,t,u,o=y.shape; target=y.reshape(-1,u,o)
    animal=np.tile(table.session.str.startswith('Logan').to_numpy(dtype=float),s)
    event=np.tile(table.event_class.eq('microsaccade').to_numpy(dtype=float),s)
    path=np.tile(table.rendered_path_length_arcmin.to_numpy(),s)
    weights=animal_weights(animal)
    ef=trial_folds(table,SEED)
    unique,inv=np.unique(images.source_canvas_sha256.to_numpy(),return_inverse=True)
    rng=np.random.default_rng(SEED+70)
    cf_unique=np.empty(len(unique),int); cf_unique[rng.permutation(len(unique))]=np.arange(len(unique))%5
    cf=cf_unique[inv]
    engagement=engagement.reshape(-1,u); dynamic=dynamic.ravel()
    pooled=engagement.mean(axis=1)
    names=['class_path','class_path_engagement','class_path_dynamic',
        'class_path_dynamic_engagement','class_path_dynamic_population_passband']
    predictions={name:np.full(target.shape,np.nan,np.float32) for name in names}
    coverage=np.zeros(s*t,int)
    for c in range(5):
        for e in range(5):
            train=np.flatnonzero(((cf[:,None]!=c)&(ef[None,:]!=e)).ravel())
            test=np.flatnonzero(((cf[:,None]==c)&(ef[None,:]==e)).ravel())
            assert not set(inv[train//t]) & set(inv[test//t])
            groups=(table.session+'|'+table.trial_idx.astype(str)).to_numpy()
            assert not set(groups[train%t]) & set(groups[test%t])
            for left in range(0,u,32):
                right=min(left+32,u)
                features,_=make_features(path,engagement[:,left:right],dynamic,animal,event,train,pooled_power=pooled)
                for name in names:
                    predictions[name][test,left:right]=fit_predict(features[name],target[:,left:right],train,test,weights)
            coverage[test]+=1
            print(f'Secondary: canvas fold {c+1}/5, eye-trial fold {e+1}/5; train={len(train)}, test={len(test)}',flush=True)
    assert np.all(coverage==1)
    assert all(np.isfinite(v).all() for v in predictions.values())
    np.savez_compressed(out/'secondary_predictions.npz',**predictions,
        canvas_fold=cf,eye_trial_fold=ef,coverage=coverage)
    return {name:(p.astype(float)-target)**2 for name,p in predictions.items()}


def strata_results(table,y,error,strict):
    # Movie rows are flattened scene first, eye second.
    n_scenes=len(y)//len(table)
    a=np.tile(table.session.str.split('_').str[0].to_numpy(),n_scenes)
    c=np.tile(table.event_class.to_numpy(),n_scenes)
    results={}
    for name,mask in [('Allen',a=='Allen'),('Logan',a=='Logan'),('drift',c=='drift'),('microsaccade',c=='microsaccade')]:
        results[name]=paired_scores(y[mask],{k:v[mask] for k,v in error.items()},animal_weights(a[mask]),None,strict)[0]
    return results


def save_per_unit(out,name,table,r2,contrasts):
    rows=[]
    for key,values in r2.items():
        for o,outcome in enumerate(['rate_percent','ssi_percent']):
            for u,value in enumerate(values[:,o]):
                rows.append({'unit_index':int(table.iloc[u].unit_index),'session':table.iloc[u].session,
                    'cid':int(table.iloc[u].cid),'strict_tuning':bool(table.iloc[u].validated_for_figure4),
                    'outcome':outcome,'model_or_contrast':key,'metric':'r2','value':float(value)})
    for key,values in contrasts.items():
        for o,outcome in enumerate(['rate_percent','ssi_percent']):
            for u,value in enumerate(values[:,o]):
                rows.append({'unit_index':int(table.iloc[u].unit_index),'session':table.iloc[u].session,
                    'cid':int(table.iloc[u].cid),'strict_tuning':bool(table.iloc[u].validated_for_figure4),
                    'outcome':outcome,'model_or_contrast':key,'metric':'error_reduction','value':float(value)})
    pd.DataFrame(rows).to_csv(out/f'{name}_per_unit.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--stage',choices=['all','primary','secondary'],default='all')
    args=parser.parse_args();out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    if not (out/'analysis_inputs.npz').exists(): load_data(out)
    design=json.loads((out/'design.json').read_text())
    for name,expected in design['source_sha256'].items():
        if digest(ROOT/name)!=expected: raise ValueError('Changed source: '+name)
    table=pd.read_csv(out/'traces.csv');images=pd.read_csv(out/'images.csv');units=pd.read_csv(out/'units.csv')
    strict=units.validated_for_figure4.to_numpy(dtype=bool)
    inputs=np.load(out/'analysis_inputs.npz')
    if args.stage in ['all','primary']:
        y=inputs['primary_y'];ep=inputs['primary_engagement'];dynamic=inputs['primary_dynamic']
        errors,shuffle_errors,folds=primary_predictions(out,table,y,ep,dynamic)
        weights=animal_weights(table.session.str.split('_').str[0].to_numpy())
        boot=bootstrap_weights(table,np.array(['fixed_image_ensemble']),1000,SEED+100)
        report,r2,contrasts=paired_scores(y,errors,weights,boot,strict)
        report['strata']=strata_results(table,y,errors,strict)
        report['uncertainty']='Source-trial bootstrap within animal; fixed 40-image ensemble and fixed trained prediction rules.'
        shuffle_scores=[]
        for err in shuffle_errors:
            mse=np.einsum('n,nuo->uo',weights,err)
            base=np.einsum('n,nuo->uo',weights,errors['class_path_dynamic'])
            shuffle_scores.append(np.median(1-mse/base,axis=0))
        report['shuffled_assignment_error_reductions']=shuffle_scores
        report['partial_rank_correlations']={}
        for event in ['all','drift','microsaccade']:
            mask=np.ones(len(table),bool) if event=='all' else table.event_class.eq(event).to_numpy()
            values=[]
            for o in range(2):
                value=partial_rank_correlations(y[...,o],ep,table.rendered_path_length_arcmin.to_numpy(),
                    table.session.str.startswith('Logan').to_numpy(dtype=float),table.event_class.eq('microsaccade').to_numpy(dtype=float),mask)
                values.append({'median':float(np.median(value)),'strict_median':float(np.median(value[strict])),'fraction_positive':float(np.mean(value>0))})
            report['partial_rank_correlations'][event]=values
        save_per_unit(out,'primary',units,r2,contrasts)
        write_json(out/'primary_summary.json',report)
        for penalty in [.001,.1]:
            errs,_,_=primary_predictions(out,table,y,ep,dynamic,penalty=penalty,shuffles=0)
            report_sensitivity=paired_scores(y,errs,weights,None,strict)[0]
            write_json(out/f'primary_sensitivity_lambda_{penalty:g}.json',report_sensitivity)
    if args.stage in ['all','secondary']:
        y=inputs['movie_y']
        errors=secondary_predictions(out,table,images,y,inputs['movie_engagement'],inputs['movie_dynamic'])
        target=y.reshape(-1,*y.shape[-2:])
        weights=np.tile(animal_weights(table.session.str.split('_').str[0].to_numpy()),len(images))/len(images)
        boot=bootstrap_weights(table,images.source_canvas_sha256.to_numpy(),1000,SEED+101)
        report,r2,contrasts=paired_scores(target,errors,weights,boot,strict)
        report['strata']=strata_results(table,target,errors,strict)
        report['uncertainty']='Crossed source-canvas and within-animal source-eye-trial bootstrap; fixed trained prediction rules.'
        save_per_unit(out,'secondary',units,r2,contrasts)
        write_json(out/'secondary_summary.json',report)
    print(f'Completed {args.stage} in {time.monotonic()-started:.1f} seconds',flush=True)


if __name__=='__main__': main()
