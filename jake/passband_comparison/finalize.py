"""Audit saved predictions and bind the text-only comparison to its sources."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from jake.passband_comparison.data import ROOT,digest,write_json
from jake.passband_comparison.statistics import animal_weights,make_features,fit_predict,paired_scores


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',type=Path,required=True)
    out=p.parse_args().out_dir
    design=json.loads((out/'design.json').read_text())
    checks={}
    def check(name,value,detail=None):
        checks[name]={'passed':bool(value),'detail':detail}
        if not value:
            write_json(out/'audit.json',{'passed':False,'checks':checks})
            raise ValueError(name)
    selection=json.loads((ROOT/'manuscript/analysis/selected_model_bundle.json').read_text())
    check('selected_checkpoint',selection['checkpoint_sha256']==design['checkpoint_sha256'])
    check('documented_analysis_design',digest(Path(__file__).with_name('README.md'))==design['readme_sha256'])
    for name,expected in design['source_sha256'].items(): check('source:'+name,digest(ROOT/name)==expected)
    check('input_numerical_audit',json.loads((out/'input_audit.json').read_text())['passed'])
    table=pd.read_csv(out/'traces.csv');images=pd.read_csv(out/'images.csv');units=pd.read_csv(out/'units.csv')
    strict=units.validated_for_figure4.to_numpy(dtype=bool)
    data=np.load(out/'analysis_inputs.npz')
    primary=json.loads((out/'primary_summary.json').read_text())
    secondary=json.loads((out/'secondary_summary.json').read_text())
    estimator=json.loads((out/'estimator_diagnostics.json').read_text())
    for name,expected in estimator['sources'].items():
        check('estimator_source:'+name,digest(ROOT/name)==expected)
    pp=np.load(out/'primary_predictions_lambda_0.01.npz')
    sp=np.load(out/'secondary_predictions.npz')
    groups=(table.session+'|'+table.trial_idx.astype(str)).to_numpy()
    folds=pp['fold_assignments'];check('primary_fold_shape',folds.shape==(3,200))
    for r in range(3):
        for k in range(5):
            check(f'primary_trial_disjoint_r{r}_k{k}',not set(groups[folds[r]==k]) & set(groups[folds[r]!=k]))
    permutations=pp['shuffle_assignments']
    check('32_complete_self_excluding_shuffles',permutations.shape==(32,725) and all(
        np.array_equal(np.sort(p),np.arange(725)) and np.all(p!=np.arange(725)) for p in permutations))
    check('secondary_full_prediction_coverage',np.all(sp['coverage']==1))
    cf=sp['canvas_fold'];ef=sp['eye_trial_fold']
    for _,frame in images.assign(fold=cf).groupby('source_canvas_sha256'):
        check('canvas_group:'+frame.source_canvas_sha256.iloc[0],frame.fold.nunique()==1)
    for k in range(5):
        check(f'secondary_trial_disjoint_{k}',not set(groups[ef==k]) & set(groups[ef!=k]))
    for name in primary['models']:
        if name.endswith('_shuffled'): continue
        check('primary_predictions:'+name,pp[name].shape==(3,200,725,2) and np.isfinite(pp[name]).all())
    for name in secondary['models']:
        check('secondary_predictions:'+name,sp[name].shape==(8000,725,2) and np.isfinite(sp[name]).all())
    # Fresh regression fits from original predictors independently reproduce
    # saved test predictions, without rerunning the neural model.
    selected=np.array([0,362,724]);maximum=0.
    for stage,archive,ys,es,ds in [('primary',pp,'primary_y','primary_engagement','primary_dynamic'),
            ('secondary',sp,'movie_y','movie_engagement','movie_dynamic')]:
        y=data[ys].reshape(-1,725,2);eng=data[es].reshape(-1,725);dyn=data[ds].ravel()
        nscene=len(y)//200
        animal=np.tile(table.session.str.startswith('Logan').to_numpy(float),nscene)
        event=np.tile(table.event_class.eq('microsaccade').to_numpy(float),nscene)
        path=np.tile(table.rendered_path_length_arcmin.to_numpy(),nscene)
        w=animal_weights(animal)
        for k in [0,2,4]:
            if stage=='primary':
                train=np.flatnonzero(folds[0]!=k);test=np.flatnonzero(folds[0]==k)
            else:
                train=np.flatnonzero(((cf[:,None]!=k)&(ef[None,:]!=k)).ravel())
                test=np.flatnonzero(((cf[:,None]==k)&(ef[None,:]==k)).ravel())
                check(f'fresh_secondary_canvas_disjoint_{k}',not set(images.source_canvas_sha256.iloc[train//200]) & set(images.source_canvas_sha256.iloc[test//200]))
            features,_=make_features(path,eng[:,selected],dyn,animal,event,train,pooled_power=eng.mean(axis=1))
            for name in ['class_path','class_path_engagement','class_path_dynamic','class_path_dynamic_engagement',
                    'class_path_dynamic_population_passband']:
                pred=fit_predict(features[name],y[:,selected],train,test,w)
                expected=archive[name][0,test][:,selected] if stage=='primary' else archive[name][test][:,selected]
                error=float(np.max(np.abs(pred-expected)));maximum=max(maximum,error)
                check(f'fresh_fit_{stage}_{k}_{name}',np.allclose(pred,expected,atol=5e-5,rtol=2e-6),error)
    # Explicitly score scene-to-scene variation for a fixed eye history.
    movie_y=data['movie_y'];target=movie_y-movie_y.mean(axis=0,keepdims=True)
    errors={}
    for name in secondary['models']:
        pred=sp[name].reshape(movie_y.shape).astype(float)
        pred-=pred.mean(axis=0,keepdims=True)
        errors[name]=((pred-target)**2).reshape(8000,725,2)
    weights=np.tile(animal_weights(table.session.str.split('_').str[0].to_numpy()),40)/40
    scene_only=paired_scores(target.reshape(8000,725,2),errors,weights,None,strict)[0]
    scene_only['definition']='Out-of-fold predictions and targets centered over scenes separately for each eye history and neuron; descriptive score isolates variation across scenes for fixed movement.'
    # A source-canvas table makes the generalization claim inspectable.
    rows=[]
    y=movie_y
    b=(sp['class_path_dynamic'].reshape(y.shape)-y)**2
    f=(sp['class_path_dynamic_engagement'].reshape(y.shape)-y)**2
    ew=animal_weights(table.session.str.split('_').str[0].to_numpy())
    for canvas,group in images.groupby('source_canvas_sha256'):
        idx=group.index.to_numpy()
        mse_b=np.einsum('t,stuo->uo',ew,b[idx])/len(idx)
        mse_f=np.einsum('t,stuo->uo',ew,f[idx])/len(idx)
        vals=np.median(1-mse_f/mse_b,axis=0)
        rows.append({'canvas_sha256':canvas,'scene_rows':','.join(map(str,idx)),
            'rate_error_reduction':float(vals[0]),'ssi_error_reduction':float(vals[1])})
    pd.DataFrame(rows).to_csv(out/'per_source_canvas.csv',index=False)
    code={str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    for name in ['paper/fig4/spatiotemporal_tuning/build_exact_cid_figure4_contract.py',
            'paper/fig4/spatiotemporal_tuning/spectral_power.py']:
        code[name]=digest(ROOT/name)
    artifacts={p.name:digest(p) for p in out.iterdir() if p.suffix in ['.npz','.csv']}
    source_note='Engagement uses nonnegative fitted Yu SF×TF predictions multiplied by measured axial-orientation weights, normalized to unit mass; it does not use raw phase-RMS responses when passband_weight is present.'
    audit={'passed':True,'checkpoint_sha256':design['checkpoint_sha256'],'checks':checks,
        'fresh_regression_max_abs_error':maximum,'source_code_sha256':code,'artifact_sha256':artifacts,
        'method_description_correction':source_note}
    write_json(out/'audit.json',audit)
    summary={'checkpoint_sha256':design['checkpoint_sha256'],'design_sha256':digest(out/'design.json'),
        'audit_sha256':digest(out/'audit.json'),'primary':primary,'secondary':secondary,
        'same_eye_scene_variation':scene_only,'per_source_canvas':rows,
        'estimator_diagnostics':estimator,
        'regularization_sensitivity':{str(v):json.loads((out/f'primary_sensitivity_lambda_{v:g}.json').read_text()) for v in [.001,.1]},
        'method_description_correction':source_note,'limits':[
            'Model responses, not measured neural-noise discrimination or attention.',
            'Primary estimates condition on the fixed 40-image ensemble.',
            'Bootstrap intervals condition on fitted cross-validation models and resample source histories/canvases, not model fits or animals.',
            'Population-average engagement control averages image-reduced unit predictors in the primary analysis; it is a common spectral predictor, not a raw uniform-power measure.',
            'A negative tuning-shuffle comparison is conditional on a finite-window spectral estimator; it cannot rule out actual tuning-specific effects. Estimator resolution and predictor collinearity were inspected but not established as the cause of the negative result.',
            'Specificity controls establish predictive value relative to tested descriptors, not a unique causal mechanism.']}
    write_json(out/'summary.json',summary)
    print(json.dumps({'passed':True,'checks':len(checks),'max_fresh_fit_error':maximum},indent=2),flush=True)


if __name__=='__main__':main()
