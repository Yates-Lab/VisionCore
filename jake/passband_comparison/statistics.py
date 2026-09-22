"""Training-only spline regression, blocked folds, and paired evaluation."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold


def trial_folds(table, seed, n_splits=5):
    groups=(table.session.astype(str)+'|'+table.trial_idx.astype(str)).to_numpy()
    strata=(table.session.str.split('_').str[0]+'|'+table.event_class).to_numpy()
    folds=np.full(len(table),-1,dtype=int)
    splitter=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    for k,(train,test) in enumerate(splitter.split(np.zeros(len(table)),strata,groups)):
        assert not set(groups[train]) & set(groups[test])
        assert set(strata[train])==set(strata)
        folds[test]=k
    assert np.all(folds>=0)
    return folds


def spline_features(x, train):
    """[sample,unit] -> [unit,sample,5], fitted without test statistics.

    The empirical percentile interpolates tied training values at their
    midrank; outside the training range it saturates at zero/one.
    """
    x=np.asarray(x,dtype=float)
    if x.ndim==1: x=x[:,None]
    q=np.empty_like(x)
    for u in range(x.shape[1]):
        ref=x[train,u]
        unique,inverse=np.unique(ref,return_inverse=True)
        rank=(rankdata(ref,method='average')-.5)/len(ref)
        mid=np.bincount(inverse,weights=rank)/np.bincount(inverse)
        q[:,u]=np.interp(x[:,u],unique,mid,left=0.,right=1.)
    knots=np.array([0,0,0,0,1/3,2/3,1,1,1,1],dtype=float)
    basis=BSpline.design_matrix(q.ravel(),knots,3).toarray()[:,:-1]
    return basis.reshape(*x.shape,5).transpose(1,0,2)


def fit_predict(x, y, train, test, weights, penalty=.01):
    """Batched independent fits. X [unit,sample,p], y [sample,unit,outcome]."""
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float).transpose(1,0,2)
    w=np.asarray(weights[train],dtype=float); w/=w.sum()
    if x.shape[0]==1:
        xt=x[0,train]; yt=y[:,train]
        xm=w@xt; ym=np.einsum('n,uno->uo',w,yt)
        scale=np.sqrt(w@((xt-xm)**2)); scale=np.where(scale>1e-10,scale,1.)
        xt=(xt-xm)/scale; xv=(x[0,test]-xm)/scale
        gram=xt.T@(w[:,None]*xt)+penalty*np.eye(xt.shape[1])
        target=(yt-ym[:,None]).transpose(1,0,2).reshape(len(train),-1)
        beta=np.linalg.solve(gram,xt.T@(w[:,None]*target))
        return (xv@beta).reshape(len(test),y.shape[0],y.shape[-1])+ym[None]
    xt=x[:,train]; yt=y[:,train]
    xm=np.einsum('n,unp->up',w,xt); ym=np.einsum('n,uno->uo',w,yt)
    scale=np.sqrt(np.einsum('n,unp->up',w,(xt-xm[:,None])**2))
    scale=np.where(scale>1e-10,scale,1.)
    xt=(xt-xm[:,None])/scale[:,None]
    xv=(x[:,test]-xm[:,None])/scale[:,None]
    gram=np.einsum('unp,unq,n->upq',xt,xt,w,optimize=True)
    gram+=penalty*np.eye(x.shape[-1])[None]
    rhs=np.einsum('unp,uno,n->upo',xt,yt-ym[:,None],w,optimize=True)
    beta=np.linalg.solve(gram,rhs)
    pred=np.einsum('unp,upo->uno',xv,beta)+ym[:,None]
    return pred.transpose(1,0,2)


def animal_weights(animal):
    a=np.asarray(animal)
    return np.array([1./np.sum(a==v)/len(np.unique(a)) for v in a])


def make_features(path, engagement, dynamic, animal, event, train, pooled_power=None):
    n,u=engagement.shape
    a=animal[None,:,None]
    c=event[None,:,None]
    p=spline_features(path,train)
    d=spline_features(dynamic,train)
    e=spline_features(engagement,train)
    pooled=spline_features(engagement.mean(axis=1) if pooled_power is None else pooled_power,train)
    def cat(*parts):
        size=max(v.shape[0] for v in parts)
        return np.concatenate([np.broadcast_to(v,(size,*v.shape[1:])) for v in parts],axis=-1)
    cp=cat(a,c,p,p*c)
    return {
        'animal':a,'class':cat(a,c),'path':cat(a,p),'engagement':cat(a,e),
        'class_path':cp,'class_path_engagement':cat(cp,e),
        'class_path_dynamic':cat(cp,d),
        'class_path_dynamic_engagement':cat(cp,d,e),
        'class_path_dynamic_population_passband':cat(cp,d,pooled),
    },e


def partial_rank_correlations(y, engagement, path, animal, event, mask=None):
    if mask is None: mask=np.ones(len(path),dtype=bool)
    yr=rankdata(y[mask],axis=0); er=rankdata(engagement[mask],axis=0)
    c=np.column_stack([np.ones(mask.sum()),rankdata(path[mask]),animal[mask],event[mask]])
    yr-=c@np.linalg.lstsq(c,yr,rcond=None)[0]
    er-=c@np.linalg.lstsq(c,er,rcond=None)[0]
    return np.sum(yr*er,axis=0)/np.maximum(np.sqrt(np.sum(yr**2,axis=0)*np.sum(er**2,axis=0)),1e-30)


def bootstrap_weights(table, canvas, repeats, seed):
    """Weights for flattened [scene,eye] rows; source trials stay together."""
    rng=np.random.default_rng(seed)
    groups=(table.session.astype(str)+'|'+table.trial_idx.astype(str)).to_numpy()
    animals=table.session.str.split('_').str[0].to_numpy()
    tw=np.zeros((repeats,len(table)))
    for animal in np.unique(animals):
        idx=np.flatnonzero(animals==animal)
        unique,inverse=np.unique(groups[idx],return_inverse=True)
        counts=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=repeats)
        w=counts[:,inverse].astype(float)
        tw[:,idx]=w/w.sum(axis=1,keepdims=True)/len(np.unique(animals))
    unique,inverse=np.unique(canvas,return_inverse=True)
    counts=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=repeats)
    sw=counts[:,inverse].astype(float); sw/=sw.sum(axis=1,keepdims=True)
    return (sw[:,:,None]*tw[:,None,:]).reshape(repeats,-1)


def paired_scores(y, errors, weights, bootstrap, strict):
    """Errors are repeated-split mean squared errors, not ensemble errors."""
    w=weights/weights.sum()
    mean=np.einsum('n,nuo->uo',w,y)
    variance=np.einsum('n,nuo->uo',w,(y-mean[None])**2)
    mse={k:np.einsum('n,nuo->uo',w,v) for k,v in errors.items()}
    r2={k:1-v/np.maximum(variance,1e-20) for k,v in mse.items()}
    model={k:{'median_r2':np.median(v,axis=0).tolist(),
        'mean_r2':np.mean(v,axis=0).tolist(),
        'strict_median_r2':np.median(v[strict],axis=0).tolist()} for k,v in r2.items()}
    pairs=[('class','engagement'),('path','engagement'),
        ('class_path','class_path_engagement'),('engagement','class_path_engagement'),
        ('class_path_dynamic','class_path_dynamic_engagement'),
        ('class_path_dynamic_population_passband','class_path_dynamic_engagement')]
    if 'class_path_dynamic_shuffled' in errors:
        pairs.append(('class_path_dynamic_shuffled','class_path_dynamic_engagement'))
    contrasts={}; per_unit={}
    # Reuse bootstrapped MSEs across contrasts. No neuron is resampled independently.
    bm={}
    if bootstrap is not None:
        for k in set(v for pair in pairs for v in pair):
            if k in errors:
                bm[k]=(bootstrap@errors[k].reshape(len(y),-1)).reshape(len(bootstrap),*mse[k].shape)
    for base,full in pairs:
        if base not in mse or full not in mse: continue
        reduction=1-mse[full]/np.maximum(mse[base],1e-20)
        delta=r2[full]-r2[base]
        key=full+'__over__'+base
        record={'median_error_reduction':np.median(reduction,axis=0).tolist(),
            'median_delta_r2':np.median(delta,axis=0).tolist(),
            'fraction_units_improved':np.mean(reduction>0,axis=0).tolist(),
            'strict_median_error_reduction':np.median(reduction[strict],axis=0).tolist(),
            'strict_median_delta_r2':np.median(delta[strict],axis=0).tolist()}
        if bootstrap is not None:
            br=1-bm[full]/np.maximum(bm[base],1e-20)
            record['error_reduction_ci95']=np.quantile(np.median(br,axis=1),[.025,.975],axis=0).T.tolist()
            record['strict_error_reduction_ci95']=np.quantile(np.median(br[:,strict],axis=1),[.025,.975],axis=0).T.tolist()
        contrasts[key]=record; per_unit[key]=reduction
    return {'models':model,'contrasts':contrasts},r2,per_unit
