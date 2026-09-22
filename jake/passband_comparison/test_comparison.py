"""Tests of leakage boundaries, clustered sampling, and inference calculations."""
import unittest
import numpy as np
import pandas as pd
from jake.passband_comparison.statistics import (
    spline_features,fit_predict,trial_folds,bootstrap_weights,paired_scores)


class ScientificContracts(unittest.TestCase):
    def test_heldout_values_do_not_change_training_transform(self):
        x=np.linspace(0,1,20); train=np.arange(15)
        a=spline_features(x,train)
        x[15:]=np.arange(5)*100+20
        b=spline_features(x,train)
        np.testing.assert_array_equal(a[:,train],b[:,train])

    def test_training_only_prediction_and_batched_common_equivalence(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(1,80,3)); y=np.stack([2*x[0,:,0]-x[0,:,1],-3*x[0,:,0]],axis=1)[...,None]
        tr=np.arange(60);te=np.arange(60,80);w=np.ones(80)
        a=fit_predict(x,y,tr,te,w,1e-9)
        b=fit_predict(np.repeat(x,2,axis=0),y,tr,te,w,1e-9)
        np.testing.assert_allclose(a,b,atol=1e-8)
        np.testing.assert_allclose(a,y[te],atol=1e-7)
        y[te]+=1000
        np.testing.assert_array_equal(a,fit_predict(x,y,tr,te,w,1e-9))

    def test_source_trial_group_never_crosses_fold(self):
        table=pd.DataFrame({'session':['Allen_a']*30+['Logan_a']*30,
            'trial_idx':list(np.repeat(np.arange(15),2))*2,'event_class':['drift','microsaccade']*30})
        fold=trial_folds(table,5)
        for _,g in table.assign(fold=fold).groupby(['session','trial_idx']):
            self.assertEqual(g.fold.nunique(),1)

    def test_crossed_bootstrap_preserves_duplicate_trials_and_canvases(self):
        table=pd.DataFrame({'session':['Allen_a']*4+['Logan_a']*4,
            'trial_idx':[1,1,2,3,1,1,2,3]})
        w=bootstrap_weights(table,np.array(['a','a','b']),100,1).reshape(100,3,8)
        np.testing.assert_allclose(w.sum(axis=(1,2)),1)
        np.testing.assert_array_equal(w[:,:,0],w[:,:,1])
        np.testing.assert_array_equal(w[:,0],w[:,1])
        np.testing.assert_allclose(w[:,:,:4].sum(axis=(1,2)),.5)

    def test_paired_error_reduction_retains_negative_prediction_scores(self):
        y=np.arange(20,dtype=float)[:,None,None]
        bad=np.full_like(y,200.);good=np.full_like(y,50.)
        result,r2,contrast=paired_scores(y,{'class_path':bad,'class_path_engagement':good},
            np.ones(20),None,np.array([True]))
        self.assertLess(r2['class_path'][0,0],0)
        self.assertAlmostEqual(result['contrasts']['class_path_engagement__over__class_path']['median_error_reduction'][0],.75)


if __name__=='__main__': unittest.main()
