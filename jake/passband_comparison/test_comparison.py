"""Tests of leakage boundaries, clustered sampling, and inference calculations."""
import unittest
import numpy as np
import pandas as pd
from jake.passband_comparison.data import digest, prepare_cached_inputs
from jake.passband_comparison.statistics import (
    spline_features,fit_predict,trial_folds,bootstrap_weights,paired_scores,
    load_released_folds,validate_fold_assignments)


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

    def test_released_folds_preserve_trial_groups_and_validate_archive(self):
        import tempfile
        from pathlib import Path
        table=pd.DataFrame({'session':['Allen_a']*10+['Logan_a']*10,
            'trial_idx':list(np.repeat(np.arange(5),2))*2,
            'event_class':['drift']*20})
        folds=np.tile(np.repeat(np.arange(5),2),2)
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/'folds.npz'
            table.to_csv(archive.parent/'traces.csv',index=False)
            expected=np.stack([folds,folds,folds])
            np.savez(archive,fold_assignments=expected)
            np.testing.assert_array_equal(load_released_folds(table,archive),expected)
            np.testing.assert_array_equal(validate_fold_assignments(table,expected),expected)
            reordered=table.copy()
            reordered.iloc[[0,1,2,3]]=table.iloc[[2,3,0,1]].to_numpy()
            with self.assertRaisesRegex(ValueError,'ordered trace'):
                load_released_folds(reordered,archive)
            broken=expected.copy();broken[0,0]=1
            np.savez(archive,fold_assignments=broken)
            with self.assertRaisesRegex(ValueError,'trial group'):
                load_released_folds(table,archive)
            np.savez(archive,fold_assignments=expected[:2])
            with self.assertRaisesRegex(ValueError,'shape'):
                load_released_folds(table,archive)

    def test_cached_replay_checks_original_and_corrected_shard_identity(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); released=root/'released';released.mkdir()
            (released/'input_audit.json').write_text('{"passed":true}')
            for name in ('traces.csv','images.csv','units.csv'):
                (released/name).write_text('identity\n')
            old=root/'old.npz';new=root/'new.npz';out=root/'out'
            axes={'image_indices':np.array([7,8]),'trace_indices':np.array([0,1]),
                  'unit_indices':np.array([0,1]),'motion_scales':np.array([0.,1.]),
                  'spatial_cpd':np.array([1.,2.]),'temporal_hz':np.array([2.,4.]),
                  'orientation_deg':np.array([0.,90.])}
            response=np.ones((2,2,2,2),dtype=np.float32);power=np.ones_like(response)
            power[:,:,0]=np.float32(1/7)
            power[1]*=np.float32(1/3)
            np.savez(old,**axes,mean_rate=response,expected_spikes=response,map_ssi=response,
                     joint_passband_power=power,total_dynamic_power=power)
            np.savez(new,**axes,mean_rate=response,expected_spikes=response,map_ssi=response,
                     joint_passband_power=2*power,total_dynamic_power=power)
            scene_engagement=power[:,:,1].astype(np.float64)-power[:,:,0].astype(np.float64)
            scene_power=scene_engagement[...,0]
            np.savez(released/'analysis_inputs.npz',primary_engagement=np.median(scene_engagement,axis=0),
                     movie_engagement=scene_engagement,primary_dynamic=np.median(scene_power,axis=0),
                     movie_dynamic=scene_power,primary_y=np.ones((2,2,2)),movie_y=np.ones((2,2,2,2)))
            (released/'design.json').write_text(json.dumps({'source_sha256':{str(old):digest(old)}}))
            prepare_cached_inputs(released,[old],[new],out)
            with np.load(out/'analysis_inputs.npz') as z:
                np.testing.assert_array_equal(z['movie_engagement'],2*scene_engagement)
                np.testing.assert_array_equal(z['primary_engagement'],np.median(2*scene_engagement,axis=0))
            design=json.loads((out/'design.json').read_text())
            self.assertEqual(design['source_sha256'][str(new)],digest(new))
            self.assertEqual(design['released_source_sha256'][str(old)],digest(old))
            with np.load(new) as z:
                arrays=dict(z)
            arrays['orientation_deg']=np.array([10.,100.])
            np.savez(new,**arrays)
            with self.assertRaisesRegex(ValueError,'identity'):
                prepare_cached_inputs(released,[old],[new],root/'bad')
            arrays['orientation_deg']=axes['orientation_deg']
            arrays['mean_rate']=np.zeros_like(response)
            np.savez(new,**arrays)
            with self.assertRaisesRegex(ValueError,'response'):
                prepare_cached_inputs(released,[old],[new],root/'bad')

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
