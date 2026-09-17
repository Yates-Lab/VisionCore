"""Scientific invariants for the intervention, renderer, and gain estimands."""
import unittest
import numpy as np
from jake.fem_band_tuning.common import (
    laplacian_bands, interventions, sweep_indices, STRENGTHS,
    gain_statistics, three_way_residual, render_basis)


class ScientificInvariants(unittest.TestCase):
    def test_pyramid_reconstructs_odd_and_even_images_and_holds_dc(self):
        rng = np.random.default_rng(42)
        for shape in [(151, 159), (240, 320)]:
            image = rng.uniform(10, 240, shape).astype(np.float32)
            bands, dc = laplacian_bands(image)
            np.testing.assert_allclose(bands.sum(0)+dc, image, atol=8e-5, rtol=1e-5)
            np.testing.assert_allclose(bands.mean((1,2)), 0, atol=2e-5)
            np.testing.assert_allclose((dc+1.4*bands[2]).mean(), dc, atol=2e-5)
        bands, dc = laplacian_bands(np.full((151,159), 113, np.float32))
        np.testing.assert_allclose(bands, 0, atol=2e-5)
        self.assertAlmostEqual(dc, 113)

    def test_full_movie_intervention_equals_rendered_basis_and_endpoint_is_fixed(self):
        import torch
        rng = np.random.default_rng(4)
        y, x = np.mgrid[:240, :240]
        image = (127+20*np.sin(x/5)+30*np.cos(y/13)).astype(np.float32)
        bands, _ = laplacian_bands(image)
        fields = np.concatenate([image[None], bands])
        trace = rng.normal(0,.02,(60,2)).astype(np.float32)
        trace -= trace[-1]
        stationary = np.zeros_like(trace)
        basis = render_basis(fields, trace, device='cpu')
        still = render_basis(fields, stationary, device='cpu')
        torch.testing.assert_close(basis[:,:,0], still[:,:,0], atol=2e-6, rtol=1e-5)
        d = np.array([.2,-.1,.1,0,.2,-.2],np.float32)
        edited = image + np.einsum('k,kyx->yx',d,bands)
        direct = render_basis(edited[None], trace, device='cpu')[0]
        combined = basis[0] + torch.einsum('k,kltyx->ltyx',torch.from_numpy(d),basis[1:])
        torch.testing.assert_close(direct, combined, atol=2e-6, rtol=1e-5)
        # The first model lag is the latest retinal frame, not the earliest.
        direct_last = render_basis(edited[None], trace[-1:], device='cpu')[0,0,0]
        torch.testing.assert_close(direct[0,0], direct_last)

    def test_gain_and_curvature_recover_known_nonlinear_tuning(self):
        a = STRENGTHS-1
        slope = np.array([1.,-2.,0.])
        quadratic = np.array([3.,0.,0.])
        y = 10 + a[None,:,None]*slope + a[None,:,None]**2*quadratic
        gain, wide, curvature = gain_statistics(y)
        np.testing.assert_allclose(gain[0], slope)
        np.testing.assert_allclose(wide[0], slope)
        np.testing.assert_allclose(curvature[0], 2*quadratic)
        offsets, rows = interventions()
        idx = sweep_indices(rows)
        for k in range(6):
            np.testing.assert_allclose(offsets[idx[k],k]+1, STRENGTHS)
            np.testing.assert_allclose(np.delete(offsets[idx[k]],k,axis=1),0)

    def test_three_way_interaction_rejects_every_two_way_only_effect(self):
        rng = np.random.default_rng(123)
        shape = (7,8,9,2)
        pairwise = (rng.normal(size=(7,8,1,2)) + rng.normal(size=(7,1,9,2))
                    + rng.normal(size=(1,8,9,2)))
        np.testing.assert_allclose(three_way_residual(pairwise),0,atol=2e-15)
        effect = rng.normal(size=shape)
        three = three_way_residual(effect)
        for axis in range(3):
            np.testing.assert_allclose(three.mean(axis=axis),0,atol=3e-16)
        np.testing.assert_allclose(three_way_residual(pairwise+three),three,atol=2e-15)

    def test_gain_controls_remove_uniform_and_per_neuron_scaling(self):
        from jake.fem_band_tuning.analyze import scalar_gain_residual
        rng=np.random.default_rng(47)
        reference=rng.normal(size=(3,6,7))
        scale=rng.uniform(.2,4,(3,5,7))
        moving=reference[:,None]*scale[:,:,None,:]
        global_residual,unit_residual,_,stats=scalar_gain_residual(moving,reference)
        np.testing.assert_allclose(unit_residual,0,atol=1e-12)
        self.assertGreater(stats['global_scalar_unexplained_energy_fraction'],.001)
        moving=reference[:,None]*scale[:,:,:1,None]
        global_residual,unit_residual,_,stats=scalar_gain_residual(moving,reference)
        np.testing.assert_allclose(global_residual,0,atol=1e-12)
        np.testing.assert_allclose(unit_residual,0,atol=1e-12)

    def test_event_weights_balance_animals_despite_unequal_counts(self):
        import pandas as pd
        from jake.fem_band_tuning.analyze import condition_weights
        tab=pd.DataFrame({'session':['A_1']*7+['B_1']*9,
            'event_class':['drift']*2+['microsaccade']*5+['drift']*6+['microsaccade']*3,
            'trial_idx':[0,0,*range(2,16)]})
        for rng in [None,np.random.default_rng(7)]:
            w=condition_weights(tab,rng)
            np.testing.assert_allclose(w.sum(1),1)
            np.testing.assert_allclose(w[:,:7].sum(1),.5)
            np.testing.assert_allclose(w[:,0],w[:,1])
            for c,event in enumerate(['drift','microsaccade']):
                np.testing.assert_allclose(w[c,tab.event_class.ne(event)],0)


if __name__ == '__main__':
    unittest.main()
