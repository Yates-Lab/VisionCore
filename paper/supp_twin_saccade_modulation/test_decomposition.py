import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from paper.supp_twin_saccade_modulation.decomposition import (
    aggregate_native_pairs,
    analytic_components,
    isolated_event_indices,
    shared_condition_mask,
    valid_event_windows,
    valid_inference_indices,
)


class BiasedToyModulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim = 2
        self.additive_dim = 2
        self.film_max_gain = 0.5
        self.input_norm = nn.Identity()
        self.encoder = nn.Sequential(nn.Linear(2, 2), nn.GELU())
        self.scale_layer = nn.Linear(2, 2)

    def forward(self, features, behavior):
        encoded = self.encoder(self.input_norm(behavior))
        gain = 1 + self.film_max_gain * torch.tanh(self.scale_layer(encoded))
        visual = features * gain[:, :, None, None, None]
        additive = encoded[:, :, None, None, None].expand(-1, -1, 1, 2, 2)
        return torch.cat((visual, additive), dim=1)


class SignedToyReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(4, 2, 1, bias=False)
        self.bias = nn.Parameter(torch.tensor([0.4, -0.7]))
        self.output_scale = 1.3
        self.register_buffer("spatial", torch.tensor([
            [[1.0, -0.5], [0.25, 0.75]],
            [[-0.4, 0.2], [0.6, -0.8]],
        ]))

    def get_spatial_weights(self):
        return self.spatial

    def forward(self, x):
        x = x[:, :, -1]
        return self.output_scale * (
            self.features(x) * self.spatial[None]
        ).sum((-2, -1)) + self.bias


class DecompositionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.modulator = BiasedToyModulator()
        self.readout = SignedToyReadout()
        with torch.no_grad():
            self.modulator.encoder[0].weight.copy_(torch.tensor([[0.5, -0.2], [-0.3, 0.7]]))
            self.modulator.encoder[0].bias.copy_(torch.tensor([0.25, -0.15]))
            self.modulator.scale_layer.weight.copy_(torch.tensor([[0.4, -0.6], [-0.5, 0.3]]))
            self.modulator.scale_layer.bias.copy_(torch.tensor([0.1, -0.2]))
            self.readout.features.weight[:, :, 0, 0].copy_(torch.tensor([
                [0.8, -0.4, 0.7, -0.2],
                [-0.3, 0.9, -0.6, 0.5],
            ]))

    def test_signed_biased_decomposition_matches_ordinary_forward(self):
        scaffold = torch.tensor([
            [[[[1.0, -2.0], [0.5, 0.2]]], [[[0.3, 0.7], [-1.0, 2.0]]]],
            [[[[0.1, 0.4], [1.5, -0.8]]], [[[-0.2, 1.0], [0.6, 0.9]]]],
        ])
        behavior = torch.tensor([[0.2, -0.5], [1.0, 0.3]])
        result = analytic_components(scaffold, behavior, self.modulator, self.readout)
        ordinary_logits = self.readout(self.modulator(scaffold, behavior))
        ordinary_rates = torch.nn.functional.softplus(ordinary_logits)

        torch.testing.assert_close(result["zfull"], ordinary_logits)
        torch.testing.assert_close(
            result["z0"] + result["delta_additive"] + result["delta_gain"],
            result["zfull"],
        )
        torch.testing.assert_close(result["rfull"], ordinary_rates)
        self.assertFalse(torch.allclose(result["gain0"], torch.ones_like(result["gain0"])))
        self.assertFalse(torch.allclose(result["additive0"], torch.zeros_like(result["additive0"])))

    def test_softplus_interaction_completes_rate_identity(self):
        scaffold = torch.randn(3, 2, 1, 2, 2)
        behavior = torch.randn(3, 2)
        result = analytic_components(scaffold, behavior, self.modulator, self.readout)
        interaction = result["interaction"]
        torch.testing.assert_close(
            result["rfull"] - result["r0"],
            (result["r_additive"] - result["r0"])
            + (result["r_gain"] - result["r0"])
            + interaction,
        )
        self.assertGreater(float(interaction.detach().abs().max()), 0.0)


class AlignmentTests(unittest.TestCase):
    def test_isolation_uses_all_neighboring_events_before_amplitude_filter(self):
        events = [
            {"start_time": 1.0, "start_x": 0.0, "start_y": 0.0, "end_x": 0.5, "end_y": 0.0},
            {"start_time": 1.1, "start_x": 0.0, "start_y": 0.0, "end_x": 2.0, "end_y": 0.0},
            {"start_time": 2.0, "start_x": 0.0, "start_y": 0.0, "end_x": 0.4, "end_y": 0.0},
        ]
        selected, audit = isolated_event_indices(events, max_amplitude=1.0, isolation_s=0.15)
        np.testing.assert_array_equal(selected, [2])
        self.assertEqual(audit["amplitude_eligible"], 2)
        self.assertEqual(audit["isolated_amplitude_eligible"], 1)

    def test_event_windows_reject_cross_trial_invalid_and_time_gap_samples(self):
        times = np.arange(18) / 240.0
        times[15:] += 0.02
        trials = np.array([0] * 6 + [1] * 6 + [2] * 6)
        valid = np.ones(18, dtype=bool)
        valid[9] = False
        events = np.array([times[3], times[5], times[8], times[14]])
        windows, audit = valid_event_windows(
            times, trials, valid, events, pre_bins=2, post_bins=2
        )
        np.testing.assert_array_equal(windows, [[1, 2, 3, 4, 5]])
        self.assertEqual(audit, {"input": 4, "outside_tolerance": 0,
            "edge_or_cross_trial": 1, "time_gap_window": 1,
            "invalid_window": 1, "kept": 1})

    def test_inference_indices_require_finite_same_trial_contiguous_history(self):
        times = np.arange(10) / 240.0
        times[8:] += 0.02
        trials = np.array([0] * 5 + [1] * 5)
        dfs = torch.ones(10, 2)
        dfs[6] = float("nan")
        indices, audit = valid_inference_indices(dfs, trials, times, history_frames=3)
        np.testing.assert_array_equal(indices, [2, 3, 4, 7])
        self.assertEqual(audit["kept"], 4)
        self.assertEqual(audit["rejected_nonfinite_or_zero_dfs"], 1)
        self.assertGreaterEqual(audit["rejected_trial_or_time_history"], 1)

    def test_shared_mask_requires_every_prediction_to_be_finite(self):
        robs = np.array([[1.0, 2.0], [3.0, np.nan]])
        dfs = np.ones_like(robs)
        full = np.array([[0.5, np.nan], [0.4, 0.2]])
        zero = np.array([[0.3, 0.2], [np.nan, 0.1]])
        np.testing.assert_array_equal(
            shared_condition_mask(robs, dfs, full, zero),
            [[True, False], [False, False]],
        )

    def test_native_pair_aggregation_rejects_cross_trial_and_preserves_identity(self):
        trials = np.array([0, 0, 0, 1, 1, 1])
        values = np.arange(12, dtype=float).reshape(6, 2)
        endpoints, summed = aggregate_native_pairs(values, trials)
        np.testing.assert_array_equal(endpoints, [1, 4])
        np.testing.assert_allclose(summed, values[[0, 3]] + values[[1, 4]])
        baseline = values
        additive = values * 0.1
        gain = values * -0.2
        interaction = values * 0.05
        _, full_sum = aggregate_native_pairs(baseline + additive + gain + interaction, trials)
        _, base_sum = aggregate_native_pairs(baseline, trials)
        _, add_sum = aggregate_native_pairs(additive, trials)
        _, gain_sum = aggregate_native_pairs(gain, trials)
        _, interaction_sum = aggregate_native_pairs(interaction, trials)
        np.testing.assert_allclose(full_sum, base_sum + add_sum + gain_sum + interaction_sum)


if __name__ == "__main__":
    unittest.main()
