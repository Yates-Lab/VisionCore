"""Selection of the illustrative post-saccade history for panel A."""
import unittest

import numpy as np

from paper.supp_stabilization_control.example import (
    N_LAGS, drift_endpoint, post_saccade_endpoints,
)


def trace(steps, n=200):
    eye = np.zeros((n, 2))
    for sample, amplitude in steps:
        eye[sample:, 0] += amplitude
    return eye


class EndpointTest(unittest.TestCase):
    def test_endpoint_follows_an_isolated_saccade_with_a_full_history(self):
        (endpoint, amplitude), = post_saccade_endpoints(trace([(100, .3)]), delay=10)
        self.assertEqual(endpoint, 110)
        self.assertAlmostEqual(amplitude, .3)

    def test_history_with_two_saccades_or_too_little_history_is_skipped(self):
        self.assertEqual(post_saccade_endpoints(trace([(80, .3), (100, .3)]), delay=10)[1:], [])
        self.assertEqual(post_saccade_endpoints(trace([(N_LAGS - 30, .3)]), delay=10), [])


class DriftTest(unittest.TestCase):
    def test_largest_saccade_free_displacement_is_selected(self):
        eye = np.zeros((300, 2))
        eye[100:200, 0] = np.linspace(0, .2, 100)      # 0.2 deg of drift in ~417 ms
        eye[200:, 0] = .2
        endpoint, displacement = drift_endpoint(eye)
        self.assertTrue(100 + N_LAGS - 1 <= endpoint < 200 + 8)
        self.assertAlmostEqual(displacement, .2 * N_LAGS / 100, delta=.02)

    def test_history_containing_a_saccade_is_not_drift(self):
        _, displacement = drift_endpoint(trace([(100, .3)], n=140))
        self.assertLess(displacement, .01)
        self.assertIsNone(drift_endpoint(trace([(40, .3)], n=110)))


if __name__ == "__main__":
    unittest.main()
