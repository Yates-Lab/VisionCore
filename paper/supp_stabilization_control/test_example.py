"""Selection of the illustrative post-saccade history for panel A."""
import unittest

import numpy as np

from paper.supp_stabilization_control.example import N_LAGS, post_saccade_endpoints


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


if __name__ == "__main__":
    unittest.main()
