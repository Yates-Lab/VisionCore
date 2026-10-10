"""Figure 3 C and D share one scored population."""
import sys
import unittest

import numpy as np

from VisionCore.paths import VISIONCORE_ROOT

sys.path.insert(0, str(VISIONCORE_ROOT / "paper/fig3"))   # the generator imports its siblings by name
import generate_figure3


class ScoredUnitsTest(unittest.TestCase):
    def test_units_must_be_valid_in_both_panels(self):
        ones = np.ones(4)
        ccnorm = {key: ones.copy() for key in ("intact", "zeroed", "stabilized")}
        fraction = {key: ones.copy() for key in ("psth", "intact", "zeroed", "stabilized")}
        ccnorm["stabilized"][1] = np.nan       # no noise-normalized correlation
        fraction["psth"][2] = np.nan           # no usable Figure 2 denominator
        abl = {"cd_population": np.array([True, True, True, False]),
               "ccnorm": ccnorm, "explainable_fraction": fraction}
        np.testing.assert_array_equal(generate_figure3._scored_units(abl),
                                      [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
