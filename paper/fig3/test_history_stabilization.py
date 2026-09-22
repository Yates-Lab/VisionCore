"""Integration checks against the real FixRSVP renderer (no model inference)."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from history_stabilization import HistoryStabilizer, render_shared_image_rois


class HistoryStabilizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from DataYatesV1.utils.io import YatesV1Session
        from DataYatesV1.utils.data.datasets import DictDataset
        from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
        cls.session = YatesV1Session("Allen_2022-04-08")
        cls.raw = DictDataset.load(cls.session.sess_dir / "datasets/fixrsvp.dset")
        cls.endpoint = 2579  # Recorded central fixation used in Figure 3B.
        cls.trial_id = int(cls.raw["trial_inds"][cls.endpoint])
        cls.trial = FixRsvpTrial(cls.session.exp["D"][cls.trial_id], cls.session.exp["S"])
        cls.stim = cls.raw["stim"].numpy()[:, 8:-8, 8:-8]
        cls.renderer = HistoryStabilizer("Allen_2022-04-08", ((cls.stim.astype(np.float32)-127)/255)[:, None], 1)

    def test_optimized_roi_render_matches_every_direct_crop(self):
        indices = np.arange(4, 12).repeat(4)
        rois = self.raw["roi"].numpy()[self.endpoint-31:self.endpoint+1]
        expected = self.trial.get_rois(indices, roi=rois)
        np.testing.assert_array_equal(render_shared_image_rois(self.trial, indices, rois), expected)

    def test_preserves_endpoint_and_historical_image_sequence(self):
        from DataYatesV1.utils.general import get_clock_functions
        lags = np.arange(60)
        actual = np.rint(self.renderer.render([self.endpoint], lags)[0, 0]*255+127)
        np.testing.assert_array_equal(actual[0], self.stim[self.endpoint])
        ptb2ephys, _ = get_clock_functions(self.session.exp)
        start = np.flatnonzero(self.trial.image_ids == 2)[0]
        times = self.raw["t_bins"].numpy().ravel()[self.endpoint-lags]
        images = np.searchsorted(ptb2ephys(self.trial.flip_times[start:]), times, side="right")-1+start
        expected = self.trial.get_rois(images, roi=self.raw["roi"].numpy()[self.endpoint])[:, 8:-8, 8:-8]
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(len(np.unique(images)), 1)
        self.assertFalse(np.array_equal(actual, np.broadcast_to(actual[0], actual.shape)))
        self.assertFalse(np.array_equal(actual, self.stim[self.endpoint-lags]))


if __name__ == "__main__":
    unittest.main()
