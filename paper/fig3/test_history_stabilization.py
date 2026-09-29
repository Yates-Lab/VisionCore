"""Integration checks against the real FixRSVP renderer (no model inference)."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from history_stabilization import HistoryStabilizer, render_shared_image_rois
from _fig3_ablation_data import build_stabilized_stim


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
        cls.embedded = ((cls.stim.astype(np.float32)-127)/255)[:, None]
        cls.renderer = HistoryStabilizer("Allen_2022-04-08", cls.embedded, 1)

    def test_optimized_roi_render_matches_every_direct_crop(self):
        indices = np.arange(4, 12).repeat(4)
        rois = self.raw["roi"].numpy()[self.endpoint-31:self.endpoint+1]
        expected = self.trial.get_rois(indices, roi=rois)
        np.testing.assert_array_equal(render_shared_image_rois(self.trial, indices, rois), expected)

    def test_default_global_render_still_uses_one_session_centroid(self):
        from DataYatesV1.utils.general import get_clock_functions

        rendered, align_maxabs, _ = build_stabilized_stim(
            "Allen_2022-04-08", self.embedded, 1
        )
        self.assertEqual(align_maxabs, 0)
        eyepos = self.raw["eyepos"].numpy()
        dpi_pix = self.raw["dpi_pix"].numpy()
        dpi_valid = self.raw["dpi_valid"].numpy().ravel() > 0
        valid = (np.hypot(eyepos[:, 0], eyepos[:, 1]) < .5) & dpi_valid
        indices = np.flatnonzero(valid)
        centroid = dpi_pix[indices].mean(axis=0)
        nearest = indices[np.argmin(((dpi_pix[indices] - centroid) ** 2).sum(axis=1))]
        anchor = self.raw["roi"].numpy()[nearest]
        rows = np.flatnonzero(self.raw["trial_inds"].numpy().astype(int).ravel() == self.trial_id)
        ptb2ephys, _ = get_clock_functions(self.session.exp)
        start = np.flatnonzero(self.trial.image_ids == 2)[0]
        images = np.searchsorted(
            ptb2ephys(self.trial.flip_times[start:]),
            self.raw["t_bins"].numpy().ravel()[rows],
            side="right",
        ) - 1 + start
        expected = self.trial.get_rois(
            images, roi=np.repeat(anchor[None], len(rows), axis=0)
        )[:, 8:-8, 8:-8]
        actual = np.rint(rendered[rows, 0] * 255 + 127).astype(np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_trial_centroid_render_uses_each_source_trials_centroid_with_fallback(self):
        from DataYatesV1.utils.general import get_clock_functions

        try:
            rendered, _, _ = build_stabilized_stim(
                "Allen_2022-04-08", self.embedded, 1,
                stabilization_reference="trial_centroid",
            )
        except TypeError as error:
            self.fail(f"trial-centroid stabilization reference is unavailable: {error}")

        trial_ids = self.raw["trial_inds"].numpy().astype(int).ravel()
        eyepos = self.raw["eyepos"].numpy()
        dpi_pix = self.raw["dpi_pix"].numpy()
        dpi_valid = self.raw["dpi_valid"].numpy().ravel() > 0
        rois = self.raw["roi"].numpy()
        times = self.raw["t_bins"].numpy().ravel()
        central = np.hypot(eyepos[:, 0], eyepos[:, 1]) < .5
        fixation = np.hypot(eyepos[:, 0], eyepos[:, 1]) < 1.
        ptb2ephys, _ = get_clock_functions(self.session.exp)

        selected = []
        for trial_id in np.unique(trial_ids):
            trial_mask = trial_ids == trial_id
            primary = trial_mask & central & dpi_valid
            fallback = trial_mask & fixation & dpi_valid
            if np.any(primary) and not selected:
                selected.append((trial_id, primary, False))
            if not np.any(primary) and np.any(fallback):
                selected.append((trial_id, fallback, True))
                break
        self.assertEqual([row[2] for row in selected], [False, True])

        anchors = []
        for trial_id, valid, _used_fallback in selected:
            indices = np.flatnonzero(valid)
            centroid = dpi_pix[indices].mean(axis=0)
            nearest = indices[np.argmin(((dpi_pix[indices] - centroid) ** 2).sum(axis=1))]
            anchor = rois[nearest]
            anchors.append(anchor)
            rows = np.flatnonzero(trial_ids == trial_id)
            trial = type(self.trial)(self.session.exp["D"][trial_id], self.session.exp["S"])
            start = np.flatnonzero(trial.image_ids == 2)[0]
            images = np.searchsorted(ptb2ephys(trial.flip_times[start:]), times[rows], side="right") - 1 + start
            expected = trial.get_rois(images, roi=np.repeat(anchor[None], len(rows), axis=0))[:, 8:-8, 8:-8]
            actual = np.rint(rendered[rows, 0] * 255 + 127).astype(np.uint8)
            np.testing.assert_array_equal(actual, expected)
            self.assertGreater(len(np.unique(images)), 1)
            self.assertFalse(np.array_equal(actual, np.broadcast_to(actual[0], actual.shape)))
        self.assertFalse(np.array_equal(anchors[0], anchors[1]))

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
