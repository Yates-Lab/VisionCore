"""Freeze gaze within each model history, preserving the actual RSVP sequence.

Each prediction has its own anchor: the ROI at its latest input sample. This
removes preceding retinal motion while retaining the current retinal position.
It is intentionally different from a single session-global stabilization.
"""
import numpy as np


def render_shared_image_rois(trial, image_indices, rois):
    """Render one bounding ROI per flashed image and take exact integer crops.

    ROI extraction in FixRsvpTrial.get_rois is a crop after image placement;
    widening that ROI does not resample the image. Check a direct native render
    for each image as well as every history's endpoint below.
    """
    output = np.empty((len(rois), *(rois[0, :, 1] - rois[0, :, 0])), dtype=np.uint8)
    for index in np.unique(image_indices):
        rows = np.flatnonzero(image_indices == index)
        selected = rois[rows]
        union = np.stack([selected[:, :, 0].min(axis=0), selected[:, :, 1].max(axis=0)], axis=1)
        canvas = trial.get_rois(np.array([index]), roi=union)[0]
        for row in rows:
            roi = rois[row] - union[:, 0, None]
            output[row] = canvas[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1]]
        direct = trial.get_rois(np.array([index]), roi=rois[rows[0]])[0]
        if not np.array_equal(output[rows[0]], direct):
            raise AssertionError("Bounding-ROI optimization differs from native rendering")
    return output


class HistoryStabilizer:
    def __init__(self, session_name, embedded_stim, factor):
        from DataYatesV1.utils.io import YatesV1Session
        from DataYatesV1.utils.general import get_clock_functions
        from DataYatesV1.utils.data.datasets import DictDataset
        from _fig3_ablation_data import _center_crop_spatial

        self.session = YatesV1Session(session_name)
        self.ptb2ephys, _ = get_clock_functions(self.session.exp)
        raw = DictDataset.load(self.session.sess_dir / "datasets" / "fixrsvp.dset")
        self.raw_stim = raw["stim"].numpy()
        self.trial_ids = raw["trial_inds"].numpy().astype(int).ravel()
        self.times = raw["t_bins"].numpy().ravel()
        self.rois = raw["roi"].numpy().astype(int)
        self.factor = factor
        self.hw = embedded_stim.shape[-2:]
        dec = _center_crop_spatial(self.raw_stim[::factor], self.hw)
        emb = np.rint(embedded_stim[:, 0] * 255 + 127).astype(int)
        self.align_maxabs = int(np.abs(dec[:len(emb)].astype(int) - emb).max())
        if self.align_maxabs:
            raise ValueError("Raw stimulus and embedded input are not pixel aligned")
        self.audit = {"anchor": "latest input sample", "alignment_maxabs": 0,
                      "current_frame_maxabs": 0, "n_histories": 0,
                      "n_histories_with_image_change": 0,
                      "n_changed_history_pixels": 0}

    def render(self, model_indices, stim_lags):
        from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
        from _fig3_ablation_data import _center_crop_spatial

        endpoints = np.asarray(model_indices, dtype=int) * self.factor
        indices = endpoints[:, None] - np.asarray(stim_lags)[None, :] * self.factor
        if indices.min() < 0:
            raise ValueError("History precedes the recorded stimulus")
        anchors = self.rois[endpoints]
        output = np.empty((*indices.shape, *self.hw), dtype=np.uint8)
        # Group by the SOURCE frame's trial, including any history crossing a
        # trial boundary. Rendering uses that frame's actual image and position.
        for trial_id in np.unique(self.trial_ids[indices]):
            rows, cols = np.where(self.trial_ids[indices] == trial_id)
            trial = FixRsvpTrial(self.session.exp["D"][trial_id], self.session.exp["S"])
            start = np.where(trial.image_ids == 2)[0][0]
            flips = self.ptb2ephys(trial.flip_times[start:])
            image_indices = np.searchsorted(flips, self.times[indices[rows, cols]],
                                           side="right") - 1 + start
            # A flashed image is repeated at many native samples. Render each
            # exact (image, anchored ROI) once; this is an exact cache, not an
            # approximation or interpolation of the retinal input.
            keys = np.column_stack([image_indices, anchors[rows].reshape(-1, 4)])
            unique, inverse = np.unique(keys, axis=0, return_inverse=True)
            rendered = render_shared_image_rois(trial, unique[:, 0], unique[:, 1:].reshape(-1, 2, 2))
            output[rows, cols] = _center_crop_spatial(rendered, self.hw)[inverse]
        current = np.where(np.asarray(stim_lags) == 0)[0]
        if len(current) != 1:
            raise ValueError("Control requires exactly one current-frame lag")
        expected = _center_crop_spatial(self.raw_stim[endpoints], self.hw)
        error = int(np.abs(output[:, current[0]].astype(int) - expected.astype(int)).max())
        if error:
            raise AssertionError(f"Current retinal frame changed by {error} pixel levels")
        intact = _center_crop_spatial(self.raw_stim[indices], self.hw)
        self.audit["n_histories"] += len(endpoints)
        self.audit["n_changed_history_pixels"] += int(np.count_nonzero(output != intact))
        # Image changes remain visible with gaze frozen (no replacement with a
        # single repeated endpoint image).
        self.audit["n_histories_with_image_change"] += int(np.sum(
            np.any(output != output[:, current[0], None], axis=(1, 2, 3))))
        return ((output.astype(np.float32) - 127.) / 255.)[:, None]
