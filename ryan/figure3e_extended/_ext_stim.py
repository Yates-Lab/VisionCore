"""Pixel-exact fixRSVP re-renderer supporting three stabilization scopes.

Figure 3 panel D contrasts the intact twin against ONE reafferent ablation:
the retinal input frozen at a single session-global gaze. This module
generalizes that to a *ladder* of stabilization scopes, from the narrowest
(inside the model's own integration window) to the broadest (one gaze for the
whole session):

  window : for every prediction time t, the 33-frame retinal input window is
           re-cropped at t's gaze. Each lag keeps the RSVP image that was
           actually on screen at that lag time, so the image-flash dynamics are
           untouched; only the eye-movement-induced motion WITHIN the window is
           removed. Across prediction times the crop still moves with the eye,
           so the gaze-contingent position shifts survive. This is the strictest
           possible reafference control: it removes retinal image motion and
           nothing else.
  trial  : the retinal input is frozen at each trial's OWN centroid gaze, so
           there is no image motion within a trial but different trials sit at
           different retinal positions.
  global : the retinal input is frozen at ONE session-global centroid gaze, so
           every trial sees a pixel-identical frozen image (this reproduces
           figure 3's `stabilized` condition).

Rendering strategy
------------------
`FixRsvpTrial.get_rois` regenerates a full image texture per frame, so the naive
route for the window condition (n_time x 33 renders per trial) is hopeless. But
`roi = dpi_pix.astype(int) + roi_src` is a pure integer translation of gaze, and
`place_gauss_image_texture` pastes through `nd_paste`, which is
translation-covariant with a constant fill outside the source. Therefore two
crops of the SAME screen content at two different gazes are two slices of one
canvas.

So each distinct screen *content* -- identified by which RSVP image was up, where
it was drawn, and the trial's background/radius, NOT by sample index -- is
rendered ONCE into a canvas spanning the bounding box of every gaze ROI in the
session, and every frame the model ever sees is a numpy slice of a canvas. The
display flips every raw sample but only flashes ~13 distinct images per trial,
reused across trials, so a whole session collapses to ~30 canvases (versus the
~500 renders *per trial* the existing fig3 stabilized render performs).

Because content is resolved per sample, a lag reaching back across a trial
boundary (37% of fixation samples at a 33-frame window) renders the PREVIOUS
trial's image at the current gaze -- exactly what the intact window holds, minus
the eye motion. No clamping, no boundary special case.

Exactness is not assumed. `verify_native_render` slices each canvas at its own
native ROI and requires bit-exact equality with the stored raw stimulus, which
proves in one gate that canvas-slicing == `get_rois`, that the `hist_idx`
reconstruction matches the stored stimulus, and that the substitution is frame
aligned. Callers must treat a failed gate as fatal for the session.

Self-contained: imports nothing from `paper/fig3`.
"""
from collections import OrderedDict

import numpy as np


FIX_RADIUS = 1.0        # deg; fixation = hypot(eyepos) < 1.0 (matches fig3)
CENTROID_RADIUS = 0.5   # deg; central window for a stabilization centroid
CANVAS_MARGIN = 4       # px slack around the gaze-ROI bounding box
CANVAS_CACHE_SIZE = 512 # distinct (trial, hist_idx) canvases held at once

STAB_SCOPES = ("window", "trial", "global")


class FixRsvpRenderer:
    """Per-session pixel-exact re-renderer for the fixRSVP retinal stimulus.

    Parameters
    ----------
    session_name : str
        e.g. "Allen_2022-02-16".
    """

    def __init__(self, session_name):
        from DataYatesV1.utils.io import YatesV1Session
        from DataYatesV1.exp.fix_rsvp import FixRsvpTrial
        from DataYatesV1.utils.general import get_clock_functions
        from DataYatesV1.utils.data.datasets import DictDataset

        self.session_name = session_name
        sess = YatesV1Session(session_name)
        self.exp = sess.exp
        self._ptb2ephys, _ = get_clock_functions(self.exp)
        raw = DictDataset.load(sess.sess_dir / "datasets" / "fixrsvp.dset")

        self.raw_stim = raw["stim"].numpy()                 # (Nraw, 51, 51) uint8
        self.roi = raw["roi"].numpy().astype(int)           # (Nraw, 2, 2)
        self.trial_inds = raw["trial_inds"].numpy().astype(int)
        self.t_bins = raw["t_bins"].numpy()
        self.dpi_pix = raw["dpi_pix"].numpy()
        self.dpi_valid = raw["dpi_valid"].numpy() > 0
        self.eyepos = raw["eyepos"].numpy()
        self.n_raw = self.raw_stim.shape[0]

        ecc = np.hypot(self.eyepos[:, 0], self.eyepos[:, 1])
        self.fixation = ecc < FIX_RADIUS
        self.central = ecc < CENTROID_RADIUS

        self.win_h = int(self.roi[0, 0, 1] - self.roi[0, 0, 0])
        self.win_w = int(self.roi[0, 1, 1] - self.roi[0, 1, 0])
        assert (self.roi[:, 0, 1] - self.roi[:, 0, 0] == self.win_h).all()
        assert (self.roi[:, 1, 1] - self.roi[:, 1, 0] == self.win_w).all()

        # Canvas span: the whole screen, grown by one window on every side, so
        # ANY on-screen gaze ROI is a slice of it. Deliberately not keyed to the
        # fixation samples -- prediction times are selected on the model's
        # average-pooled eye position, which admits samples whose raw-frame gaze
        # sits just outside a fixation-derived bounding box. There are only ~30
        # canvases per session, so spanning the screen costs ~1 MB each and
        # removes the failure mode entirely. `_crop` still asserts containment,
        # which now fires only on a genuinely nonsensical (off-screen) ROI.
        x0, y0, x1, y1 = np.asarray(self.exp["S"]["screenRect"], dtype=int)
        self.canvas_roi = np.array([
            [y0 - self.win_h - CANVAS_MARGIN, y1 + self.win_h + CANVAS_MARGIN],
            [x0 - self.win_w - CANVAS_MARGIN, x1 + self.win_w + CANVAS_MARGIN],
        ], dtype=int)
        self.canvas_shape = (
            int(self.canvas_roi[0, 1] - self.canvas_roi[0, 0]),
            int(self.canvas_roi[1, 1] - self.canvas_roi[1, 0]),
        )
        # Samples whose stored ROI can be served as a canvas slice. Blinks and
        # lost-track samples carry a gaze far off screen (or a non-finite
        # `dpi_pix` cast to a garbage int), so their ROI is unrenderable from any
        # finite canvas. They are never prediction times and never supply a crop
        # ROI -- only content -- but the default gate has to skip them.
        self.roi_on_canvas = (
            (self.roi[:, 0, 0] >= self.canvas_roi[0, 0])
            & (self.roi[:, 0, 0] + self.win_h <= self.canvas_roi[0, 1])
            & (self.roi[:, 1, 0] >= self.canvas_roi[1, 0])
            & (self.roi[:, 1, 0] + self.win_w <= self.canvas_roi[1, 1])
        )

        # Per-trial FixRsvpTrial objects and the content index (`hist_idx`) of
        # every raw sample, reconstructed exactly as
        # `DataYatesV1.exp.dataset_generation.generate_fixrsvp_dataset` does.
        self._trials = {}
        self.hist_idx = np.full(self.n_raw, -1, dtype=int)
        for iT in np.unique(self.trial_inds):
            m = self.trial_inds == iT
            trial = FixRsvpTrial(self.exp["D"][iT], self.exp["S"])
            start_idx = int(np.where(trial.image_ids == 2)[0][0])
            flip_times = self._ptb2ephys(trial.flip_times[start_idx:])
            self.hist_idx[m] = (
                np.searchsorted(flip_times, self.t_bins[m], side="right") - 1 + start_idx
            )
            self._trials[int(iT)] = trial
        assert (self.hist_idx >= 0).all(), "unassigned hist_idx"

        # Content identity per raw sample. `get_rois` renders from exactly
        # (image_ids[hist_idx], positions[hist_idx], bkgnd, radius); the display
        # flips every sample but reuses the same handful of flashed images, so
        # keying the canvas on content rather than on sample index collapses a
        # session from ~12k renders to ~30. `_content_src[cid]` records one
        # (trial, hist_idx) able to render content `cid`.
        self.content_id = np.full(self.n_raw, -1, dtype=int)
        self._content_src = []
        seen = {}
        for iT in np.unique(self.trial_inds):
            idx = np.where(self.trial_inds == iT)[0]
            trial = self._trials[int(iT)]
            h = self.hist_idx[idx]
            im_ids = trial.image_ids[h]
            pos = trial.positions[h]
            for loc, k in enumerate(idx):
                key = (int(im_ids[loc]), float(pos[loc, 0]), float(pos[loc, 1]),
                       float(trial.bkgnd), float(trial.radius))
                cid = seen.get(key)
                if cid is None:
                    cid = len(self._content_src)
                    seen[key] = cid
                    self._content_src.append((int(iT), int(h[loc])))
                self.content_id[k] = cid
        assert (self.content_id >= 0).all(), "unassigned content id"
        self.n_contents = len(self._content_src)

        self._canvas_cache = OrderedDict()
        self.n_canvas_renders = 0

    # -- canvas machinery ---------------------------------------------------
    def _canvas(self, cid):
        """Full-span render of screen content `cid`, cached."""
        cid = int(cid)
        hit = self._canvas_cache.get(cid)
        if hit is not None:
            self._canvas_cache.move_to_end(cid)
            return hit
        itrial, hidx = self._content_src[cid]
        canvas = self._trials[itrial].get_rois(
            np.array([hidx]), roi=self.canvas_roi[None])[0]
        self.n_canvas_renders += 1
        self._canvas_cache[cid] = canvas
        if len(self._canvas_cache) > CANVAS_CACHE_SIZE:
            self._canvas_cache.popitem(last=False)
        return canvas

    def _crop(self, cid, row0, col0):
        """The 51x51 retinal crop of content `cid` at gaze ROI origin
        `(row0, col0)` -- bit-identical to `get_rois` with that ROI."""
        i0 = int(row0) - int(self.canvas_roi[0, 0])
        j0 = int(col0) - int(self.canvas_roi[1, 0])
        assert 0 <= i0 and i0 + self.win_h <= self.canvas_shape[0], "ROI outside canvas"
        assert 0 <= j0 and j0 + self.win_w <= self.canvas_shape[1], "ROI outside canvas"
        return self._canvas(cid)[i0:i0 + self.win_h, j0:j0 + self.win_w]

    def crop_at(self, content_idx, roi_idx):
        """Content of raw sample `content_idx` cropped at raw sample `roi_idx`'s
        gaze. `content_idx == roi_idx` reproduces the stored stimulus exactly."""
        return self._crop(self.content_id[content_idx],
                          self.roi[roi_idx, 0, 0], self.roi[roi_idx, 1, 0])

    # -- gates --------------------------------------------------------------
    def verify_native_render(self, indices=None):
        """Max |canvas-slice - stored raw stim| over `indices` (every raw sample
        with an on-canvas ROI by default). MUST be 0: this is the single gate
        proving canvas-slicing == get_rois, that `hist_idx`/content resolution
        matches the stored stimulus, and that the re-render carries no artifact.

        Callers should pass the raw indices they will actually crop at, so the
        gate covers exactly the substitution being made."""
        if indices is None:
            indices = np.where(self.roi_on_canvas)[0]
        worst = 0
        for k in np.asarray(indices, dtype=int):
            d = int(np.abs(self.crop_at(k, k).astype(int)
                           - self.raw_stim[k].astype(int)).max())
            worst = max(worst, d)
            if worst:
                break
        return worst

    def alignment_maxabs(self, embedded_stim, factor):
        """Max |decimate(raw stored stim) - embedded stim| in raw pixel units.
        MUST be 0 or the raw <-> model frame mapping (raw index = factor * model
        index) does not hold and no substitution is frame-aligned."""
        emb = np.asarray(embedded_stim)
        emb_px = np.rint(emb.reshape(emb.shape[0], *emb.shape[-2:]) * 255 + 127).astype(int)
        keep = (self.n_raw // factor) * factor
        dec = self.raw_stim[:keep:factor].astype(int)
        n = min(len(dec), len(emb_px))
        return int(np.abs(dec[:n] - emb_px[:n]).max())

    # -- frozen-gaze (trial / global) stabilization --------------------------
    def _centroid_roi(self, mask):
        """ROI of the sample nearest the `dpi_pix` centroid of `mask`.

        Realizing the centroid as an actual sample's ROI (rather than rounding
        the centroid itself) keeps the frozen image on the integer pixel grid the
        stimulus was rendered on. Returns None if `mask` selects nothing."""
        idx = np.where(mask)[0]
        if not len(idx):
            return None
        centroid = self.dpi_pix[idx].mean(axis=0)
        med = int(idx[np.argmin(((self.dpi_pix[idx] - centroid) ** 2).sum(1))])
        return self.roi[med]

    def global_frozen_roi(self):
        """The single session-global stabilization ROI (fig3's `stabilized`):
        centroid over valid samples inside the central CENTROID_RADIUS, falling
        back to the fixation window if nothing lands that centrally."""
        roi = self._centroid_roi(self.central & self.dpi_valid)
        if roi is None:
            roi = self._centroid_roi(self.fixation & self.dpi_valid)
        assert roi is not None, "no valid sample for the global stabilization gaze"
        return roi

    def build_frozen_stim(self, scope, factor, n_embedded):
        """Return `(stim, n_trials_frozen)` for a frozen-gaze scope.

        `stim` is float32 `(n_embedded, 1, 51, 51)`, pixel-normalized
        ((raw-127)/255) -- a drop-in for `dset['stim']` -- built by rendering in
        the raw 240 Hz frame and decimating to the model's 120 Hz frame, exactly
        as the training pipeline's `downsample_stimulus` does.

        scope='global' freezes every trial at one session-global centroid gaze,
        so the frozen retinal image is pixel-identical across trials (a true
        extraretinal-only control). scope='trial' freezes each trial at its own
        centroid gaze, so there is no image motion within a trial but the frozen
        image still differs between trials.

        Trials with no valid fixation sample keep their stored frames, matching
        fig3's stabilized render.
        """
        assert scope in ("trial", "global"), scope
        roi_global = self.global_frozen_roi() if scope == "global" else None

        frozen_raw = self.raw_stim.copy()
        n_frozen = 0
        for iT in np.unique(self.trial_inds):
            m = self.trial_inds == iT
            if not np.any(m & self.fixation & self.dpi_valid):
                continue
            if scope == "global":
                roi_f = roi_global
            else:
                roi_f = self._centroid_roi(m & self.central & self.dpi_valid)
                if roi_f is None:
                    roi_f = self._centroid_roi(m & self.fixation & self.dpi_valid)
                if roi_f is None:
                    continue
            for k in np.where(m)[0]:
                frozen_raw[k] = self._crop(self.content_id[k], roi_f[0, 0], roi_f[1, 0])
            n_frozen += 1

        keep = (self.n_raw // factor) * factor
        dec = frozen_raw[:keep:factor].astype(np.float32)
        stim = ((dec - 127.0) / 255.0)[:, None][:n_embedded]
        return stim.astype(np.float32), n_frozen

    # -- within-window stabilization ----------------------------------------
    def build_window_stabilized_cube(self, emb_indices, lags, factor):
        """Return `(cube, n_clamped)` for the within-window condition.

        `cube` is float32 `(N, n_lags, 1, 51, 51)`, pixel-normalized, holding for
        each prediction time t (model index `emb_indices[i]`) and each lag l:

            frame(t, l) = crop( content at model index t - l,  gaze ROI at t )

        i.e. each lag keeps its own screen content (the RSVP flash dynamics are
        untouched) but every lag is cropped at the LAST frame's gaze, so no eye
        motion survives inside the window. Model index t maps to raw index
        `factor * t` (guaranteed by `alignment_maxabs == 0`).

        `n_clamped` counts (t, l) pairs whose lag ran off the start of the
        session and was clamped to raw sample 0; lags crossing a *trial*
        boundary are rendered exactly (previous trial's content at the current
        gaze), not clamped.
        """
        emb_indices = np.asarray(emb_indices, dtype=int)
        lags = np.asarray(lags, dtype=int)
        cur_raw = emb_indices * factor
        lag_raw = lags * factor

        out = np.empty((len(emb_indices), len(lags), self.win_h, self.win_w),
                       dtype=np.uint8)
        n_clamped = 0
        for i, k in enumerate(cur_raw):
            row0 = self.roi[k, 0, 0]
            col0 = self.roi[k, 1, 0]
            content = k - lag_raw
            n_clamped += int((content < 0).sum())
            np.maximum(content, 0, out=content)
            for j, cid in enumerate(self.content_id[content]):
                out[i, j] = self._crop(cid, row0, col0)

        cube = ((out.astype(np.float32) - 127.0) / 255.0)[:, :, None]
        return cube, n_clamped

    def window_displacement_px(self, emb_indices, lags, factor):
        """Median |gaze displacement| (px) between the newest and each other
        frame of the window -- how much retinal motion the window condition
        removes.

        Pairs where either endpoint has invalid eye tracking (blink, lost track)
        are excluded: `dpi_pix` is unconstrained there and would otherwise
        dominate. Purely diagnostic -- the stabilized cube never reads a lag
        sample's gaze, only its content, so an invalid lag sample cannot corrupt
        the stimulus. Median rather than mean for the same robustness reason."""
        emb_indices = np.asarray(emb_indices, dtype=int)
        cur_raw = emb_indices * factor
        lag_raw = np.asarray(lags, dtype=int) * factor
        content = np.maximum(cur_raw[:, None] - lag_raw[None, :], 0)
        d = self.dpi_pix[content] - self.dpi_pix[cur_raw][:, None, :]
        ok = (self.dpi_valid[content] & self.dpi_valid[cur_raw][:, None]
              & np.isfinite(d).all(axis=-1))
        if not ok.any():
            return float("nan")
        return float(np.median(np.hypot(d[..., 0], d[..., 1])[ok]))
