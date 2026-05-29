r"""TDD spec for the eye-position-distribution-matched LOTC estimator.

Ground truth comes from ``synthetic.py`` (closed-form decompositions under the full
fixational distribution p and the close-pair distribution p^2). The estimator under
test, ``estimators.decompose``, takes a ``target`` distribution and must:

  * be a NO-OP for a homogeneous stimulus (rate independent of absolute eye pos),
  * recover the p decomposition with target='full'  (Direction 1),
  * recover the p^2 decomposition with target='central' (Direction 2),
  * leave a much smaller error than the inconsistent NAIVE estimator,
  * be MORE STABLE for target='central' than 'full' on an eccentric-sensitive cell
    (the unbounded-1/p tail-variance cost of Direction 1),
  * remove independent Poisson noise so a pure-Poisson cell has Fano ~ 1.

Recovery tolerances allow ~0.06 for the residual finite-threshold smoothing that is
shared with the original McFarland estimator (it shrinks as the threshold shrinks).

Run:  uv run --with pytest pytest test_estimators.py -q   (from this folder)
"""
import numpy as np

from synthetic import make_session
from estimators import decompose

NTR, NPH, SIG = 600, 100, 0.15


def _seed_stats(kinds, target, key, seeds=range(6), deterministic=True,
                density="gaussian", threshold=0.05, **mk):
    """Per-cell mean and std of a decompose() field across seeds."""
    vals = []
    for s in seeds:
        sess = make_session(kinds, n_trials=NTR, n_phases=NPH, sigma_eye=SIG,
                            seed=s, **mk)
        counts = sess["rate"] if deterministic else sess["spikes"]
        d = decompose(counts, sess["eye"], target=target, density=density,
                      threshold=threshold)
        vals.append(d[key])
    vals = np.array(vals, float)
    return np.nanmean(vals, 0), np.nanstd(vals, 0), sess


def test_homogeneous_stimulus_correction_is_noop():
    """Flat profile -> rate independent of absolute eye position, so the eye
    distribution is irrelevant: 1-alpha is 0 under every target."""
    sess = make_session(["flat", "flat"], n_trials=NTR, n_phases=NPH,
                        sigma_eye=SIG, seed=0)
    for target in ("naive", "full", "central"):
        d = decompose(sess["rate"], sess["eye"], target=target, density="gaussian")
        assert np.allclose(d["one_minus_alpha"], 0.0, atol=0.05), target


def test_full_target_recovers_p_decomposition():
    """Direction 1: target='full' recovers the closed-form p decomposition."""
    kinds = ["central", "eccentric", "linear"]
    oma, _, sess = _seed_stats(kinds, "full", "one_minus_alpha")
    gt = np.array([sess["truth"][c]["p"]["one_minus_alpha"] for c in range(len(kinds))])
    assert np.allclose(oma, gt, atol=0.06), f"got {oma}, want {gt}"


def test_central_target_recovers_p2_decomposition():
    """Direction 2: target='central' recovers the closed-form p^2 decomposition.

    Tested on cells whose spatial feature is broad relative to the close-pair
    threshold (eccentric, linear); a feature narrower than the threshold is
    smoothed -- a shared finite-threshold limitation tested separately below.
    """
    kinds = ["eccentric", "linear"]
    oma, _, sess = _seed_stats(kinds, "central", "one_minus_alpha")
    gt = np.array([sess["truth"][c]["p2"]["one_minus_alpha"] for c in range(len(kinds))])
    assert np.allclose(oma, gt, atol=0.06), f"got {oma}, want {gt}"


def test_finite_threshold_bias_shrinks_with_threshold():
    """The residual under-estimation of 1-alpha is finite-threshold smoothing:
    a narrow central feature is recovered better as the threshold shrinks."""
    kinds = ["central"]
    oma_05, _, sess = _seed_stats(kinds, "central", "one_minus_alpha", threshold=0.05)
    oma_03, _, _ = _seed_stats(kinds, "central", "one_minus_alpha", threshold=0.03)
    gt = sess["truth"][0]["p2"]["one_minus_alpha"]
    assert abs(oma_03[0] - gt) < abs(oma_05[0] - gt), \
        f"thr=0.03 err {abs(oma_03[0]-gt):.3f} !< thr=0.05 err {abs(oma_05[0]-gt):.3f}"


def test_naive_is_grossly_biased_where_corrected_is_not():
    """The naive (unmatched) estimator's error on a central-sensitive cell is far
    larger than the matched estimator's."""
    kinds = ["central"]
    oma_naive, _, sess = _seed_stats(kinds, "naive", "one_minus_alpha")
    oma_full, _, _ = _seed_stats(kinds, "full", "one_minus_alpha")
    gt = sess["truth"][0]["p"]["one_minus_alpha"]
    err_naive = abs(np.nan_to_num(oma_naive[0], nan=1.0) - gt)
    err_full = abs(oma_full[0] - gt)
    assert err_full < 0.5 * err_naive, f"full err {err_full} not << naive err {err_naive}"


def test_direction2_is_more_stable_than_direction1_for_eccentric():
    """Direction 1's unbounded 1/p weights make it noisier in the periphery, where
    an eccentric cell's variance lives; Direction 2 (bounded weights) is steadier."""
    _, std_full, _ = _seed_stats(["eccentric"], "full", "one_minus_alpha", threshold=0.03)
    _, std_cent, _ = _seed_stats(["eccentric"], "central", "one_minus_alpha", threshold=0.03)
    assert std_cent[0] < std_full[0], f"central std {std_cent[0]} !< full std {std_full[0]}"


def test_full_target_removes_poisson_fano_to_one():
    """Pure Poisson + non-homogeneous rate: the matched 'full' estimator removes
    Poisson and the distribution contamination, giving Fano ~ 1; naive is worse."""
    kinds = ["central", "eccentric", "linear", "central"]
    fano_full, _, _ = _seed_stats(kinds, "full", "fano", seeds=range(8),
                                  deterministic=False)
    fano_naive, _, _ = _seed_stats(kinds, "naive", "fano", seeds=range(8),
                                   deterministic=False)
    med_full = np.nanmedian(fano_full)
    med_naive = np.nanmedian(fano_naive)
    assert abs(med_full - 1.0) < 0.12, f"full Fano median {med_full}"
    assert abs(med_full - 1.0) < abs(med_naive - 1.0), \
        f"full {med_full} not closer to 1 than naive {med_naive}"


def test_naive_path_matches_existing_pipeline():
    """Sanity: target='naive' reproduces the existing close-pair pipeline
    (pipeline_one_minus_alpha) on the same deterministic rates."""
    from VisionCore.covariance import pipeline_one_minus_alpha
    sess = make_session(["central", "eccentric", "linear"], n_trials=200,
                        n_phases=100, sigma_eye=SIG, seed=3)
    rate, eye = sess["rate"], sess["eye"]
    d = decompose(rate, eye, target="naive", density="gaussian")
    ref = pipeline_one_minus_alpha(rate, eye, threshold=0.05,
                                   min_trials_per_phase=10, device="cpu")
    a, b = d["one_minus_alpha"], ref["one_minus_alpha"]
    ok = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(a[ok], b[ok], atol=0.05), f"{a} vs {b}"
