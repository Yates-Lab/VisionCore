r"""TDD spec for the eye-position-distribution-matched LOTC estimator.

Ground truth comes from ``synthetic.py``: a unified stationary-GP rate field
with a per-cell spatial mask M(e). The (A1) violation is variable n_t; the
(A2) violation is a non-constant mask. The estimator under test,
``estimators.decompose``, takes a ``target`` distribution and must:

  * recover the closed-form 1-alpha^p under (A2) ('flat' mask) and target in
    {'naive', 'full'} (Direction 1: the actual viewing distribution),
  * recover the closed-form 1-alpha^p^2 under (A2) with target='central'
    (Direction 2: the close-pair distribution),
  * recover each direction's truth on non-homogeneous masks
    (central/eccentric/linear),
  * be MUCH less biased than the naive estimator on a centrally-modulated cell,
  * be MORE STABLE for target='central' than 'full' on an eccentric-modulated
    cell (the unbounded-1/p tail-variance cost of Direction 1),
  * remove independent Poisson noise so a pure-Poisson cell has Fano ~ 1.

Section "Random-field sanity check" verifies that McFarland's estimator
(target='naive', constant n_t) recovers the analytical 1-alpha^p across an
ell/sigma sweep -- the regime McFarland claimed but the additive synthetic
could not test.

Run:  uv run --with pytest pytest test_estimators.py -q   (from this folder)
"""
import numpy as np

from synthetic import make_session, make_trajectory_session, ground_truth
from estimators import decompose, decompose_trajectory

NTR, NPH, SIG = 600, 100, 0.15

# Trajectory-mode test budget: smaller than the single-bin tests to keep the
# combined runtime manageable while still giving a stable seed mean.
T_NPER, T_NPH, T_TWIN = 25, 30, 5


def _seed_stats(kinds, target, key, seeds=range(6), deterministic=True,
                density="gaussian", threshold=0.05, **mk):
    """Per-cell mean and std of a decompose() field across seeds."""
    vals = []
    for s in seeds:
        sess = make_session(kinds, n_trials=NTR, n_time_bins=NPH, sigma_eye=SIG,
                            seed=s, **mk)
        counts = sess["rate"] if deterministic else sess["spikes"]
        d = decompose(counts, sess["eye"], target=target, density=density,
                      threshold=threshold)
        vals.append(d[key])
    vals = np.array(vals, float)
    return np.nanmean(vals, 0), np.nanstd(vals, 0), sess


# ---------------------------------------------------------------------------
# Eye-position-distribution matching (Extension 2)
# ---------------------------------------------------------------------------

def test_homogeneous_mask_correction_is_noop_for_full_target():
    """(A2) holds (flat mask). Naive and full targets both recover the
    closed-form 1-alpha^p; central recovers 1-alpha^p^2 (a different but
    consistent target). All three are consistent with their respective truths.
    Averaged across seeds because single-seed estimates have non-trivial
    finite-sample variance under the random-field model (the cross-term
    2*mu_0*overline{M*s_t} in Crate is the main source).
    """
    oma_naive, _, sess = _seed_stats(["flat"], "naive", "one_minus_alpha")
    oma_full, _, _ = _seed_stats(["flat"], "full", "one_minus_alpha")
    oma_cent, _, _ = _seed_stats(["flat"], "central", "one_minus_alpha")
    gt_p = sess["truth"][0]["p"]["one_minus_alpha"]
    gt_p2 = sess["truth"][0]["p2"]["one_minus_alpha"]
    assert abs(oma_naive[0] - gt_p) < 0.06, \
        f"naive {oma_naive[0]} not near gt_p {gt_p}"
    assert abs(oma_full[0] - gt_p) < 0.06, \
        f"full {oma_full[0]} not near gt_p {gt_p}"
    assert abs(oma_cent[0] - gt_p2) < 0.06, \
        f"central {oma_cent[0]} not near gt_p^2 {gt_p2}"


def test_full_target_recovers_p_decomposition():
    """Direction 1: target='full' recovers the closed-form p decomposition on
    non-homogeneous masks (central/eccentric/linear)."""
    kinds = ["central", "eccentric", "linear"]
    oma, _, sess = _seed_stats(kinds, "full", "one_minus_alpha")
    gt = np.array([sess["truth"][c]["p"]["one_minus_alpha"]
                   for c in range(len(kinds))])
    assert np.allclose(oma, gt, atol=0.08), f"got {oma}, want {gt}"


def test_central_target_recovers_p2_decomposition():
    """Direction 2: target='central' recovers the closed-form p^2 decomposition.

    Tested on masks whose spatial scale is broad enough relative to the
    close-pair threshold (eccentric, linear) that finite-threshold smoothing
    is small; the narrow central mask is exercised separately in the
    threshold-shrinks test.
    """
    kinds = ["eccentric", "linear"]
    oma, _, sess = _seed_stats(kinds, "central", "one_minus_alpha")
    gt = np.array([sess["truth"][c]["p2"]["one_minus_alpha"]
                   for c in range(len(kinds))])
    assert np.allclose(oma, gt, atol=0.08), f"got {oma}, want {gt}"


def test_finite_threshold_bias_shrinks_with_threshold():
    """Under a narrow central mask the residual is finite-threshold smoothing:
    the recovery improves as the threshold shrinks."""
    kinds = ["central"]
    oma_05, _, sess = _seed_stats(kinds, "central", "one_minus_alpha",
                                  threshold=0.05)
    oma_03, _, _ = _seed_stats(kinds, "central", "one_minus_alpha",
                               threshold=0.03)
    gt = sess["truth"][0]["p2"]["one_minus_alpha"]
    assert abs(oma_03[0] - gt) < abs(oma_05[0] - gt), \
        (f"thr=0.03 err {abs(oma_03[0]-gt):.3f} "
         f"!< thr=0.05 err {abs(oma_05[0]-gt):.3f}")


def test_naive_is_grossly_biased_where_corrected_is_not():
    """On a centrally-modulated cell the naive (unmatched) estimator's typical
    error is much larger than the matched estimator's, against truth on p.
    Uses MEDIAN over 12 seeds: the cross-term 2*mu_0*overline{M*s_t} in Crate
    creates heavy-tailed seed-to-seed fluctuations (occasional Crate ~ 0 -> alpha
    clipped to 1, 1-alpha = 0), and the mean is sensitive to those outliers."""
    naive_vals, full_vals = [], []
    gt = None
    for s in range(12):
        sess = make_session(["central"], n_trials=NTR, n_time_bins=NPH,
                            sigma_eye=SIG, seed=s)
        if gt is None:
            gt = sess["truth"][0]["p"]["one_minus_alpha"]
        d_naive = decompose(sess["rate"], sess["eye"], target="naive",
                            density="gaussian")
        d_full = decompose(sess["rate"], sess["eye"], target="full",
                           density="gaussian")
        naive_vals.append(d_naive["one_minus_alpha"][0])
        full_vals.append(d_full["one_minus_alpha"][0])
    err_naive = abs(float(np.nanmedian(naive_vals)) - gt)
    err_full = abs(float(np.nanmedian(full_vals)) - gt)
    assert err_full < 0.5 * err_naive, \
        f"full err {err_full:.3f} not << naive err {err_naive:.3f}"


def test_direction2_is_more_stable_than_direction1_for_eccentric():
    """Direction 1's unbounded 1/p weights make it noisier in the periphery,
    where an eccentric-modulated cell's variance lives; Direction 2 (bounded
    weights) is steadier across seeds.

    Threshold and seed choices reflect McFarland M6 Cpsth (the default after
    the M6/split-half switch): a sweep over thresholds {0.03, 0.02, 0.015,
    0.01} at 8 seeds gives std_cent < std_full clearly at 0.02 (std 0.039 vs
    0.060) but only marginally at 0.03 (0.052 vs 0.053) where seed-sample
    noise on a 6-seed SD estimate routinely flips the ordering. At very tight
    thresholds (≤0.015) Direction 1's std actually DROPS (peripheral close
    pairs become too rare to contribute, and the moderate-eccentricity central
    region dominates) while Direction 2's stays roughly flat -- so the
    §4.5 stability gap is largest at moderate thresholds, not tight ones.
    Pin the test at the regime where the §4.5 argument bites cleanly.
    """
    _, std_full, _ = _seed_stats(["eccentric"], "full", "one_minus_alpha",
                                 threshold=0.02, seeds=range(12))
    _, std_cent, _ = _seed_stats(["eccentric"], "central", "one_minus_alpha",
                                 threshold=0.02, seeds=range(12))
    assert std_cent[0] < std_full[0], \
        f"central std {std_cent[0]} !< full std {std_full[0]}"


def test_full_target_removes_poisson_fano_to_one():
    """Pure Poisson + non-homogeneous masks: the matched 'full' estimator gives
    Fano ~ 1; the naive estimator is biased further from 1."""
    kinds = ["central", "eccentric", "linear", "central"]
    fano_full, _, _ = _seed_stats(kinds, "full", "fano", seeds=range(8),
                                  deterministic=False)
    fano_naive, _, _ = _seed_stats(kinds, "naive", "fano", seeds=range(8),
                                   deterministic=False)
    med_full = np.nanmedian(fano_full)
    med_naive = np.nanmedian(fano_naive)
    assert abs(med_full - 1.0) < 0.15, f"full Fano median {med_full}"
    assert abs(med_full - 1.0) < abs(med_naive - 1.0), \
        f"full {med_full} not closer to 1 than naive {med_naive}"


def test_naive_path_matches_existing_pipeline():
    """Sanity: target='naive' reproduces the existing close-pair pipeline
    (pipeline_one_minus_alpha) on the same deterministic rates."""
    from VisionCore.covariance import pipeline_one_minus_alpha
    sess = make_session(["central", "eccentric", "linear"], n_trials=200,
                        n_time_bins=100, sigma_eye=SIG, seed=3)
    rate, eye = sess["rate"], sess["eye"]
    d = decompose(rate, eye, target="naive", density="gaussian")
    # NB: production pipeline (VisionCore/covariance.py) still uses the
    # original 'min_trials_per_phase' kwarg name; do not touch that here.
    ref = pipeline_one_minus_alpha(rate, eye, threshold=0.05,
                                   min_trials_per_phase=10, device="cpu")
    a, b = d["one_minus_alpha"], ref["one_minus_alpha"]
    ok = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(a[ok], b[ok], atol=0.05), f"{a} vs {b}"


# ---------------------------------------------------------------------------
# Extension 1: consistent time-bin weighting under variable n_t
# ---------------------------------------------------------------------------

def _staircase_n_t(n_time_bins, lo=20, hi=80):
    """Monotone-decaying per-bin trial counts mimicking fixRSVP fixation
    durations."""
    return np.linspace(hi, lo, n_time_bins).round().astype(int)


def test_variable_n_t_session_masks_correctly():
    """Sanity: passing n_trials_per_time_bin produces a valid-mask whose row sums
    equal the requested per-time-bin counts and whose invalid entries are NaN."""
    nt = _staircase_n_t(20, lo=10, hi=40)
    sess = make_session(["flat"], n_trials=int(nt.max()), n_time_bins=20,
                        sigma_eye=SIG, seed=0, n_trials_per_time_bin=nt)
    assert sess["valid"].shape == (int(nt.max()), 20)
    assert np.array_equal(sess["valid"].sum(0), nt)
    assert np.all(np.isnan(sess["spikes"][~sess["valid"]]))
    assert np.all(np.isfinite(sess["spikes"][sess["valid"]]))


def test_pair_count_and_uniform_directions_both_recover_truth_under_variable_nt():
    """Both consistent w_t directions (pair-count and uniform) recover the
    closed-form 1-alpha^p under variable n_t + envelope on flat-mask + mixed
    masks.

    The truth 1-alpha is invariant under w_t in the unified model (envelope
    cancels in the ratio; §A.5), so both directions target the same value.
    They differ in finite-sample efficiency, which is exhibited in Fig 1C
    with many more seeds; here we only verify both are unbiased.
    """
    nt = _staircase_n_t(NPH, lo=15, hi=int(NTR * 0.6))
    env = np.linspace(1.0, 0.05, NPH)
    kinds = ["flat", "central", "linear"]
    truth = None
    oma_pair, oma_uni = [], []
    for s in range(6):
        sess = make_session(kinds, n_trials=NTR, n_time_bins=NPH, sigma_eye=SIG,
                            seed=s, n_trials_per_time_bin=nt, psth_envelope=env)
        if truth is None:
            truth = np.array([sess["truth"][c]["p"]["one_minus_alpha"]
                              for c in range(len(kinds))])
        d_pair = decompose(sess["rate"], sess["eye"], target="full",
                           density="gaussian", time_bin_weighting="pair_count")
        d_uni = decompose(sess["rate"], sess["eye"], target="full",
                          density="gaussian", time_bin_weighting="uniform")
        oma_pair.append(d_pair["one_minus_alpha"])
        oma_uni.append(d_uni["one_minus_alpha"])
    m_pair = np.nanmean(oma_pair, 0)
    m_uni = np.nanmean(oma_uni, 0)
    err_pair = np.abs(m_pair - truth)
    err_uni = np.abs(m_uni - truth)
    assert np.all(err_pair < 0.10), \
        f"pair_count off truth: got {m_pair}, want {truth}"
    assert np.all(err_uni < 0.10), \
        f"uniform off truth: got {m_uni}, want {truth}"


def test_estimator_diagonals_recover_closed_form_under_variable_nt():
    """All three estimators (Ctotal, Cpsth, Crate) on flat-mask diagonals
    recover the closed-form values for BOTH consistent w_t directions under
    variable n_t + envelope, with deterministic rates.

    Under (A2)+flat the closed-form diagonals are:
        Ctotal = Crate = E_w[alpha^2] * tau^2
        Cpsth          = E_w[alpha^2] * tau^2 * ell^2 / (ell^2 + 2 sigma^2)

    Since the envelope is correlated with n_t, E_{w_pair}[alpha^2] differs
    from E_{w_uni}[alpha^2], so pair-count and uniform target DIFFERENT
    diagonal values -- and each must match its own closed form. This pins
    the §1.5 table's w_t column: every estimator respects its w_t parameter.
    """
    nt = _staircase_n_t(NPH, lo=15, hi=int(NTR * 0.6))
    env = np.linspace(1.0, 0.05, NPH)
    tau = 1.0
    ell = SIG
    sig = SIG

    w_pair = nt * (nt - 1) / 2.0
    w_pair = w_pair / w_pair.sum()
    w_uni = np.ones(NPH) / NPH
    E_a2_pair = float((w_pair * env**2).sum())
    E_a2_uni = float((w_uni * env**2).sum())
    I_MKD = tau**2 * ell**2 / (ell**2 + 2.0 * sig**2)

    truth = {
        "pair_count": {"Ctotal": E_a2_pair * tau**2,
                       "Cpsth":  E_a2_pair * I_MKD,
                       "Crate":  E_a2_pair * tau**2},
        "uniform":    {"Ctotal": E_a2_uni  * tau**2,
                       "Cpsth":  E_a2_uni  * I_MKD,
                       "Crate":  E_a2_uni  * tau**2},
    }

    # Sanity: envelope/n_t correlation produces genuinely different truths.
    assert abs(truth["pair_count"]["Ctotal"] - truth["uniform"]["Ctotal"]) > 0.1

    acc = {pw: {k: [] for k in ("Ctotal", "Cpsth", "Crate")}
           for pw in ("pair_count", "uniform")}
    for s in range(6):
        sess = make_session(["flat"], n_trials=NTR, n_time_bins=NPH,
                            sigma_eye=sig, ell=ell, tau=tau, seed=s,
                            n_trials_per_time_bin=nt, psth_envelope=env)
        for pw in acc:
            d = decompose(sess["rate"], sess["eye"], target="full",
                          density="gaussian", time_bin_weighting=pw)
            for k in acc[pw]:
                acc[pw][k].append(float(d[k][0, 0]))

    # Tolerance matches Ext-1's existing 1-alpha tolerance; the seed-mean SE of
    # Ctotal at pair_count under heavy concentration on early bins is ~0.06 with
    # 6 seeds (effective T is small), so 0.05 was too tight.
    tol = 0.10
    fails = []
    for pw in acc:
        for k in acc[pw]:
            m = float(np.mean(acc[pw][k]))
            t = truth[pw][k]
            if abs(m - t) >= tol:
                fails.append(f"{pw}/{k}: got {m:.4f}, expected {t:.4f} closed form")
    assert not fails, "; ".join(fails)


# ---------------------------------------------------------------------------
# Random-field sanity check: McFarland recovers analytical 1-alpha^p under (A2)
# ---------------------------------------------------------------------------

def _seed_stats_sweep(target, key, ell, sigma_eye=SIG, n_trials=NTR,
                      seeds=range(6), threshold=0.05):
    """Per-seed decompose() values on a single flat-mask cell at the requested
    ell. Returns (mean, std, gt_p, gt_p2) over the seeds."""
    vals = []
    for s in seeds:
        sess = make_session(["flat"], n_trials=n_trials, n_time_bins=NPH,
                            sigma_eye=sigma_eye, ell=ell, seed=s)
        d = decompose(sess["rate"], sess["eye"], target=target,
                      density="gaussian", threshold=threshold)
        vals.append(d[key])
    vals = np.array(vals, float)
    gt_p = sess["truth"][0]["p"]["one_minus_alpha"]
    gt_p2 = sess["truth"][0]["p2"]["one_minus_alpha"]
    return np.nanmean(vals, 0)[0], np.nanstd(vals, 0)[0], gt_p, gt_p2


def test_random_field_closed_form_one_minus_alpha():
    """The closed-form 1-alpha^p = 2 sigma^2 / (ell^2 + 2 sigma^2) and
    1-alpha^p^2 = sigma^2 / (ell^2 + sigma^2) match ``ground_truth`` for
    'flat' across an ell/sigma sweep that covers (0, 1)."""
    sig = SIG
    for r in (0.5, 1.0, 2.0, 4.0):
        ell = r * sig
        gt = ground_truth("flat", sig, ell=ell)
        oma_p_ref = 2.0 * sig**2 / (ell**2 + 2.0 * sig**2)
        oma_p2_ref = sig**2 / (ell**2 + sig**2)
        assert abs(gt["p"]["one_minus_alpha"] - oma_p_ref) < 1e-9, \
            f"r={r}: gt_p {gt['p']['one_minus_alpha']} != ref {oma_p_ref}"
        assert abs(gt["p2"]["one_minus_alpha"] - oma_p2_ref) < 1e-9, \
            f"r={r}: gt_p2 {gt['p2']['one_minus_alpha']} != ref {oma_p2_ref}"


def test_random_field_var_total_is_K_zero_under_A2():
    """Under (A2) the total rate variance equals K(0) = tau^2 -- a key
    D-invariance the rest of the closed form relies on. Verify empirically
    on the synthesized rates."""
    sig, ell, tau = SIG, SIG, 1.0
    sess = make_session(["flat"], n_trials=NTR, n_time_bins=NPH, sigma_eye=sig,
                        ell=ell, tau=tau, seed=0)
    r = sess["rate"][..., 0]
    var_total = float(np.nanvar(r))
    assert abs(var_total - tau**2) < 0.05, \
        f"empirical Var_total {var_total} != tau^2 {tau**2}"


def test_mcfarland_recovers_one_minus_alpha_p_under_A2():
    """McFarland's estimator (decompose(target='naive'), constant n_t, no
    envelope) on the random-field synthetic recovers the analytical
    1-alpha^p = 2 sigma^2 / (ell^2 + 2 sigma^2) across an ell/sigma sweep --
    the regime McFarland claimed."""
    sig = SIG
    for r in (0.5, 1.0, 2.0, 4.0):
        ell = r * sig
        mean, _, gt_p, _ = _seed_stats_sweep("naive", "one_minus_alpha", ell)
        assert abs(mean - gt_p) < 0.06, \
            f"r={r}: McFarland estimate {mean:.3f} != truth {gt_p:.3f}"


# ---------------------------------------------------------------------------
# §4.6 trajectory-mode estimator (multi-bin eye trajectories)
# ---------------------------------------------------------------------------

def _traj_threshold(sigma_drift, base=0.05):
    """Inflate the RMS-trajectory close-pair threshold so the effective
    centroid-distance threshold stays ~constant across sigma_drift values.

    E[RMS^2] for two trajectories with centroid distance d and per-bin drift
    sigma_drift is d^2 + 4 sigma_drift^2; pick thr so the centroid-matching
    radius is `base` regardless of sigma_drift.
    """
    return float(np.sqrt(base ** 2 + 4.0 * float(sigma_drift) ** 2))


def _traj_seed_stats(kinds, target, sigma_drift, seeds=range(6),
                     n_per=T_NPER, n_T=T_NPH, t_window=T_TWIN, base_thr=0.05,
                     **mk):
    """Per-cell mean and std of decompose_trajectory across seeds."""
    thr = _traj_threshold(sigma_drift, base=base_thr)
    vals = []
    sess = None
    for s in seeds:
        sess = make_trajectory_session(
            kinds, n_samples_per_time_bin=n_per, n_time_bins=n_T,
            t_window=t_window, sigma_eye=SIG, sigma_drift=sigma_drift,
            seed=s, **mk)
        d = decompose_trajectory(sess["counts"], sess["trajectories"],
                                 sess["T_idx"], target=target, threshold=thr)
        vals.append(d["one_minus_alpha"])
    vals = np.array(vals, float)
    return np.nanmean(vals, 0), np.nanstd(vals, 0), sess


def test_trajectory_flat_limit_recovers_truth():
    """sigma_drift = 0: trajectories collapse to their centroid, the
    pooled-per-bin KDE coincides with the centroid KDE, and the trajectory-mode
    estimator must recover §4.4's centroid-distribution 1-alpha on a mix of
    homogeneous and non-homogeneous masks. This is the regime where the
    flat-trajectory approximation is exact, so the recovery should match the
    single-bin §4.4 tests' tolerance."""
    kinds = ["flat", "central", "eccentric", "linear"]
    oma_full, _, sess = _traj_seed_stats(kinds, "full", sigma_drift=0.0)
    oma_cent, _, _ = _traj_seed_stats(kinds, "central", sigma_drift=0.0)
    gt_p = np.array([sess["traj_truth"][c]["p"]["one_minus_alpha"]
                     for c in range(len(kinds))])
    gt_p2 = np.array([sess["traj_truth"][c]["p2"]["one_minus_alpha"]
                      for c in range(len(kinds))])
    assert np.allclose(oma_full, gt_p, atol=0.08), \
        f"flat-limit full got {oma_full}, want {gt_p}"
    assert np.allclose(oma_cent, gt_p2, atol=0.10), \
        f"flat-limit central got {oma_cent}, want {gt_p2}"


def test_trajectory_moderate_drift_recovers_per_bin_marginal_truth():
    """sigma_drift = sigma/5: the per-bin marginal is N(0, (sigma^2 + sigma_drift^2) I),
    slightly broader than the centroid distribution. The pooled-per-bin KDE
    targets that per-bin marginal directly, so the estimator should recover
    `traj_truth` (the §4.4 closed form at sigma_traj = sqrt(sigma^2 + sigma_drift^2))
    within a slightly looser tolerance than the flat limit. Tests the
    smoothing-doesn't-blow-up regime."""
    kinds = ["flat", "central", "linear"]
    sd = SIG / 5.0
    oma_full, _, sess = _traj_seed_stats(kinds, "full", sigma_drift=sd)
    gt_p = np.array([sess["traj_truth"][c]["p"]["one_minus_alpha"]
                     for c in range(len(kinds))])
    err = np.abs(oma_full - gt_p)
    assert np.all(err < 0.12), \
        f"moderate-drift full got {oma_full}, want {gt_p} (err {err})"


def test_trajectory_strong_drift_documents_bias():
    """sigma_drift = sigma: the per-bin pooled KDE is heavily smoothed relative
    to the centroid distribution, AND the inflated close-pair RMS threshold
    captures most pairs (close-pair filter is no longer selective). The
    estimator under-states 1-alpha by ~0.3 in this regime -- the flat-trajectory
    approximation is broken and the importance reweighting is barely applied.
    This test documents the degradation by asserting the bias is bounded
    (not unbounded); the actual ~0.3 under-shoot is the writeup §4.6 caveat
    that the estimator is meaningful only for sigma_drift << sigma_eye."""
    sd = SIG  # full sigma_drift
    oma_full, _, sess = _traj_seed_stats(["flat"], "full", sigma_drift=sd,
                                         seeds=range(4))
    gt_p = sess["traj_truth"][0]["p"]["one_minus_alpha"]
    err = abs(float(oma_full[0]) - gt_p)
    # Bound is generous: the estimator is approximate at this drift level
    # (empirically err ~ 0.3). A 2x regression (err > 0.6) would signal a
    # structural problem (e.g. the close-pair filter degenerating to "all
    # pairs", or the KDE returning NaN at the centroid).
    assert err < 0.45, \
        f"strong-drift bias unexpectedly large: got {oma_full[0]} vs truth {gt_p}"


def test_trajectory_naive_is_biased_where_matched_is_not_on_central():
    """Naive trajectory-mode estimator (no importance reweighting on close pairs)
    still over-states 1-alpha on a centrally-modulated cell -- the same §4.3
    mechanism applies in the multi-bin setting because the close-pair midpoint
    density is still concentrated where M^2 is large. The matched 'full'
    estimator recovers truth at modest drift (sigma/5)."""
    sd = SIG / 5.0
    oma_naive, _, sess = _traj_seed_stats(["central"], "naive", sigma_drift=sd)
    oma_full, _, _ = _traj_seed_stats(["central"], "full", sigma_drift=sd)
    gt_p = sess["traj_truth"][0]["p"]["one_minus_alpha"]
    err_naive = abs(float(oma_naive[0]) - gt_p)
    err_full = abs(float(oma_full[0]) - gt_p)
    assert err_full < err_naive, \
        f"matched err {err_full:.3f} !< naive err {err_naive:.3f} (truth {gt_p:.3f})"


def test_t_floor_on_sd_one_minus_alpha():
    """Appendix A.6: the across-time-bin floor on sd[1-alpha-hat] is

        sd_floor = alpha* * sqrt(2 / (T - 1))

    where alpha* = ell^2 / (ell^2 + 2 sigma^2) under the flat-mask synthetic.
    Derivation: under (A2) and constant n_t, the per-time-bin projection
    G_t = integral M(e) s_t(e) p(e) de is iid N(0, V_p) with V_p = alpha*tau^2,
    so the sample-variance estimator of V_p has std 2 V_p^2 / (T-1), giving
    sd[alpha-hat] = sd[V_p-hat]/tau^2 = alpha* sqrt(2/(T-1)) in the large-N
    limit.

    Test at (N=400, T=200, ell=sigma): the analytical floor is ~0.033;
    empirical sd[1-alpha-hat] over 10 seeds sits in [0.5 floor, 4 floor].
    The upper bound is generous so within-bin noise (which adds
    quadratically) does not trip the test; the lower bound guards against
    a formula sign / 2x error in the derivation. Catches a 10x regression.
    """
    sig = SIG
    ell, N, T = sig, 400, 200
    a_star = ell ** 2 / (ell ** 2 + 2.0 * sig ** 2)
    floor = a_star * np.sqrt(2.0 / (T - 1))
    vals = []
    for s in range(10):
        sess = make_session(["flat"], n_trials=N, n_time_bins=T, sigma_eye=sig,
                            ell=ell, seed=s)
        d = decompose(sess["rate"], sess["eye"], target="naive",
                      density="gaussian", threshold=0.05)
        vals.append(float(d["one_minus_alpha"][0]))
    sd = float(np.nanstd(vals))
    assert sd > 0.5 * floor, \
        f"empirical sd {sd:.4f} suspiciously below T-floor {floor:.4f}"
    assert sd < 4.0 * floor, \
        f"empirical sd {sd:.4f} much above T-floor {floor:.4f}"
