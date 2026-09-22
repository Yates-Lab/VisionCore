"""Export figure-2 panel data for the general-exam oral deck.

Panels A-I of the FEM/V1 manuscript figure 2, reduced to the arrays each one
actually plots. The talk uses these as a staged reveal:

    A  two-trial eye traces + spike rates      the raw observation
    B  unaccounted-variance vs mismatch curve  how rate variance is estimated
    D  covariance decomposition matrices       the law of total covariance
    C  f_FEM histogram                         how much of the modulation is FEM
    E  population Fano factor                  consequence for coding
    F  pairwise noise correlation              consequence for coding
    G  participation ratio per component       the FEM component is compact
    I  subspace alignment vs shuffle           and it shares the stimulus's dims

Panel H is a data-free schematic and is redrawn on the deck side rather than
exported.

G and I are the only panels here that need the SUBJECT-FILTERED bundle: they are
per-session quantities, so the omitted subject has to be dropped before the
across-session mean means anything. Both therefore go through
``_filter_subjects``/``_pool_subjects_for_plotting`` (the same preparation
``generate_figure2.compose`` does) rather than the raw bundle the other
exporters read.

Everything comes from the derived bundle (``covdecomp_derived.pkl``) plus the
example-trial cache; no recomputation, no GPU, no raw sessions.

Panel E also has a companion export, ``E'``/``fig2_e_fano_scatter``: the
per-neuron points the Fano slope is fitted to, which the oral deck animates.

Usage (on solo):
    uv run --directory ~/v1-fovea/VisionCore ryan/general-exam-export/export_fig2.py
    uv run --directory ~/v1-fovea/VisionCore ryan/general-exam-export/export_fig2.py e_scatter
    uv run --directory ~/v1-fovea/VisionCore ryan/general-exam-export/export_fig2.py g i
"""

from __future__ import annotations

import numpy as np

from _export_common import add_paper_path, save_panel

add_paper_path("fig2")

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")

from VisionCore.covariance import project_to_psd              # noqa: E402
from VisionCore.paths import CACHE_DIR                        # noqa: E402
from compute_fig2_data import load_fig2_data                  # noqa: E402
from compute_fig2_data import _slope_through_origin           # noqa: E402
from compute_fig2_data import compute_alignment_aggregate     # noqa: E402
import generate_panel_example as ex                           # noqa: E402
from generate_panel_fano import _per_session_slopes           # noqa: E402
from generate_figure2 import TARGET_SESSION, WINDOW_IDX       # noqa: E402
from generate_figure2 import _filter_subjects                 # noqa: E402
from generate_figure2 import _pool_subjects_for_plotting      # noqa: E402


DERIVED_CACHE = CACHE_DIR / "covdecomp_derived.pkl"
EMPIRICAL_CACHE = CACHE_DIR / "covdecomp_empirical.pkl"
WINDOW_MS = 25.0            # the fig2 standard counting window


def _nearest_window(windows_ms, target):
    arr = np.asarray(windows_ms, dtype=float)
    return windows_ms[int(np.argmin(np.abs(arr - float(target))))]


# ── A: two-trial eye traces + spike rates ───────────────────────────
def export_panel_a():
    pair = ex._load_trial_pair(ex.SESSION)
    robs, eyepos = pair["robs"], pair["eyepos"]
    j = int(np.where(np.asarray(pair["neuron_mask"]) == ex.EXAMPLE_UNIT)[0][0])

    W = ex.WINDOW_BINS
    t_ms = np.arange(W) * ex.DT * 1000.0
    e_a = eyepos[ex.TRIAL_A, :W, ex.EYE_AXIS]
    e_b = eyepos[ex.TRIAL_B, :W, ex.EYE_AXIS]
    t_spk, r_a = ex._binned_rate(robs[ex.TRIAL_A, :W, j])
    _, r_b = ex._binned_rate(robs[ex.TRIAL_B, :W, j])

    save_panel(
        "fig2_a_example",
        {
            "t_ms": t_ms,
            "eye_a": e_a,
            "eye_b": e_b,
            "t_spk_ms": t_spk,
            "rate_a": r_a,
            "rate_b": r_b,
            # W1 = divergent, W2 = matched (bin indices into the trial window)
            "win_divergent_bins": np.asarray(ex.W1),
            "win_matched_bins": np.asarray(ex.W2),
            "delta_divergent_deg": ex._window_delta(e_a, e_b, ex.W1),
            "delta_matched_deg": ex._window_delta(e_a, e_b, ex.W2),
            "dt_s": ex.DT,
            "session": ex.SESSION,
            "unit": ex.EXAMPLE_UNIT,
            "trials": np.asarray([ex.TRIAL_A, ex.TRIAL_B]),
            "eye_axis": ex.EYE_AXIS,
        },
        source="paper/fig2/generate_panel_example.py:plot_eye_rate_example",
        caches=[EMPIRICAL_CACHE],
        notes="Eye axis 0 = horizontal. Rates binned at RATE_BIN_FACTOR "
              "(25 ms), matching the panel B counting window.",
    )


# ── B: unaccounted variance vs eye-trajectory mismatch ──────────────
def export_panel_b():
    d = ex._compute_unaccounted_curve()
    save_panel(
        "fig2_b_mismatch",
        {
            "bin_centers_deg": np.asarray(d["bin_centers"], float),
            "cum_crate": np.asarray(d["cum_crate"], float),
            "c_total": float(d["Ctotal"]),
            "c_rate": float(d["Crate"]),
            "c_psth": float(d["Cpsth"]),
            "sigma_int": float(d["sigma_int"]),
            "radius": float(d["radius"]),
            "unit": ex.EXAMPLE_UNIT,
            "session": ex.SESSION,
        },
        source="paper/fig2/generate_panel_example.py:_compute_unaccounted_curve",
        caches=[EMPIRICAL_CACHE],
        notes="Plot U = c_total - cum_crate against bin_centers_deg (x axis "
              "reversed). Reference levels: c_total, c_total - c_psth "
              "(eye-blind asymptote), sigma_int (internal-noise floor).",
    )


# ── C: f_FEM histogram ──────────────────────────────────────────────
def export_panel_c(data):
    w_idx = int(np.argmin(np.abs(np.asarray(data["WINDOWS_MS"], float) - WINDOW_MS)))
    m0 = np.asarray(data["m_by_window"][w_idx], float)
    labels = np.asarray(data["subject_per_neuron_by_window"][w_idx]).astype("U32")
    subjects = np.asarray(list(data["SUBJECTS"])).astype("U32")
    colors = np.asarray([data["SUBJECT_COLORS"][s] for s in data["SUBJECTS"]]).astype("U32")

    save_panel(
        "fig2_c_femfraction",
        {
            "f_fem": m0,
            "subject_per_neuron": labels,
            "subjects": subjects,
            "subject_colors": colors,
            "window_ms": WINDOW_MS,
        },
        source="paper/fig2/generate_panel_femfraction.py:plot_panel_c",
        caches=[DERIVED_CACHE],
        notes="Stacked histogram over subjects, 31 bins spanning the finite "
              "range; median marker per subject.",
    )


# ── D: covariance decomposition matrices ────────────────────────────
def export_panel_d(data):
    sr = next((s for s in data["session_results"]
               if s["session"] == TARGET_SESSION), None)
    if sr is None:
        avail = [s["session"] for s in data["session_results"]]
        raise ValueError(f"{TARGET_SESSION} not in bundle. Available: {avail}")

    mats = sr["mats"][WINDOW_IDX]
    crate_raw = mats["Intercept"]
    valid = np.isfinite(np.diag(crate_raw)) & np.isfinite(np.diag(mats["PSTH"]))
    ix = np.ix_(valid, valid)

    c_total = project_to_psd(mats["Total"][ix])
    c_psth = project_to_psd(mats["PSTH"][ix])
    c_fem = project_to_psd(crate_raw[ix] - mats["PSTH"][ix])
    c_resid = project_to_psd(mats["Total"][ix] - crate_raw[ix])
    c_resid_unc = project_to_psd(mats["Total"][ix] - mats["PSTH"][ix])

    save_panel(
        "fig2_d_covdecomp",
        {
            "c_total": c_total,
            "c_stimulus": c_psth,
            "c_fem": c_fem,
            "c_residual": c_resid,
            "c_residual_uncorrected": c_resid_unc,
            "vlim": 0.35 * float(np.nanmax(c_total)),
            "session": TARGET_SESSION,
            "window_idx": WINDOW_IDX,
            "n_units": int(c_total.shape[0]),
        },
        source="paper/fig2/generate_figure2.py:_plot_compact_cov_decomp",
        caches=[DERIVED_CACHE],
        notes="Colormap seismic_r, symmetric Normalize(-vlim, vlim). "
              "Top row: total = stimulus + uncorrected residual. "
              "Bottom row: total = stimulus + FEM + residual.",
    )


# ── E: population Fano factor ───────────────────────────────────────
def export_panel_e(data):
    w = _nearest_window(data["WINDOWS_MS"], WINDOW_MS)
    s = data["fano_stats"][w]
    su = np.asarray(s.get("sess_slope_unc", []), dtype=float)
    sc = np.asarray(s.get("sess_slope_cor", []), dtype=float)
    if su.size == 0 or sc.size == 0:
        su, sc = _per_session_slopes(s)

    null_lo, null_hi = s.get("mean_sess_cor_null_ci", (np.nan, np.nan))
    save_panel(
        "fig2_e_fano",
        {
            "slope_uncorrected": su,
            "slope_corrected": sc,
            "null_ci": np.asarray([null_lo, null_hi], float),
            "p_emp": float(s.get("p_emp_mean_sess", np.nan)),
            "n_shuffles": int(s.get("n_shuff_sess") or 0),
            "window_ms": float(w),
        },
        source="paper/fig2/generate_panel_fano.py:plot_fano_population",
        caches=[DERIVED_CACHE],
        notes="One faint line per session joining uncorrected -> corrected; "
              "across-session mean +/- SD overlaid; Poisson reference at 1.0.",
    )


# ── E': the per-neuron scatter the Fano slope is fitted to ──────────
def export_panel_e_scatter(data):
    """Per-neuron (mean spike count, residual variance) pairs.

    ``fig2_e_fano`` carries only the 19 per-session slopes; the oral deck
    animates the fit itself — the cloud of neurons dropping from ``var_u`` to
    ``var_c`` and the through-origin line rotating down with it — which needs
    the points the slope is computed from.
    """
    w = _nearest_window(data["WINDOWS_MS"], WINDOW_MS)
    s = data["fano_stats"][w]

    erate = np.asarray(s["erate"], dtype=float)
    var_u = np.asarray(s["var_u"], dtype=float)
    var_c = np.asarray(s["var_c"], dtype=float)

    save_panel(
        "fig2_e_fano_scatter",
        {
            "erate": erate,
            "var_uncorrected": var_u,
            "var_corrected": var_c,
            "session_per_neuron": np.asarray(s["session_per_neuron"]).astype(str),
            "subject_per_neuron": np.asarray(s["subject_per_neuron"]).astype(str),
            # Pooled through-origin slopes over exactly these points, so the
            # animated line is guaranteed to agree with the fit the manuscript
            # reports rather than being re-derived downstream.
            "slope_uncorrected": float(_slope_through_origin(erate, var_u)),
            "slope_corrected": float(_slope_through_origin(erate, var_c)),
            "window_ms": float(w),
        },
        source="paper/covariance_decomposition/derive.py:_compute_fano_stats",
        caches=[DERIVED_CACHE],
        notes="One point per neuron: residual variance (spk^2) against mean "
              "spike count in the counting window. Fano = var / erate; the "
              "population Fano is the through-origin slope sum(e*v)/sum(e^2). "
              "var_uncorrected conditions on the stimulus alone, var_corrected "
              "on stimulus and eye position.",
    )


# ── F: pairwise noise correlation ───────────────────────────────────
def export_panel_f(data):
    w = _nearest_window(data["WINDOWS_MS"], WINDOW_MS)
    s = data["nc_stats"][w]
    dr_lo, dr_hi = s["null_dr_ci"]
    r_u = float(s["r_u_mean"])

    save_panel(
        "fig2_f_noisecorr",
        {
            "rho_uncorrected": np.asarray(s["rho_u"], float),
            "rho_corrected": np.asarray(s["rho_c"], float),
            "mean_uncorrected": r_u,
            "mean_corrected": float(s["r_c_mean"]),
            "sd_uncorrected": float(s["r_u_sd"]),
            "sd_corrected": float(s["r_c_sd"]),
            "null_ci": np.asarray([r_u + dr_lo, r_u + dr_hi], float),
            "p_emp": float(s["p_emp_dr"]),
            "n_shuffles": int(s.get("n_shuff_dr") or 0),
            "window_ms": float(w),
        },
        source="paper/fig2/generate_panel_noisecorr.py:plot_nc_violin",
        caches=[DERIVED_CACHE],
        notes="Grey violins of the per-pair distribution, uncorrected vs "
              "FEM-corrected, with across-dataset mean +/- SD and a "
              "shuffle-null band on the corrected marker.",
    )


# ── G: participation ratio per covariance component ─────────────────
def export_panel_g(prepared):
    """Per-session participation ratio, one value per component.

    Exported in the MANUSCRIPT'S component order (residual, stimulus, FEM); the
    deck reorders to stimulus/FEM/residual so the axis reads in the same order
    as the four-term decomposition already on screen. Keeping the export in the
    upstream order means a future diff against panel G is a straight comparison.
    """
    pr_resid = np.asarray(prepared["pr_resid_list"], dtype=float)
    pr_psth = np.asarray(prepared["pr_psth_list"], dtype=float)
    pr_fem = np.asarray(prepared["pr_fem_list"], dtype=float)
    names = np.asarray(list(prepared.get("sub_names", []))).astype("U64")

    ok = np.isfinite(pr_resid) & np.isfinite(pr_psth) & np.isfinite(pr_fem)

    # Paired sign test on stimulus vs FEM, the comparison panel G annotates.
    # Reported here rather than recomputed on the deck side so the slide and the
    # manuscript quote one number from one place.
    from scipy.stats import binomtest, wilcoxon

    def _sign_p(a, b):
        d = a - b
        d = d[np.isfinite(d) & (d != 0)]
        if d.size == 0:
            return np.nan, 0, 0
        p = binomtest(int((d > 0).sum()), d.size, 0.5,
                      alternative="two-sided").pvalue
        return float(p), int((d > 0).sum()), int(d.size)

    p_sf, n_sf_pos, n_sf = _sign_p(pr_psth[ok], pr_fem[ok])

    def _wilcox_p(a, b):
        try:
            return float(wilcoxon(a, b).pvalue)
        except ValueError:
            return np.nan

    save_panel(
        "fig2_g_pr",
        {
            "pr_residual": pr_resid[ok],
            "pr_stimulus": pr_psth[ok],
            "pr_fem": pr_fem[ok],
            "session_names": names[ok] if names.size == ok.size else names,
            "p_wilcoxon_resid_vs_stimulus": _wilcox_p(pr_resid[ok], pr_psth[ok]),
            "p_wilcoxon_resid_vs_fem": _wilcox_p(pr_resid[ok], pr_fem[ok]),
            "p_signtest_stimulus_vs_fem": p_sf,
            "n_signtest_positive": n_sf_pos,
            "n_signtest": n_sf,
            "window_ms": WINDOW_MS,
        },
        source="paper/fig2/generate_figure2.py:_plot_pr_comparison",
        caches=[DERIVED_CACHE],
        notes="Participation ratio (tr S)^2 / tr(S^2) per session for the "
              "corrected residual, stimulus-locked, and FEM components. "
              "Subject-filtered bundle. Log y upstream; the sign test is the "
              "stimulus-vs-FEM comparison, Wilcoxon the two residual brackets.",
    )


# ── I: subspace alignment against the eye-trajectory shuffle ────────
def export_panel_i(prepared):
    """Both directions of the alignment measure, with the shuffle null.

    The inference is on the ACROSS-SESSION MEAN, and its null is built by
    ``compute_alignment_aggregate`` (average within session, then across); that
    aggregation is not something the deck should reimplement, so the aggregated
    null distribution is exported alongside the per-session observations.
    """
    agg = compute_alignment_aggregate(prepared)

    arrays = {
        "subspace_k": int(prepared.get("SUBSPACE_K", 4)),
        "n_sessions": int(agg["n_sessions"]),
        "window_ms": WINDOW_MS,
    }
    # x = stimulus variance lying in the FEM subspace
    # y = FEM variance lying in the stimulus subspace
    for tag, name in (("x", "stimulus_in_fem"), ("y", "fem_in_stimulus")):
        a = agg[tag]
        null_mean = np.asarray(a["null_mean"], float)
        null_mean = null_mean[np.isfinite(null_mean)]
        arrays[f"{name}_observed"] = np.asarray(a["observed"], float)
        arrays[f"{name}_mean"] = float(a["mean"])
        arrays[f"{name}_sd"] = float(a["sd"])
        arrays[f"{name}_null_ci"] = np.percentile(null_mean, [2.5, 97.5])
        arrays[f"{name}_null_mean"] = float(null_mean.mean())
        arrays[f"{name}_p_emp"] = float(a["p"])
        arrays[f"{name}_n_shuffles"] = int(a["n_shuff"])
        arrays[f"{name}_n_sig05"] = int(a["n_sig05"])

    save_panel(
        "fig2_i_alignment",
        arrays,
        source="paper/fig2/generate_figure2.py:"
               "_plot_subspace_alignment_vs_shuffle",
        caches=[DERIVED_CACHE],
        notes="Fraction of one component's variance lying in the other's "
              "leading-k subspace, both directions. Per-session observations, "
              "across-session mean +/- SD, and the 95% interval of the "
              "eye-trajectory shuffle null OF THE MEAN (not of single "
              "sessions). Subject-filtered bundle.",
    )


def main(only=()):
    """Export every panel, or just the named ones.

    ``only`` takes panel letters ("a", "e", "e_scatter", ...). Selective runs
    exist so a newly added panel can be exported without rewriting the
    already-snapshotted ones — re-exporting all of them would silently pick up
    any upstream cache change and move numbers already on slides.
    """
    want = {s.lower() for s in only}

    def run(letter):
        return not want or letter in want

    if run("a"):
        print("== panel A (example trials)")
        export_panel_a()
    if run("b"):
        print("== panel B (mismatch curve)")
        export_panel_b()

    derived = {"c", "d", "e", "e_scatter", "f", "g", "i"}
    if not want or (want & derived):
        print("== loading derived bundle (this is the slow part)")
        data = load_fig2_data()

    # G and I are per-session, so they read the subject-filtered/pooled bundle.
    # Prepared from the already-loaded `data` rather than by calling
    # `load_prepared_data()`, which would read the 7-8 GB cache a second time.
    prepared = None
    if not want or (want & {"g", "i"}):
        prepared = _pool_subjects_for_plotting(_filter_subjects(data))

    if run("c"):
        print("== panel C (f_FEM)")
        export_panel_c(data)
    if run("d"):
        print("== panel D (covariance decomposition)")
        export_panel_d(data)
    if run("e"):
        print("== panel E (Fano)")
        export_panel_e(data)
    if run("e_scatter"):
        print("== panel E' (Fano per-neuron scatter)")
        export_panel_e_scatter(data)
    if run("f"):
        print("== panel F (noise correlation)")
        export_panel_f(data)
    if run("g"):
        print("== panel G (participation ratio)")
        export_panel_g(prepared)
    if run("i"):
        print("== panel I (subspace alignment)")
        export_panel_i(prepared)


if __name__ == "__main__":
    import sys as _sys
    main(_sys.argv[1:])
