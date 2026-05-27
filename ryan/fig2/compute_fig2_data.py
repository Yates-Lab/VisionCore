"""
Figure 2 precomputation: load raw data, run LOTC covariance decomposition,
then derive every per-window/per-panel statistic the fig2 panel scripts
need. Cached as a single bundle so panel scripts load instantly.

Bundle keys returned by ``load_fig2_data(refresh=False)``:
    session_results, metrics, m_by_window, subject_per_neuron_by_window,
    alpha_stats, fano_stats, nc_stats,
    sub_names, sub_subjects, pr_fem_list, pr_psth_list,
    overlap_k1_list, overlap_k_list, var_p_given_f, var_f_given_p,
    spectra_psth, spectra_fem,
    WINDOWS_MS, WINDOWS_BINS, SUBJECTS, SUBJECT_COLORS,
    session_names, subjects, n_sessions,
    SUBSPACE_WINDOW_IDX, SUBSPACE_K,
    config (dict of analysis parameters).

Two caches on disk:
    CACHE_DIR/fig2_decomposition.pkl   raw per-session decompositions
    CACHE_DIR/fig2_derived.pkl         derived bundle (everything above)

Set REFRESH=True at the top, pass refresh=True to load_fig2_data(), or
delete the cache file to force recompute.

Usage:
    from compute_fig2_data import load_fig2_data
    data = load_fig2_data()
"""
import sys
from pathlib import Path

import numpy as np
import dill

from VisionCore.paths import VISIONCORE_ROOT, CACHE_DIR
from VisionCore.covariance import (
    cov_to_corr,
    project_to_psd,
    get_upper_triangle,
    align_fixrsvp_trials,
    run_covariance_decomposition,
)
from VisionCore.stats import (
    geomean,
    iqr_25_75,
    bootstrap_mean_ci,
    fisher_z_mean,
    emp_p_one_sided,
    wilcoxon_signed_rank,
    paired_valid,
)
from VisionCore.subspace import (
    participation_ratio,
    symmetric_subspace_overlap,
    directional_variance_capture,
)
from DataYatesV1 import get_free_device


# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
REFRESH = False              # set True to force recompute of derived bundle
DT = 1 / 120                 # seconds per bin (native 240 Hz sampling)
WINDOW_BINS = [1, 2, 3, 6]   # counting windows in bins (6 @ 120 Hz = 50 ms = stim refresh)
N_SHUFFLES = 100             # shuffle null iterations
MIN_RATE_HZ = 2.0            # firing-rate inclusion threshold
MIN_PSTH_R2 = 0.05           # split-half PSTH R^2 inclusion threshold
N_PSTH_SPLITS = 100          # random halvings for split-half PSTH reliability
INTERCEPT_MODE = "below_threshold"
INTERCEPT_THRESHOLD = 0.05
INTERCEPT_KWARGS = {"threshold": INTERCEPT_THRESHOLD}
MIN_VAR = 0                  # minimum variance for correlation computation
EPS_RHO = 1e-3               # floor for correlation denominators
SUBJECTS = ["Allen", "Logan", "Luke"]
SUBJECT_COLORS = {"Allen": "tab:blue", "Logan": "tab:green", "Luke": "tab:orange"}

DATASET_CONFIGS_PATH = (
    VISIONCORE_ROOT / "experiments" / "dataset_configs" / "multi_basic_120_long.yaml"
)

SUBSPACE_WINDOW_IDX = 1      # second window (4 bins ≈ 16.67 ms)
SUBSPACE_K = 5

DECOMP_CACHE = CACHE_DIR / "fig2_decomposition.pkl"
DERIVED_CACHE = CACHE_DIR / "fig2_derived.pkl"


# ---------------------------------------------------------------------------
# Stage 1: per-session decomposition (cached as fig2_decomposition.pkl)
# ---------------------------------------------------------------------------

def _load_contam_rate(session_name, subject, n_neurons_total):
    """Per-neuron min contamination rate from QC data (or None)."""
    if subject in ("Allen", "Logan"):
        from DataYatesV1.utils.io import YatesV1Session
        try:
            sess = YatesV1Session(session_name)
            refractory = np.load(
                sess.sess_dir / 'qc' / 'refractory' / 'refractory.npz'
            )
            min_contam_props = refractory['min_contam_props']
            return np.array([
                np.min(min_contam_props[i]) for i in range(len(min_contam_props))
            ])
        except Exception as e:
            print(f"  Warning: Could not load QC data: {e}")
            return None
    raise NotImplementedError(
        f"QC loading not implemented for subject {subject}"
    )


def _compute_session_results():
    """Run LOTC decomposition for every session listed in the dataset config."""
    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))

    from models.config_loader import load_dataset_configs
    from models.data import prepare_data

    device = get_free_device()
    dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))
    session_results = []

    for cfg in dataset_configs:
        session_name = cfg["session"]
        subject = session_name.split("_")[0]
        if subject not in SUBJECTS:
            continue

        if "fixrsvp" not in cfg["types"]:
            cfg["types"] = cfg["types"] + ["fixrsvp"]

        print(f"\n--- {session_name} ({subject}) ---")
        try:
            train_data, val_data, cfg = prepare_data(cfg, strict=False)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        try:
            dset_idx = train_data.get_dataset_index("fixrsvp")
        except (ValueError, KeyError):
            print("  Skipping: no fixrsvp data")
            continue
        fixrsvp_dset = train_data.dsets[dset_idx]

        robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(
            fixrsvp_dset,
            valid_time_bins=120,
            min_fix_dur=20,
            min_total_spikes=0,
        )
        if robs is None or robs.shape[0] < 10:
            print(f"  Skipping: insufficient data ({meta})")
            continue
        print(f"  Trials: {meta['n_trials_good']}/{meta['n_trials_total']}, "
              f"Neurons: {meta['n_neurons_used']}/{meta['n_neurons_total']}")

        # Per-unit firing rate (Hz) from valid (fixation) bins.
        n_units = robs.shape[2]
        n_spikes_per_unit = np.nansum(robs, axis=(0, 1))
        n_valid_bins_per_unit = np.sum(np.isfinite(robs), axis=(0, 1))
        rate_hz = np.where(
            n_valid_bins_per_unit > 0,
            n_spikes_per_unit / np.maximum(n_valid_bins_per_unit, 1) / DT,
            np.nan,
        )

        # Per-unit split-half PSTH R^2 over N_PSTH_SPLITS random trial halvings.
        rng_r2 = np.random.default_rng(42)
        r2_sum = np.zeros(n_units)
        r2_cnt = np.zeros(n_units, dtype=int)
        n_trials_r2 = robs.shape[0]
        for _ in range(N_PSTH_SPLITS):
            perm = rng_r2.permutation(n_trials_r2)
            h = n_trials_r2 // 2
            psth_a = np.nanmean(robs[perm[:h]], axis=0)
            psth_b = np.nanmean(robs[perm[h:2 * h]], axis=0)
            for j in range(n_units):
                a, b = psth_a[:, j], psth_b[:, j]
                ok_t = np.isfinite(a) & np.isfinite(b)
                if ok_t.sum() < 3 or np.std(a[ok_t]) == 0 or np.std(b[ok_t]) == 0:
                    continue
                r = np.corrcoef(a[ok_t], b[ok_t])[0, 1]
                if np.isfinite(r):
                    r2_sum[j] += r * r
                    r2_cnt[j] += 1
        psth_r2 = np.where(r2_cnt > 0, r2_sum / np.maximum(r2_cnt, 1), np.nan)

        results, mats = run_covariance_decomposition(
            robs, eyepos, valid_mask,
            window_sizes_bins=WINDOW_BINS,
            dt=DT,
            n_shuffles=N_SHUFFLES,
            intercept_mode=INTERCEPT_MODE,
            intercept_kwargs=INTERCEPT_KWARGS,
            seed=42,
            device=str(device),
        )

        psth = robs.mean(axis=0)

        try:
            contam_rate = _load_contam_rate(
                session_name, subject, meta['n_neurons_total']
            )
        except NotImplementedError:
            contam_rate = None
            print(f"  QC: contamination not available for {subject}")

        session_results.append({
            "session": session_name,
            "subject": subject,
            "results": results,
            "mats": mats,
            "neuron_mask": neuron_mask,
            "meta": meta,
            "psth": psth,
            "rate_hz": rate_hz,
            "psth_r2": psth_r2,
            "qc": {"contam_rate": contam_rate},
        })

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DECOMP_CACHE, "wb") as f:
        dill.dump(session_results, f)
    print(f"\nCached {len(session_results)} sessions to {DECOMP_CACHE}")
    return session_results


def _load_or_compute_session_results(refresh=False):
    if DECOMP_CACHE.exists() and not refresh:
        print(f"Loading cached decomposition from {DECOMP_CACHE}")
        with open(DECOMP_CACHE, "rb") as f:
            return dill.load(f)
    return _compute_session_results()


# ---------------------------------------------------------------------------
# Stage 2: derive per-window metrics
# ---------------------------------------------------------------------------

def _compute_metrics(session_results, windows_ms, windows_bins):
    n_windows = len(windows_ms)
    metrics = []
    for w_idx in range(n_windows):
        all_alpha, all_ff_uncorr, all_ff_corr, all_erate = [], [], [], []
        all_rho_uncorr, all_rho_corr = [], []
        rho_u_meanz_by_ds, rho_c_meanz_by_ds, rho_delta_meanz_by_ds = [], [], []
        all_Ctotal, all_Cpsth, all_Crate, all_CnoiseU, all_CnoiseC, all_Cfem = (
            [], [], [], [], [], []
        )
        shuff_alphas = []
        shuff_rho_delta_meanz, shuff_rho_c_meanz, shuff_rho_subject = [], [], []
        subject_by_ds, subject_per_neuron, subject_per_pair = [], [], []

        for sr in session_results:
            if w_idx >= len(sr["results"]):
                continue
            res = sr["results"][w_idx]
            mats = sr["mats"][w_idx]

            Ctotal = mats["Total"]
            Cpsth = mats["PSTH"]
            Crate = mats["Intercept"]
            Cfem = mats["FEM"]

            CnoiseU = 0.5 * ((Ctotal - Cpsth) + (Ctotal - Cpsth).T)
            CnoiseC = 0.5 * ((Ctotal - Crate) + (Ctotal - Crate).T)

            erate = res["Erates"]
            rate_hz_ds = sr["rate_hz"]
            psth_r2_ds = sr["psth_r2"]
            valid = (
                np.isfinite(erate)
                & np.isfinite(rate_hz_ds)
                & (rate_hz_ds > MIN_RATE_HZ)
                & np.isfinite(psth_r2_ds)
                & (psth_r2_ds > MIN_PSTH_R2)
                & (np.diag(Ctotal) > MIN_VAR)
                & np.isfinite(np.diag(Crate))
                & np.isfinite(np.diag(Cpsth))
            )
            if valid.sum() < 3:
                continue

            subject_by_ds.append(sr["subject"])

            diag_psth = np.diag(Cpsth)[valid]
            diag_rate = np.diag(Crate)[valid]
            alpha = diag_psth / diag_rate
            all_alpha.append(alpha)
            subject_per_neuron.extend([sr["subject"]] * valid.sum())

            ff_u = np.diag(CnoiseU)[valid] / erate[valid]
            ff_c = np.diag(CnoiseC)[valid] / erate[valid]
            all_ff_uncorr.append(ff_u)
            all_ff_corr.append(ff_c)
            all_erate.append(erate[valid])

            NoiseCorrU = cov_to_corr(
                project_to_psd(CnoiseU[np.ix_(valid, valid)]), min_var=MIN_VAR
            )
            NoiseCorrC = cov_to_corr(
                project_to_psd(CnoiseC[np.ix_(valid, valid)]), min_var=MIN_VAR
            )
            rho_u = get_upper_triangle(NoiseCorrU)
            rho_c = get_upper_triangle(NoiseCorrC)

            pair_ok = np.isfinite(rho_u) & np.isfinite(rho_c)
            rho_u = rho_u[pair_ok]
            rho_c = rho_c[pair_ok]

            all_rho_uncorr.append(rho_u)
            all_rho_corr.append(rho_c)
            subject_per_pair.extend([sr["subject"]] * len(rho_u))

            if len(rho_u) > 0:
                rho_u_meanz_by_ds.append(fisher_z_mean(rho_u, eps=EPS_RHO))
                rho_c_meanz_by_ds.append(fisher_z_mean(rho_c, eps=EPS_RHO))
                rho_delta_meanz_by_ds.append(
                    fisher_z_mean(rho_c, eps=EPS_RHO)
                    - fisher_z_mean(rho_u, eps=EPS_RHO)
                )

            all_Ctotal.append(Ctotal[np.ix_(valid, valid)])
            all_Cpsth.append(Cpsth[np.ix_(valid, valid)])
            all_Crate.append(Crate[np.ix_(valid, valid)])
            all_CnoiseU.append(CnoiseU[np.ix_(valid, valid)])
            all_CnoiseC.append(CnoiseC[np.ix_(valid, valid)])
            all_Cfem.append(Cfem[np.ix_(valid, valid)])

            if "Shuffled_Intercepts" in mats and len(mats["Shuffled_Intercepts"]) > 0:
                for Crate_shuf in mats["Shuffled_Intercepts"]:
                    diag_rate_shuf = np.diag(Crate_shuf)[valid]
                    alpha_shuf = diag_psth / diag_rate_shuf
                    shuff_alphas.append(1 - alpha_shuf)

                    CnoiseC_shuf = Ctotal - Crate_shuf
                    CnoiseC_shuf = 0.5 * (CnoiseC_shuf + CnoiseC_shuf.T)
                    NC_shuf = cov_to_corr(
                        project_to_psd(CnoiseC_shuf[np.ix_(valid, valid)]),
                        min_var=MIN_VAR,
                    )
                    rho_c_shuf = get_upper_triangle(NC_shuf)
                    ok = np.isfinite(rho_c_shuf) & pair_ok
                    if ok.sum() > 0:
                        shuff_rho_c_meanz.append(
                            fisher_z_mean(rho_c_shuf[ok], eps=EPS_RHO)
                        )
                        shuff_rho_delta_meanz.append(
                            fisher_z_mean(rho_c_shuf[ok], eps=EPS_RHO)
                            - fisher_z_mean(rho_u[ok[:len(rho_u)]], eps=EPS_RHO)
                        )
                        shuff_rho_subject.append(sr["subject"])

        metrics.append({
            "window_ms": windows_ms[w_idx],
            "window_bins": windows_bins[w_idx],
            "alpha": np.concatenate(all_alpha) if all_alpha else np.array([]),
            "uncorr": np.concatenate(all_ff_uncorr) if all_ff_uncorr else np.array([]),
            "corr": np.concatenate(all_ff_corr) if all_ff_corr else np.array([]),
            "erate": np.concatenate(all_erate) if all_erate else np.array([]),
            "rho_uncorr": (
                np.concatenate(all_rho_uncorr) if all_rho_uncorr else np.array([])
            ),
            "rho_corr": (
                np.concatenate(all_rho_corr) if all_rho_corr else np.array([])
            ),
            "rho_u_meanz_by_ds": np.array(rho_u_meanz_by_ds),
            "rho_c_meanz_by_ds": np.array(rho_c_meanz_by_ds),
            "rho_delta_meanz_by_ds": np.array(rho_delta_meanz_by_ds),
            "subject_by_ds": subject_by_ds,
            "subject_per_neuron": np.array(subject_per_neuron),
            "subject_per_pair": np.array(subject_per_pair),
            "Ctotal": all_Ctotal,
            "Cpsth": all_Cpsth,
            "Crate": all_Crate,
            "CnoiseU": all_CnoiseU,
            "CnoiseC": all_CnoiseC,
            "Cfem": all_Cfem,
            "shuff_alphas": shuff_alphas,
            "shuff_rho_delta_meanz": np.array(shuff_rho_delta_meanz),
            "shuff_rho_c_meanz": np.array(shuff_rho_c_meanz),
            "shuff_rho_subject": np.array(shuff_rho_subject),
        })

        m = metrics[-1]
        print(f"Window {windows_ms[w_idx]:.1f} ms ({windows_bins[w_idx]} bins): "
              f"{len(m['alpha'])} neurons, "
              f"{len(m['rho_uncorr'])} pairs, "
              f"{len(m['shuff_alphas'])} shuffle iterations")
    return metrics


# ---------------------------------------------------------------------------
# Stage 3: per-panel summaries
# ---------------------------------------------------------------------------

def _compute_alpha_stats(metrics, windows_ms):
    m_by_window = []
    subject_per_neuron_by_window = []
    alpha_stats = {}
    for w_idx, m_dict in enumerate(metrics):
        alpha = m_dict["alpha"]
        m_raw = 1 - alpha
        subj_raw = m_dict["subject_per_neuron"]

        in_range = np.isfinite(m_raw) & (m_raw >= 0.0) & (m_raw <= 1.0)
        n_total = int(np.isfinite(m_raw).sum())
        n_dropped = int(n_total - in_range.sum())
        m = m_raw[in_range]
        m_by_window.append(m)
        subject_per_neuron_by_window.append(subj_raw[in_range])

        mean_m, (ci_lo, ci_hi) = bootstrap_mean_ci(m, nboot=5000, seed=0)
        med_m = float(np.nanmedian(m))
        q25, q75 = iqr_25_75(m)

        shuff_m = [
            s[np.isfinite(s) & (s >= 0.0) & (s <= 1.0)]
            for s in m_dict["shuff_alphas"]
        ]
        shuff_m = [s for s in shuff_m if s.size > 0]
        if len(shuff_m) > 0:
            null_means = np.array([np.nanmean(s) for s in shuff_m])
            null_mean_ci = (
                float(np.percentile(null_means, 2.5)),
                float(np.percentile(null_means, 97.5)),
            )
            p_emp = emp_p_one_sided(null_means, mean_m, direction="less")
        else:
            null_mean_ci = (np.nan, np.nan)
            p_emp = np.nan

        alpha_stats[windows_ms[w_idx]] = {
            "n": len(m), "mean": mean_m, "ci": (ci_lo, ci_hi),
            "median": med_m, "iqr": (q25, q75),
            "null_ci": null_mean_ci, "p_emp": p_emp,
            "n_dropped": n_dropped, "n_total": n_total,
        }
    return m_by_window, subject_per_neuron_by_window, alpha_stats


def _compute_fano_stats(metrics, windows_ms):
    fano_stats = {}
    for w_idx, m_dict in enumerate(metrics):
        ff_u, ff_c, erate = m_dict["uncorr"], m_dict["corr"], m_dict["erate"]
        ff_u_v, ff_c_v, mask = paired_valid(ff_u, ff_c, positive=True)
        erate_v = erate[mask]
        subject_labels_v = m_dict["subject_per_neuron"][mask]
        n_valid = len(ff_u_v)

        g_unc = geomean(ff_u_v)
        g_cor = geomean(ff_c_v)
        ratio = g_cor / g_unc
        pct_red = (1 - ratio) * 100

        _, p_wil = wilcoxon_signed_rank(ff_c_v, ff_u_v, alternative="less")

        var_u = ff_u_v * erate_v
        var_c = ff_c_v * erate_v

        slope_unc = float(np.sum(erate_v * var_u) / np.sum(erate_v ** 2))
        slope_cor = float(np.sum(erate_v * var_c) / np.sum(erate_v ** 2))

        rng = np.random.default_rng(0)
        nboot = 5000
        slopes_unc_boot = np.empty(nboot)
        slopes_cor_boot = np.empty(nboot)
        for b in range(nboot):
            idx = rng.integers(0, n_valid, size=n_valid)
            e_b = erate_v[idx]
            slopes_unc_boot[b] = np.sum(e_b * var_u[idx]) / np.sum(e_b ** 2)
            slopes_cor_boot[b] = np.sum(e_b * var_c[idx]) / np.sum(e_b ** 2)

        diff_boot = slopes_unc_boot - slopes_cor_boot
        slope_diff = slope_unc - slope_cor
        slope_diff_ci = (
            float(np.percentile(diff_boot, 2.5)),
            float(np.percentile(diff_boot, 97.5)),
        )
        p_slope = float(np.mean(diff_boot <= 0))

        fano_stats[windows_ms[w_idx]] = {
            "n": n_valid, "g_unc": g_unc, "g_cor": g_cor,
            "ratio": ratio, "pct_red": pct_red, "p_wil": p_wil,
            "slope_unc": slope_unc, "slope_cor": slope_cor,
            "slope_diff": slope_diff, "slope_diff_ci": slope_diff_ci,
            "p_slope": p_slope, "null_ratio_ci": (np.nan, np.nan),
            "erate": erate_v, "var_u": var_u, "var_c": var_c,
            "subject_per_neuron": subject_labels_v,
        }
    return fano_stats


def _compute_nc_stats(metrics, windows_ms):
    nc_stats = {}
    for w_idx, m_dict in enumerate(metrics):
        rho_u = m_dict["rho_uncorr"]
        rho_c = m_dict["rho_corr"]
        n_pairs = len(rho_u)

        z_u_ds = m_dict["rho_u_meanz_by_ds"]
        z_c_ds = m_dict["rho_c_meanz_by_ds"]
        dz_ds = m_dict["rho_delta_meanz_by_ds"]
        n_ds = len(z_u_ds)

        z_u_mean, z_u_ci = bootstrap_mean_ci(z_u_ds, nboot=5000, seed=0)
        z_c_mean, z_c_ci = bootstrap_mean_ci(z_c_ds, nboot=5000, seed=0)
        dz_mean, dz_ci = bootstrap_mean_ci(dz_ds, nboot=5000, seed=0)

        if n_ds >= 5:
            _, p_wil = wilcoxon_signed_rank(z_c_ds, z_u_ds, alternative="less")
        else:
            p_wil = np.nan

        shuff_dz = m_dict["shuff_rho_delta_meanz"]
        shuff_subj = m_dict["shuff_rho_subject"]
        if len(shuff_dz) > 0:
            null_dz_ci = (
                float(np.percentile(shuff_dz, 2.5)),
                float(np.percentile(shuff_dz, 97.5)),
            )
            p_emp_dz = emp_p_one_sided(shuff_dz, dz_mean, direction="less")
        else:
            null_dz_ci = (np.nan, np.nan)
            p_emp_dz = np.nan

        null_dz_ci_by_subject = {}
        for subj in SUBJECTS:
            s_mask = shuff_subj == subj
            if s_mask.sum() > 0:
                null_dz_ci_by_subject[subj] = (
                    float(np.percentile(shuff_dz[s_mask], 2.5)),
                    float(np.percentile(shuff_dz[s_mask], 97.5)),
                )
            else:
                null_dz_ci_by_subject[subj] = (np.nan, np.nan)

        nc_stats[windows_ms[w_idx]] = {
            "n_pairs": n_pairs, "n_ds": n_ds,
            "z_u_mean": z_u_mean, "z_u_ci": z_u_ci,
            "z_c_mean": z_c_mean, "z_c_ci": z_c_ci,
            "dz_mean": dz_mean, "dz_ci": dz_ci,
            "p_wil": p_wil, "null_dz_ci": null_dz_ci, "p_emp_dz": p_emp_dz,
            "null_dz_ci_by_subject": null_dz_ci_by_subject,
            "rho_u": rho_u, "rho_c": rho_c,
        }
    return nc_stats


def _compute_subspace(session_results, session_names, subjects):
    w_idx = SUBSPACE_WINDOW_IDX
    sub_names, sub_subjects = [], []
    pr_fem_list, pr_psth_list = [], []
    overlap_k1_list, overlap_k_list = [], []
    var_p_given_f, var_f_given_p = [], []
    spectra_psth, spectra_fem = [], []

    for ds_idx, sr in enumerate(session_results):
        if w_idx >= len(sr["mats"]):
            continue
        mats = sr["mats"][w_idx]
        Cpsth = mats["PSTH"]
        Crate = mats["Intercept"]
        Ctotal = mats["Total"]
        Cfem = Crate - Cpsth

        erate = sr["results"][w_idx]["Erates"]
        rate_hz_ds = sr["rate_hz"]
        psth_r2_ds = sr["psth_r2"]
        valid = (
            np.isfinite(erate)
            & np.isfinite(rate_hz_ds)
            & (rate_hz_ds > MIN_RATE_HZ)
            & np.isfinite(psth_r2_ds)
            & (psth_r2_ds > MIN_PSTH_R2)
            & (np.diag(Ctotal) > MIN_VAR)
            & np.isfinite(np.diag(Crate))
            & np.isfinite(np.diag(Cpsth))
        )
        if valid.sum() < SUBSPACE_K + 1:
            continue

        Cpsth_v = Cpsth[np.ix_(valid, valid)]
        Cfem_v = Cfem[np.ix_(valid, valid)]
        Ctotal_v = Ctotal[np.ix_(valid, valid)]

        Cpsth_psd = project_to_psd(Cpsth_v)
        Cfem_psd = project_to_psd(Cfem_v)

        w_psth, V_psth = np.linalg.eigh(Cpsth_psd)
        w_fem, V_fem = np.linalg.eigh(Cfem_psd)
        w_psth, V_psth = w_psth[::-1], V_psth[:, ::-1]
        w_fem, V_fem = w_fem[::-1], V_fem[:, ::-1]

        pr_psth_list.append(participation_ratio(Cpsth_psd))
        pr_fem_list.append(participation_ratio(Cfem_psd))

        k = min(SUBSPACE_K, valid.sum() - 1)
        U_psth = V_psth[:, :k]
        U_fem = V_fem[:, :k]
        overlap_k_list.append(symmetric_subspace_overlap(U_psth, U_fem))
        overlap_k1_list.append(
            symmetric_subspace_overlap(V_psth[:, :1], V_fem[:, :1])
        )

        var_p_given_f.append(directional_variance_capture(Cpsth_psd, U_fem))
        var_f_given_p.append(directional_variance_capture(Cfem_psd, U_psth))

        tr_total = np.trace(Ctotal_v)
        spectra_psth.append(w_psth / tr_total)
        spectra_fem.append(w_fem / tr_total)

        sub_names.append(session_names[ds_idx])
        sub_subjects.append(subjects[ds_idx])

    return dict(
        sub_names=sub_names,
        sub_subjects=sub_subjects,
        pr_fem_list=pr_fem_list,
        pr_psth_list=pr_psth_list,
        overlap_k1_list=overlap_k1_list,
        overlap_k_list=overlap_k_list,
        var_p_given_f=var_p_given_f,
        var_f_given_p=var_f_given_p,
        spectra_psth=spectra_psth,
        spectra_fem=spectra_fem,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_fig2_data(refresh=False):
    """Load the derived fig2 bundle, recomputing if needed."""
    refresh = refresh or REFRESH
    if DERIVED_CACHE.exists() and not refresh:
        print(f"Loading cached fig2 derived bundle from {DERIVED_CACHE}")
        with open(DERIVED_CACHE, "rb") as f:
            return dill.load(f)

    session_results = _load_or_compute_session_results(refresh=refresh)

    windows_ms = [r["window_ms"] for r in session_results[0]["results"]]
    windows_bins = [r["window_bins"] for r in session_results[0]["results"]]
    session_names = [sr["session"] for sr in session_results]
    subjects = [sr["subject"] for sr in session_results]
    n_sessions = len(session_results)
    print(f"\nLoaded {n_sessions} sessions: {session_names}")
    print(f"Windows (bins): {windows_bins} -> (ms): "
          f"{[f'{w:.1f}' for w in windows_ms]}")

    metrics = _compute_metrics(session_results, windows_ms, windows_bins)
    m_by_window, subject_per_neuron_by_window, alpha_stats = _compute_alpha_stats(
        metrics, windows_ms
    )
    fano_stats = _compute_fano_stats(metrics, windows_ms)
    nc_stats = _compute_nc_stats(metrics, windows_ms)
    subspace = _compute_subspace(session_results, session_names, subjects)

    bundle = dict(
        session_results=session_results,
        metrics=metrics,
        m_by_window=m_by_window,
        subject_per_neuron_by_window=subject_per_neuron_by_window,
        alpha_stats=alpha_stats,
        fano_stats=fano_stats,
        nc_stats=nc_stats,
        WINDOWS_MS=windows_ms,
        WINDOWS_BINS=windows_bins,
        SUBJECTS=SUBJECTS,
        SUBJECT_COLORS=SUBJECT_COLORS,
        session_names=session_names,
        subjects=subjects,
        n_sessions=n_sessions,
        SUBSPACE_WINDOW_IDX=SUBSPACE_WINDOW_IDX,
        SUBSPACE_K=SUBSPACE_K,
        config=dict(
            DT=DT, WINDOW_BINS=WINDOW_BINS, N_SHUFFLES=N_SHUFFLES,
            MIN_RATE_HZ=MIN_RATE_HZ, MIN_PSTH_R2=MIN_PSTH_R2,
            N_PSTH_SPLITS=N_PSTH_SPLITS, INTERCEPT_MODE=INTERCEPT_MODE,
            INTERCEPT_KWARGS=INTERCEPT_KWARGS, MIN_VAR=MIN_VAR, EPS_RHO=EPS_RHO,
        ),
        **subspace,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DERIVED_CACHE, "wb") as f:
        dill.dump(bundle, f)
    print(f"\nCached fig2 derived bundle to {DERIVED_CACHE}")
    return bundle


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Precompute fig2 derived data.")
    p.add_argument("-r", "--refresh", action="store_true",
                   help="Force recompute of derived bundle (keeps decomposition cache).")
    p.add_argument("--recompute-decomposition", action="store_true",
                   help="Also drop decomposition cache and rerun raw decomposition.")
    args, _ = p.parse_known_args()

    if args.recompute_decomposition and DECOMP_CACHE.exists():
        DECOMP_CACHE.unlink()
    load_fig2_data(refresh=args.refresh or args.recompute_decomposition)
