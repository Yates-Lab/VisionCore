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

from VisionCore.paths import VISIONCORE_ROOT
from .covariance import (
    cov_to_corr,
    project_to_psd,
    get_upper_triangle,
    align_fixrsvp_trials,
    run_covariance_decomposition,
)
from .stats import (
    geomean,
    iqr_25_75,
    bootstrap_mean_ci,
    fisher_z_mean,
    emp_p_one_sided,
    wilcoxon_signed_rank,
    paired_valid,
)
from .subspace import (
    participation_ratio,
    symmetric_subspace_overlap,
    directional_variance_capture,
)
from DataYatesV1 import get_free_device

# Snapshot-local cache directory. The original CACHE_DIR was VisionCore-global;
# we keep the snapshot self-contained so its outputs never collide with the
# live pipeline's caches.
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "legacy_native"


# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
REFRESH = False              # set True to force recompute of derived bundle
DT = 1 / 120                 # seconds per bin (native 240 Hz sampling)
WINDOW_BINS = [1, 2, 3, 6]   # counting windows in bins (6 @ 120 Hz = 50 ms = stim refresh)
N_SHUFFLES = 100             # shuffle null iterations
N_STAGE1_WORKERS = 8         # parallel session decompositions (set >1 to fan out)
N_STAGE1_GPUS = 2            # GPUs to distribute workers across (round-robin)
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


def _compute_one_session(cfg):
    """Per-session work: returns the result dict, or None to skip.

    Module-level so it's picklable for multiprocessing. Imports its own
    heavy modules so spawned workers initialize cleanly.
    """
    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))
    from models.data import prepare_data

    session_name = cfg["session"]
    subject = session_name.split("_")[0]
    if subject not in SUBJECTS:
        return None

    if "fixrsvp" not in cfg["types"]:
        cfg["types"] = cfg["types"] + ["fixrsvp"]

    print(f"\n--- {session_name} ({subject}) ---")
    try:
        train_data, val_data, cfg = prepare_data(cfg, strict=False)
    except Exception as e:
        print(f"  [{session_name}] Skipping: {e}")
        return None

    try:
        dset_idx = train_data.get_dataset_index("fixrsvp")
    except (ValueError, KeyError):
        print(f"  [{session_name}] Skipping: no fixrsvp data")
        return None
    fixrsvp_dset = train_data.dsets[dset_idx]

    robs, eyepos, valid_mask, neuron_mask, meta = align_fixrsvp_trials(
        fixrsvp_dset,
        valid_time_bins=120,
        min_fix_dur=20,
        min_total_spikes=0,
    )
    if robs is None or robs.shape[0] < 10:
        print(f"  [{session_name}] Skipping: insufficient data ({meta})")
        return None
    print(f"  [{session_name}] Trials: {meta['n_trials_good']}/{meta['n_trials_total']}, "
          f"Neurons: {meta['n_neurons_used']}/{meta['n_neurons_total']}")

    n_units = robs.shape[2]
    n_spikes_per_unit = np.nansum(robs, axis=(0, 1))
    n_valid_bins_per_unit = np.sum(np.isfinite(robs), axis=(0, 1))
    rate_hz = np.where(
        n_valid_bins_per_unit > 0,
        n_spikes_per_unit / np.maximum(n_valid_bins_per_unit, 1) / DT,
        np.nan,
    )

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

    # If a worker pool pinned us to a single GPU via CUDA_VISIBLE_DEVICES,
    # use cuda:0 directly. get_free_device queries nvidia-smi and may
    # return a physical index that doesn't exist in this worker's CUDA view.
    import os, torch
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if torch.cuda.is_available() and cvd and "," not in cvd:
        device = torch.device("cuda:0")
        print(f"  [{session_name}] using pinned device cuda:0 "
              f"(CUDA_VISIBLE_DEVICES={cvd})")
    else:
        device = get_free_device()
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
        print(f"  [{session_name}] QC: contamination not available for {subject}")

    return {
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
    }


def _worker_init(gpu_queue):
    """Pool initializer: claim a GPU and pin this worker to it.

    Sets CUDA_VISIBLE_DEVICES BEFORE any CUDA op runs. Subsequent
    `torch.cuda` queries in this process see exactly one GPU (renumbered 0).
    """
    import os
    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[worker pid={os.getpid()}] pinned to physical GPU {gpu_id}", flush=True)


def _run_session_pool(cfgs, n_workers, n_gpus=1):
    """Run _compute_one_session over cfgs, returning results in input order.

    n_workers=1 → in-process sequential (bit-identical to legacy path).
    n_workers>1 → spawn-Pool with workers round-robin pinned to n_gpus GPUs.
    """
    if n_workers <= 1:
        return [_compute_one_session(cfg) for cfg in cfgs]

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    n_gpus = max(int(n_gpus), 1)
    gpu_queue = ctx.Queue()
    for i in range(n_workers):
        gpu_queue.put(i % n_gpus)
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(gpu_queue,),
    ) as pool:
        return pool.map(_compute_one_session, cfgs)


def _compute_session_results(n_workers=None, n_gpus=None):
    """Run LOTC decomposition for every session listed in the dataset config."""
    if n_workers is None:
        n_workers = N_STAGE1_WORKERS
    if n_gpus is None:
        n_gpus = N_STAGE1_GPUS

    if str(VISIONCORE_ROOT) not in sys.path:
        sys.path.insert(0, str(VISIONCORE_ROOT))
    from models.config_loader import load_dataset_configs

    dataset_configs = load_dataset_configs(str(DATASET_CONFIGS_PATH))
    per_session = _run_session_pool(dataset_configs, n_workers, n_gpus)
    session_results = [r for r in per_session if r is not None]

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

def _metrics_one(sr, w_idx):
    """Per-(window, session) contribution to the metrics dict for w_idx.
    Returns None if the session has too few valid neurons in this window."""
    if w_idx >= len(sr["results"]):
        return None
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
        return None

    diag_psth = np.diag(Cpsth)[valid]
    diag_rate = np.diag(Crate)[valid]
    alpha = diag_psth / diag_rate

    ff_u = np.diag(CnoiseU)[valid] / erate[valid]
    ff_c = np.diag(CnoiseC)[valid] / erate[valid]

    NoiseCorrU = cov_to_corr(
        project_to_psd(CnoiseU[np.ix_(valid, valid)]), min_var=MIN_VAR
    )
    NoiseCorrC = cov_to_corr(
        project_to_psd(CnoiseC[np.ix_(valid, valid)]), min_var=MIN_VAR
    )
    rho_u_full = get_upper_triangle(NoiseCorrU)
    rho_c_full = get_upper_triangle(NoiseCorrC)
    pair_ok = np.isfinite(rho_u_full) & np.isfinite(rho_c_full)
    rho_u = rho_u_full[pair_ok]
    rho_c = rho_c_full[pair_ok]

    if len(rho_u) > 0:
        rho_u_meanz = fisher_z_mean(rho_u, eps=EPS_RHO)
        rho_c_meanz = fisher_z_mean(rho_c, eps=EPS_RHO)
        rho_delta_meanz = rho_c_meanz - rho_u_meanz
    else:
        rho_u_meanz = rho_c_meanz = rho_delta_meanz = np.nan

    n_valid_ds = int(valid.sum())
    shuff_alphas = []
    ds_shuff_var_c = []
    shuff_rho_c_meanz_list, shuff_rho_delta_meanz_list = [], []
    shuff_rho_subject_list = []
    if "Shuffled_Intercepts" in mats and len(mats["Shuffled_Intercepts"]) > 0:
        for Crate_shuf in mats["Shuffled_Intercepts"]:
            diag_rate_shuf = np.diag(Crate_shuf)[valid]
            alpha_shuf = diag_psth / diag_rate_shuf
            shuff_alphas.append(1 - alpha_shuf)

            CnoiseC_shuf = Ctotal - Crate_shuf
            CnoiseC_shuf = 0.5 * (CnoiseC_shuf + CnoiseC_shuf.T)
            ds_shuff_var_c.append(np.diag(CnoiseC_shuf)[valid])
            NC_shuf = cov_to_corr(
                project_to_psd(CnoiseC_shuf[np.ix_(valid, valid)]),
                min_var=MIN_VAR,
            )
            rho_c_shuf = get_upper_triangle(NC_shuf)
            ok = np.isfinite(rho_c_shuf) & pair_ok
            if ok.sum() > 0:
                shuff_rho_c_meanz_list.append(
                    fisher_z_mean(rho_c_shuf[ok], eps=EPS_RHO)
                )
                shuff_rho_delta_meanz_list.append(
                    fisher_z_mean(rho_c_shuf[ok], eps=EPS_RHO)
                    - fisher_z_mean(rho_u_full[ok[:len(rho_u_full)]], eps=EPS_RHO)
                )
                shuff_rho_subject_list.append(sr["subject"])

    return dict(
        subject=sr["subject"],
        session=sr["session"],
        n_valid=n_valid_ds,
        alpha=alpha,
        ff_uncorr=ff_u,
        ff_corr=ff_c,
        erate=erate[valid],
        rho_uncorr=rho_u,
        rho_corr=rho_c,
        rho_u_meanz=rho_u_meanz,
        rho_c_meanz=rho_c_meanz,
        rho_delta_meanz=rho_delta_meanz,
        Ctotal=Ctotal[np.ix_(valid, valid)],
        Cpsth=Cpsth[np.ix_(valid, valid)],
        Crate=Crate[np.ix_(valid, valid)],
        CnoiseU=CnoiseU[np.ix_(valid, valid)],
        CnoiseC=CnoiseC[np.ix_(valid, valid)],
        Cfem=Cfem[np.ix_(valid, valid)],
        shuff_alphas=shuff_alphas,
        ds_shuff_var_c=np.asarray(ds_shuff_var_c) if ds_shuff_var_c else None,
        shuff_rho_c_meanz=shuff_rho_c_meanz_list,
        shuff_rho_delta_meanz=shuff_rho_delta_meanz_list,
        shuff_rho_subject=shuff_rho_subject_list,
    )


def _compute_metrics(session_results, windows_ms, windows_bins, n_jobs=-1):
    """Parallelized across (window, session) pairs.

    The original loop body was pure numpy and per-session-independent — the
    only cross-session step is final concatenation, which we do after the
    parallel fan-out below. joblib preserves input order, so session order is
    preserved exactly.
    """
    from joblib import Parallel, delayed

    n_windows = len(windows_ms)
    tasks = [(w_idx, sr_i) for w_idx in range(n_windows)
             for sr_i in range(len(session_results))]
    flat = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_metrics_one)(session_results[sr_i], w_idx)
        for (w_idx, sr_i) in tasks
    )

    by_w = [[] for _ in range(n_windows)]
    for (w_idx, _), r in zip(tasks, flat):
        by_w[w_idx].append(r)

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
        session_per_neuron = []
        shuff_var_c_blocks, shuff_var_c_nvalid = [], []

        for r in by_w[w_idx]:
            if r is None:
                continue
            subject_by_ds.append(r["subject"])
            all_alpha.append(r["alpha"])
            subject_per_neuron.extend([r["subject"]] * r["n_valid"])
            session_per_neuron.extend([r["session"]] * r["n_valid"])
            all_ff_uncorr.append(r["ff_uncorr"])
            all_ff_corr.append(r["ff_corr"])
            all_erate.append(r["erate"])
            all_rho_uncorr.append(r["rho_uncorr"])
            all_rho_corr.append(r["rho_corr"])
            subject_per_pair.extend([r["subject"]] * len(r["rho_uncorr"]))

            if len(r["rho_uncorr"]) > 0:
                rho_u_meanz_by_ds.append(r["rho_u_meanz"])
                rho_c_meanz_by_ds.append(r["rho_c_meanz"])
                rho_delta_meanz_by_ds.append(r["rho_delta_meanz"])

            all_Ctotal.append(r["Ctotal"])
            all_Cpsth.append(r["Cpsth"])
            all_Crate.append(r["Crate"])
            all_CnoiseU.append(r["CnoiseU"])
            all_CnoiseC.append(r["CnoiseC"])
            all_Cfem.append(r["Cfem"])

            shuff_alphas.extend(r["shuff_alphas"])
            shuff_rho_c_meanz.extend(r["shuff_rho_c_meanz"])
            shuff_rho_delta_meanz.extend(r["shuff_rho_delta_meanz"])
            shuff_rho_subject.extend(r["shuff_rho_subject"])

            shuff_var_c_blocks.append(r["ds_shuff_var_c"])
            shuff_var_c_nvalid.append(r["n_valid"])

        # Pool shuffle-null corrected variances across sessions into [N_total, S_min].
        # Sessions are truncated to the smallest shuffle count so each column b is one
        # coherent null draw pooled over all neurons; the row order matches `erate`.
        # Sessions without shuffles contribute NaN rows (dropped per-column later).
        present_S = [b.shape[0] for b in shuff_var_c_blocks if b is not None]
        if present_S:
            S_min = min(present_S)
            rows = []
            for blk, nv in zip(shuff_var_c_blocks, shuff_var_c_nvalid):
                if blk is None:
                    rows.append(np.full((nv, S_min), np.nan))
                else:
                    rows.append(blk[:S_min].T)  # [n_valid, S_min]
            shuff_var_c = np.concatenate(rows, axis=0)
        else:
            shuff_var_c = np.empty((0, 0))

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
            "session_per_neuron": np.array(session_per_neuron),
            "subject_per_pair": np.array(subject_per_pair),
            "shuff_var_c": shuff_var_c,
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


def _slope_through_origin(erate, var):
    """LS slope of var = slope * erate forced through the origin (nan-safe)."""
    ok = np.isfinite(erate) & np.isfinite(var) & (erate > 0) & (var >= 0)
    if ok.sum() < 3:
        return np.nan
    e, v = erate[ok], var[ok]
    return float(np.sum(e * v) / np.sum(e ** 2))


def _clustered_slope_bootstrap(erate, var_u, var_c, sessions, nboot=5000, seed=0):
    """
    Session-clustered bootstrap of slope-through-origin Fano factors.

    Neurons within a session share the same trials and eye-movement traces, so
    they are not independent samples. A naive neuron-level bootstrap therefore
    underestimates the true uncertainty. We instead resample whole sessions with
    replacement, pool every neuron from the drawn sessions, and refit the pooled
    slopes each iteration -- propagating session-level uncertainty.

    Fallback: with a single session we cannot estimate between-session variance,
    so we resample neurons within that session (the only uncertainty available)
    rather than return a degenerate, zero-width interval.

    Returns dict with 95% CIs for the uncorrected slope, corrected slope, and
    their difference (uncorrected - corrected), plus a one-sided bootstrap p for
    H0: corrected slope >= uncorrected slope (i.e. no FEM-driven inflation).

    Vectorized: per-session sufficient statistics (sum e*v and sum e^2) are
    precomputed, then bootstrap draws are gathered+summed in a single matrix
    op. Equivalent to the per-iteration Python loop but ~100x faster.
    """
    erate = np.asarray(erate, dtype=float)
    var_u = np.asarray(var_u, dtype=float)
    var_c = np.asarray(var_c, dtype=float)
    sessions = np.asarray(sessions)
    uniq = np.unique(sessions)
    rng = np.random.default_rng(seed)
    n = len(erate)

    if uniq.size >= 2:
        # Sufficient statistics per session: slope = sum(e*v) / sum(e^2). The
        # cluster bootstrap concatenates whole sessions, so a session draw is
        # just a sum over the per-session sufficient statistics.
        K = uniq.size
        sum_ee = np.zeros(K)
        sum_eu = np.zeros(K)
        sum_ec = np.zeros(K)
        for i, s in enumerate(uniq):
            m = sessions == s
            e = erate[m]; vu = var_u[m]; vc = var_c[m]
            ok = np.isfinite(e) & np.isfinite(vu) & np.isfinite(vc)
            e, vu, vc = e[ok], vu[ok], vc[ok]
            sum_ee[i] = np.sum(e * e)
            sum_eu[i] = np.sum(e * vu)
            sum_ec[i] = np.sum(e * vc)

        # draws: [nboot, K] session indices, with replacement
        draws = rng.integers(0, K, size=(nboot, K))
        D_ee = sum_ee[draws].sum(axis=1)
        D_eu = sum_eu[draws].sum(axis=1)
        D_ec = sum_ec[draws].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            su = np.where(D_ee > 0, D_eu / D_ee, np.nan)
            sc = np.where(D_ee > 0, D_ec / D_ee, np.nan)
    else:
        # Neuron-level bootstrap (single-session fallback). Same vectorized
        # gather pattern.
        ok = np.isfinite(erate) & np.isfinite(var_u) & np.isfinite(var_c)
        e_v = erate[ok]; vu_v = var_u[ok]; vc_v = var_c[ok]
        if e_v.size == 0:
            su = np.full(nboot, np.nan)
            sc = np.full(nboot, np.nan)
        else:
            idx = rng.integers(0, e_v.size, size=(nboot, e_v.size))
            E = e_v[idx]
            D_ee = np.sum(E * E, axis=1)
            D_eu = np.sum(E * vu_v[idx], axis=1)
            D_ec = np.sum(E * vc_v[idx], axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                su = np.where(D_ee > 0, D_eu / D_ee, np.nan)
                sc = np.where(D_ee > 0, D_ec / D_ee, np.nan)

    diff = su - sc

    def _ci(a):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    diff_f = diff[np.isfinite(diff)]
    p = (float((np.sum(diff_f <= 0) + 1) / (diff_f.size + 1))
         if diff_f.size else np.nan)
    return {
        "unc_ci": _ci(su), "cor_ci": _ci(sc), "diff_ci": _ci(diff),
        "p": p, "n_sessions": int(uniq.size),
    }


def _compute_fano_stats(metrics, windows_ms):
    fano_stats = {}
    for w_idx, m_dict in enumerate(metrics):
        ff_u, ff_c, erate = m_dict["uncorr"], m_dict["corr"], m_dict["erate"]
        ff_u_v, ff_c_v, mask = paired_valid(ff_u, ff_c, positive=True)
        erate_v = erate[mask]
        subject_labels_v = m_dict["subject_per_neuron"][mask]
        session_labels_v = m_dict["session_per_neuron"][mask]
        n_valid = len(ff_u_v)

        g_unc = geomean(ff_u_v)
        g_cor = geomean(ff_c_v)
        ratio = g_cor / g_unc
        pct_red = (1 - ratio) * 100

        _, p_wil = wilcoxon_signed_rank(ff_c_v, ff_u_v, alternative="less")

        var_u = ff_u_v * erate_v
        var_c = ff_c_v * erate_v

        # Pooled slope-through-origin Fano factors with session-clustered CIs.
        slope_unc = _slope_through_origin(erate_v, var_u)
        slope_cor = _slope_through_origin(erate_v, var_c)
        boot = _clustered_slope_bootstrap(
            erate_v, var_u, var_c, session_labels_v, nboot=5000, seed=0
        )
        slope_unc_ci = boot["unc_ci"]
        slope_cor_ci = boot["cor_ci"]
        slope_diff = slope_unc - slope_cor
        slope_diff_ci = boot["diff_ci"]
        p_slope = boot["p"]

        # Per-subject slopes + clustered CIs (drive the per-subject fitted lines
        # and shaded bands in panel D).
        per_subject = {}
        for subj in np.unique(subject_labels_v):
            s_mask = subject_labels_v == subj
            e_s, vu_s, vc_s = erate_v[s_mask], var_u[s_mask], var_c[s_mask]
            sess_s = session_labels_v[s_mask]
            boot_s = _clustered_slope_bootstrap(
                e_s, vu_s, vc_s, sess_s, nboot=2000, seed=0
            )
            su_s = _slope_through_origin(e_s, vu_s)
            sc_s = _slope_through_origin(e_s, vc_s)
            per_subject[str(subj)] = {
                "slope_unc": su_s, "slope_cor": sc_s,
                "slope_unc_ci": boot_s["unc_ci"], "slope_cor_ci": boot_s["cor_ci"],
                "slope_diff": su_s - sc_s, "slope_diff_ci": boot_s["diff_ci"],
                "p_slope": boot_s["p"], "n_sessions": boot_s["n_sessions"],
                "n": int(s_mask.sum()),
            }

        # FEM-shuffle null on the corrected slope. `shuff_var_c` holds, per neuron,
        # the corrected noise variance recomputed with eye positions shuffled
        # ([N_neurons, S]); column b is one null draw pooled across sessions. For
        # each draw we refit the pooled corrected slope and measure how much
        # variance the correction strips off by chance. The real reduction is
        # significant if it exceeds this null reduction.
        shuff_var_c = m_dict.get("shuff_var_c", np.empty((0, 0)))
        if shuff_var_c.size and shuff_var_c.shape[0] == mask.shape[0]:
            svc = shuff_var_c[mask]  # [n_valid, S]
            null_slope_cor = np.array([
                _slope_through_origin(erate_v, svc[:, b])
                for b in range(svc.shape[1])
            ])
            null_slope_cor = null_slope_cor[np.isfinite(null_slope_cor)]
        else:
            null_slope_cor = np.array([])

        if null_slope_cor.size:
            null_reduction = slope_unc - null_slope_cor
            obs_reduction = slope_unc - slope_cor
            slope_cor_null_ci = (
                float(np.percentile(null_slope_cor, 2.5)),
                float(np.percentile(null_slope_cor, 97.5)),
            )
            p_emp_slope = emp_p_one_sided(
                null_reduction, obs_reduction, direction="greater"
            )
        else:
            slope_cor_null_ci = (np.nan, np.nan)
            p_emp_slope = np.nan

        fano_stats[windows_ms[w_idx]] = {
            "n": n_valid, "g_unc": g_unc, "g_cor": g_cor,
            "ratio": ratio, "pct_red": pct_red, "p_wil": p_wil,
            "slope_unc": slope_unc, "slope_cor": slope_cor,
            "slope_unc_ci": slope_unc_ci, "slope_cor_ci": slope_cor_ci,
            "slope_diff": slope_diff, "slope_diff_ci": slope_diff_ci,
            "p_slope": p_slope, "n_sessions": boot["n_sessions"],
            "slope_cor_null_ci": slope_cor_null_ci, "p_emp_slope": p_emp_slope,
            "null_ratio_ci": (np.nan, np.nan),
            "per_subject": per_subject,
            "erate": erate_v, "var_u": var_u, "var_c": var_c,
            "subject_per_neuron": subject_labels_v,
            "session_per_neuron": session_labels_v,
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


def _subspace_one_session(sr, session_name, subject):
    """Pure per-session subspace computation. Returns None if the session
    has too few valid neurons to participate in panel G."""
    w_idx = SUBSPACE_WINDOW_IDX
    if w_idx >= len(sr["mats"]):
        return None
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
        return None

    Cpsth_v = Cpsth[np.ix_(valid, valid)]
    Cfem_v = Cfem[np.ix_(valid, valid)]
    Ctotal_v = Ctotal[np.ix_(valid, valid)]

    Cpsth_psd = project_to_psd(Cpsth_v)
    Cfem_psd = project_to_psd(Cfem_v)

    w_psth, V_psth = np.linalg.eigh(Cpsth_psd)
    w_fem, V_fem = np.linalg.eigh(Cfem_psd)
    w_psth, V_psth = w_psth[::-1], V_psth[:, ::-1]
    w_fem, V_fem = w_fem[::-1], V_fem[:, ::-1]

    k = min(SUBSPACE_K, int(valid.sum()) - 1)
    U_psth = V_psth[:, :k]
    U_fem = V_fem[:, :k]
    tr_total = np.trace(Ctotal_v)

    # Shuffle null on Cfem only: same Cpsth (and its subspace) as real, but
    # FEM rebuilt from eye-shuffled Intercepts.
    null_x, null_y, null_ok, null_ok1 = [], [], [], []
    shuff_intercepts = mats.get("Shuffled_Intercepts", []) or []
    for Crate_shuf in shuff_intercepts:
        Crate_shuf = np.asarray(Crate_shuf, dtype=np.float64)
        Cfem_shuf = Crate_shuf[np.ix_(valid, valid)] - Cpsth_v
        Cfem_shuf_psd = project_to_psd(Cfem_shuf)
        w_shuf, V_shuf = np.linalg.eigh(Cfem_shuf_psd)
        V_shuf = V_shuf[:, ::-1]
        U_fem_shuf = V_shuf[:, :k]
        null_x.append(directional_variance_capture(Cpsth_psd, U_fem_shuf))
        null_y.append(directional_variance_capture(Cfem_shuf_psd, U_psth))
        null_ok.append(symmetric_subspace_overlap(U_psth, U_fem_shuf))
        null_ok1.append(symmetric_subspace_overlap(V_psth[:, :1], V_shuf[:, :1]))

    return dict(
        session_name=session_name,
        subject=subject,
        pr_psth=participation_ratio(Cpsth_psd),
        pr_fem=participation_ratio(Cfem_psd),
        overlap_k=symmetric_subspace_overlap(U_psth, U_fem),
        overlap_k1=symmetric_subspace_overlap(V_psth[:, :1], V_fem[:, :1]),
        var_p_given_f=directional_variance_capture(Cpsth_psd, U_fem),
        var_f_given_p=directional_variance_capture(Cfem_psd, U_psth),
        spectrum_psth=w_psth / tr_total,
        spectrum_fem=w_fem / tr_total,
        null_var_p_given_f=null_x,
        null_var_f_given_p=null_y,
        null_overlap_k=null_ok,
        null_overlap_k1=null_ok1,
    )


def _compute_subspace(session_results, session_names, subjects, n_jobs=-1):
    """Per-session work runs in parallel across CPU cores (joblib).

    Stage 2 only — pure numpy, no torch/CUDA, safe to fan out.
    """
    from joblib import Parallel, delayed

    per_session = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_subspace_one_session)(sr, session_names[i], subjects[i])
        for i, sr in enumerate(session_results)
    )

    sub_names, sub_subjects = [], []
    pr_fem_list, pr_psth_list = [], []
    overlap_k1_list, overlap_k_list = [], []
    var_p_given_f, var_f_given_p = [], []
    spectra_psth, spectra_fem = [], []
    null_var_p_given_f, null_var_f_given_p = [], []
    null_overlap_k, null_overlap_k1 = [], []
    null_session_idx, null_subjects = [], []

    for r in per_session:
        if r is None:
            continue
        sub_names.append(r["session_name"])
        sub_subjects.append(r["subject"])
        pr_psth_list.append(r["pr_psth"])
        pr_fem_list.append(r["pr_fem"])
        overlap_k_list.append(r["overlap_k"])
        overlap_k1_list.append(r["overlap_k1"])
        var_p_given_f.append(r["var_p_given_f"])
        var_f_given_p.append(r["var_f_given_p"])
        spectra_psth.append(r["spectrum_psth"])
        spectra_fem.append(r["spectrum_fem"])

        sess_pos = len(sub_names) - 1
        n_draws = len(r["null_var_p_given_f"])
        null_var_p_given_f.extend(r["null_var_p_given_f"])
        null_var_f_given_p.extend(r["null_var_f_given_p"])
        null_overlap_k.extend(r["null_overlap_k"])
        null_overlap_k1.extend(r["null_overlap_k1"])
        null_session_idx.extend([sess_pos] * n_draws)
        null_subjects.extend([r["subject"]] * n_draws)

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
        null_var_p_given_f=null_var_p_given_f,
        null_var_f_given_p=null_var_f_given_p,
        null_overlap_k=null_overlap_k,
        null_overlap_k1=null_overlap_k1,
        null_session_idx=null_session_idx,
        null_subjects=null_subjects,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_fig2_data(refresh=False, refresh_decomposition=False):
    """Load the derived fig2 bundle, recomputing if needed.

    ``refresh`` rebuilds only the derived bundle (stage 2). The expensive
    per-session decomposition (stage 1, GPU) is preserved unless
    ``refresh_decomposition=True``.
    """
    refresh = refresh or REFRESH
    if DERIVED_CACHE.exists() and not refresh:
        print(f"Loading cached fig2 derived bundle from {DERIVED_CACHE}")
        with open(DERIVED_CACHE, "rb") as f:
            return dill.load(f)

    session_results = _load_or_compute_session_results(refresh=refresh_decomposition)

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

    load_fig2_data(
        refresh=args.refresh or args.recompute_decomposition,
        refresh_decomposition=args.recompute_decomposition,
    )
