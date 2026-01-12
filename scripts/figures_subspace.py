# scripts/figures_subspace.py

import numpy as np
import matplotlib.pyplot as plt

# Import shared utilities (use relative import for same-directory scripts)
try:
    from figure_common import set_pub_style
except ImportError:
    from scripts.figure_common import set_pub_style

# ----------------------------
# numerics / PSD utilities
# ----------------------------
def sym(A):
    A = np.asarray(A, dtype=np.float64)
    return 0.5 * (A + A.T)

def psd_project(A, eps=0.0, return_eigs=False):
    """
    Project symmetric matrix onto PSD cone by clamping eigenvalues.
    eps: floor for eigenvalues (0 -> PSD; >0 -> PD-ish).
    """
    A = sym(A)
    # guard: NaNs will break eigh
    if not np.all(np.isfinite(A)):
        # replace non-finite with 0 (diagnostics will flag)
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    w, V = np.linalg.eigh(A)
    w_clamped = np.maximum(w, eps)
    A_psd = (V * w_clamped) @ V.T
    A_psd = sym(A_psd)
    if return_eigs:
        return A_psd, w, w_clamped
    return A_psd

def safe_trace(A):
    return float(np.trace(A))

def topk_evecs_psd(C_psd, k):
    """
    Top-k eigenvectors of PSD matrix (descending eigenvalues).
    Returns U (N,k).
    """
    w, V = np.linalg.eigh(sym(C_psd))
    idx = np.argsort(w)[::-1]
    idx = idx[:k]
    return V[:, idx]

def participation_ratio(C_psd, eps=1e-12):
    """
    PR = (tr C)^2 / tr(C^2). For PSD C, PR is well-defined.
    """
    C = sym(C_psd)
    tr = np.trace(C)
    tr2 = np.trace(C @ C)
    if tr2 <= eps:
        return np.nan
    return float((tr * tr) / tr2)

def symmetric_subspace_overlap(Ua, Ub):
    """
    ||Ua^T Ub||_F^2 / k in [0,1] if Ua,Ub orthonormal (columns).
    """
    k = Ua.shape[1]
    X = Ua.T @ Ub
    return float((np.linalg.norm(X, ord="fro") ** 2) / k)

def directional_variance_capture(C_target_psd, U_source, eps=1e-12):
    """
    Fraction of target variance captured by source basis:
      tr(U^T C U) / tr(C)
    In [0,1] for PSD C.
    """
    C = sym(C_target_psd)
    denom = np.trace(C)
    if denom <= eps:
        return np.nan
    numer = np.trace(U_source.T @ C @ U_source)
    return float(numer / denom)

def eigenspectrum_frac(C_psd, norm_trace):
    """
    Return eigenvalues sorted desc, normalized by norm_trace.
    """
    w = np.linalg.eigvalsh(sym(C_psd))[::-1]
    if norm_trace <= 0:
        return w * np.nan
    return w / norm_trace

# ----------------------------
# diagnostics
# ----------------------------
def cov_diag_stats(A):
    A = sym(A)
    d = np.diag(A)
    frac_bad = float(np.mean(~np.isfinite(d)))
    frac_nonpos = float(np.mean((d <= 0) & np.isfinite(d)))
    return dict(
        diag_min=float(np.nanmin(d)),
        diag_med=float(np.nanmedian(d)),
        diag_bad_frac=frac_bad,
        diag_nonpos_frac=frac_nonpos,
        any_nonfinite=bool(np.any(~np.isfinite(A))),
    )

def psd_stats_from_eigs(w_raw, w_clamped):
    w_raw = np.asarray(w_raw)
    w_clamped = np.asarray(w_clamped)
    return dict(
        eig_min=float(np.min(w_raw)),
        eig_neg_frac=float(np.mean(w_raw < 0)),
        eig_clamp_mass=float(np.sum(w_clamped - w_raw)),  # total added "ridge" in spectrum
    )

# ----------------------------
# main computation
# ----------------------------
def compute_subspace_stats(
    outputs,
    model=None,
    window_idx=1,
    rep_k=5,
    max_dims_to_plot=50,
    min_total_spikes=50,
    psd_eps=0.0,
    n_shuffles_diag=500,  # for diagnostics: cap how many shuffles you scan
):
    """
    Computes subspace + spectra stats for real and shuffle controls.

    Real matrices:
      Cpsth = mats['PSTH']
      Crate = mats['Intercept']
      Cfem  = Crate - Cpsth
      Ctotal = mats['Total']

    Shuffles:
      Cfem_shuff = Crate_shuff - Cpsth  (then PSD project)

    Returns dict with:
      - per_session arrays for dims/overlaps/variance-capture
      - null distributions for variance-capture metrics (flattened over sessions & shuffles)
      - spectra arrays for plotting
      - diagnostics per session (PSD clamp amounts, etc.)
    """
    n_sessions = len(outputs)

    # outputs
    per = dict(
        name=[],
        n_valid=[],

        pr_fem=[],
        pr_psth=[],

        overlap_k1=[],
        overlap_k=[],

        var_f_given_p=[],
        var_p_given_f=[],
    )

    spectra = dict(psth=[], fem=[], total_norm=[])

    # shuffle nulls (flattened)
    null = dict(
        var_f_given_p=[],
        var_p_given_f=[],
        overlap_k=[],
        overlap_k1=[],
    )

    diag = dict(
        window_ms=None,
        session=[],
    )

    for si in range(n_sessions):
        res = outputs[si]["results"][window_idx]
        mats = outputs[si]["last_mats"][window_idx]

        window_ms = res.get("window_ms", None)
        diag["window_ms"] = window_ms

        Er = np.asarray(res["Erates"], dtype=np.float64).reshape(-1)
        ns = float(res["n_samples"])
        spike_mask = (Er * ns) > min_total_spikes

        # name
        if model is not None and hasattr(model, "names"):
            nm = model.names[si]
        else:
            nm = outputs[si].get("sess", f"session_{si}")
        per["name"].append(nm)

        # load matrices (numpy)
        Cpsth_raw = np.asarray(mats["PSTH"], dtype=np.float64)
        Crate_raw = np.asarray(mats["Intercept"], dtype=np.float64)
        Ctotal_raw = np.asarray(mats["Total"], dtype=np.float64)

        Cfem_raw = Crate_raw - Cpsth_raw

        # validity mask: finite diagonals + spike_mask
        d_ok = np.isfinite(np.diag(Cpsth_raw)) & np.isfinite(np.diag(Cfem_raw)) & np.isfinite(np.diag(Ctotal_raw))
        valid = d_ok & spike_mask

        n_valid = int(np.sum(valid))
        per["n_valid"].append(n_valid)
        if n_valid < max(5, rep_k):
            # store NaNs for this session
            for k in ["pr_fem","pr_psth","overlap_k1","overlap_k","var_f_given_p","var_p_given_f"]:
                per[k].append(np.nan)
            diag["session"].append(dict(
                name=nm, n_valid=n_valid, skipped=True,
                reason="too_few_valid_neurons",
                Cpsth=cov_diag_stats(Cpsth_raw),
                Cfem=cov_diag_stats(Cfem_raw),
                Ctotal=cov_diag_stats(Ctotal_raw),
            ))
            continue

        # index to valid neurons
        Cpsth = Cpsth_raw[np.ix_(valid, valid)]
        Cfem  = Cfem_raw[np.ix_(valid, valid)]
        Ctotal = Ctotal_raw[np.ix_(valid, valid)]

        # PSD project (critical!)
        Cpsth_psd, wp_raw, wp_cl = psd_project(Cpsth, eps=psd_eps, return_eigs=True)
        Cfem_psd, wf_raw, wf_cl  = psd_project(Cfem,  eps=psd_eps, return_eigs=True)
        Ctotal_psd, wt_raw, wt_cl = psd_project(Ctotal, eps=psd_eps, return_eigs=True)

        # normalization for spectra
        total_norm = safe_trace(Ctotal_psd)
        spectra["total_norm"].append(total_norm)

        sp_p = eigenspectrum_frac(Cpsth_psd, total_norm)[:max_dims_to_plot]
        sp_f = eigenspectrum_frac(Cfem_psd,  total_norm)[:max_dims_to_plot]
        spectra["psth"].append(sp_p)
        spectra["fem"].append(sp_f)

        # subspaces
        curr_k = min(rep_k, n_valid)
        Up1 = topk_evecs_psd(Cpsth_psd, 1)
        Uf1 = topk_evecs_psd(Cfem_psd,  1)
        Upk = topk_evecs_psd(Cpsth_psd, curr_k)
        Ufk = topk_evecs_psd(Cfem_psd,  curr_k)

        # metrics
        per["pr_fem"].append(participation_ratio(Cfem_psd))
        per["pr_psth"].append(participation_ratio(Cpsth_psd))
        per["overlap_k1"].append(symmetric_subspace_overlap(Up1, Uf1))
        per["overlap_k"].append(symmetric_subspace_overlap(Upk, Ufk))
        per["var_f_given_p"].append(directional_variance_capture(Cfem_psd, Upk))
        per["var_p_given_f"].append(directional_variance_capture(Cpsth_psd, Ufk))

        # diagnostics
        diag["session"].append(dict(
            name=nm, n_valid=n_valid, skipped=False,
            Cpsth_raw=cov_diag_stats(Cpsth),
            Cfem_raw=cov_diag_stats(Cfem),
            Ctotal_raw=cov_diag_stats(Ctotal),
            Cpsth_psd=psd_stats_from_eigs(wp_raw, wp_cl),
            Cfem_psd=psd_stats_from_eigs(wf_raw, wf_cl),
            Ctotal_psd=psd_stats_from_eigs(wt_raw, wt_cl),
        ))

        # shuffle nulls (if present)
        sh_list = mats.get("Shuffled_Intercepts", [])
        if sh_list is None:
            sh_list = []
        if len(sh_list) > 0:
            # limit shuffles for diagnostics speed if needed
            take = min(len(sh_list), int(n_shuffles_diag))
            for s in range(take):
                Cr_s = np.asarray(sh_list[s], dtype=np.float64)
                Cr_s = Cr_s[np.ix_(valid, valid)]
                # FEM for shuffle
                Cf_s = Cr_s - Cpsth  # use same Cpsth (indexed) as real
                Cf_s_psd = psd_project(Cf_s, eps=psd_eps)

                # subspaces in shuffle
                Uf_s_k = topk_evecs_psd(Cf_s_psd, curr_k)
                Uf_s_1 = topk_evecs_psd(Cf_s_psd, 1)

                # null captures/overlap
                null["var_p_given_f"].append(directional_variance_capture(Cpsth_psd, Uf_s_k))
                null["var_f_given_p"].append(directional_variance_capture(Cf_s_psd, Upk))  # FEM|PSTH for shuffle FEM
                null["overlap_k"].append(symmetric_subspace_overlap(Upk, Uf_s_k))
                null["overlap_k1"].append(symmetric_subspace_overlap(Up1, Uf_s_1))

    # convert lists to arrays
    for k in per:
        if k != "name":
            per[k] = np.asarray(per[k], dtype=np.float64)
    # for k in spectra:
    #     spectra[k] = np.asarray(spectra[k], dtype=np.float64)
    # spectra["psth"] and spectra["fem"] are ragged lists of 1D arrays (different lengths per session).
# Keep them as lists; we'll truncate+stack only when plotting summary bands.
    spectra["total_norm"] = np.asarray(spectra["total_norm"], dtype=np.float64)

    for k in null:
        null[k] = np.asarray(null[k], dtype=np.float64)

    out = dict(per=per, spectra=spectra, null=null, diag=diag,
               params=dict(window_idx=window_idx, rep_k=rep_k, psd_eps=psd_eps,
                           min_total_spikes=min_total_spikes, max_dims_to_plot=max_dims_to_plot))
    return out

# ----------------------------
# stats / reporting
# ----------------------------
def _mean_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, np.nan
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(x.size))

def _median_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, (np.nan, np.nan)
    q = np.percentile(x, [25, 50, 75])
    return float(q[1]), (float(q[0]), float(q[2]))

def empirical_p_less(null, obs):
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(obs):
        return np.nan
    return float((np.sum(null <= obs) + 1) / (null.size + 1))

def print_subspace_report(S):
    per = S["per"]
    null = S["null"]
    names = per["name"]

    n_sess = len(names)
    ok = np.isfinite(per["var_f_given_p"]) & np.isfinite(per["var_p_given_f"])
    n_ok = int(np.sum(ok))

    print(f"\n--- SUBSPACE REPORT (window_idx={S['params']['window_idx']}, k={S['params']['rep_k']}) ---")
    print(f"Sessions: {n_sess} (usable: {n_ok})")
    for key, label in [
        ("pr_fem", "Participation ratio (FEM)"),
        ("pr_psth","Participation ratio (PSTH)"),
        ("overlap_k1", "Subspace overlap (k=1)"),
        ("overlap_k",  "Subspace overlap (k=k)"),
        ("var_f_given_p","FEM variance captured by PSTH subspace (Y)"),
        ("var_p_given_f","PSTH variance captured by FEM subspace (X)"),
    ]:
        mu, se = _mean_sem(per[key])
        med, (q1, q3) = _median_iqr(per[key])
        print(f"  {label}: mean={mu:.3f} ± {se:.3f}, median={med:.3f} (IQR [{q1:.3f}, {q3:.3f}])")

    # directional asymmetry summary
    d = per["var_f_given_p"] - per["var_p_given_f"]
    mu, se = _mean_sem(d)
    med, (q1, q3) = _median_iqr(d)
    print(f"  Asymmetry (Y-X): mean={mu:.3f} ± {se:.3f}, median={med:.3f} (IQR [{q1:.3f}, {q3:.3f}])")

    # compare real to null (empirical p-values)
    # Here: we test whether real Y is larger than null Y? depends on framing.
    # But you previously compared "vs chance", so show both:
    real_y = np.nanmean(per["var_f_given_p"])
    real_x = np.nanmean(per["var_p_given_f"])
    p_y = empirical_p_less(null["var_f_given_p"], real_y)  # how often null <= real (i.e., real is "large")
    p_x = empirical_p_less(null["var_p_given_f"], real_x)

    # also show where real is above null 97.5th percentile
    def ci95(a):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (np.nan, np.nan)
        return tuple(np.percentile(a, [2.5, 97.5]))

    y_ci = ci95(null["var_f_given_p"])
    x_ci = ci95(null["var_p_given_f"])
    print("  Null comparisons (flattened over sessions×shuffles):")
    print(f"    Null Y 95% CI: [{y_ci[0]:.3f}, {y_ci[1]:.3f}], real mean(Y)={real_y:.3f}, p_emp(null<=real)={p_y:.3g}")
    print(f"    Null X 95% CI: [{x_ci[0]:.3f}, {x_ci[1]:.3f}], real mean(X)={real_x:.3f}, p_emp(null<=real)={p_x:.3g}")

    # outliers where X > Y
    print("\n  Sessions with X>Y (FEM explains PSTH more than vice versa):")
    for nm, x, y in zip(names, per["var_p_given_f"], per["var_f_given_p"]):
        if np.isfinite(x) and np.isfinite(y) and (x > y):
            print(f"    {nm}:  X={x:.3f}, Y={y:.3f}, (Y-X)={y-x:.3f}")

# ----------------------------
# plotting
# ----------------------------
def _stack_truncated(spec_list, max_dims):
    """
    Ragged -> rectangular by NaN padding (NOT truncating to global min length).

    Accepts:
      - list/tuple of 1D spectra arrays/tensors with varying lengths (ragged)
      - OR a 2D numpy array (n_sessions, n_dims)

    Returns:
      M: (S, L) where L = min(max_dims, max_len_across_sessions)
         padded with NaNs where a session has fewer dims
      L: number of dims returned

    Use np.nanmedian / np.nanpercentile on M to get bands.
    """
    if spec_list is None:
        return np.empty((0, 0), dtype=np.float64), 0

    # Already a proper 2D array
    if isinstance(spec_list, np.ndarray) and spec_list.ndim == 2:
        S, D = spec_list.shape
        L = int(min(D, max_dims))
        return np.asarray(spec_list[:, :L], dtype=np.float64), L

    # Must be list/tuple of per-session spectra
    if not isinstance(spec_list, (list, tuple)):
        # try to coerce; allow 2D
        arr = np.asarray(spec_list)
        if arr.ndim == 2:
            S, D = arr.shape
            L = int(min(D, max_dims))
            return np.asarray(arr[:, :L], dtype=np.float64), L
        raise ValueError(f"_stack_truncated: expected list/tuple or 2D array, got shape {arr.shape}")

    specs = []
    for idx, s in enumerate(spec_list):
        if s is None:
            continue
        if hasattr(s, "detach"):  # torch
            s = s.detach().cpu().numpy()
        s = np.asarray(s)

        # enforce 1D spectrum
        if s.ndim == 2:
            # allow (1, D) or (D, 1)
            if 1 in s.shape:
                s = s.reshape(-1)
            else:
                raise ValueError(
                    f"_stack_truncated: spectrum element {idx} is 2D with shape {s.shape}; "
                    "expected 1D eigenvalue vector"
                )
        elif s.ndim != 1:
            raise ValueError(
                f"_stack_truncated: spectrum element {idx} has ndim={s.ndim}; expected 1D"
            )

        s = s.astype(np.float64, copy=False)
        if s.size > 0:
            specs.append(s)

    if len(specs) == 0:
        return np.empty((0, 0), dtype=np.float64), 0

    max_len = max(s.size for s in specs)
    L = int(min(max_len, max_dims))
    if L <= 0:
        return np.empty((0, 0), dtype=np.float64), 0

    M = np.full((len(specs), L), np.nan, dtype=np.float64)
    for i, s in enumerate(specs):
        Li = min(s.size, L)
        M[i, :Li] = s[:Li]

    return M, L



def plot_subspace_main_figure(S, title=None, max_dims_to_plot=50):
    """
    2-panel main figure:
      (A) spectra PSTH vs FEM (normalized by tr(Ctotal))
      (B) scatter of X vs Y for sessions
    """
    set_pub_style()
    per = S["per"]
    sp = S["spectra"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    # ----------------------------
    # A) spectra panel
    # ----------------------------
    ax = axes[0]

    Mp, Lp = _stack_truncated(sp["psth"], max_dims_to_plot)
    Mf, Lf = _stack_truncated(sp["fem"],  max_dims_to_plot)

    L = int(min(Lp, Lf))
    if L >= 1:
        Mp = Mp[:, :L]
        Mf = Mf[:, :L]
        dims = np.arange(1, L + 1)

        # light per-session spectra
        for i in range(Mp.shape[0]):
            ax.loglog(dims, Mp[i], alpha=0.12, lw=1)
            ax.loglog(dims, Mf[i], alpha=0.12, lw=1)

        # median + IQR band (your “median band” behavior; truncation = your original logic)
        p_med = np.nanmedian(Mp, axis=0)
        p_lo  = np.nanpercentile(Mp, 25, axis=0)
        p_hi  = np.nanpercentile(Mp, 75, axis=0)


        f_med = np.median(Mf, axis=0)
        f_lo  = np.percentile(Mf, 25, axis=0)
        f_hi  = np.percentile(Mf, 75, axis=0)

        ax.fill_between(dims, p_lo, p_hi, alpha=0.18)
        ax.fill_between(dims, f_lo, f_hi, alpha=0.18)
        ax.loglog(dims, p_med, lw=2.8, label="Stimulus (PSTH)")
        ax.loglog(dims, f_med, lw=2.8, label="Eye movement (FEM)")

        print(f"Dim L = {L}")
        ax.set_xlim(1, 100)
    else:
        ax.text(0.5, 0.5, "No valid spectra to plot", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Dimension (PC)")
    ax.set_ylabel("Fraction of total shared variance")
    ax.set_title("Eigenspectra magnitude")
    ax.legend(frameon=False, loc="best")
    ax.set_ylim(1e-6, 1)

    # ----------------------------
    # B) scatter panel
    # ----------------------------
    ax = axes[1]
    x = np.asarray(per["var_p_given_f"], dtype=np.float64)  # PSTH var captured by FEM
    y = np.asarray(per["var_f_given_p"], dtype=np.float64)  # FEM var captured by PSTH
    ok = np.isfinite(x) & np.isfinite(y)

    ax.scatter(x[ok], y[ok], s=80, alpha=0.9, edgecolors="white", linewidths=0.8)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("PSTH var captured by FEM subspace (X)\n(Do eyes explain the image?)")
    ax.set_ylabel("FEM var captured by PSTH subspace (Y)\n(Does image explain the eyes?)")
    ax.set_title(f"Subspace alignment (k={S['params']['rep_k']})")

    mx, my = np.nanmean(x), np.nanmean(y)
    ax.text(
        0.05, 0.95, f"Mean Y={my:.2f}\nMean X={mx:.2f}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)

    return fig, axes


def plot_subspace_vs_shuffle_figure(S, title=None):
    """
    Supplemental: real vs chance scatter + null marginals.
    """
    set_pub_style()
    per = S["per"]
    null = S["null"]

    x = np.asarray(per["var_p_given_f"], dtype=np.float64)
    y = np.asarray(per["var_f_given_p"], dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)

    xs = np.asarray(null["var_p_given_f"], dtype=np.float64).reshape(-1)
    ys = np.asarray(null["var_f_given_p"], dtype=np.float64).reshape(-1)
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)

    # scatter
    ax = axes[0]
    if xs.size and ys.size:
        ax.scatter(xs, ys, s=10, alpha=0.15, color="gray", label="Shuffled (chance)")
        ax.plot(np.mean(xs), np.mean(ys), "kx", ms=10, mew=2, label="Chance mean")
    ax.scatter(x[ok], y[ok], s=90, alpha=0.9, edgecolors="white", linewidths=0.8, label="Real sessions")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("X: PSTH var captured by FEM subspace")
    ax.set_ylabel("Y: FEM var captured by PSTH subspace")
    ax.set_title("Real vs chance")
    ax.legend(frameon=False, loc="best")

    # marginal null histograms with real mean overlay
    ax = axes[1]
    ax.hist(xs, bins=np.linspace(0, 1, 60), density=True, alpha=0.6, color="gray")
    ax.axvline(np.nanmean(x), color="k", linestyle="--", lw=1.5)
    ax.set_xlabel("X (null)")
    ax.set_ylabel("Density")
    ax.set_title("Null marginal: X")

    ax = axes[2]
    ax.hist(ys, bins=np.linspace(0, 1, 60), density=True, alpha=0.6, color="gray")
    ax.axvline(np.nanmean(y), color="k", linestyle="--", lw=1.5)
    ax.set_xlabel("Y (null)")
    ax.set_ylabel("Density")
    ax.set_title("Null marginal: Y")

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)

    return fig, axes