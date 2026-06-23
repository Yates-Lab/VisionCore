# %% Imports and configuration
"""
Per-session overview figures for Figure 2 data.

Loads the cached decomposition from generate_figure2.py and produces
a two-page PDF per session: page 1 for QC metrics, page 2 for all
analyses that feed the population analysis.

Requires running generate_figure2.py with RECOMPUTE=True first to
populate the cache with PSTH and QC fields.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import dill

from VisionCore.paths import CACHE_DIR, FIGURES_DIR
from VisionCore.covariance import (
    cov_to_corr,
    project_to_psd,
    get_upper_triangle,
)
from VisionCore.stats import fisher_z
from VisionCore.subspace import (
    participation_ratio,
    symmetric_subspace_overlap,
    directional_variance_capture,
)

# Matplotlib publication defaults
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# ---------------------------------------------------------------------------
# Parameters (must match generate_figure2.py)
# ---------------------------------------------------------------------------
DT = 1 / 240
WINDOW_BINS = [2, 4, 8, 16]
MIN_TOTAL_SPIKES = 500
MIN_VAR = 0
EPS_RHO = 1e-3
SUBSPACE_K = 5
HEATMAP_WINDOW_IDX = 3  # 80 ms window for covariance heatmaps

# Output
FIG_DIR = FIGURES_DIR / "fig2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = FIG_DIR / "session_overviews.pdf"

# %% Load cached decomposition
cache_path = CACHE_DIR / "fig2_decomposition.pkl"
assert cache_path.exists(), (
    f"Cache not found at {cache_path}. "
    "Run generate_figure2.py with RECOMPUTE=True first."
)

with open(cache_path, "rb") as f:
    session_results = dill.load(f)

# Validate cache has extended fields
sr0 = session_results[0]
assert "psth" in sr0, "Cache missing 'psth' field. Rerun generate_figure2.py with RECOMPUTE=True."
assert "qc" in sr0, "Cache missing 'qc' field. Rerun generate_figure2.py with RECOMPUTE=True."

WINDOWS_MS = [r["window_ms"] for r in sr0["results"]]
n_sessions = len(session_results)
print(f"Loaded {n_sessions} sessions")
print(f"Windows (ms): {[f'{w:.1f}' for w in WINDOWS_MS]}")


# %% Page 1: QC Overview

def plot_qc_page(sr, fig):
    """Plot QC overview page for a single session.

    Parameters
    ----------
    sr : dict
        Single session entry from session_results.
    fig : matplotlib.figure.Figure
        Figure to draw on.
    """
    session_name = sr["session"]
    subject = sr["subject"]
    meta = sr["meta"]
    psth = sr["psth"]  # (n_time, n_neurons_used)
    contam_rate_raw = sr["qc"]["contam_rate"]
    neuron_mask = sr["neuron_mask"]

    # Align contam_rate to the neuron subset used by the model
    if contam_rate_raw is not None and neuron_mask is not None:
        contam_rate = contam_rate_raw[neuron_mask]
    else:
        contam_rate = contam_rate_raw

    # Erates from first window for firing rate display
    res0 = sr["results"][0]
    erate = res0["Erates"]
    n_samples = res0["n_samples"]

    # Neuron inclusion mask (same logic as generate_figure2.py metrics extraction)
    mats0 = sr["mats"][0]
    Ctotal = mats0["Total"]
    Crate = mats0["Intercept"]
    Cpsth = mats0["PSTH"]
    valid = (
        np.isfinite(erate)
        & (erate * n_samples >= MIN_TOTAL_SPIKES)
        & (np.diag(Ctotal) > MIN_VAR)
        & np.isfinite(np.diag(Crate))
        & np.isfinite(np.diag(Cpsth))
    )
    n_valid = valid.sum()
    n_total = len(erate)

    fig.suptitle(
        f"{session_name} ({subject}) — QC Overview\n"
        f"Neurons: {n_valid} included / {n_total} total",
        fontsize=13,
    )

    axs = fig.subplots(2, 2)

    # --- Top-left: PSTH heatmap ---
    ax = axs[0, 0]
    psth_T = psth.T  # (n_neurons_used, n_time)
    peak_order = np.argsort(np.max(psth_T, axis=1))[::-1]
    psth_sorted = psth_T[peak_order]
    im = ax.imshow(
        psth_sorted, aspect="auto", interpolation="nearest", cmap="hot",
    )
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Neuron (sorted by peak rate)")
    ax.set_title("Trial-averaged PSTH")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="spk/bin")

    # --- Top-right: Firing rate bar chart ---
    ax = axs[0, 1]
    erate_hz = erate[valid] / DT  # counts/bin -> Hz
    rate_order = np.argsort(erate_hz)[::-1]
    ax.bar(range(n_valid), erate_hz[rate_order], color="steelblue", width=1.0)
    ax.set_xlabel("Neuron (sorted)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Firing rates (included neurons)")

    # --- Bottom-left: Total spike count ---
    ax = axs[1, 0]
    total_spikes = erate * n_samples
    spike_order = np.argsort(total_spikes)[::-1]
    colors = ["steelblue" if valid[i] else "lightcoral" for i in spike_order]
    ax.bar(range(n_total), total_spikes[spike_order], color=colors, width=1.0)
    ax.axhline(MIN_TOTAL_SPIKES, color="red", linestyle="--", linewidth=1,
               label=f"threshold={MIN_TOTAL_SPIKES}")
    ax.set_xlabel("Neuron (sorted)")
    ax.set_ylabel("Total spike count")
    ax.set_title("Spike counts (red = excluded)")
    ax.legend(fontsize=7, frameon=False)

    # --- Bottom-right: Contamination rate ---
    ax = axs[1, 1]
    if contam_rate is not None:
        contam_pct = contam_rate * 100
        contam_order = np.argsort(contam_pct)
        colors_c = [
            "steelblue" if valid[i] else "lightcoral" for i in contam_order
        ]
        ax.bar(range(n_total), contam_pct[contam_order], color=colors_c, width=1.0)
        ax.set_xlabel("Neuron (sorted)")
        ax.set_ylabel("Min contamination (%)")
        ax.set_title("Refractory contamination")
    else:
        ax.text(0.5, 0.5, "QC data not available\nfor this session",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("Refractory contamination")
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.93])


# %% Page 2: Analysis Overview

def plot_analysis_page(sr, fig):
    """Plot analysis overview page for a single session.

    Parameters
    ----------
    sr : dict
        Single session entry from session_results.
    fig : matplotlib.figure.Figure
        Figure to draw on.
    """
    session_name = sr["session"]
    subject = sr["subject"]

    fig.suptitle(
        f"{session_name} ({subject}) — Analysis Overview",
        fontsize=13,
    )

    # GridSpec: rows 0-1 are 3 columns, row 2 is 5 columns
    gs = GridSpec(3, 15, figure=fig, hspace=0.45, wspace=0.5,
                  top=0.92, bottom=0.06)

    # Row 0: 3 equal panels (each 5 columns wide)
    ax_alpha = fig.add_subplot(gs[0, 0:5])
    ax_meanvar = fig.add_subplot(gs[0, 5:10])
    ax_fano_win = fig.add_subplot(gs[0, 10:15])

    # Row 1: 3 equal panels
    ax_fisherz = fig.add_subplot(gs[1, 0:5])
    ax_deltaz = fig.add_subplot(gs[1, 5:10])
    ax_eigen = fig.add_subplot(gs[1, 10:15])

    # Row 2: 4 heatmaps (3 cols each) + scorecard (3 cols)
    ax_cov_total = fig.add_subplot(gs[2, 0:3])
    ax_cov_fem = fig.add_subplot(gs[2, 3:6])
    ax_cov_psth = fig.add_subplot(gs[2, 6:9])
    ax_cov_noise = fig.add_subplot(gs[2, 9:12])
    ax_scorecard = fig.add_subplot(gs[2, 12:15])

    # ---------------------------------------------------------------
    # Compute per-window metrics for this session
    # ---------------------------------------------------------------
    n_windows = len(sr["results"])
    windows_ms = [r["window_ms"] for r in sr["results"]]
    per_window = []

    for w_idx in range(n_windows):
        res = sr["results"][w_idx]
        mats = sr["mats"][w_idx]

        Ctotal = mats["Total"]
        Cpsth = mats["PSTH"]
        Crate = mats["Intercept"]
        CnoiseU = 0.5 * ((Ctotal - Cpsth) + (Ctotal - Cpsth).T)
        CnoiseC = 0.5 * ((Ctotal - Crate) + (Ctotal - Crate).T)

        erate = res["Erates"]
        n_samples = res["n_samples"]
        total_spikes = erate * n_samples

        valid = (
            np.isfinite(erate)
            & (total_spikes >= MIN_TOTAL_SPIKES)
            & (np.diag(Ctotal) > MIN_VAR)
            & np.isfinite(np.diag(Crate))
            & np.isfinite(np.diag(Cpsth))
        )
        if valid.sum() < 3:
            per_window.append(None)
            continue

        # Alpha
        diag_psth = np.diag(Cpsth)[valid]
        diag_rate = np.diag(Crate)[valid]
        alpha = np.clip(diag_psth / diag_rate, 0, 1)

        # Fano factors
        ff_u = np.diag(CnoiseU)[valid] / erate[valid]
        ff_c = np.diag(CnoiseC)[valid] / erate[valid]
        erate_v = erate[valid]

        # Variance for mean-var scatter
        var_u = ff_u * erate_v
        var_c = ff_c * erate_v

        # Population FF (slope through origin)
        ok_ff = (np.isfinite(ff_u) & np.isfinite(ff_c)
                 & (ff_u > 0) & (ff_c > 0) & (erate_v > 0))
        e_ok = erate_v[ok_ff]
        vu_ok = var_u[ok_ff]
        vc_ok = var_c[ok_ff]
        if len(e_ok) > 0:
            slope_unc = float(np.sum(e_ok * vu_ok) / np.sum(e_ok ** 2))
            slope_cor = float(np.sum(e_ok * vc_ok) / np.sum(e_ok ** 2))
        else:
            slope_unc = np.nan
            slope_cor = np.nan

        # Pairwise noise correlations
        NoiseCorrU = cov_to_corr(CnoiseU[np.ix_(valid, valid)], min_var=MIN_VAR)
        NoiseCorrC = cov_to_corr(CnoiseC[np.ix_(valid, valid)], min_var=MIN_VAR)
        rho_u = get_upper_triangle(NoiseCorrU)
        rho_c = get_upper_triangle(NoiseCorrC)
        pair_ok = np.isfinite(rho_u) & np.isfinite(rho_c)
        rho_u = rho_u[pair_ok]
        rho_c = rho_c[pair_ok]

        # Fisher z
        z_u = fisher_z(rho_u, eps=EPS_RHO)
        z_c = fisher_z(rho_c, eps=EPS_RHO)
        z_u_mean = float(np.nanmean(z_u))
        z_c_mean = float(np.nanmean(z_c))
        z_u_sem = float(np.nanstd(z_u) / np.sqrt(np.sum(np.isfinite(z_u))))
        z_c_sem = float(np.nanstd(z_c) / np.sqrt(np.sum(np.isfinite(z_c))))
        dz = z_c - z_u
        dz_mean = float(np.nanmean(dz))
        dz_sem = float(np.nanstd(dz) / np.sqrt(np.sum(np.isfinite(dz))))

        per_window.append({
            "alpha": alpha,
            "erate_v": erate_v,
            "var_u": var_u, "var_c": var_c,
            "slope_unc": slope_unc, "slope_cor": slope_cor,
            "z_u_mean": z_u_mean, "z_c_mean": z_c_mean,
            "z_u_sem": z_u_sem, "z_c_sem": z_c_sem,
            "dz_mean": dz_mean, "dz_sem": dz_sem,
        })

    # ---------------------------------------------------------------
    # Row 0, Col 0: 1-alpha histogram (first window)
    # ---------------------------------------------------------------
    pw0 = per_window[0]
    if pw0 is not None:
        m0 = 1 - pw0["alpha"]
        ax_alpha.hist(m0, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
        ax_alpha.axvline(np.nanmedian(m0), color="red", linewidth=2,
                         label=f"median={np.nanmedian(m0):.3f}")
        ax_alpha.set_xlabel("1 - α")
        ax_alpha.set_ylabel("Count")
        ax_alpha.set_title(f"FEM modulation ({windows_ms[0]:.1f} ms)")
        ax_alpha.legend(fontsize=7, frameon=False)

    # ---------------------------------------------------------------
    # Row 0, Col 1: Mean-rate vs variance scatter (first window)
    # ---------------------------------------------------------------
    if pw0 is not None:
        ax_meanvar.scatter(pw0["erate_v"], pw0["var_u"], s=8, alpha=0.4,
                           c="tab:blue", label=f"Uncorr FF={pw0['slope_unc']:.3f}")
        ax_meanvar.scatter(pw0["erate_v"], pw0["var_c"], s=8, alpha=0.4,
                           c="tab:red", label=f"Corr FF={pw0['slope_cor']:.3f}")
        x_line = np.linspace(0, pw0["erate_v"].max(), 100)
        ax_meanvar.plot(x_line, pw0["slope_unc"] * x_line, "b--", linewidth=1)
        ax_meanvar.plot(x_line, pw0["slope_cor"] * x_line, "r--", linewidth=1)
        ax_meanvar.set_xlabel("Mean rate")
        ax_meanvar.set_ylabel("Variance")
        ax_meanvar.set_title(f"Mean-variance ({windows_ms[0]:.1f} ms)")
        ax_meanvar.legend(fontsize=7, frameon=False)

    # ---------------------------------------------------------------
    # Row 0, Col 2: Fano factor vs window
    # ---------------------------------------------------------------
    slopes_u = [pw["slope_unc"] if pw else np.nan for pw in per_window]
    slopes_c = [pw["slope_cor"] if pw else np.nan for pw in per_window]
    ax_fano_win.plot(windows_ms, slopes_u, "o-", color="tab:blue", label="Uncorrected")
    ax_fano_win.plot(windows_ms, slopes_c, "o-", color="tab:red", label="Corrected")
    ax_fano_win.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Poisson")
    ax_fano_win.set_xlabel("Window (ms)")
    ax_fano_win.set_ylabel("Population FF")
    ax_fano_win.set_title("Fano factor vs window")
    ax_fano_win.set_xscale("log")
    ax_fano_win.set_xticks(windows_ms)
    ax_fano_win.set_xticklabels([f"{w:.0f}" for w in windows_ms])
    ax_fano_win.legend(fontsize=7, frameon=False)

    # ---------------------------------------------------------------
    # Row 1, Col 0: Mean Fisher z vs window
    # ---------------------------------------------------------------
    zu_means = [pw["z_u_mean"] if pw else np.nan for pw in per_window]
    zc_means = [pw["z_c_mean"] if pw else np.nan for pw in per_window]
    zu_sems = [pw["z_u_sem"] if pw else np.nan for pw in per_window]
    zc_sems = [pw["z_c_sem"] if pw else np.nan for pw in per_window]
    ax_fisherz.errorbar(windows_ms, zu_means, yerr=zu_sems,
                        fmt="o-", color="tab:blue", capsize=3, label="Uncorrected")
    ax_fisherz.errorbar(windows_ms, zc_means, yerr=zc_sems,
                        fmt="o-", color="tab:red", capsize=3, label="Corrected")
    ax_fisherz.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax_fisherz.set_xlabel("Window (ms)")
    ax_fisherz.set_ylabel("Mean Fisher z")
    ax_fisherz.set_title("Noise corr vs window")
    ax_fisherz.set_xscale("log")
    ax_fisherz.set_xticks(windows_ms)
    ax_fisherz.set_xticklabels([f"{w:.0f}" for w in windows_ms])
    ax_fisherz.legend(fontsize=7, frameon=False)

    # ---------------------------------------------------------------
    # Row 1, Col 1: Delta Fisher z vs window
    # ---------------------------------------------------------------
    dz_means = [pw["dz_mean"] if pw else np.nan for pw in per_window]
    dz_sems = [pw["dz_sem"] if pw else np.nan for pw in per_window]
    ax_deltaz.errorbar(windows_ms, dz_means, yerr=dz_sems,
                       fmt="o-", color="black", capsize=3)
    ax_deltaz.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax_deltaz.set_xlabel("Window (ms)")
    ax_deltaz.set_ylabel("Δz (corr - uncorr)")
    ax_deltaz.set_title("Effect size vs window")
    ax_deltaz.set_xscale("log")
    ax_deltaz.set_xticks(windows_ms)
    ax_deltaz.set_xticklabels([f"{w:.0f}" for w in windows_ms])

    # ---------------------------------------------------------------
    # Row 1, Col 2: Eigenspectrum (PSTH + FEM)
    # ---------------------------------------------------------------
    sub_w_idx = min(1, n_windows - 1)
    res_sub = sr["results"][sub_w_idx]
    mats_sub = sr["mats"][sub_w_idx]
    erate_sub = res_sub["Erates"]
    n_samples_sub = res_sub["n_samples"]
    Ctotal_sub = mats_sub["Total"]
    Cpsth_sub = mats_sub["PSTH"]
    Crate_sub = mats_sub["Intercept"]
    Cfem_sub = Crate_sub - Cpsth_sub
    total_spikes_sub = erate_sub * n_samples_sub
    valid_sub = (
        np.isfinite(erate_sub)
        & (total_spikes_sub >= MIN_TOTAL_SPIKES)
        & (np.diag(Ctotal_sub) > MIN_VAR)
        & np.isfinite(np.diag(Crate_sub))
        & np.isfinite(np.diag(Cpsth_sub))
    )

    # Scorecard values (defaults)
    pr_psth_val = np.nan
    pr_fem_val = np.nan
    overlap_k1 = np.nan
    overlap_k5 = np.nan
    var_capture_x = np.nan
    var_capture_y = np.nan

    if valid_sub.sum() > SUBSPACE_K + 1:
        ix_sub = np.ix_(valid_sub, valid_sub)
        Cpsth_psd = project_to_psd(Cpsth_sub[ix_sub])
        Cfem_psd = project_to_psd(Cfem_sub[ix_sub])
        Ctotal_v = Ctotal_sub[ix_sub]

        w_psth, V_psth = np.linalg.eigh(Cpsth_psd)
        w_fem, V_fem = np.linalg.eigh(Cfem_psd)
        w_psth, V_psth = w_psth[::-1], V_psth[:, ::-1]
        w_fem, V_fem = w_fem[::-1], V_fem[:, ::-1]

        tr_total = np.trace(Ctotal_v)
        spec_psth = w_psth / tr_total
        spec_fem = w_fem / tr_total

        dims = np.arange(1, len(spec_psth) + 1)
        ax_eigen.plot(dims, spec_psth, color="tab:blue", label="PSTH")
        ax_eigen.plot(dims, spec_fem, color="tab:red", label="FEM")
        ax_eigen.set_xscale("log")
        ax_eigen.set_yscale("log")
        ax_eigen.set_xlabel("Eigenvalue rank")
        ax_eigen.set_ylabel("Frac. total var")
        ax_eigen.set_title(f"Eigenspectra ({windows_ms[sub_w_idx]:.1f} ms)")
        ax_eigen.legend(fontsize=7, frameon=False)

        # Scorecard computations
        pr_psth_val = participation_ratio(Cpsth_psd)
        pr_fem_val = participation_ratio(Cfem_psd)

        k = min(SUBSPACE_K, valid_sub.sum() - 1)
        U_psth = V_psth[:, :k]
        U_fem = V_fem[:, :k]
        overlap_k5 = symmetric_subspace_overlap(U_psth, U_fem)
        overlap_k1 = symmetric_subspace_overlap(V_psth[:, :1], V_fem[:, :1])
        var_capture_x = directional_variance_capture(Cpsth_psd, U_fem)
        var_capture_y = directional_variance_capture(Cfem_psd, U_psth)
    else:
        ax_eigen.text(0.5, 0.5, "Insufficient neurons",
                      ha="center", va="center", transform=ax_eigen.transAxes)
        ax_eigen.set_title("Eigenspectra")

    # ---------------------------------------------------------------
    # Row 2: Covariance heatmaps (80 ms window)
    # ---------------------------------------------------------------
    hm_idx = min(HEATMAP_WINDOW_IDX, n_windows - 1)
    mats_hm = sr["mats"][hm_idx]
    Crate_hm = mats_hm["Intercept"]
    hm_valid = np.isfinite(np.diag(Crate_hm)) & np.isfinite(np.diag(mats_hm["PSTH"]))
    ix_hm = np.ix_(hm_valid, hm_valid)

    Ctotal_hm = project_to_psd(mats_hm["Total"][ix_hm])
    Cpsth_hm = project_to_psd(mats_hm["PSTH"][ix_hm])
    Cfem_hm = project_to_psd(Crate_hm[ix_hm] - mats_hm["PSTH"][ix_hm])
    CnoiseC_hm = project_to_psd(mats_hm["Total"][ix_hm] - Crate_hm[ix_hm])

    # PR for heatmap window
    pr_psth_hm = participation_ratio(Cpsth_hm)
    pr_fem_hm = participation_ratio(Cfem_hm)

    cmap = plt.get_cmap("RdBu")

    if Ctotal_hm.size > 0:
        v = np.nanmax(np.abs(Ctotal_hm)) * 0.5
        for ax, mat, title in [
            (ax_cov_total, Ctotal_hm, "Total"),
            (ax_cov_fem, Cfem_hm, f"FEM (PR={pr_fem_hm:.1f})"),
            (ax_cov_psth, Cpsth_hm, f"PSTH (PR={pr_psth_hm:.1f})"),
            (ax_cov_noise, CnoiseC_hm, "Noise (Corr)"),
        ]:
            vscale = 0.5 if "PSTH" in title else 1.0
            ax.imshow(mat, cmap=cmap, interpolation="nearest",
                      vmin=-v * vscale, vmax=v * vscale)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    else:
        for ax in [ax_cov_total, ax_cov_fem, ax_cov_psth, ax_cov_noise]:
            ax.text(0.5, 0.5, "Insufficient neurons",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    # ---------------------------------------------------------------
    # Row 2, Col 4: Scorecard
    # ---------------------------------------------------------------
    ax_scorecard.axis("off")
    scorecard_text = (
        f"PR(PSTH): {pr_psth_val:.2f}\n"
        f"PR(FEM):  {pr_fem_val:.2f}\n"
        f"\n"
        f"Overlap k=1: {overlap_k1:.3f}\n"
        f"Overlap k={SUBSPACE_K}: {overlap_k5:.3f}\n"
        f"\n"
        f"X (PSTH→FEM): {var_capture_x:.3f}\n"
        f"Y (FEM→PSTH): {var_capture_y:.3f}"
    )
    ax_scorecard.text(
        0.1, 0.95, scorecard_text,
        transform=ax_scorecard.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
    )
    ax_scorecard.set_title("Scorecard", fontsize=9)


# %% Generate all session overview PDFs

with PdfPages(PDF_PATH) as pdf:
    for ds_idx, sr in enumerate(session_results):
        session_name = sr["session"]
        print(f"[{ds_idx + 1}/{n_sessions}] {session_name}")

        # Page 1: QC
        fig1 = plt.figure(figsize=(12, 8))
        plot_qc_page(sr, fig1)
        pdf.savefig(fig1, bbox_inches="tight", dpi=150)
        #plt.close(fig1)
        plt.show()

        # Page 2: Analysis
        fig2 = plt.figure(figsize=(16, 12))
        plot_analysis_page(sr, fig2)
        pdf.savefig(fig2, bbox_inches="tight", dpi=150)
        #plt.close(fig2)
        plt.show()
        plt.close("all")

print(f"\nDone. {n_sessions} sessions ({n_sessions * 2} pages) saved to {PDF_PATH}")
