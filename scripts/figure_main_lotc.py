# scripts/figure_main_lotc.py
"""
Unified main figure for LOTC manuscript (Panels C-K).

Layout (3 rows x 3 columns):
  C: Alpha distribution histogram (1 - alpha)
  D: Mean-variance relationship (Fano factors, single window)
  E: Population Fano factor vs counting window
  
  F: Noise correlations scatter (corrected vs uncorrected, single window)
  G: Mean noise correlations vs window
  H: Effect size vs shuffle control
  
  I: Eigenspectra (PSTH vs FEM)
  J: Subspace alignment visualization (placeholder - needs specific plot)
  K: Subspace summary (X vs Y scatter with shuffle)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import shared utilities (use relative import for same-directory scripts)
try:
    from figure_common import set_pub_style, _finite, iqr_25_75, bootstrap_mean_ci
except ImportError:
    from scripts.figure_common import set_pub_style, _finite, iqr_25_75, bootstrap_mean_ci


def create_main_figure(
    metrics,
    fano_stats,
    nc_stats,
    alpha_stats,
    subspace_stats,
    window_idx=0,  # which window to use for single-window panels (C, D, F)
    figsize=(11, 9),
    savepath=None,
):
    """
    Create unified main figure with panels C-K.
    
    Parameters
    ----------
    metrics : list of dict
        Per-window metrics from mcfarland analysis
    fano_stats : dict
        Output from compute_all_fano_stats
    nc_stats : dict  
        Output from compute_all_noisecorr_stats
    alpha_stats : dict
        Output from compute_all_alpha_stats
    subspace_stats : dict
        Output from compute_subspace_stats
    window_idx : int
        Index into metrics for single-window panels
    figsize : tuple
        Figure size
    savepath : str, optional
        Path to save figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : dict of axes keyed by panel letter
    """
    set_pub_style()
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    axs = {}
    axs['C'] = fig.add_subplot(gs[0, 0])
    axs['D'] = fig.add_subplot(gs[0, 1])
    axs['E'] = fig.add_subplot(gs[0, 2])
    axs['F'] = fig.add_subplot(gs[1, 0])
    axs['G'] = fig.add_subplot(gs[1, 1])
    axs['H'] = fig.add_subplot(gs[1, 2])
    axs['I'] = fig.add_subplot(gs[2, 0])
    axs['J'] = fig.add_subplot(gs[2, 1])
    axs['K'] = fig.add_subplot(gs[2, 2])
    
    # Get window info
    m = metrics[window_idx]
    window_ms = int(m['window_ms'])
    
    # --- Panel C: Alpha histogram ---
    _plot_alpha_histogram(axs['C'], m, window_ms)
    
    # --- Panel D: Mean-variance scatter ---
    _plot_mean_variance(axs['D'], m, window_ms)
    
    # --- Panel E: Population Fano vs window ---
    _plot_fano_vs_window(axs['E'], fano_stats)
    
    # --- Panel F: Noise correlation scatter ---
    _plot_noisecorr_scatter(axs['F'], m, window_ms)
    
    # --- Panel G: Mean noise correlations vs window ---
    _plot_noisecorr_vs_window(axs['G'], nc_stats)
    
    # --- Panel H: Effect size vs shuffle ---
    _plot_effect_size_vs_shuffle(axs['H'], nc_stats)
    
    # --- Panel I: Eigenspectra ---
    _plot_eigenspectra(axs['I'], subspace_stats)
    
    # --- Panel J: Subspace alignment visualization ---
    _plot_subspace_viz(axs['J'], subspace_stats)
    
    # --- Panel K: Subspace summary scatter ---
    _plot_subspace_summary(axs['K'], subspace_stats)
    
    # Add panel labels (lowercase)
    for letter, ax in axs.items():
        ax.text(-0.12, 1.08, letter.lower(), transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')
    
    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=300)
        print(f"Saved: {savepath}")
    
    return fig, axs


# ----------------------------
# Individual panel plotting functions
# ----------------------------

def _plot_alpha_histogram(ax, m, window_ms):
    """Panel C: Distribution of 1-alpha."""
    alpha = np.asarray(m.get('alpha', []), dtype=float)
    one_minus_alpha = 1.0 - alpha
    one_minus_alpha = one_minus_alpha[np.isfinite(one_minus_alpha)]

    ax.hist(one_minus_alpha, bins=np.linspace(0, 1, 31), alpha=0.75, color='0.5', edgecolor='white')
    
    if one_minus_alpha.size:
        med = np.median(one_minus_alpha)
        mean_val = np.mean(one_minus_alpha)
        ax.axvline(med, color='k', linestyle='--', lw=1.5, label=f'Median={med:.2f}')
        ax.axvline(mean_val, color='crimson', linestyle='--', lw=1.5, label=f'Mean={mean_val:.2f}')
    
    ax.set_xlabel('Fraction rate modulation\ndue to FEM (1 - α)')
    ax.set_ylabel('Count')
    ax.set_title(f'Window {window_ms}ms')
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8)


# Define consistent colors for uncorrected (blue) and corrected (green)
COLOR_UNCORR = 'C0'  # blue
COLOR_CORR = '#2ca02c'  # nice green


def _plot_mean_variance(ax, m, window_ms):
    """Panel D: Mean-variance relationship with Fano fits."""
    erate = np.asarray(m.get('erate', []), dtype=float)
    ff_unc = np.asarray(m.get('uncorr', []), dtype=float)
    ff_cor = np.asarray(m.get('corr', []), dtype=float)

    # variance = fano * mean
    var_unc = ff_unc * erate
    var_cor = ff_cor * erate

    ok = np.isfinite(erate) & np.isfinite(var_unc) & np.isfinite(var_cor) & (erate > 0)
    x, yu, yc = erate[ok], var_unc[ok], var_cor[ok]

    ax.scatter(x, yu, s=12, alpha=0.5, color=COLOR_UNCORR, label='Uncorrected')
    ax.scatter(x, yc, s=12, alpha=0.5, color=COLOR_CORR, label='Corrected')

    # Fit slopes through origin
    if x.size >= 3:
        slope_unc = np.dot(x, yu) / np.dot(x, x)
        slope_cor = np.dot(x, yc) / np.dot(x, x)
        xfit = np.linspace(0, np.max(x), 50)
        ax.plot(xfit, slope_unc * xfit, '-', color=COLOR_UNCORR, lw=2, label=f'Uncorrected FF = {slope_unc:.2f}')
        ax.plot(xfit, slope_cor * xfit, '-', color=COLOR_CORR, lw=2, label=f'Corrected FF = {slope_cor:.2f}')

    ax.set_xlabel('Mean Rate (spk/s)')
    ax.set_ylabel('Variance')
    ax.set_title(f'Window {window_ms}ms')
    ax.legend(frameon=False, fontsize=7, loc='upper left')


def _plot_fano_vs_window(ax, stats):
    """Panel E: Population Fano (slope) vs counting window."""
    w = np.asarray(stats['window_ms'])
    order = np.argsort(w)
    w = w[order]

    slope_u = np.asarray(stats['slope_unc'])[order]
    slope_c = np.asarray(stats['slope_cor'])[order]
    slope_u_ci = np.asarray(stats['slope_unc_ci'])[order]
    slope_c_ci = np.asarray(stats['slope_cor_ci'])[order]

    u_yerr = np.vstack([slope_u - slope_u_ci[:, 0], slope_u_ci[:, 1] - slope_u])
    c_yerr = np.vstack([slope_c - slope_c_ci[:, 0], slope_c_ci[:, 1] - slope_c])

    ax.errorbar(w, slope_u, yerr=u_yerr, fmt='o-', capsize=3, lw=1.5, color=COLOR_UNCORR, label='Uncorrected slope-FF (95% CI)')
    ax.errorbar(w, slope_c, yerr=c_yerr, fmt='o-', capsize=3, lw=1.5, color=COLOR_CORR, label='Corrected slope-FF (95% CI)')
    ax.axhline(1.0, ls='--', lw=1, alpha=0.6, color='gray')

    ax.set_xlabel('Counting Window (ms)')
    ax.set_ylabel('Slope Fano (Var vs Mean)')
    ax.set_title('Population Fano (Churchland slope)')
    ax.legend(frameon=False, fontsize=7)


def _plot_noisecorr_scatter(ax, m, window_ms):
    """Panel F: Noise correlations scatter (corrected vs uncorrected)."""
    rho_unc = np.asarray(m.get('rho_uncorr', []), dtype=float)
    rho_cor = np.asarray(m.get('rho_corr', []), dtype=float)

    ok = np.isfinite(rho_unc) & np.isfinite(rho_cor)
    x, y = rho_unc[ok], rho_cor[ok]

    # 2D histogram for density (grayscale: white to black)
    bins = np.linspace(-0.4, 0.4, 81)
    h, xe, ye = np.histogram2d(x, y, bins=[bins, bins])
    h = np.log1p(h)

    ax.imshow(h.T, origin='lower', extent=[xe[0], xe[-1], ye[0], ye[-1]],
              aspect='auto', cmap='Greys', interpolation='nearest')

    # means
    ax.plot(np.mean(x), np.mean(y), 'ro', ms=8, label='Mean')
    ax.plot([-0.4, 0.4], [-0.4, 0.4], 'k--', alpha=0.5, lw=1)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, alpha=0.3)

    ax.set_xlabel('ρ (uncorrected)')
    ax.set_ylabel('ρ (corrected)')
    ax.set_title(f'Window {window_ms}ms')
    ax.legend(frameon=False, fontsize=8, loc='upper left')


def _plot_noisecorr_vs_window(ax, stats):
    """Panel G: Mean Fisher-z noise correlations vs window."""
    w = np.asarray(stats['window_ms'], dtype=float)
    order = np.argsort(w)
    w = w[order]

    z_u = np.asarray(stats['z_u_mean'])[order]
    z_c = np.asarray(stats['z_c_mean'])[order]
    z_u_ci = np.asarray(stats['z_u_ci'])[order]
    z_c_ci = np.asarray(stats['z_c_ci'])[order]

    u_yerr = np.vstack([z_u - z_u_ci[:, 0], z_u_ci[:, 1] - z_u])
    c_yerr = np.vstack([z_c - z_c_ci[:, 0], z_c_ci[:, 1] - z_c])

    ax.errorbar(w + 0.5, z_u, yerr=u_yerr, fmt='o-', capsize=3, lw=1.5, color=COLOR_UNCORR, label='Uncorrected (mean z)')
    ax.errorbar(w, z_c, yerr=c_yerr, fmt='o-', capsize=3, lw=1.5, color=COLOR_CORR, label='Corrected (mean z)')
    ax.axhline(0, color='k', lw=1, alpha=0.35)

    ax.set_xlabel('Window (ms)')
    ax.set_ylabel('Mean Fisher z(noise corr)')
    ax.set_title('Noise correlations')
    ax.legend(frameon=False, fontsize=7)


def _plot_effect_size_vs_shuffle(ax, stats):
    """Panel H: Effect size (delta z) vs shuffle control."""
    w = np.asarray(stats['window_ms'], dtype=float)
    order = np.argsort(w)
    w = w[order]

    dz = np.asarray(stats['dz_mean'])[order]
    dz_ci = np.asarray(stats['dz_ci'])[order]
    null_ci = np.asarray(stats['null_dz_ci95'])[order]

    dz_yerr = np.vstack([dz - dz_ci[:, 0], dz_ci[:, 1] - dz])

    ax.errorbar(w, dz, yerr=dz_yerr, fmt='o-', capsize=3, lw=1.5, label='Real Δz = zC - zU')
    ax.fill_between(w, null_ci[:, 0], null_ci[:, 1], alpha=0.18, color='gray',
                    label='Shuffle null 95% CI (Δz)')
    ax.axhline(0, color='k', lw=1, alpha=0.35)

    ax.set_xlabel('Window (ms)')
    ax.set_ylabel('Δ mean Fisher z')
    ax.set_title('Effect size vs shuffle control')
    ax.legend(frameon=False, fontsize=7)


def _plot_eigenspectra(ax, S, max_dims=50):
    """Panel I: Eigenspectra of PSTH and FEM covariance."""
    sp = S.get('spectra', {})
    psth_specs = sp.get('psth', [])
    fem_specs = sp.get('fem', [])

    if not psth_specs or not fem_specs:
        ax.text(0.5, 0.5, 'No spectra data', ha='center', va='center', transform=ax.transAxes)
        return

    # Stack and truncate spectra
    def stack_truncate(specs, L):
        M = np.full((len(specs), L), np.nan)
        for i, s in enumerate(specs):
            s = np.asarray(s)
            Li = min(s.size, L)
            M[i, :Li] = s[:Li]
        return M

    L = max_dims
    Mp = stack_truncate(psth_specs, L)
    Mf = stack_truncate(fem_specs, L)

    dims = np.arange(1, L + 1)

    # Per-session traces (light)
    for i in range(Mp.shape[0]):
        ax.loglog(dims, Mp[i], alpha=0.12, lw=1, color=COLOR_UNCORR)
        ax.loglog(dims, Mf[i], alpha=0.12, lw=1, color=COLOR_CORR)

    # Median + IQR
    p_med = np.nanmedian(Mp, axis=0)
    f_med = np.nanmedian(Mf, axis=0)
    p_lo, p_hi = np.nanpercentile(Mp, [25, 75], axis=0)
    f_lo, f_hi = np.nanpercentile(Mf, [25, 75], axis=0)

    ax.fill_between(dims, p_lo, p_hi, alpha=0.18, color=COLOR_UNCORR)
    ax.fill_between(dims, f_lo, f_hi, alpha=0.18, color=COLOR_CORR)
    ax.loglog(dims, p_med, lw=2.5, color=COLOR_UNCORR, label='Stimulus (PSTH)')
    ax.loglog(dims, f_med, lw=2.5, color=COLOR_CORR, label='Eye movement (FEM)')

    ax.set_xlabel('Dimension (PC)')
    ax.set_ylabel('Eigenspectra magnitude')
    ax.set_title('Eigenspectra magnitude')
    ax.set_xlim(1, 100)
    ax.set_ylim(1e-6, 1)
    ax.legend(frameon=False, fontsize=7)


def _plot_subspace_viz(ax, S):
    """Panel J: Subspace alignment visualization (participation ratio)."""
    # Show participation ratio distribution or similar
    per = S.get('per', {})
    pr_fem = per.get('pr_fem', [])
    pr_psth = per.get('pr_psth', [])

    pr_fem = np.asarray(pr_fem, dtype=float)
    pr_psth = np.asarray(pr_psth, dtype=float)

    # Filter to sessions where both are valid
    ok = np.isfinite(pr_fem) & np.isfinite(pr_psth)
    pr_fem = pr_fem[ok]
    pr_psth = pr_psth[ok]

    if pr_fem.size == 0 and pr_psth.size == 0:
        ax.text(0.5, 0.5, 'No PR data', ha='center', va='center', transform=ax.transAxes)
        return

    # Sort sessions by PSTH dimensionality (participation ratio)
    sort_idx = np.argsort(pr_psth)
    pr_fem = pr_fem[sort_idx]
    pr_psth = pr_psth[sort_idx]

    # Bar plot showing PR for PSTH vs FEM
    sessions = np.arange(len(pr_fem))
    width = 0.35

    if len(pr_fem) == len(pr_psth):
        ax.bar(sessions - width/2, pr_psth, width, label='PSTH', alpha=0.7, color=COLOR_UNCORR)
        ax.bar(sessions + width/2, pr_fem, width, label='FEM', alpha=0.7, color=COLOR_CORR)
        ax.set_xlabel('Session (sorted by PSTH dim.)')
    else:
        # Just show FEM participation ratio
        ax.bar(sessions, pr_fem, alpha=0.7, color=COLOR_CORR, label='FEM PR')
        ax.axhline(np.mean(pr_fem), color='r', ls='--', lw=1.5,
                   label=f'Mean PR={np.mean(pr_fem):.2f}')
        ax.set_xlabel('Session')

    ax.set_ylabel('Participation ratio')
    ax.set_title('Subspace dimensionality')
    ax.legend(frameon=False, fontsize=7)


def _plot_subspace_summary(ax, S):
    """Panel K: Subspace alignment summary (X vs Y with shuffle controls)."""
    per = S.get('per', {})
    null = S.get('null', {})

    x = np.asarray(per.get('var_p_given_f', []), dtype=float)  # PSTH var captured by FEM
    y = np.asarray(per.get('var_f_given_p', []), dtype=float)  # FEM var captured by PSTH
    ok = np.isfinite(x) & np.isfinite(y)

    # Shuffle controls
    xs = np.asarray(null.get('var_p_given_f', []), dtype=float).reshape(-1)
    ys = np.asarray(null.get('var_f_given_p', []), dtype=float).reshape(-1)
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]

    # Plot shuffle cloud
    if xs.size and ys.size:
        ax.scatter(xs, ys, s=10, alpha=0.15, color='gray', label='Shuffled (chance)')
        ax.plot(np.mean(xs), np.mean(ys), 'kx', ms=10, mew=2, label='Chance mean')

    # Plot real sessions
    ax.scatter(x[ok], y[ok], s=80, alpha=0.9, edgecolors='white', linewidths=0.8,
               label='Real sessions', zorder=5)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, lw=1)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('X: PSTH var captured by FEM subspace')
    ax.set_ylabel('Y: FEM var captured by PSTH subspace')
    ax.set_title('Real vs chance')

    # Annotate means
    mx, my = np.nanmean(x), np.nanmean(y)
    ax.text(0.05, 0.95, f'Mean X={mx:.2f}\nMean Y={my:.2f}',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(frameon=False, fontsize=6, loc='lower right')
