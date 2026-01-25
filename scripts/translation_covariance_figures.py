"""
Translation-Induced Covariance: plotting utilities

Loads saved outputs from the covariance pipeline and generates exploratory figures:
 - Eigenspectra per stimulus (from Σ)
 - Principal-angle histograms between PCA and gradient subspaces
 - Capture fraction (top-2 PCA) histogram
 - Capture matrix heatmap (f_{j|i})
 - Pairwise subspace alignment across stimuli (mean cos^2 of principal angles)

Usage:
    python scripts/translation_covariance_figures.py \
        --results-dir declan/translation_covariance \
        --max-stimuli 50 \
        [--shuf-results-dir declan/translation_covariance] \
        [--plot-within] [--plot-baselines]

Outputs are written into <results-dir>/figures/.
"""

import os
import sys
import argparse
import pickle
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles


# Ensure repository root is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)


def _ensure_figdir(results_dir: str) -> str:
    figdir = os.path.join(results_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    return figdir


def load_cov_results(results_dir: str) -> Dict[str, dict]:
    """Load covariance results.

    Prefers all_cov_results.pkl; falls back to scanning per-stimulus *.npz files.
    Returns mapping: stimulus -> {Sigma, evals2, U_pca2, U_grad2, principal_angles, capture2}
    """
    pkl_path = os.path.join(results_dir, 'all_cov_results.pkl')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data

    # Fallback: reconstruct from npz
    out: Dict[str, dict] = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('_cov_products.npz'):
            continue
        path = os.path.join(results_dir, fname)
        try:
            npz = np.load(path, allow_pickle=True)
            stim = str(npz.get('stimulus', fname))
            out[stim] = {
                'Sigma': npz['Sigma'],
                'evals2': npz['evals2'],
                'U_pca2': npz['U_pca2'],
                'U_grad2': npz['U_grad2'],
                'principal_angles': npz['principal_angles'],
                'capture2': float(npz['capture2']) if 'capture2' in npz else np.nan,
            }
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
    return out


def load_cov_results_shuf(results_dir: str) -> Dict[str, dict]:
    """Load shuffled cov results from all_cov_results_shuf.pkl if present."""
    pkl_path = os.path.join(results_dir, 'all_cov_results_shuf.pkl')
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load shuffle results: {e}")
    return {}


def load_capture_matrix(results_dir: str) -> Tuple[np.ndarray, List[str]]:
    mat_path = os.path.join(results_dir, 'capture_matrix.npy')
    keys_path = os.path.join(results_dir, 'all_cov_results.pkl')
    keys: List[str] = []
    if os.path.exists(keys_path):
        try:
            with open(keys_path, 'rb') as f:
                data = pickle.load(f)
            keys = list(data.keys())
        except Exception:
            pass
    if os.path.exists(mat_path):
        return np.load(mat_path), keys
    return np.array([]), keys


def _top_k_eigs(Sigma: np.ndarray, k: int = 20, normalize: bool = True) -> np.ndarray:
    w = np.linalg.eigvalsh(Sigma)
    w = np.sort(np.real(w))[::-1]
    if normalize:
        s = np.sum(np.maximum(w, 0))
        if s > 0 and np.isfinite(s):
            w = w / s
    return w[:k]


def _annotate_heatmap(A: np.ndarray, ax, fmt: str = ".2f", threshold: float = 0.5, fontsize: int = 7) -> None:
    """Annotate a heatmap with numeric values.
    Uses white text below threshold, black above for readability.
    Skips non-finite entries.
    """
    nrows, ncols = A.shape
    for i in range(nrows):
        for j in range(ncols):
            val = A[i, j]
            if not np.isfinite(val):
                continue
            color = 'white' if val < threshold else 'black'
            ax.text(j, i, format(val, fmt), ha='center', va='center', color=color, fontsize=fontsize)


def plot_eigenspectra(cov_results: Dict[str, dict], results_dir: str, max_stimuli: int = 50, k: int = 20) -> str:
    if len(cov_results) == 0:
        print('[INFO] No covariance results found for eigenspectra plot.')
        return ''
    figdir = _ensure_figdir(results_dir)
    plt.figure(figsize=(8, 5))
    for i, (stim, rec) in enumerate(list(cov_results.items())[:max_stimuli]):
        Sigma = rec['Sigma']
        vals = _top_k_eigs(Sigma, k=k, normalize=True)
        plt.plot(np.arange(1, len(vals) + 1), vals, alpha=0.6, lw=1, marker='o', markersize=2)
    plt.xlabel('Eigen index (desc)')
    plt.ylabel('Explained frac (norm)')
    plt.title('Eigenspectra across stimuli (normalized)')
    plt.grid(True, alpha=0.2)
    out_path = os.path.join(figdir, 'eigenspectra.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_angles_hist(cov_results: Dict[str, dict], results_dir: str) -> str:
    if len(cov_results) == 0:
        print('[INFO] No covariance results found for angles histogram.')
        return ''
    figdir = _ensure_figdir(results_dir)
    angles = []
    for rec in cov_results.values():
        th = np.asarray(rec['principal_angles']).ravel()
        if th.size >= 2 and np.all(np.isfinite(th)):
            angles.append(np.degrees(th))
    if len(angles) == 0:
        print('[INFO] No valid principal angles to plot.')
        return ''
    A = np.vstack(angles)
    plt.figure(figsize=(8, 4))
    bins = list(np.linspace(0, 90, 19))
    plt.hist(A[:, 0], bins=bins, alpha=0.6, label='theta1')
    plt.hist(A[:, 1], bins=bins, alpha=0.6, label='theta2')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Count')
    plt.title('Principal angles: U_PCA vs U_grad (per stimulus)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    out_path = os.path.join(figdir, 'angles_hist.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_capture_hist(cov_results: Dict[str, dict], results_dir: str) -> str:
    if len(cov_results) == 0:
        print('[INFO] No covariance results found for capture histogram.')
        return ''
    figdir = _ensure_figdir(results_dir)
    vals = [rec.get('capture2', np.nan) for rec in cov_results.values()]
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        print('[INFO] No valid capture2 values to plot.')
        return ''
    plt.figure(figsize=(7, 4))
    bins = list(np.linspace(0, 1, 21))
    plt.hist(v, bins=bins, color='#4477aa', edgecolor='white')
    plt.xlabel('Fraction of variance captured by top-2 PCA')
    plt.ylabel('Count')
    plt.title('Capture fraction (per stimulus)')
    plt.grid(True, alpha=0.2)
    out_path = os.path.join(figdir, 'capture2_hist.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_capture_matrix_heatmap(results_dir: str) -> str:
    M, keys = load_capture_matrix(results_dir)
    if M.size == 0:
        print('[INFO] capture_matrix.npy not found; skipping heatmap.')
        return ''
    figdir = _ensure_figdir(results_dir)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(M, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('f_{j|i}')
    ax.set_title('Capture matrix: variance of j in subspace i')
    n = M.shape[0]
    if keys and len(keys) == n:
        short = [str(k).split('/')[-1][:12] for k in keys]
        ax.set_xticks(range(n), short)
        ax.set_yticks(range(n), short)
        for label in ax.get_xticklabels():
            label.set_rotation(90)
            label.set_fontsize(7)
        for label in ax.get_yticklabels():
            label.set_fontsize(7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    _annotate_heatmap(M, ax, fmt=".2f", threshold=0.5, fontsize=7)
    fig.tight_layout()
    out_path = os.path.join(figdir, 'capture_matrix_heatmap.png')
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_pairwise_subspace_alignment(cov_results: Dict[str, dict], results_dir: str, basis_key: str = 'U_pca2') -> Tuple[str, str]:
    if len(cov_results) < 2:
        print('[INFO] Need >=2 stimuli for pairwise subspace alignment.')
        return '', ''
    figdir = _ensure_figdir(results_dir)
    keys = list(cov_results.keys())
    S = len(keys)
    A = np.full((S, S), np.nan, dtype=np.float32)
    vals = []
    for i in range(S):
        Ui = cov_results[keys[i]][basis_key]
        for j in range(S):
            Uj = cov_results[keys[j]][basis_key]
            th = subspace_angles(Ui, Uj)  # radians
            # mean cos^2 across the two principal angles (higher = more aligned)
            score = float(np.mean(np.cos(th) ** 2))
            A[i, j] = score
            if i < j:
                vals.append(score)

    # Heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(A, cmap='magma', vmin=0, vmax=1)
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('mean cos^2 (U_i vs U_j)')
    ax.set_title('Pairwise subspace alignment (PCA 2D bases)')
    n = A.shape[0]
    short = [str(k).split('/')[-1][:12] for k in keys]
    ax.set_xticks(range(n), short)
    ax.set_yticks(range(n), short)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_fontsize(7)
    for label in ax.get_yticklabels():
        label.set_fontsize(7)
    _annotate_heatmap(A, ax, fmt=".2f", threshold=0.6, fontsize=7)
    fig.tight_layout()
    heatmap_path = os.path.join(figdir, 'pairwise_subspace_alignment_heatmap.png')
    fig.savefig(heatmap_path, dpi=220)
    plt.close(fig)

    # Histogram of unique pairs
    if len(vals) > 0:
        plt.figure(figsize=(7, 4))
        bins = list(np.linspace(0, 1, 21))
        plt.hist(vals, bins=bins, color='#aa7744', edgecolor='white')
        plt.xlabel('mean cos^2 across principal angles')
        plt.ylabel('Count')
        plt.title('Across-stimulus subspace alignment (PCA)')
        plt.grid(True, alpha=0.2)
        hist_path = os.path.join(figdir, 'pairwise_alignment_hist.png')
        plt.tight_layout()
        plt.savefig(hist_path, dpi=200)
        plt.close()
    else:
        hist_path = ''

    return heatmap_path, hist_path


def plot_within_angles_hist(cov_results: Dict[str, dict], results_dir: str) -> str:
    """Histogram of within-stimulus split-half PCA plane angles."""
    angles = []
    for rec in cov_results.values():
        th = np.asarray(rec.get('principal_angles_within_pca2', [np.nan, np.nan])).ravel()
        if th.size >= 2 and np.all(np.isfinite(th)):
            angles.append(np.degrees(th))
    if len(angles) == 0:
        print('[INFO] No valid within-stimulus angles to plot.')
        return ''
    A = np.vstack(angles)
    figdir = _ensure_figdir(results_dir)
    plt.figure(figsize=(8, 4))
    bins = list(np.linspace(0, 90, 19))
    plt.hist(A[:, 0], bins=bins, alpha=0.6, label='theta1 (within)')
    plt.hist(A[:, 1], bins=bins, alpha=0.6, label='theta2 (within)')
    plt.xlabel('Within-stimulus angle (degrees)')
    plt.ylabel('Count')
    plt.title('Split-half PCA plane angles (per stimulus)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    out_path = os.path.join(figdir, 'within_angles_hist.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_pca_vs_regressor_scores(cov_results: Dict[str, dict], results_dir: str) -> str:
    """Boxplots of PCA-vs-regressor alignment scores across stimuli.
    Shows position, velocity, and best filtered-position (over tau in {1,2,4})."""
    pos_scores = []
    vel_scores = []
    fpos_best = []
    for rec in cov_results.values():
        s_pos = rec.get('score_pca_vs_pos', np.nan)
        s_vel = rec.get('score_pca_vs_vel', np.nan)
        f_dict = rec.get('score_pca_vs_fpos', {})
        if isinstance(f_dict, dict) and len(f_dict) > 0:
            s_fbest = np.nanmax(np.asarray(list(f_dict.values()), dtype=float))
        else:
            # Handle fields saved per tau in npz
            s_candidates = []
            for tau in (1, 2, 4):
                key = f'score_pca_vs_fpos_tau{tau}'
                if key in rec:
                    s_candidates.append(float(rec[key]))
            s_fbest = np.nanmax(np.asarray(s_candidates, dtype=float)) if s_candidates else np.nan
        pos_scores.append(s_pos)
        vel_scores.append(s_vel)
        fpos_best.append(s_fbest)
    pos_scores = np.asarray(pos_scores, dtype=float)
    vel_scores = np.asarray(vel_scores, dtype=float)
    fpos_best = np.asarray(fpos_best, dtype=float)
    # Filter finite
    series = [pos_scores[np.isfinite(pos_scores)], vel_scores[np.isfinite(vel_scores)], fpos_best[np.isfinite(fpos_best)]]
    labels = ['pos', 'vel', 'fpos(best)']
    if all(s.size == 0 for s in series):
        print('[INFO] No regressor comparison scores to plot.')
        return ''
    figdir = _ensure_figdir(results_dir)
    plt.figure(figsize=(7, 4))
    bp = plt.boxplot(series)
    plt.xticks([1, 2, 3], labels)
    plt.ylim(0, 1)
    plt.ylabel('mean cos^2 (U_PCA vs U_reg2)')
    plt.title('Regressor choice vs PCA plane (per stimulus)')
    plt.grid(True, alpha=0.2)
    out_path = os.path.join(figdir, 'pca_vs_regressor_scores_boxplot.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def load_random_baseline(results_dir: str) -> np.ndarray:
    p = os.path.join(results_dir, 'random_baseline_alignment.npy')
    if os.path.exists(p):
        try:
            return np.load(p)
        except Exception:
            return np.array([])
    return np.array([])


def plot_alignment_real_vs_shuf_and_baseline(
    cov_results: Dict[str, dict],
    shuf_results: Dict[str, dict],
    results_dir: str,
    baseline_vals: Optional[np.ndarray] = None,
) -> str:
    """Overlay histograms: real off-diagonal alignments, shuffled, and random baseline."""
    figdir = _ensure_figdir(results_dir)
    vals_real = []
    keys_real = list(cov_results.keys())
    for i in range(len(keys_real)):
        Ui = cov_results[keys_real[i]]['U_pca2']
        for j in range(i+1, len(keys_real)):
            Uj = cov_results[keys_real[j]]['U_pca2']
            th = subspace_angles(Ui, Uj)
            vals_real.append(float(np.mean(np.cos(th) ** 2)))
    vals_shuf = []
    keys_shuf = list(shuf_results.keys())
    if len(keys_shuf) >= 2:
        for i in range(len(keys_shuf)):
            Ui = shuf_results[keys_shuf[i]]['U_pca2']
            for j in range(i+1, len(keys_shuf)):
                Uj = shuf_results[keys_shuf[j]]['U_pca2']
                th = subspace_angles(Ui, Uj)
                vals_shuf.append(float(np.mean(np.cos(th) ** 2)))
    if len(vals_real) == 0 and len(vals_shuf) == 0 and (baseline_vals is None or baseline_vals.size == 0):
        print('[INFO] No data for real/shuf/baseline alignment histogram.')
        return ''
    plt.figure(figsize=(7.5, 4.5))
    bins = list(np.linspace(0, 1, 21))
    if len(vals_real) > 0:
        plt.hist(vals_real, bins=bins, alpha=0.6, label='real (off-diag)')
    if len(vals_shuf) > 0:
        plt.hist(vals_shuf, bins=bins, alpha=0.6, label='shuffle (off-diag)')
    if baseline_vals is not None and baseline_vals.size > 0:
        plt.hist(baseline_vals, bins=bins, alpha=0.4, label='random baseline')
    plt.xlabel('mean cos^2 (U_i vs U_j)')
    plt.ylabel('Count')
    plt.title('Across-stimulus PCA subspace alignment: real vs shuffle vs baseline')
    plt.legend()
    plt.grid(True, alpha=0.2)
    out_path = os.path.join(figdir, 'alignment_real_vs_shuf_vs_baseline_hist.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def _angles_sorted_radians(angles) -> np.ndarray:
    a = np.asarray(angles, dtype=float).reshape(-1)
    return np.sort(a)


def compute_pairwise_alignment(cov_results: Dict[str, dict], basis_key: str = 'U_pca2') -> Tuple[np.ndarray, List[str]]:
    """Compute mean cos^2 alignment matrix across principal angles for all stimulus pairs."""
    keys = list(cov_results.keys())
    S = len(keys)
    A = np.full((S, S), np.nan, dtype=np.float32)
    for i in range(S):
        Ui = cov_results[keys[i]][basis_key]
        for j in range(S):
            Uj = cov_results[keys[j]][basis_key]
            ang = _angles_sorted_radians(subspace_angles(Ui, Uj))
            A[i, j] = float(np.mean(np.cos(ang) ** 2))
    return A, keys


def write_text_summary(cov_results: Dict[str, dict], results_dir: str, shuf_results: Optional[Dict[str, dict]] = None) -> str:
    """Write a human-readable summary of figure metrics to a text file."""
    figdir = _ensure_figdir(results_dir)
    out_path = os.path.join(figdir, 'summary.txt')
    lines: List[str] = []
    keys = list(cov_results.keys())
    S = len(keys)
    lines.append(f"Summary for {S} stimuli\n")

    # Per-stimulus metrics
    lines.append("Per-stimulus metrics (capture2, angles_deg, within, best_regressor):")
    for k in keys:
        rec = cov_results[k]
        cap = float(rec.get('capture2', np.nan))
        ang_deg = np.degrees(_angles_sorted_radians(rec.get('principal_angles', [np.nan, np.nan])))
        Sigma = rec.get('Sigma')
        within_ang = rec.get('principal_angles_within_pca2', [np.nan, np.nan])
        within_ang_deg = np.degrees(_angles_sorted_radians(within_ang))
        within_score = float(rec.get('within_alignment_score', np.nan))
        # Best regressor among pos/vel/filtered
        s_pos = float(rec.get('score_pca_vs_pos', np.nan))
        s_vel = float(rec.get('score_pca_vs_vel', np.nan))
        s_fbest = np.nan
        f_dict = rec.get('score_pca_vs_fpos', {})
        if isinstance(f_dict, dict) and len(f_dict) > 0:
            s_fbest = float(np.nanmax(np.asarray(list(f_dict.values()), dtype=float)))
        else:
            candidates = []
            for tau in (1, 2, 4):
                key = f'score_pca_vs_fpos_tau{tau}'
                if key in rec:
                    candidates.append(float(rec[key]))
            s_fbest = float(np.nanmax(np.asarray(candidates, dtype=float))) if candidates else np.nan
        # Choose best label
        scores = {'pos': s_pos, 'vel': s_vel, 'fpos(best)': s_fbest}
        best_label = max(scores, key=lambda z: (scores[z] if np.isfinite(scores[z]) else -1))
        best_score = scores[best_label]
        top5 = _top_k_eigs(Sigma, k=5, normalize=True) if Sigma is not None else np.array([np.nan]*5)
        short = str(k).split('/')[-1]
        lines.append(f"- {short}: capture2={cap:.3f}, angles=[{ang_deg[0]:.2f}, {ang_deg[1]:.2f}], within=[{within_ang_deg[0]:.2f}, {within_ang_deg[1]:.2f}] score={within_score:.3f}, best_reg={best_label}:{best_score:.3f}, eigs5={np.round(top5, 4).tolist()}")

    # Pairwise PCA subspace alignment
    A_align, align_keys = compute_pairwise_alignment(cov_results, basis_key='U_pca2')
    if A_align.size:
        vals = []
        pairs = []
        for i in range(A_align.shape[0]):
            for j in range(A_align.shape[1]):
                if i == j:
                    continue
                vals.append(A_align[i, j])
                pairs.append((i, j))
        vals_arr = np.asarray(vals, dtype=float)
        lines.append("\nPairwise PCA subspace alignment (mean cos^2):")
        lines.append(f"- Off-diagonal mean={np.nanmean(vals_arr):.3f}, median={np.nanmedian(vals_arr):.3f}, std={np.nanstd(vals_arr):.3f}")
        # Top-5 aligned pairs
        idx_sorted = np.argsort(vals_arr)[::-1][:5]
        lines.append("- Top aligned pairs:")
        for idx in idx_sorted:
            i, j = pairs[idx]
            name_i = str(align_keys[i]).split('/')[-1]
            name_j = str(align_keys[j]).split('/')[-1]
            lines.append(f"  * {name_i} vs {name_j}: {vals_arr[idx]:.3f}")

    # Within-stimulus split-half stats
    within_scores = []
    for rec in cov_results.values():
        sc = rec.get('within_alignment_score', np.nan)
        if np.isfinite(sc):
            within_scores.append(sc)
    if len(within_scores) > 0:
        arr = np.asarray(within_scores, dtype=float)
        lines.append("\nWithin-stimulus PCA plane stability:")
        lines.append(f"- Mean={np.nanmean(arr):.3f}, median={np.nanmedian(arr):.3f}, std={np.nanstd(arr):.3f}")

    # Capture matrix stats
    M, cap_keys = load_capture_matrix(results_dir)
    if M.size:
        lines.append("\nCapture matrix f_{j|i} stats:")
        # Off-diagonal stats
        od = M.copy()
        for d in range(min(od.shape)):
            od[d, d] = np.nan
        lines.append(f"- Off-diagonal mean={np.nanmean(od):.3f}, median={np.nanmedian(od):.3f}, std={np.nanstd(od):.3f}")
        # Diagonal vs capture2 consistency check
        diag = np.diag(M) if M.shape[0] == M.shape[1] else np.array([])
        if diag.size:
            lines.append(f"- Diagonal mean (self-capture)={np.nanmean(diag):.3f}")
        # Top capturing pairs
        flat = M.flatten()
        idxs = np.argsort(flat)[::-1][:5]
        lines.append("- Top f_{j|i} pairs:")
        for idx in idxs:
            i = idx // M.shape[1]
            j = idx % M.shape[1]
            name_i = str(cap_keys[i]).split('/')[-1] if cap_keys and i < len(cap_keys) else str(i)
            name_j = str(cap_keys[j]).split('/')[-1] if cap_keys and j < len(cap_keys) else str(j)
            lines.append(f"  * U_{name_i} capturing Σ({name_j}): {M[i, j]:.3f}")

    # Real vs shuffle alignment and capture comparisons
    if shuf_results and len(shuf_results) >= 2:
        As, keys_s = compute_pairwise_alignment(shuf_results, basis_key='U_pca2')
        if As.size:
            vals_s = []
            for i in range(As.shape[0]):
                for j in range(As.shape[1]):
                    if i == j:
                        continue
                    vals_s.append(As[i, j])
            arr_s = np.asarray(vals_s, dtype=float)
            lines.append("\nAcross-stimulus alignment: real vs shuffled")
            if 'vals_arr' in locals():
                lines.append(f"- Real off-diag mean={np.nanmean(vals_arr):.3f} vs Shuf mean={np.nanmean(arr_s):.3f}")
            else:
                lines.append(f"- Shuf off-diag mean={np.nanmean(arr_s):.3f}")
        # Capture matrix shuf
        mat_shuf_path = os.path.join(results_dir, 'capture_matrix_shuf.npy')
        if os.path.exists(mat_shuf_path):
            Ms = np.load(mat_shuf_path)
            od_s = Ms.copy()
            for d in range(min(od_s.shape)):
                od_s[d, d] = np.nan
            lines.append("- Capture off-diag mean (shuf)=" + f"{np.nanmean(od_s):.3f}")

    # Random baseline stats
    baseline_path = os.path.join(results_dir, 'random_baseline_alignment.npy')
    if os.path.exists(baseline_path):
        try:
            b = np.load(baseline_path)
            q25, q50, q75 = np.nanpercentile(b, [25, 50, 75])
            lines.append("\nRandom baseline alignment (off-diag)")
            lines.append(f"- Mean={np.nanmean(b):.3f}, std={np.nanstd(b):.3f}, quantiles=({q25:.3f}, {q50:.3f}, {q75:.3f})")
        except Exception:
            pass

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Plot translation-induced covariance results.')
    parser.add_argument('--results-dir', type=str, default=os.path.join(ROOT, 'declan', 'translation_covariance'))
    parser.add_argument('--max-stimuli', type=int, default=50)
    parser.add_argument('--topk', type=int, default=20, help='Top-k eigenvalues to display per stimulus')
    parser.add_argument('--shuf-results-dir', type=str, default=None, help='Directory containing all_cov_results_shuf.pkl (defaults to results-dir)')
    parser.add_argument('--plot-within', action='store_true', help='Plot within-stimulus split-half angles histogram')
    parser.add_argument('--plot-baselines', action='store_true', help='Plot real vs shuffle vs random baseline alignment histogram')
    args = parser.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    cov_results = load_cov_results(results_dir)
    shuf_dir = args.shuf_results_dir or results_dir
    shuf_results = load_cov_results_shuf(shuf_dir)

    print(f"[INFO] Loaded {len(cov_results)} stimulus result(s) from {results_dir}")
    if shuf_results:
        print(f"[INFO] Loaded {len(shuf_results)} shuffled result(s) from {shuf_dir}")

    out_paths = []
    out_paths.append(plot_eigenspectra(cov_results, results_dir, max_stimuli=args.max_stimuli, k=args.topk))
    out_paths.append(plot_angles_hist(cov_results, results_dir))
    out_paths.append(plot_capture_hist(cov_results, results_dir))
    out_paths.append(plot_capture_matrix_heatmap(results_dir))
    pair_heat, pair_hist = plot_pairwise_subspace_alignment(cov_results, results_dir, basis_key='U_pca2')
    out_paths.extend([pair_heat, pair_hist])

    if args.plot_within:
        out_paths.append(plot_within_angles_hist(cov_results, results_dir))
        out_paths.append(plot_pca_vs_regressor_scores(cov_results, results_dir))

    if args.plot_baselines:
        baseline_vals = load_random_baseline(results_dir)
        out_paths.append(plot_alignment_real_vs_shuf_and_baseline(cov_results, shuf_results, results_dir, baseline_vals))

    out_paths = [p for p in out_paths if p]
    if out_paths:
        print('[INFO] Saved figures:')
        for p in out_paths:
            print('  -', os.path.relpath(p, ROOT))
        # Also write a text summary of metrics
        summary_path = write_text_summary(cov_results, results_dir, shuf_results=shuf_results if shuf_results else None)
        print('[INFO] Wrote summary:')
        print('  -', os.path.relpath(summary_path, ROOT))
    else:
        print('[INFO] No figures were produced (likely missing inputs).')


if __name__ == '__main__':
    main()
