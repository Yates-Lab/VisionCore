"""Adapter: map the methods derived bundle into the exact dict schema that
`ryan/fig2/compute_fig2_data.load_fig2_data()` returns, for one target.

Design (see note_consistency.md §1.3): the methods `metrics[target]` list
already carries every per-window key that production's Stage-3 statistics
functions consume (`alpha/uncorr/corr/erate/subject_per_neuron/
session_per_neuron/shuff_var_c/rho_*/shuff_*`). So we feed that list through the
*literal* production functions `_compute_alpha_stats`, `_compute_fano_stats`,
`_compute_nc_stats`. This guarantees zero statistic-function divergence — any
NAIVE-vs-PROD residual is then provably upstream (decomposition / inclusion) —
and it reconstructs the two per-subject sub-keys the methods bundle omits
(`fano_stats[...]['per_subject']`, `nc_stats[...]['null_dz_ci_by_subject']`).

Usage:
    from adapter import methods_to_fig2_schema, load_methods_bundle
    md = load_methods_bundle()
    data_naive = methods_to_fig2_schema(md, "naive")
    data_full  = methods_to_fig2_schema(md, "full")
"""
from __future__ import annotations

import sys
from pathlib import Path

import dill

THIS_DIR = Path(__file__).resolve().parent
METHODS_DIR = THIS_DIR.parent
FIG2_DIR = METHODS_DIR.parent / "fig2"
FIG3_DIR = METHODS_DIR.parent / "fig3"
CACHE = METHODS_DIR / "cache"
for p in (METHODS_DIR, FIG2_DIR, FIG3_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Literal production Stage-3 functions + constants (the reference statistics).
from compute_fig2_data import (  # noqa: E402
    _compute_alpha_stats,
    _compute_fano_stats,
    _compute_nc_stats,
    SUBJECTS,
    SUBJECT_COLORS,
)


def load_methods_bundle(path=None):
    path = Path(path) if path else (CACHE / "methods_derived.pkl")
    with open(path, "rb") as f:
        return dill.load(f)


def methods_to_fig2_schema(md, target, null_from=None):
    """Return a dict matching load_fig2_data()'s schema for the given target.

    `md` is the derive_methods bundle (cache/methods_derived.pkl).
    `target` is one of md['targets'] ('naive' | 'full' | 'central').

    `null_from`: optional target name whose **NC shuffle-null** arrays
    (`shuff_rho_subject`, `shuff_rho_delta_meanz`) are borrowed into this
    target's per-window metrics before computing `nc_stats`. The pipeline only
    runs the eye-shuffle null for `target='naive'`, so panel 3D's shuffle band is
    undefined for `full`. Setting `null_from='naive'` plots the full observed
    delta-z against the **naive** null band (same shuffle procedure) — the
    reference-band choice documented in note_consistency.md. Only the NC null
    fields are affected; observed delta-z, Fano, and alpha all stay this
    target's own (no other panel reads these arrays).
    """
    if target not in md["targets"]:
        raise ValueError(f"target {target!r} not in {md['targets']}")
    if null_from is not None and null_from not in md["targets"]:
        raise ValueError(f"null_from {null_from!r} not in {md['targets']}")

    windows_ms = list(md["windows_ms"])
    windows_bins = list(md["windows_bins"])
    metrics = md["metrics"][target]  # list of per-window dicts, prod-compatible

    if null_from is not None and null_from != target:
        src = md["metrics"][null_from]
        patched = []
        for m_tgt, m_src in zip(metrics, src):
            m = dict(m_tgt)  # shallow copy; only swap the NC shuffle arrays
            m["shuff_rho_subject"] = m_src["shuff_rho_subject"]
            m["shuff_rho_delta_meanz"] = m_src["shuff_rho_delta_meanz"]
            patched.append(m)
        metrics = patched

    # Run the *production* Stage-3 statistics on the methods metrics.
    m_by_window, subj_pn_by_window, alpha_stats = _compute_alpha_stats(
        metrics, windows_ms
    )
    fano_stats = _compute_fano_stats(metrics, windows_ms)
    nc_stats = _compute_nc_stats(metrics, windows_ms)

    return {
        "SUBJECTS": SUBJECTS,
        "SUBJECT_COLORS": SUBJECT_COLORS,
        "WINDOWS_MS": windows_ms,
        "WINDOWS_BINS": windows_bins,
        "metrics": metrics,
        "m_by_window": m_by_window,
        "subject_per_neuron_by_window": subj_pn_by_window,
        "alpha_stats": alpha_stats,
        "fano_stats": fano_stats,
        "nc_stats": nc_stats,
        "session_names": list(md["session_names"]),
        "subjects": list(md["subjects"]),
        "n_sessions": len(md["session_names"]),
        "_target": target,
    }
