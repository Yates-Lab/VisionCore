"""Covariance-decomposition pipeline (empirical + model digital-twin).

The eye-position-distribution-matched Law-of-Total-Covariance decomposition
shared by Figures 2 and 3. Stage 1 (per-session decomposition) and stage 2/3
(derived metrics + subspace) are split across ``data_loading`` / ``decompose`` /
``derive``; the model digital-twin decomposition lives in ``model_decompose``.

Modules use flat imports + sys.path self-insertion (the convention in this
repo's analysis folders), so they are importable both as a package
(``from paper.covariance_decomposition import load_empirical_data`` once the repo
root is on the path) and as flat modules from within this directory.

Public API:
    load_empirical_data(refresh=False, refresh_decomposition=False)
        Derived empirical bundle (the schema the fig2 panels read).
    load_model_data(refresh=False)
        Per-cell digital-twin model 1-alpha bundle (fig3 panel D / ablation).
"""
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from derive import load_empirical_data  # noqa: E402

__all__ = ["load_empirical_data", "load_model_data"]


def load_model_data(*args, **kwargs):
    # Imported lazily so the empirical path has no dependency on the fig3
    # digital-twin model loader.
    from model_decompose import load_model_data as _load
    return _load(*args, **kwargs)
