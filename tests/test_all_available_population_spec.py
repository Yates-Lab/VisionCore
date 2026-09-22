from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from paper.fig4.spatiotemporal_tuning import build_all_available_population_spec as population


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "paper"
    / "fig4"
    / "spatiotemporal_tuning"
    / "build_all_available_population_spec.py"
)


def test_direct_entrypoint_bootstraps_repository_imports(tmp_path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    code = (
        "import runpy; "
        f"namespace = runpy.run_path({str(SCRIPT)!r}); "
        "import paper.fig4.upstream.real_trace_matrix.model; "
        "assert str(namespace['ROOT']) in __import__('sys').path"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        check=True,
    )


@pytest.mark.parametrize("has_phase", [False, True])
def test_population_release_accepts_matching_checkpoint_architecture(tmp_path, monkeypatch, has_phase):
    """An absent phase branch is valid when the checkpoint also lacks it."""
    import json
    from types import SimpleNamespace

    checkpoint = tmp_path / "checkpoint.ckpt"
    dataset = tmp_path / "dataset.yaml"
    recorded = tmp_path / "recorded.pkl"
    for path in (checkpoint, dataset, recorded):
        path.write_bytes(b"fixture")
    canonical = pd.DataFrame({"canonical_channel": [0], "session": ["session"],
                              "cid": [2], "model_readout_row": [0], "available": [True]})
    units = canonical.assign(unit_index=0, unit_label="u000")
    identity = np.ones((1, 1), dtype=np.float32)
    audits = {"native_cid_mapping": {"passed": True},
              "scalar_logit_equivalence": {"passed": True},
              "includes_phase_branch": has_phase, "phase_branch_matches_checkpoint": True}
    monkeypatch.setattr(population, "runtime_identity_view", lambda **kwargs:
                        (canonical, units, identity, identity, np.array([0]), audits))
    monkeypatch.setattr(population, "parse_args", lambda: SimpleNamespace(
        checkpoint=checkpoint, dataset_config=dataset, mcfarland_outputs=recorded,
        out_dir=tmp_path / "population", version="fixture", force=False, device="cpu"))
    assert population.main() == 0
    summary = json.loads((tmp_path / "population/summary.json").read_text())
    assert summary["readout_audits"]["includes_phase_branch"] is has_phase
    assert all(summary["gates"].values())
