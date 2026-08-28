from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


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
