"""Produce the manuscript-facing orientation-corrected Figure 4: zero tests, manuscript-layout PDF, and statistics.

The manuscript scripts (render_figures.py figure 4 branch, export_figure4_zero_tests.py, sync_stats.py) run unmodified
against an overlay source root: Jake's selected bundle with only the rebuilt Figure 4 directories swapped in
(matrix_spectral_replay, passband_vs_path_length, top_passband_stage_trajectory_10img_x_10fix, production_figure4).
Manuscript-level inputs are redirected to a scratch manuscript directory, so nothing under manuscript/ is written:
  * analysis/selected_model_bundle.json  - pins the corrected results_provenance.json hash
  * analysis/passband_comparison.json    - points at passband_comparison_corrected
  * analysis/empirical_stats.json        - copied unchanged (Figure 2)

Outputs:
  outputs/figures/fig4_orientation_fix/figure4.pdf    manuscript layout, drop-in for manuscript/figures/figure4.pdf
  outputs/stats/fig4_orientation_fix/source_root/manuscript_fix/generated_stats.tex  drop-in for manuscript/generated_stats.tex
  outputs/stats/fig4_orientation_fix/macro_changes.csv                every macro that differs from the release

Usage (after build_shards.py, the GPU stage trajectory, the release render/audit/provenance, passband_comparison.py):
    .venv/bin/python declan/fig4_orientation_fix/manuscript_outputs.py
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUNDLE, CORRECTED, FIG_OUT, JAKE, OUT, sha256  # noqa: E402

from VisionCore.paths import VISIONCORE_ROOT  # noqa: E402

REPO = VISIONCORE_ROOT
ROOT = OUT / "source_root"
FAKE = ROOT / "manuscript_fix"         # inside ROOT: the zero-test exporter records paths relative to it
REBUILT = ("matrix_spectral_replay", "passband_vs_path_length", "top_passband_stage_trajectory_10img_x_10fix",
           "production_figure4")
EMPIRICAL_CACHE = Path("/home/ryanress/v1-fovea/VisionCore/outputs/cache")   # hash-identical, readable Figure 2 cache


def _link(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def _overlay(source: Path, dest: Path, replace: dict[str, Path | None]) -> None:
    """Make `dest` a directory of symlinks to `source`'s entries, except names in `replace` (None = real dir)."""
    dest.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if entry.name not in replace:
            _link(entry, dest / entry.name)
    for name, target in replace.items():
        if target is not None:
            _link(target, dest / name)


def build_source_root() -> None:
    rank1 = BUNDLE.relative_to(JAKE)
    _overlay(JAKE / "outputs", ROOT / "outputs", {rank1.parts[1]: None, "stats": None})
    _overlay(JAKE / "outputs/stats", ROOT / "outputs/stats", {OUT.name: OUT})
    _overlay(BUNDLE.parent, ROOT / rank1.parent, {"rank1": None})
    _overlay(BUNDLE, ROOT / rank1, {"figure4": None})
    _overlay(BUNDLE / "figure4", ROOT / rank1 / "figure4", {name: CORRECTED / name for name in REBUILT})
    for name in ("paper", "scripts", "jake", "manuscript"):
        _link(REPO / name, ROOT / name)
    # The manuscript render replays the release's recorded build command; its absolute bundle paths resolve through
    # this overlay to the corrected shards and stage trajectory (render() checks this).
    manifest = json.loads((BUNDLE / "figure4/production_figure4/run_manifest.json").read_text())
    manifest["orientation_correction"] = {
        "released_manifest": str(BUNDLE / "figure4/production_figure4/run_manifest.json"),
        "note": "figure/ and audit/ rebuilt with orientation-corrected shards and stage trajectory; recorded commands "
                "are the release's and resolve through the overlay source root",
        "corrected_shards_sha256": {str(p): sha256(p) for p in sorted(CORRECTED.glob("matrix_spectral_replay/*/*.npz"))}}
    (CORRECTED / "production_figure4/run_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def build_fake_manuscript() -> None:
    analysis = FAKE / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    selection = json.loads((REPO / "manuscript/analysis/selected_model_bundle.json").read_text())
    provenance = ROOT / selection["bundle"] / "figure4/production_figure4/figure/results_provenance.json"
    selection["released_figure4_results_sha256"] = selection["figure4_results_sha256"]
    selection["figure4_results_sha256"] = sha256(provenance)
    selection["orientation_correction"] = ("Figure 4 spectral predictors recomputed with the grating-assay orientation "
                                           "convention; see declan/fig4_orientation_fix")
    (analysis / "selected_model_bundle.json").write_text(json.dumps(selection, indent=2) + "\n")
    shutil.copy2(REPO / "manuscript/analysis/empirical_stats.json", analysis / "empirical_stats.json")
    comparison = json.loads((REPO / "manuscript/analysis/passband_comparison.json").read_text())
    base = OUT / "passband_comparison_corrected"
    comparison["summary"] = str((base / "summary.json").relative_to(REPO))
    comparison["summary_sha256"] = sha256(base / "summary.json")
    comparison["normalized_overlap"] = {
        "summary": str((base / "normalized_overlap/summary.json").relative_to(REPO)),
        "summary_sha256": sha256(base / "normalized_overlap/summary.json")}
    comparison["orientation_correction"] = "engagement recomputed from orientation-corrected spectral shards"
    (analysis / "passband_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")


def patch_selection():
    """Point the manuscript's selection helpers at the overlay root and corrected selection, then import them."""
    sys.path.insert(0, str(REPO / "manuscript"))
    sys.path.insert(0, str(REPO))
    import analysis_selection
    analysis_selection.SOURCE_ROOT = ROOT
    analysis_selection.SELECTION = FAKE / "analysis/selected_model_bundle.json"
    import figure4_selection
    importlib.reload(figure4_selection)
    return analysis_selection, figure4_selection


def zero_tests() -> Path:
    import export_figure4_zero_tests as zt
    zt.HERE, zt.ROOT = FAKE, ROOT
    zt.main()
    return FAKE / "analysis/figure4_zero_tests/summary.json"


def render(zero: Path, selection, figure4_selection) -> Path:
    """render_figures.py's figure-4 branch, writing to FIG_OUT instead of manuscript/figures."""
    bundle = ROOT / selection.selected_analysis()["bundle"]
    example = figure4_selection.selected_example_dir(bundle)
    manifest = json.loads((bundle / "figure4/production_figure4/run_manifest.json").read_text())
    recorded = shlex.split(manifest["commands"][0])
    command = [sys.executable, str(REPO / "paper/fig4/spatiotemporal_tuning/build_figure4.py")]
    command += [str(selection.source_path(v)) if Path(v).is_absolute() else v for v in recorded[2:]]
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    command[command.index("--out-dir") + 1] = str(FIG_OUT)
    command[command.index("--panel-a-audit") + 1] = str(example)
    if figure4_selection.spectrum_update(bundle):
        raise RuntimeError("an interim spectrum update is selected; not handled here")
    command += ["--layout", "manuscript", "--zero-tests", str(zero)]
    for flag in ("--population-shards", "--stage-trajectory"):
        resolved = Path(command[command.index(flag) + 1]).resolve()
        if not resolved.is_relative_to(CORRECTED):
            raise RuntimeError(f"{flag} does not resolve to the corrected bundle: {resolved}")
    env = {**os.environ, "VISIONCORE_MIN_FIGURE_FONT_PT": "6.1", "MPLCONFIGDIR": str(OUT / "mpl"),
           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONPATH": str(REPO),
           "VISIONCORE_MANUSCRIPT_SOURCE_ROOT": str(ROOT)}
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)
    return FIG_OUT / "figure4.pdf"


def stats() -> Path:
    os.environ["VISIONCORE_MANUSCRIPT_EMPIRICAL_CACHE_DIR"] = str(EMPIRICAL_CACHE)
    import sync_stats
    sync_stats.MANUSCRIPT = FAKE
    sys.argv = ["sync_stats.py"]
    sync_stats.main()
    return FAKE / "generated_stats.tex"


def macro_changes(corrected: Path) -> Path:
    pattern = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}$")
    def parse(path):
        return {m[1]: m[2] for line in path.read_text().splitlines() if (m := pattern.match(line))}
    old, new = parse(REPO / "manuscript/generated_stats.tex"), parse(corrected)
    rows = [{"macro": k, "released": old.get(k, ""), "corrected": new.get(k, "")}
            for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
    out = OUT / "macro_changes.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["macro", "released", "corrected"])
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['macro']:40s} {r['released']:>24s} -> {r['corrected']}")
    return out


def main() -> int:
    build_source_root()
    build_fake_manuscript()
    selection, figure4_selection = patch_selection()
    zero = zero_tests()
    pdf = render(zero, selection, figure4_selection)
    tex = stats()
    macro_changes(tex)
    print(pdf, tex, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
