"""Figure 4 must render pixel-identical to the reference build.

`paper/fig4/reference/figure4_reference.pdf` is Declan Rowley's original
composite, the artifact the fig4 refactor is measured against. The refactor's
whole contract is that it changes structure and nothing else, so the check is
exact: render both PDFs to raster and require a maximum channel difference of
zero. One differing pixel is a failed refactor, not a rounding artifact.

The build reads ~590 MB of cached inputs and takes ~35 s, so the build test
skips (rather than fails) when the caches are absent -- a fresh clone has the
code but not the caches. `test_reference_pdf_intact` and
`test_diff_harness_detects_a_single_pixel` always run: they guard the baseline
itself and prove the comparison is not vacuous.

Usage:
    uv run pytest tests/test_fig4_regression.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys

import numpy as np
import pytest

from VisionCore.paths import VISIONCORE_ROOT, FIGURES_DIR

fitz = pytest.importorskip("fitz", reason="pymupdf is required to rasterize PDFs")

REFERENCE_PDF = VISIONCORE_ROOT / "paper" / "fig4" / "reference" / "figure4_reference.pdf"
REFERENCE_MD5 = "48e092df402c7d43bc0e5e84fafb3d06"
ENTRY_POINT = VISIONCORE_ROOT / "paper" / "fig4" / "generate_figure4.py"
BUILT_PDF = FIGURES_DIR / "fig4" / "figure4.pdf"

RENDER_DPI = 110


def render(pdf_path, dpi: int = RENDER_DPI) -> np.ndarray:
    """Rasterize page 1 of a PDF to an (H, W, 3) uint8 RGB array."""
    with fitz.open(str(pdf_path)) as doc:
        pixmap = doc[0].get_pixmap(dpi=dpi)
    raster = np.frombuffer(pixmap.samples, dtype=np.uint8)
    raster = raster.reshape(pixmap.height, pixmap.width, pixmap.n)
    return raster[:, :, :3]


def max_channel_difference(actual: np.ndarray, expected: np.ndarray) -> int:
    """Largest per-channel absolute difference, or a loud failure on a shape
    mismatch -- a page that changed size would otherwise fail to broadcast with
    a numpy error that says nothing about the figure."""
    if actual.shape != expected.shape:
        pytest.fail(
            f"rendered page geometry changed: {actual.shape} vs reference {expected.shape}. "
            "The composite page size or panel layout moved."
        )
    return int(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max())


def test_reference_pdf_intact():
    """The baseline itself must not drift. If this fails, the reference was
    overwritten and the rest of the suite is measuring against nothing."""
    assert REFERENCE_PDF.exists(), f"reference build missing: {REFERENCE_PDF}"
    digest = hashlib.md5(REFERENCE_PDF.read_bytes()).hexdigest()
    assert digest == REFERENCE_MD5, (
        f"reference PDF changed: {digest} != {REFERENCE_MD5}. "
        "Restore it from git rather than re-baselining."
    )


def test_diff_harness_detects_a_single_pixel():
    """Guard against a vacuous assertion: a comparison that always returns 0
    would let every regression through. Perturbing one channel of one pixel
    must register."""
    reference = render(REFERENCE_PDF)
    assert max_channel_difference(reference, reference) == 0

    perturbed = reference.copy()
    perturbed[0, 0, 0] = 255 - int(perturbed[0, 0, 0])
    assert max_channel_difference(perturbed, reference) > 0


def test_figure4_matches_reference():
    """Rebuild figure 4 from cached inputs and require an exact pixel match."""
    if not ENTRY_POINT.exists():
        pytest.skip(f"fig4 entry point not in place yet: {ENTRY_POINT}")

    result = subprocess.run(
        [sys.executable, str(ENTRY_POINT)],
        cwd=str(VISIONCORE_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        combined = f"{result.stdout}\n{result.stderr}"
        # A missing cache is an environment gap, not a regression; the build
        # itself is responsible for saying so explicitly (see the fig4
        # preflight), and that message is what distinguishes the two.
        lowered = combined.lower()
        if (
            "required fig4 inputs are missing" in lowered
            or ("cache" in lowered and "not found" in lowered)
        ):
            pytest.skip(f"fig4 caches unavailable:\n{combined}")
        pytest.fail(f"fig4 build failed (exit {result.returncode}):\n{combined}")

    assert BUILT_PDF.exists(), f"build reported success but produced no {BUILT_PDF}"

    difference = max_channel_difference(render(BUILT_PDF), render(REFERENCE_PDF))
    assert difference == 0, (
        f"figure 4 regressed: max channel difference {difference} at {RENDER_DPI} dpi "
        f"between {BUILT_PDF} and {REFERENCE_PDF}."
    )
