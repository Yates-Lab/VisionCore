from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper.fig4.mechanism_audit_v1.causal_low_rank import plot_results as plotting


def _rank_rows() -> pd.DataFrame:
    rows = []
    for contrast in plotting.CONTRAST_ORDER:
        for fold, learned in enumerate((0.30, 0.60, 0.90)):
            for rank in (4, 8):
                for method, offset in (
                    ("learned", learned),
                    ("movement_pca", 0.20),
                    ("readout_svd", 0.25),
                ):
                    rows.append(
                        {
                            "stage": "crossval",
                            "contrast": contrast,
                            "fold": fold,
                            "rank": rank,
                            "method": method,
                            "map_r2_sufficiency": offset,
                            "map_r2_necessity": offset - 0.03,
                            "ssi_fraction_transferred": offset + 0.05,
                            "ssi_fraction_removed": offset + 0.02,
                        }
                    )
            rows.append(
                {
                    "stage": "crossval",
                    "contrast": contrast,
                    "fold": fold,
                    "rank": 128,
                    "method": "identity",
                    "map_r2_sufficiency": 1.0,
                    "map_r2_necessity": 1.0,
                    "ssi_fraction_transferred": 1.0,
                    "ssi_fraction_removed": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _random_rows() -> pd.DataFrame:
    rows = []
    for contrast in plotting.CONTRAST_ORDER:
        for rank in (4, 8):
            for draw in range(10):
                rows.append(
                    {
                        "stage": "crossval",
                        "contrast": contrast,
                        "fold": draw % 3,
                        "rank": rank,
                        "method": "random_haar",
                        "draw": draw,
                        "map_r2_sufficiency": 0.05 + 0.005 * draw,
                        "map_r2_necessity": 0.04 + 0.004 * draw,
                        "ssi_fraction_transferred": 0.08 + 0.003 * draw,
                        "ssi_fraction_removed": 0.07 + 0.002 * draw,
                    }
                )
    return pd.DataFrame(rows)


def _fake_export(monkeypatch):
    exported = []

    def save(fig, destination):
        plt.close(fig)
        paths = [str(destination.with_suffix(suffix)) for suffix in (".pdf", ".svg", ".png")]
        exported.extend(paths)
        return paths

    monkeypatch.setattr(plotting, "save_figure", save)
    return exported


def test_plot_module_is_saved_product_only() -> None:
    tree = ast.parse(Path(plotting.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {"torch", "h5py", "models", "scripts.spatial_info"}
    assert not forbidden.intersection(imported)


def test_export_contract_is_pdf_svg_and_600dpi_png(tmp_path, monkeypatch) -> None:
    calls = []

    class Figure:
        def savefig(self, path, **options):
            calls.append((Path(path).suffix, options))

    monkeypatch.setattr(plotting.plt, "close", lambda figure: None)
    outputs = plotting.save_figure(Figure(), tmp_path / "candidate")
    assert [Path(path).suffix for path in outputs] == [".pdf", ".svg", ".png"]
    assert [suffix for suffix, _ in calls] == [".pdf", ".svg", ".png"]
    assert calls[2][1]["dpi"] == 600


def test_figure1_contains_all_required_metrics_and_baselines(tmp_path, monkeypatch) -> None:
    exported = _fake_export(monkeypatch)
    bootstrap = pd.DataFrame(
        [
            {
                "contrast": contrast,
                "rank": rank,
                "method": "learned",
                "metric": metric,
                "mean": 0.6,
                "ci_low": 0.52,
                "ci_high": 0.72,
            }
            for contrast in plotting.CONTRAST_ORDER
            for rank in (4, 8)
            for metric in (
                "map_r2_sufficiency",
                "map_r2_necessity",
                "ssi_fraction_transferred",
                "ssi_fraction_removed",
            )
        ]
    )
    plotting.figure1(_rank_rows(), _random_rows(), bootstrap, tmp_path, tmp_path)
    data = pd.read_csv(tmp_path / "figure1_rank_curves.csv")
    assert set(plotting.CONTRAST_ORDER).issubset(data.contrast)
    assert {
        "map_r2_sufficiency",
        "map_r2_necessity",
        "ssi_fraction_transferred",
        "ssi_fraction_removed",
    }.issubset(data.metric)
    assert {"learned", "movement_pca", "readout_svd", "random_haar"}.issubset(data.method)
    assert "crossed bootstrap" in set(data.ci_source)
    assert {Path(path).suffix for path in exported} == {".pdf", ".svg", ".png"}


def test_figure2_uses_median_heldout_fold_and_saves_exact_maps(tmp_path, monkeypatch) -> None:
    _fake_export(monkeypatch)
    names = [
        "representative_gain_a",
        "representative_gain_b",
        "representative_target_minus_baseline",
        "representative_gain_sufficiency",
        "representative_sufficiency_residual",
        "representative_gain_necessity",
        "representative_necessity_residual",
    ]
    maps = []
    metadata = []
    axis = np.linspace(-1.0, 1.0, 9)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    for contrast in plotting.CONTRAST_ORDER:
        for fold in range(3):
            baseline = 1.0 + 0.05 * np.exp(-(xx**2 + yy**2) / 0.5)
            target = 1.0 + (0.08 + 0.01 * fold) * np.exp(-((xx - 0.1) ** 2 + yy**2) / 0.3)
            sufficient = baseline + 0.8 * (target - baseline)
            necessity = target - 0.75 * (target - baseline)
            maps.append(
                np.stack(
                    (
                        baseline,
                        target,
                        target - baseline,
                        sufficient,
                        target - sufficient,
                        necessity,
                        necessity - baseline,
                    )
                )
            )
            metadata.append(
                {"stage": "crossval", "contrast": contrast, "fold": fold, "rank": 4}
            )
    plotting.figure2(
        _rank_rows(),
        np.stack(maps),
        names,
        metadata,
        {contrast: 4 for contrast in plotting.CONTRAST_ORDER},
        tmp_path,
        tmp_path,
    )
    selection = pd.read_csv(tmp_path / "figure2_objective_example_metadata.csv")
    assert set(selection.fold.astype(int)) == {1}
    required = {
        "map_r2_sufficiency",
        "map_r2_necessity",
        "ssi_a_bits",
        "ssi_b_bits",
        "ssi_sufficiency_bits",
        "ssi_necessity_bits",
        "ssi_fraction_transferred",
        "ssi_fraction_removed",
    }
    assert required.issubset(selection.columns)
    with np.load(tmp_path / "figure2_objective_maps.npz") as archive:
        assert len(archive.files) == 6 * len(plotting.CONTRAST_ORDER)


def test_figure3_is_conservative_without_stable_axes(tmp_path, monkeypatch) -> None:
    _fake_export(monkeypatch)
    cross_scale = []
    for contrast in ("low_0_to_2", "high_0_to_1"):
        for fold in range(3):
            for scale in (0.5, 1.0, 2.0, 3.0):
                baseline = 0.2 + 0.005 * fold
                target = baseline + 0.02 * scale
                cross_scale.append(
                    {
                        "contrast": contrast,
                        "rank": 4,
                        "method": "learned",
                        "fold": fold,
                        "scale_a": 0.0,
                        "scale_b": scale,
                        "ssi_a_bits": baseline,
                        "ssi_b_bits": target,
                        "ssi_sufficiency_bits": baseline + 0.8 * (target - baseline),
                        "ssi_necessity_bits": target - 0.7 * (target - baseline),
                    }
                )
    units = []
    for contrast in plotting.CONTRAST_ORDER:
        for fold in range(3):
            for unit in range(100):
                units.append(
                    {
                        "stage": "crossval",
                        "contrast": contrast,
                        "fold": fold,
                        "rank": 4,
                        "method": "learned",
                        "unit_index": unit,
                        "sf_split_metric": unit / 99.0,
                        "map_r2_sufficiency": 0.5,
                        "map_r2_necessity": 0.45,
                    }
                )
    _, detail = plotting.figure3(
        pd.DataFrame(cross_scale),
        pd.DataFrame(units),
        {contrast: 4 for contrast in plotting.CONTRAST_ORDER},
        False,
        tmp_path,
        tmp_path,
    )
    assert detail["canonical_axes_explicitly_stable"] is False
    assert detail["latent_axes_shown"] is False
    dose = pd.read_csv(tmp_path / "figure3_ssi_dose_curves.csv")
    assert {"intact", "sufficient", "necessity"} == set(dose.condition)
    assert set(dose.scale) == {0.0, 0.5, 1.0, 2.0, 3.0}
