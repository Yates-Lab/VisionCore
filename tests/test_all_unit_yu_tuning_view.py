import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper.fig4.spatiotemporal_tuning import build_all_unit_yu_tuning_view as view


def test_all_unit_view_retains_failed_units_with_exact_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    measurement = tmp_path / "measurement"
    audit_dir = measurement / "audit"
    audit_dir.mkdir(parents=True)
    audit = pd.DataFrame(
        {
            "unit_index": [0, 1, 2],
            "session": ["s0", "s0", "s1"],
            "cid": [10, 11, 20],
            "canonical_channel": [0, 1, 2],
            "yu_preferred_sf_cpd": [1.0, 2.0, 4.0],
            "yu_preferred_tf_hz": [16.0, 4.0, 2.0],
            "preferred_motion_direction_deg": [0.0, 90.0, 180.0],
            "validated_for_figure4": [True, False, False],
            "failed_checks": ["", "coherent_surface", "recorded_sf_twin_match"],
        }
    )
    audit.to_csv(audit_dir / "unit_measurement_audit.csv", index=False)
    source_provenance = {
        "checkpoint_sha256": "abc123",
        "model_label": "fresh",
    }
    (measurement / "provenance.json").write_text(json.dumps(source_provenance))
    (measurement / "units.csv").write_text("unit_index\n0\n1\n2\n")
    (measurement / "conditions.csv").write_text("condition_index\n0\n")
    np.savez(measurement / "responses.npz", placeholder=np.zeros((1, 3)))
    (audit_dir / "release_audit.json").write_text(
        json.dumps(
            {
                "figure4_unblocked": True,
                "n_units": 3,
                "source_measurement": str(measurement),
                "source_provenance": source_provenance,
            }
        )
    )
    response_units = audit[
        ["unit_index", "session", "cid", "canonical_channel"]
    ]
    response_path = tmp_path / "response_units.csv"
    response_units.to_csv(response_path, index=False)
    out_dir = tmp_path / "out"

    def grouped(selected, source_dir, *, require_optimizer_success):
        assert source_dir == measurement
        assert require_optimizer_success is False
        assert selected.source_unit_index.tolist() == [0, 1, 2]
        tuning = pd.DataFrame(
            {
                "unit_index": [0, 1, 2],
                "passband_weight": [0.2, 0.3, 0.4],
            }
        )
        fits = pd.DataFrame(
            {
                "unit_index": [0, 1, 2],
                "optimizer_success": [True, False, True],
            }
        )
        return tuning, fits

    monkeypatch.setattr(view, "_grouped_tuning", grouped)
    monkeypatch.setattr(
        view,
        "parse_args",
        lambda: argparse.Namespace(
            audit_dir=audit_dir,
            response_unit_table=response_path,
            out_dir=out_dir,
            force=False,
        ),
    )
    assert view.main() == 0

    report = json.loads((out_dir / "summary.json").read_text())
    assert report["population_policy"] == "all_checkpoint_available"
    assert report["n_units"] == 3
    assert report["n_figure4_validated_units"] == 1
    assert report["n_validation_failures_retained"] == 2
    assert report["n_finite_nonconverged_fits_retained"] == 1
    assert report["gates"] == {
        "all_checkpoint_available_units_included_once": True,
        "response_identity_order_exact": True,
        "no_rr_pooling": True,
        "no_tuning_quality_gate_applied": True,
        "every_exported_passband_surface_finite": True,
    }
    tuning_summary = pd.read_csv(out_dir / "tuning_summary.csv")
    assert tuning_summary.included_in_exploratory_population.astype(bool).all()
    assert tuning_summary.validated_for_figure4.tolist() == [True, False, False]
    assert np.array_equal(tuning_summary.source_unit_index, np.arange(3))
