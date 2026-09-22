#!/usr/bin/env python3
"""Build the Yu SFxTF view for every checkpoint-available exact unit.

The strict Figure-4 release gate remains attached to every row but is not used
to select the all-unit analysis population.  This product is the explicit
bridge between the 725-unit response population and the controlled drifting-
grating measurement.  Validation failures are retained and disclosed; they
are never relabeled as validated units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.build_exact_cid_figure4_contract import (  # noqa: E402
    _grouped_tuning,
    _load_release,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--response-unit-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    audit_dir = args.audit_dir.resolve()
    response_unit_path = args.response_unit_table.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        out_dir / "frequency_tuning_grouped.csv",
        out_dir / "all_yu_fits.csv",
        out_dir / "tuning_summary.csv",
        out_dir / "summary.json",
    )
    if not args.force and any(path.exists() for path in outputs):
        raise FileExistsError("all-unit tuning-view outputs exist; pass --force")

    audit_path = audit_dir / "unit_measurement_audit.csv"
    release_path = audit_dir / "release_audit.json"
    audit, release, measurement_dir = _load_release(audit_dir)
    audit = audit.sort_values(
        "unit_index", kind="mergesort"
    ).reset_index(drop=True)
    response_units = pd.read_csv(response_unit_path).sort_values(
        "unit_index", kind="mergesort"
    ).reset_index(drop=True)
    if not np.array_equal(
        audit.unit_index.to_numpy(dtype=int), np.arange(len(audit))
    ):
        raise ValueError("Yu audit rows are not in exact-unit response order")
    if not np.array_equal(
        response_units.unit_index.to_numpy(dtype=int),
        np.arange(len(response_units)),
    ):
        raise ValueError("response-unit rows are not contiguous and zero based")
    if len(audit) != len(response_units):
        raise ValueError("Yu audit and response matrix contain different unit counts")
    for column in ("session", "cid", "canonical_channel"):
        left = audit[column].astype(str).to_numpy()
        right = response_units[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"Yu audit and response matrix differ at {column}")
    coordinates = audit[
        ["yu_preferred_sf_cpd", "yu_preferred_tf_hz"]
    ].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("all-unit Yu view contains non-finite coordinates")

    selected = audit.copy()
    selected.insert(0, "source_unit_index", selected.unit_index.to_numpy(dtype=int))
    tuning, fits = _grouped_tuning(
        selected,
        measurement_dir,
        require_optimizer_success=False,
    )
    tuning.to_csv(outputs[0], index=False)
    fits.to_csv(outputs[1], index=False)
    summary_table = pd.DataFrame(
        {
            "unit_index": audit.unit_index.astype(int),
            "source_unit_index": audit.unit_index.astype(int),
            "canonical_channel": audit.canonical_channel.astype(int),
            "session": audit.session.astype(str),
            "cid": audit.cid.astype(int),
            # The historical column name remains part of the renderer contract;
            # population_policy below is the authoritative production label.
            "included_in_exploratory_population": True,
            "validated_for_figure4": audit.validated_for_figure4.astype(bool),
            "validation_failed_checks": audit.failed_checks.fillna("").astype(str),
            "exact_twin_yu_preferred_sf_cpd": audit.yu_preferred_sf_cpd.astype(float),
            "exact_twin_yu_preferred_tf_hz": audit.yu_preferred_tf_hz.astype(float),
        }
    )
    summary_table.to_csv(outputs[2], index=False)
    checkpoint_sha256 = str(release["source_provenance"]["checkpoint_sha256"])
    report = {
        "analysis": "all-checkpoint-available exact-CID Yu SFxTF view",
        "artifact_status": "analysis population; strict validation failures retained and disclosed",
        "model_label": str(release["source_provenance"]["model_label"]),
        "checkpoint_sha256": checkpoint_sha256,
        "population_policy": "all_checkpoint_available",
        "n_units": int(len(audit)),
        "n_figure4_validated_units": int(
            audit.validated_for_figure4.astype(bool).sum()
        ),
        "n_validation_failures_retained": int(
            (~audit.validated_for_figure4.astype(bool)).sum()
        ),
        "n_optimizer_converged_fits": int(fits.optimizer_success.astype(bool).sum()),
        "n_finite_nonconverged_fits_retained": int(
            (~fits.optimizer_success.astype(bool)).sum()
        ),
        "coordinate_assay": "exact_cid_yu_sf_tf",
        "coordinate_selection_gate": "none for the all-checkpoint-available view",
        "unit_indices": audit.unit_index.astype(int).tolist(),
        "identity_contract": "exact ordered (session, cid, canonical_channel) match to all-unit response matrix",
        "source_unit_audit": str(audit_path),
        "source_unit_audit_sha256": sha256_file(audit_path),
        "source_release": str(release_path),
        "source_release_sha256": sha256_file(release_path),
        "source_measurement": str(measurement_dir),
        "source_measurement_provenance_sha256": sha256_file(
            measurement_dir / "provenance.json"
        ),
        "response_unit_table": str(response_unit_path),
        "response_unit_table_sha256": sha256_file(response_unit_path),
        "files": {
            "tuning_table": str(outputs[0]),
            "all_yu_fits": str(outputs[1]),
            "tuning_summary": str(outputs[2]),
        },
        "gates": {
            "all_checkpoint_available_units_included_once": True,
            "response_identity_order_exact": True,
            "no_rr_pooling": True,
            "no_tuning_quality_gate_applied": True,
            "every_exported_passband_surface_finite": bool(
                np.isfinite(tuning.passband_weight.to_numpy(dtype=float)).all()
            ),
        },
    }
    outputs[3].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(outputs[3], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
