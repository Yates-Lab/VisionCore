#!/usr/bin/env python3
"""Build the exact-unit population and tuning contract consumed by Figure 4.

This adapter is deliberately one-to-one: every output population row selects
one canonical checkpoint readout and retains its exact ``(session, cid)``
identity.  It never clusters, pools, averages, or substitutes RR units.
Only units released by ``audit_exact_cid_drifting_tuning.py`` are included.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.fig4.spatiotemporal_tuning.audit_exact_cid_drifting_tuning import (  # noqa: E402
    fit_yu_passband,
    preferred_direction,
    response_cube,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--population-version",
        default=None,
        help="Optional explicit name; defaults to <released-model-label>_exactCID_Figure4_v1.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_release(audit_dir: Path) -> tuple[pd.DataFrame, dict, Path]:
    table_path = audit_dir / "unit_measurement_audit.csv"
    report_path = audit_dir / "release_audit.json"
    if not table_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            "exact-CID contract requires unit_measurement_audit.csv and release_audit.json"
        )
    table = pd.read_csv(table_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not bool(report.get("figure4_unblocked", False)):
        raise ValueError("exact-CID tuning release has not unblocked Figure 4")
    if int(report.get("n_units", -1)) != len(table):
        raise ValueError("release report and unit table disagree")
    required = {
        "unit_index",
        "canonical_channel",
        "session",
        "cid",
        "validated_for_figure4",
        "yu_preferred_sf_cpd",
        "yu_preferred_tf_hz",
        "preferred_motion_direction_deg",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"unit audit lacks {missing}")
    if table.duplicated(["session", "cid"]).any():
        raise ValueError("exact biological identities are not unique")
    if table.canonical_channel.duplicated().any():
        raise ValueError("canonical channels are not unique")
    measurement_dir = Path(report["source_measurement"])
    if measurement_dir.resolve() != audit_dir.parent.resolve():
        raise ValueError("audit directory is not beside its declared source measurement")
    return table, report, measurement_dir


def _selected_units(table: pd.DataFrame, report: dict) -> pd.DataFrame:
    selected = table.loc[table.validated_for_figure4.astype(bool)].copy()
    selected = selected.sort_values("unit_index", kind="mergesort").reset_index(drop=True)
    if len(selected) != int(report["n_validated_for_figure4"]):
        raise ValueError("validated exact-unit count changed")
    if len(selected) < 24:
        raise ValueError("too few validated exact units for Figure 4")
    selected = selected.rename(columns={"unit_index": "source_unit_index"})
    selected.insert(0, "unit_index", np.arange(len(selected), dtype=int))
    if not np.isfinite(
        selected[
            [
                "yu_preferred_sf_cpd",
                "yu_preferred_tf_hz",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("validated units contain non-finite tuning coordinates")
    return selected


def _crossed_yu_examples(fits: pd.DataFrame) -> pd.DataFrame:
    """Select audited crossed examples deterministically and without responses.

    Biological SF is anchored by the recorded-grating assay.  The controlled
    drifting-grating replay supplies the matched twin SFxTF surface.  A valid
    example therefore requires the recorded neuron and twin to occupy the same
    SF tail, while twin TF occupies the opposite tail.  ``crossed_group`` and
    ``crossed_extremity_octaves`` are computed by the release audit only after
    all quality gates have passed.

    The low-SF/high-TF example is the most crossed member of its audited group.
    For the high-SF/low-TF example, an extreme boundary-hugging unit is visually
    misleading, so the released example is the log-SF/log-TF medoid of that
    audited group.  This makes the example representative of the observed
    cluster while remaining deterministic, tuning-only, and independent of all
    retinal-motion response analyses.
    """
    required = {
        "unit_index",
        "source_unit_index",
        "preferred_sf_cpd",
        "preferred_tf_hz",
        "full_support_r2",
        "recorded_data_preferred_sf_cpd",
        "crossed_group",
        "crossed_extremity_octaves",
    }
    missing = sorted(required.difference(fits.columns))
    if missing:
        raise ValueError(f"Yu fit table lacks crossed-example fields: {missing}")
    numeric_required = (
        "unit_index",
        "source_unit_index",
        "preferred_sf_cpd",
        "preferred_tf_hz",
        "full_support_r2",
        "recorded_data_preferred_sf_cpd",
    )
    values = fits[list(numeric_required)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Yu fit table contains non-finite crossed-example fields")
    specifications = (
        (
            "low SF / high TF",
            "low recorded SF / high twin TF",
            "crossed-extremity leader",
        ),
        (
            "high SF / low TF",
            "high recorded SF / low twin TF",
            "log-SF/log-TF group medoid",
        ),
    )
    selected: list[pd.Series] = []
    for role, audited_group, policy in specifications:
        candidates = fits.loc[fits.crossed_group.eq(audited_group)].copy()
        if candidates.empty:
            raise ValueError(f"validated Yu population has no {role} candidate")
        if not np.isfinite(
            candidates.crossed_extremity_octaves.to_numpy(dtype=float)
        ).all():
            raise ValueError(f"{role} candidates lack audited extremity scores")
        if policy == "log-SF/log-TF group medoid":
            log_sf = np.log2(candidates.preferred_sf_cpd.to_numpy(dtype=float))
            log_tf = np.log2(candidates.preferred_tf_hz.to_numpy(dtype=float))
            center_sf = float(np.median(log_sf))
            center_tf = float(np.median(log_tf))
            candidates["example_selection_score_octaves"] = np.hypot(
                log_sf - center_sf,
                log_tf - center_tf,
            )
            candidates = candidates.sort_values(
                [
                    "example_selection_score_octaves",
                    "high_mode_count",
                    "full_support_r2",
                    "contrast_surface_corr",
                    "peak_delta_f0_expected_count",
                    "source_unit_index",
                ],
                ascending=[True, True, False, False, False, True],
                kind="mergesort",
            )
        else:
            candidates["example_selection_score_octaves"] = (
                candidates.crossed_extremity_octaves.to_numpy(dtype=float)
            )
            candidates = candidates.sort_values(
                [
                    "crossed_extremity_octaves",
                    "high_mode_count",
                    "full_support_r2",
                    "contrast_surface_corr",
                    "peak_delta_f0_expected_count",
                    "source_unit_index",
                ],
                ascending=[False, True, False, False, False, True],
                kind="mergesort",
            )
        leader = candidates.iloc[0].copy()
        leader["role"] = role
        leader["example_selection_policy"] = policy
        leader["validated_for_figure4"] = True
        selected.append(leader)
    return pd.DataFrame(selected).reset_index(drop=True)


def _population_spec(
    selected: pd.DataFrame,
    report: dict,
    out_dir: Path,
    version: str,
) -> tuple[Path, Path]:
    n_channels = int(report["source_provenance"]["n_canonical_units"])
    channels = selected.canonical_channel.to_numpy(dtype=int)
    if np.any(channels < 0) or np.any(channels >= n_channels):
        raise ValueError("validated canonical channel outside population support")
    membership = np.zeros((len(selected), n_channels), dtype=np.float32)
    membership[np.arange(len(selected)), channels] = 1.0
    cluster_membership = membership.copy()
    labels = np.full(n_channels, -1, dtype=np.int32)
    labels[channels] = np.arange(len(selected), dtype=np.int32)
    representatives = [
        {
            "rep_idx": int(row.unit_index),
            "selected_channel": int(row.canonical_channel),
            "rep_channel": int(row.canonical_channel),
            "members": [int(row.canonical_channel)],
            "n_members": 1,
            "pooling_mode": "exact_identity",
            "session": str(row.session),
            "cid": int(row.cid),
            "source_unit_index": int(row.source_unit_index),
        }
        for row in selected.itertuples(index=False)
    ]
    stem = f"population_spec_{version}"
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"
    np.savez_compressed(
        npz_path,
        membership=membership,
        cluster_membership=cluster_membership,
        labels=labels,
    )
    payload = {
        "version": str(version),
        "analysis": "one-to-one exact-CID Figure-4 population view",
        "pooling_mode": "exact_identity",
        "n_representatives": int(len(selected)),
        "n_input_channels": n_channels,
        "identity_key": "(session, cid)",
        "selection_gate": "validated_for_figure4",
        "rr_clustering": False,
        "membership_contract": (
            "one nonzero value of exactly 1.0 per row at canonical_channel; "
            "no pooling, medoid substitution, or unavailable-channel replacement"
        ),
        "checkpoint_sha256": str(
            report["source_provenance"]["checkpoint_sha256"]
        ),
        "source_release": str((Path(report["source_measurement"]) / "audit" / "release_audit.json").resolve()),
        "representatives": representatives,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return npz_path, json_path


def _grouped_tuning(
    selected: pd.DataFrame,
    measurement_dir: Path,
    *,
    require_optimizer_success: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditions = pd.read_csv(measurement_dir / "conditions.csv")
    source_units = pd.read_csv(measurement_dir / "units.csv")
    with np.load(measurement_dir / "responses.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if not np.array_equal(
        source_units.unit_index.to_numpy(dtype=int), np.arange(len(source_units))
    ):
        raise ValueError("measurement response columns are not in exact-unit row order")
    if not np.array_equal(
        conditions.condition_index.to_numpy(dtype=int), np.arange(len(conditions))
    ):
        raise ValueError("measurement conditions are not in saved response order")
    sf, tf, directions, delta_cube = response_cube(
        conditions, arrays["delta_f0_expected_count"]
    )
    dynamic = tf > 0
    dynamic_tf = tf[dynamic]
    axial = np.mod(conditions.bar_orientation_deg.to_numpy(dtype=float), 180.0)
    axial_values = np.sort(np.unique(axial))
    output_rows: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    measurement_unit_lookup = {
        int(row.unit_index): position
        for position, row in enumerate(source_units.itertuples(index=False))
    }
    for selected_row in selected.itertuples(index=False):
        source_unit = int(selected_row.source_unit_index)
        if source_unit not in measurement_unit_lookup:
            raise ValueError(f"source unit {source_unit} missing from measurement")
        position = measurement_unit_lookup[source_unit]
        unit_delta = delta_cube[..., position]
        direction_index = preferred_direction(unit_delta, dynamic)
        surface = np.maximum(
            unit_delta[:, dynamic, direction_index], 0.0
        ).T
        yu = fit_yu_passband(sf, dynamic_tf, surface)
        optimizer_success = bool(yu.get("fit_success", False))
        has_finite_fit = all(
            key in yu and np.isfinite(np.asarray(yu[key], dtype=float)).all()
            for key in ("parameters", "prediction", "preferred_sf_cpd", "preferred_tf_hz")
        )
        if not has_finite_fit or (require_optimizer_success and not optimizer_success):
            status = "non-converged" if has_finite_fit else "missing/non-finite"
            raise ValueError(
                f"source unit {source_unit} has a {status} Yu fit; "
                f"require_optimizer_success={require_optimizer_success}"
            )
        if not np.isclose(
            float(yu["preferred_sf_cpd"]),
            float(selected_row.yu_preferred_sf_cpd),
            atol=1e-6,
            rtol=0.0,
        ) or not np.isclose(
            float(yu["preferred_tf_hz"]),
            float(selected_row.yu_preferred_tf_hz),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(f"Yu refit changed for released source unit {source_unit}")
        parameters = np.asarray(yu["parameters"], dtype=float)
        prediction = np.maximum(np.asarray(yu["prediction"], dtype=float), 0.0)
        prediction /= max(float(prediction.max()), 1e-12)
        peak_tf, peak_sf = np.unravel_index(int(np.argmax(surface)), surface.shape)
        direction_response = np.maximum(
            unit_delta[peak_sf, np.flatnonzero(dynamic)[peak_tf]], 0.0
        )
        orientation_weight = np.zeros(len(axial_values), dtype=float)
        direction_bars = np.mod(directions + 90.0, 180.0)
        for orientation_index, orientation in enumerate(axial_values):
            members = np.flatnonzero(np.isclose(direction_bars, orientation))
            orientation_weight[orientation_index] = float(
                np.mean(direction_response[members])
            )
        if float(orientation_weight.max()) <= 0:
            preferred_bar = float(direction_bars[direction_index])
            orientation_weight[np.argmin(np.abs(axial_values - preferred_bar))] = 1.0
        else:
            orientation_weight /= float(orientation_weight.max())

        unit_conditions = conditions.copy()
        unit_conditions["bar_orientation_deg"] = axial
        for name, key in (
            ("mean_rate", "f0_expected_count"),
            ("delta_f0_expected_count", "delta_f0_expected_count"),
            ("response_amp_rms", "phase_modulation_rms_expected_count"),
            ("f1_amplitude", "f1_expected_count_amplitude"),
            ("minimum_rate", "minimum_expected_count"),
            ("maximum_rate", "maximum_expected_count"),
        ):
            unit_conditions[name] = np.asarray(arrays[key])[:, position]
        preferred_direction_surface = np.zeros(len(unit_conditions), dtype=float)
        sf_lookup = {float(value): index for index, value in enumerate(sf)}
        tf_lookup = {float(value): index for index, value in enumerate(dynamic_tf)}
        for condition_index, condition in enumerate(
            unit_conditions.itertuples(index=False)
        ):
            temporal_hz = float(condition.temporal_hz)
            if temporal_hz <= 0:
                continue
            preferred_direction_surface[condition_index] = surface[
                tf_lookup[temporal_hz], sf_lookup[float(condition.spatial_cpd)]
            ]
        # This is the exact raw surface supplied to the Yu fit.  It is repeated
        # across axial-orientation rows only so the shared table schema can be
        # consumed by the display helper without averaging opposite directions.
        unit_conditions[
            "preferred_direction_delta_f0_expected_count"
        ] = preferred_direction_surface
        grouped = (
            unit_conditions.groupby(
                ["spatial_cpd", "temporal_hz", "bar_orientation_deg"],
                as_index=False,
                sort=True,
            )[
                [
                    "mean_rate",
                    "delta_f0_expected_count",
                    "response_amp_rms",
                    "f1_amplitude",
                    "minimum_rate",
                    "maximum_rate",
                    "preferred_direction_delta_f0_expected_count",
                ]
            ]
            .mean()
            .rename(columns={"bar_orientation_deg": "probe_orientation_deg"})
        )
        ori_lookup = {float(value): index for index, value in enumerate(axial_values)}
        passband = np.zeros(len(grouped), dtype=float)
        for row_index, row in enumerate(grouped.itertuples(index=False)):
            if float(row.temporal_hz) <= 0:
                continue
            passband[row_index] = (
                prediction[
                    tf_lookup[float(row.temporal_hz)],
                    sf_lookup[float(row.spatial_cpd)],
                ]
                * orientation_weight[ori_lookup[float(row.probe_orientation_deg)]]
            )
        grouped.insert(0, "unit_label", f"u{int(selected_row.unit_index):03d}")
        grouped.insert(0, "unit_index", int(selected_row.unit_index))
        grouped["source_unit_index"] = source_unit
        grouped["canonical_channel"] = int(selected_row.canonical_channel)
        grouped["session"] = str(selected_row.session)
        grouped["cid"] = int(selected_row.cid)
        grouped["passband_weight"] = passband
        output_rows.append(grouped)

        preferred_bar = float(direction_bars[direction_index])
        fit_rows.append(
            {
                "unit_index": int(selected_row.unit_index),
                "source_unit_index": source_unit,
                "canonical_channel": int(selected_row.canonical_channel),
                "session": str(selected_row.session),
                "cid": int(selected_row.cid),
                "preferred_orientation_deg": preferred_bar,
                "preferred_motion_direction_deg": float(directions[direction_index]),
                "response_column": "preferred_direction_delta_f0_expected_count",
                "preferred_sf_cpd": float(yu["preferred_sf_cpd"]),
                "preferred_tf_hz": float(yu["preferred_tf_hz"]),
                "selected_model": str(yu["selected_model"]),
                "optimizer_success": optimizer_success,
                "full_support_r2": float(yu["selected_r2"]),
                "p_r1_full_support": float(yu["p_r1"]),
                "sigma_s": float(parameters[2]),
                "zeta_s": float(parameters[3]),
                "sigma_t": float(parameters[5]),
                "zeta_t": float(parameters[6]),
                "q": float(parameters[7]) if str(yu["selected_model"]) == "R1" else 0.0,
                "fit_contract": "released exact-CID Yu R0/R1 fit to every acquired dynamic SFxTF cell at the preferred motion direction",
                "measured_min_sf_cpd": float(sf.min()),
                "measured_max_sf_cpd": float(sf.max()),
                "measured_min_tf_hz": float(dynamic_tf.min()),
                "measured_max_tf_hz": float(dynamic_tf.max()),
                "n_full_support_conditions": int(surface.size),
                "recorded_data_preferred_sf_cpd": float(
                    selected_row.recorded_data_preferred_sf_cpd
                ),
                "crossed_group": str(selected_row.crossed_group),
                "crossed_extremity_octaves": float(
                    selected_row.crossed_extremity_octaves
                ),
                "high_mode_count": int(selected_row.high_mode_count),
                "contrast_surface_corr": float(
                    selected_row.contrast_surface_corr
                ),
                "peak_delta_f0_expected_count": float(
                    selected_row.peak_delta_f0_expected_count
                ),
            }
        )
    tuning = pd.concat(output_rows, ignore_index=True)
    fits = pd.DataFrame(fit_rows)
    expected_dynamic = len(selected) * len(sf) * len(dynamic_tf) * len(axial_values)
    if int(tuning.temporal_hz.gt(0).sum()) != expected_dynamic:
        raise RuntimeError("exact-unit dynamic tuning table is incomplete")
    return tuning, fits


def _write_contract(
    selected: pd.DataFrame,
    report: dict,
    tuning: pd.DataFrame,
    fits: pd.DataFrame,
    out_dir: Path,
    version: str,
    spec_paths: tuple[Path, Path],
) -> None:
    checkpoint = str(report["source_provenance"]["checkpoint"])
    checkpoint_digest = str(report["source_provenance"]["checkpoint_sha256"])
    dataset_config = str(report["source_provenance"]["dataset_config"])
    selected.to_csv(out_dir / "exact_unit_mapping.csv", index=False)
    tuning.to_csv(out_dir / "frequency_tuning_grouped.csv", index=False)
    fits.to_csv(out_dir / "all_validated_yu_fits.csv", index=False)

    examples = _crossed_yu_examples(fits)
    examples.to_csv(out_dir / "crossed_example_fits.csv", index=False)

    provenance = {
        "analysis": "exact-CID Figure-4 grouped tuning contract",
        "model_label": str(report["source_provenance"]["model_label"]),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_digest,
        "dataset_config": dataset_config,
        "dataset_config_sha256": str(
            report["source_provenance"]["dataset_config_sha256"]
        ),
        "n_units": int(len(selected)),
        "population_version": str(version),
        "population_identity": "one-to-one exact (session, cid); no RR pooling",
        "passband_weight": (
            "released Yu SFxTF prediction multiplied by measured axial-orientation tuning; "
            "used only for rendered-movie spectral engagement"
        ),
        "source_release": str(Path(report["files"]["unit_audit"]).resolve()),
    }
    (out_dir / "periodic_tuning_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    tuning_summary = pd.DataFrame(
        {
            "unit_index": selected.unit_index.astype(int),
            "source_unit_index": selected.source_unit_index.astype(int),
            "canonical_channel": selected.canonical_channel.astype(int),
            "session": selected.session.astype(str),
            "cid": selected.cid.astype(int),
            "audit_category": "trusted",
            "validated_tuning": True,
            # One chain of custody: both coordinates come from the released
            # exact-CID Yu SFxTF fit used to construct every passband below.
            "validated_preferred_sf_cpd": selected.yu_preferred_sf_cpd.astype(float),
            "validated_preferred_tf_hz": selected.yu_preferred_tf_hz.astype(float),
            "exact_twin_yu_preferred_sf_cpd": selected.yu_preferred_sf_cpd.astype(float),
            "exact_twin_yu_preferred_tf_hz": selected.yu_preferred_tf_hz.astype(float),
        }
    )
    tuning_summary.to_csv(out_dir / "tuning_summary.csv", index=False)
    summary = {
        "analysis": "exact-CID Figure-4 population contract",
        "model_label": str(report["source_provenance"]["model_label"]),
        "checkpoint_sha256": checkpoint_digest,
        "n_units": int(len(selected)),
        "release_ready": True,
        "selected_units_pass_validated_tuning_gate": True,
        "trusted_tuning_contract": {
            "coordinate_assay": "exact_cid_yu_sf_tf",
            "unit_indices": selected.unit_index.astype(int).tolist(),
            "sf_column": "validated_preferred_sf_cpd",
            "sf_source": "released exact-CID Yu preferred SF",
            "tf_column": "validated_preferred_tf_hz",
            "tf_source": "released exact-twin Yu preferred TF",
            "coordinate_source_file": str(
                Path(report["files"]["unit_audit"]).resolve()
            ),
            "coordinate_selection_gate": "validated_for_figure4",
            "forbidden_substitutions": [
                "recorded_data_preferred_sf_cpd",
                "recorded_model_preferred_sf_cpd",
                "RR100 pooled tuning",
            ],
            "population_version": str(version),
            "identity_mapping": "exact_unit_mapping.csv",
        },
        "gates": {
            "source_exact_cid_release_unblocked": bool(report["figure4_unblocked"]),
            "all_rows_validated_for_figure4": bool(
                selected.validated_for_figure4.astype(bool).all()
            ),
            "one_hot_identity_membership": True,
            "no_rr_pooling": True,
            "all_tuning_rows_complete": True,
            "sf_and_tf_coordinates_equal_released_yu_fits": bool(
                np.allclose(
                    tuning_summary.validated_preferred_sf_cpd,
                    selected.yu_preferred_sf_cpd,
                    atol=0.0,
                    rtol=0.0,
                )
                and np.allclose(
                    tuning_summary.validated_preferred_tf_hz,
                    selected.yu_preferred_tf_hz,
                    atol=0.0,
                    rtol=0.0,
                )
            ),
        },
        "files": {
            "population_spec_npz": str(spec_paths[0].resolve()),
            "population_spec_json": str(spec_paths[1].resolve()),
            "unit_mapping": str((out_dir / "exact_unit_mapping.csv").resolve()),
            "tuning_table": str((out_dir / "frequency_tuning_grouped.csv").resolve()),
            "all_fits": str((out_dir / "all_validated_yu_fits.csv").resolve()),
            "example_fits": str((out_dir / "crossed_example_fits.csv").resolve()),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    exemplar_summary = {
        "analysis": "released crossed exact-CID tuning exemplars",
        "model_label": str(report["source_provenance"]["model_label"]),
        "checkpoint_sha256": checkpoint_digest,
        "release_ready": True,
        "selected_units_pass_validated_tuning_gate": True,
        "selection": (
            "deterministic tuning-only selection after recorded-neuron and exact-twin "
            "SF agree on one outer third and twin TF occupies the other: low-SF/high-TF "
            "uses the crossed-extremity leader; high-SF/low-TF uses the audited "
            "group medoid in log SFxTF space"
        ),
        "selection_reads_retinal_motion_response": False,
        "unit_indices": examples.unit_index.astype(int).tolist(),
        "canonical_channels": examples.canonical_channel.astype(int).tolist(),
        "roles": examples.role.astype(str).tolist(),
    }
    (out_dir / "crossed_example_summary.json").write_text(
        json.dumps(exemplar_summary, indent=2) + "\n", encoding="utf-8"
    )
    model_spec = {
        "schema_version": 1,
        "status": "figure4_exact_cid_analysis",
        "label": str(report["source_provenance"]["model_label"]),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_digest,
        "dataset_config": dataset_config,
        "dataset_config_sha256": str(
            report["source_provenance"]["dataset_config_sha256"]
        ),
        "population_version": str(version),
        "population_contract": "one-to-one exact-CID released Figure-4 units",
    }
    (out_dir / "model_spec.yaml").write_text(
        yaml.safe_dump(model_spec, sort_keys=False), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    audit_dir = args.audit_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    table, report, measurement_dir = _load_release(audit_dir)
    population_version = str(
        args.population_version
        or f"{report['source_provenance']['model_label']}_exactCID_Figure4_v1"
    )
    selected = _selected_units(table, report)
    spec_paths = _population_spec(
        selected, report, out_dir, population_version
    )
    tuning, fits = _grouped_tuning(selected, measurement_dir)
    _write_contract(
        selected,
        report,
        tuning,
        fits,
        out_dir,
        population_version,
        spec_paths,
    )
    print(out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
