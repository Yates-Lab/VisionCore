#!/usr/bin/env python3
"""Run the recovered production schematic-map analysis with a selected twin."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set for selected-twin Figure 4 rendering.")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _recovered_python_root() -> Path:
    root = _required_path("FIG4_RECOVERED_ROOT")
    return root.parent if root.name == "declan" else root


def _out_dir() -> Path | None:
    values = sys.argv[1:]
    for index, value in enumerate(values):
        if value == "--out-dir" and index + 1 < len(values):
            return Path(values[index + 1]).expanduser().resolve()
        if value.startswith("--out-dir="):
            return Path(value.split("=", 1)[1]).expanduser().resolve()
    return None


def main() -> None:
    checkpoint = _required_path("FIG4_TWIN_CHECKPOINT")
    dataset_configs = _required_path("FIG4_DATASET_CONFIGS")
    population_spec_dir = _required_path("FIG4_RR100_POPULATION_SPEC_DIR")
    unit_map_dir = _required_path("FIG4_UNIT_MAP_DIR")
    sf_group_csv = _required_path("FIG4_SF_TUNING_GROUP_CSV")
    recovered_root = _recovered_python_root()
    mcfarland_value = os.environ.get("FIG4_MCFARLAND_OUTPUTS")
    mcfarland_outputs = (
        Path(mcfarland_value).expanduser().resolve() if mcfarland_value else None
    )
    if mcfarland_outputs is not None and not mcfarland_outputs.exists():
        raise FileNotFoundError(f"FIG4_MCFARLAND_OUTPUTS does not exist: {mcfarland_outputs}")

    for path in (ROOT, recovered_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from paper.fig4.upstream.real_trace_matrix.core import sha256_file
    from paper.fig4.upstream.real_trace_matrix.model import (
        LegacyCanonicalTwinScorerAdapter,
        RealTraceMatrixScorer,
    )
    from paper.fig4.upstream.run_selected_twin_frequency_tuning_probe import (
        _declared_time_contract,
    )
    from paper.fig4.upstream.run_selected_twin_instantaneous_unit_maps import (
        LEGACY_SCHEMATIC_TRACE_RATE_HZ,
        _trace_on_selected_output_grid,
    )
    from declan.fig_ssi import make_ssi_contour_schematic as schematic_source

    # As in the instantaneous-map adapter, execute the recovered production
    # code while resolving its retained, model-independent inputs from this
    # checkout rather than from the recovery tree's empty outputs directory.
    schematic_source.NEW_BANK_IMAGE_TABLE = (
        ROOT
        / "outputs"
        / "active_sensing_movie_information"
        / "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1"
        / "merged"
        / "image_feature_table.csv"
    )
    schematic_source.SCHEMATIC_REAL_TRACE_CENTER40_CSV = (
        ROOT
        / "outputs"
        / "fig_ssi"
        / "trace_provenance"
        / "schematic_crop_real_backimage_trace_center40.csv"
    )

    from declan.fig_ssi import compute_schematic_rr100_final_maps as production

    # Unit annotations are checkpoint-specific outputs of the two preceding
    # stages.  Never fall back to similarly named products under the recovery
    # tree, which would silently mix twins inside one production panel.
    production.RUN_DIR = unit_map_dir
    production.ORIENTATION_GROUP_CSV = unit_map_dir / "orientation_tuning_groups.csv"
    production.SF_GROUP_CSV = sf_group_csv
    recovered_load_population_view = production.load_population_view

    def selected_load_population_view(spec_dir=None, *, version_name=None):
        return recovered_load_population_view(
            population_spec_dir if spec_dir is None else spec_dir,
            version_name=version_name,
        )

    production.load_population_view = selected_load_population_view

    time_contract = _declared_time_contract(dataset_configs)
    adapters: dict[str, LegacyCanonicalTwinScorerAdapter] = {}

    class SelectedTwinScorer(LegacyCanonicalTwinScorerAdapter):
        def __init__(
            self,
            *,
            device: str,
            batch_size: int,
            empty_cache_every_batch: bool = False,
        ) -> None:
            resolved_device = "cuda:0" if str(device) == "auto" else str(device)
            key = f"{resolved_device}:{int(batch_size)}:{int(bool(empty_cache_every_batch))}"
            cached = adapters.get(key)
            if cached is not None:
                self.__dict__ = cached.__dict__
                return
            scorer = RealTraceMatrixScorer.load(
                checkpoint_path=checkpoint,
                dataset_configs=dataset_configs,
                population_spec_dir=population_spec_dir,
                rr100_version=str(production.RR100_MOVIE_MEDOID_VERSION),
                device=resolved_device,
                strict=True,
                mcfarland_outputs=mcfarland_outputs,
            )
            loaded_contract = {
                "input_rate_hz": int(scorer.input_rate_hz),
                "output_rate_hz": int(scorer.output_rate_hz),
                "temporal_factor": int(scorer.temporal_factor),
                "supervision_phase": int(scorer.supervision_phase),
                "native_history_frames": int(scorer.n_lags),
            }
            if loaded_contract != time_contract:
                raise RuntimeError(
                    "Selected-twin scorer disagrees with the declared dataset "
                    f"time contract: {loaded_contract} != {time_contract}"
                )
            super().__init__(
                scorer,
                batch_size=int(batch_size),
                empty_cache_every_batch=bool(empty_cache_every_batch),
            )
            adapters[key] = self

    production.CanonicalTwinScorer = SelectedTwinScorer
    production.rate_map_for_trace = (
        lambda scorer, patch, trace: scorer.rate_map_for_trace(
            patch,
            _trace_on_selected_output_grid(
                trace,
                output_rate_hz=scorer._scorer.output_rate_hz,
                torch=scorer.torch,
            ),
        )
    )
    original_compute_maps = production.compute_maps

    def selected_compute_maps(args):
        payload = original_compute_maps(args)
        adapter = next(iter(adapters.values()))
        scorer = adapter._scorer
        payload["meta"].update(
            {
                "selected_twin_checkpoint": str(checkpoint),
                "selected_twin_checkpoint_sha256": sha256_file(checkpoint),
                "selected_twin_dataset_configs": str(dataset_configs),
                "selected_twin_dataset_configs_sha256": sha256_file(dataset_configs),
                "selected_twin_native_history_frames": int(scorer.n_lags),
                "selected_twin_input_rate_hz": int(scorer.input_rate_hz),
                "selected_twin_output_rate_hz": int(scorer.output_rate_hz),
                "selected_twin_history_seconds": (
                    float(scorer.n_lags) / float(scorer.input_rate_hz)
                ),
                "selected_twin_temporal_contract": time_contract,
                "retained_schematic_trace_rate_hz": LEGACY_SCHEMATIC_TRACE_RATE_HZ,
                "retained_trace_resampling": (
                    "endpoint-anchored linear interpolation with held boundaries"
                ),
                "history_prefix_policy": "hold_initial_gaze_causally",
            }
        )
        payload["patch_meta_json"] = np.asarray(
            [json.dumps(production.json_ready(payload["meta"]), sort_keys=True)]
        )
        return payload

    production.compute_maps = selected_compute_maps
    production.main()

    out_dir = _out_dir()
    if out_dir is not None:
        provenance = {
            "analysis_entrypoint": str(Path(production.__file__).resolve()),
            "analysis_entrypoint_sha256": sha256_file(Path(production.__file__).resolve()),
            "adapter_entrypoint": str(Path(__file__).resolve()),
            "adapter_entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "dataset_configs": str(dataset_configs),
            "dataset_configs_sha256": sha256_file(dataset_configs),
            "population_spec_dir": str(population_spec_dir),
            "unit_map_dir": str(unit_map_dir),
            "sf_tuning_group_csv": str(sf_group_csv),
            "sf_tuning_group_csv_sha256": sha256_file(sf_group_csv),
            "orientation_groups_sha256": sha256_file(
                unit_map_dir / "orientation_tuning_groups.csv"
            ),
            "recovered_python_root": str(recovered_root),
            "selected_twin_temporal_contract": time_contract,
            "retained_schematic_trace_rate_hz": LEGACY_SCHEMATIC_TRACE_RATE_HZ,
            "retained_trace_resampling": (
                "endpoint-anchored linear interpolation with held boundaries"
            ),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "selected_twin_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
