#!/usr/bin/env python3
"""Run the recovered production unit-map script with a selected digital twin.

The original analysis is retained verbatim in the recovered tree.  This
wrapper replaces only its hard-coded Ryan checkpoint loader with the selected
checkpoint scorer and records enough provenance to make every cache model
specific.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
LEGACY_SCHEMATIC_TRACE_RATE_HZ = 120


def _trace_on_selected_output_grid(
    trace: np.ndarray,
    *,
    output_rate_hz: int,
    torch: Any,
    source_rate_hz: int = LEGACY_SCHEMATIC_TRACE_RATE_HZ,
) -> np.ndarray:
    """Endpoint-interpolate a retained 120-Hz trace for a native-rate twin."""
    from paper.fig4.upstream.real_trace_matrix.model import _trace_on_output_grid

    return _trace_on_output_grid(
        trace,
        source_rate_hz=int(source_rate_hz),
        output_rate_hz=int(output_rate_hz),
        torch=torch,
    )


def _required_path(env_name: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"{env_name} must be set for selected-twin Figure 4 rendering.")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{env_name} does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recovered_python_root() -> Path:
    root = _required_path("FIG4_RECOVERED_ROOT")
    return root.parent if root.name == "declan" else root


def _output_dir_from_argv() -> Path | None:
    for index, value in enumerate(sys.argv[1:]):
        if value == "--out-dir" and index + 2 <= len(sys.argv[1:]):
            return Path(sys.argv[index + 2]).expanduser().resolve()
        if value.startswith("--out-dir="):
            return Path(value.split("=", 1)[1]).expanduser().resolve()
    return None


def main() -> None:
    checkpoint = _required_path("FIG4_TWIN_CHECKPOINT")
    dataset_configs = _required_path("FIG4_DATASET_CONFIGS")
    population_spec_dir = _required_path("FIG4_RR100_POPULATION_SPEC_DIR")
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
    from declan.active_sensing_movie_information import (
        plot_backimage_rr100_instantaneous_unit_maps as production,
    )
    from declan.fig_ssi import make_ssi_contour_schematic as schematic_source

    # The recovered producer resolves ``ROOT`` relative to Ryan's recovery
    # tree.  Its code is the production source, but the retained data products
    # live in this checkout.  Repoint only the two model-independent inputs
    # used to reconstruct the pinned schematic trial; otherwise the recovery
    # tree appears to contain the script while silently returning no stimulus.
    source_image_table = (
        ROOT
        / "outputs"
        / "active_sensing_movie_information"
        / "backimage_real_trace_ssi_matrix_large_contour_no_driftgate_ms200_n100x1000_v1"
        / "merged"
        / "image_feature_table.csv"
    )
    source_trace_csv = (
        ROOT
        / "outputs"
        / "fig_ssi"
        / "trace_provenance"
        / "schematic_crop_real_backimage_trace_center40.csv"
    )
    schematic_source.NEW_BANK_IMAGE_TABLE = source_image_table
    schematic_source.SCHEMATIC_REAL_TRACE_CENTER40_CSV = source_trace_csv
    SCHEMATIC_NEW_BANK_IMAGE_INDEX = schematic_source.SCHEMATIC_NEW_BANK_IMAGE_INDEX
    load_new_bank_stimulus_patch = schematic_source.load_new_bank_stimulus_patch

    checkpoint_sha = sha256_file(checkpoint)
    dataset_sha = sha256_file(dataset_configs)
    time_contract = _declared_time_contract(dataset_configs)
    adapter_cache: dict[str, LegacyCanonicalTwinScorerAdapter] = {}

    recovered_load_population_view = production.load_population_view

    def selected_load_population_view(spec_dir=None, *, version_name=None):
        return recovered_load_population_view(
            population_spec_dir if spec_dir is None else spec_dir,
            version_name=version_name,
        )

    production.load_population_view = selected_load_population_view

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
            cached = adapter_cache.get(key)
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
            adapter_cache[key] = self

    # Cache identities in the recovered script predate model-selectable
    # rendering.  Enrich every identity before comparison or serialization so
    # a cache from another checkpoint can never be reused silently.
    original_identity_text = production.identity_text

    def selected_identity_text(identity: dict[str, Any]) -> str:
        enriched = dict(identity)
        enriched.update(
            {
                "selected_twin_checkpoint": str(checkpoint),
                "selected_twin_checkpoint_sha256": checkpoint_sha,
                "selected_twin_dataset_configs": str(dataset_configs),
                "selected_twin_dataset_configs_sha256": dataset_sha,
                "selected_twin_temporal_contract": time_contract,
                "retained_schematic_trace_rate_hz": LEGACY_SCHEMATIC_TRACE_RATE_HZ,
            }
        )
        return original_identity_text(enriched)

    production.CanonicalTwinScorer = SelectedTwinScorer
    production.rate_map_for_trace = lambda scorer, patch, trace: scorer.rate_map_for_trace(
        patch,
        _trace_on_selected_output_grid(
            trace,
            output_rate_hz=scorer._scorer.output_rate_hz,
            torch=scorer.torch,
        ),
    )
    production.identity_text = selected_identity_text

    # The recovered handoff retained the exact model-independent schematic
    # stimulus, but not the earlier axis-screen cache used only to select it.
    # Reconstruct the one-row production trial from that retained source.  This
    # preserves the paper's source row, natural image patch, 40-sample trace,
    # and contour axis while avoiding dependence on a missing old-model cache.
    schematic = load_new_bank_stimulus_patch(SCHEMATIC_NEW_BANK_IMAGE_INDEX)
    if schematic is None:
        raise RuntimeError("Could not reconstruct the retained Figure 4 schematic source.")
    selected_row = dict(schematic["row"])
    selected_row["trial_id"] = int(selected_row.get("source_row", schematic["source_row"]))
    selected_row["source_trace"] = schematic["real_trace_center40"]

    def selected_source_trials(_args: Any):
        import pandas as pd

        return (
            pd.DataFrame([selected_row]),
            {
                "selection_source": "retained_schematic_new_bank_stimulus",
                "source_row": int(schematic["source_row"]),
                "image_index": int(schematic["image_index"]),
            },
        )

    def selected_extract_patch(*_args: Any, **_kwargs: Any):
        return (
            schematic["model_source_patch"],
            {
                "selection_source": "retained_schematic_new_bank_stimulus",
                "source_row": int(schematic["source_row"]),
                "image_index": int(schematic["image_index"]),
            },
        )

    def selected_load_npz(path: Path):
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                return {key: np.asarray(data[key]) for key in data.files}
        return {"movie_trial_id": np.asarray([selected_row["trial_id"]], dtype=np.int64)}

    production.select_source_trials = selected_source_trials
    production._extract_patch = selected_extract_patch
    production.load_npz = selected_load_npz
    production.main()

    out_dir = _output_dir_from_argv()
    if out_dir is not None:
        provenance = {
            "analysis_entrypoint": str(Path(production.__file__).resolve()),
            "analysis_entrypoint_sha256": _sha256(Path(production.__file__).resolve()),
            "adapter_entrypoint": str(Path(__file__).resolve()),
            "adapter_entrypoint_sha256": _sha256(Path(__file__).resolve()),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "dataset_configs": str(dataset_configs),
            "dataset_configs_sha256": dataset_sha,
            "population_spec_dir": str(population_spec_dir),
            "mcfarland_outputs": str(mcfarland_outputs) if mcfarland_outputs else None,
            "mcfarland_outputs_sha256": (
                _sha256(mcfarland_outputs) if mcfarland_outputs is not None else None
            ),
            "schematic_image_table": str(source_image_table),
            "schematic_image_table_sha256": _sha256(source_image_table),
            "schematic_trace_csv": str(source_trace_csv),
            "schematic_trace_csv_sha256": _sha256(source_trace_csv),
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
