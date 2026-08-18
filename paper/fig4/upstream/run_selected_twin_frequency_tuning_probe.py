#!/usr/bin/env python3
"""Run the recovered RR100 SF/TF probe on a selected twin's native grid."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]


def _output_phase_for_prefixed_movie(
    *,
    native_history_frames: int,
    temporal_factor: int,
    supervision_phase: int,
) -> int:
    """Map source-grid phase to the lagged movie's valid-output index."""
    prefix_endpoint_phase = (int(native_history_frames) - 1) % int(temporal_factor)
    return (
        int(supervision_phase) - prefix_endpoint_phase
    ) % int(temporal_factor)


def _declared_time_contract(dataset_configs: Path) -> dict[str, int]:
    """Read the selected checkpoint's temporal grid from its dataset YAML."""
    config = yaml.safe_load(Path(dataset_configs).read_text(encoding="utf-8")) or {}
    sampling = config.get("sampling") or {}
    supervision = config.get("supervision") or {}
    input_rate = int(sampling.get("target_rate", sampling.get("source_rate", 120)))
    output_rate = int(supervision.get("target_rate", input_rate))
    if input_rate < output_rate or input_rate % output_rate:
        raise ValueError(
            f"Non-integral selected-twin time contract: {input_rate} -> {output_rate} Hz"
        )
    temporal_factor = input_rate // output_rate
    supervision_phase = (
        int(supervision.get("phase", temporal_factor - 1))
        if temporal_factor > 1
        else 0
    )
    if not 0 <= supervision_phase < temporal_factor:
        raise ValueError(
            f"Invalid supervision phase {supervision_phase} for factor "
            f"{temporal_factor}"
        )
    stim_lags = (config.get("keys_lags") or {}).get("stim")
    native_history_frames = (
        len(stim_lags) if isinstance(stim_lags, list) and stim_lags else 32
    )
    return {
        "input_rate_hz": input_rate,
        "output_rate_hz": output_rate,
        "temporal_factor": temporal_factor,
        "supervision_phase": supervision_phase,
        "native_history_frames": int(native_history_frames),
    }


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
    recovered_root = _recovered_python_root()
    mcfarland_value = os.environ.get("FIG4_MCFARLAND_OUTPUTS")
    mcfarland_outputs = (
        Path(mcfarland_value).expanduser().resolve() if mcfarland_value else None
    )
    time_contract = _declared_time_contract(dataset_configs)

    for path in (ROOT, recovered_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from paper.fig4.upstream.real_trace_matrix.core import sha256_file
    from paper.fig4.upstream.real_trace_matrix.model import (
        LegacyCanonicalTwinScorerAdapter,
        RealTraceMatrixScorer,
    )
    from declan.active_sensing_movie_information import (
        run_backimage_rr100_frequency_tuning_probe as production,
    )

    adapters: dict[str, LegacyCanonicalTwinScorerAdapter] = {}
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
    original_grating_movie = production.make_windowed_drifting_grating_movie

    def native_grating_movie(**kwargs):
        scored_rate_hz = float(kwargs["frame_rate_hz"])
        scored_valid_frames = int(kwargs["n_valid_frames"])
        duration_s = float(scored_valid_frames) / scored_rate_hz
        kwargs["frame_rate_hz"] = float(time_contract["input_rate_hz"])
        kwargs["n_lags"] = int(time_contract["native_history_frames"])
        kwargs["n_valid_frames"] = int(
            round(duration_s * float(time_contract["input_rate_hz"]))
        )
        return original_grating_movie(**kwargs)

    def selected_twin_movie_maps(scorer, view, movie_uint, *, n_lags):
        movie = (np.asarray(movie_uint, dtype=np.float32) - 127.0) / 255.0
        native_lags = int(scorer._scorer.n_lags)
        stim = production.embed_time_lags_local(
            production.torch.from_numpy(movie),
            n_lags=native_lags,
        )
        full_map = scorer._compute_rate_map_batched(stim)
        full_np = full_map.detach().cpu().numpy().astype(np.float32, copy=False)
        rr100 = production.apply_population_view(full_np, view).astype(
            np.float32, copy=False
        )
        factor = int(scorer._scorer.temporal_factor)
        output_phase = _output_phase_for_prefixed_movie(
            native_history_frames=native_lags,
            temporal_factor=factor,
            supervision_phase=int(scorer._scorer.supervision_phase),
        )
        rr100 = rr100[output_phase::factor]
        del stim, full_map, full_np
        if scorer.device.startswith("cuda") and scorer.torch.cuda.is_available():
            scorer.torch.cuda.empty_cache()
        return rr100

    production.make_windowed_drifting_grating_movie = native_grating_movie
    production.compute_rr100_movie_maps = selected_twin_movie_maps
    original_identity_text = production.identity_text
    checkpoint_sha = sha256_file(checkpoint)
    dataset_sha = sha256_file(dataset_configs)

    def selected_identity_text(identity):
        enriched = dict(identity)
        enriched.update(
            {
                "selected_twin_checkpoint": str(checkpoint),
                "selected_twin_checkpoint_sha256": checkpoint_sha,
                "selected_twin_dataset_configs": str(dataset_configs),
                "selected_twin_dataset_configs_sha256": dataset_sha,
                "native_stimulus_rate_hz": time_contract["input_rate_hz"],
                "native_history_frames": time_contract["native_history_frames"],
                "scored_response_rate_hz": time_contract["output_rate_hz"],
                "native_frames_per_scored_sample": time_contract["temporal_factor"],
                "supervision_phase": time_contract["supervision_phase"],
            }
        )
        return original_identity_text(enriched)

    production.identity_text = selected_identity_text
    production.main()

    out_dir = _out_dir()
    if out_dir is not None:
        provenance = {
            "analysis_entrypoint": str(Path(production.__file__).resolve()),
            "analysis_entrypoint_sha256": sha256_file(Path(production.__file__).resolve()),
            "adapter_entrypoint": str(Path(__file__).resolve()),
            "adapter_entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "dataset_configs": str(dataset_configs),
            "dataset_configs_sha256": dataset_sha,
            "population_spec_dir": str(population_spec_dir),
            "native_stimulus_rate_hz": time_contract["input_rate_hz"],
            "native_history_frames": time_contract["native_history_frames"],
            "scored_response_rate_hz": time_contract["output_rate_hz"],
            "native_frames_per_scored_sample": time_contract["temporal_factor"],
            "supervision_phase": time_contract["supervision_phase"],
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "selected_twin_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
