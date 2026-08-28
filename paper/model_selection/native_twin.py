"""Small identity-preserving helpers shared by native-rate model assays."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TwinEncodingModel:
    model: object
    device: str

    def zero_behavior(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor | None:
        modulator = getattr(self.model.model, "modulator", None)
        behavior_dim = (
            getattr(modulator, "behavior_dim", None) if modulator is not None else None
        )
        if behavior_dim is None:
            return None
        return torch.zeros(
            int(batch_size), int(behavior_dim), device=self.device, dtype=dtype
        )


def canonical_population_rows(model, outputs: list[dict]) -> list[dict]:
    """Map evaluation rows to exact checkpoint-native ``(session, cid)`` IDs."""
    output_by_session = {str(value["sess"]): value for value in outputs}
    rows: list[dict] = []
    for model_index, session in enumerate(model.names):
        if session not in output_by_session:
            continue
        output = output_by_session[session]
        scores = np.asarray(output["ccnorm"]["ccnorm"])
        source_cids = None
        for key in ("cids_used", "cids"):
            candidate = np.asarray(output.get(key, []))
            if candidate.ndim == 1 and candidate.size == scores.size:
                source_cids = candidate.astype(int, copy=False)
                break
        if source_cids is None:
            raise ValueError(
                f"McFarland output {session!r} has {scores.size} CCnorm rows "
                "but no equally sized cids_used/cids array"
            )
        for source_position in np.flatnonzero(scores > 0.5):
            rows.append(
                {
                    "session": str(session),
                    "model_readout_index": int(model_index),
                    "historical_source_position": int(source_position),
                    "source_cid": int(source_cids[source_position]),
                    "ccnorm": float(scores[source_position]),
                }
            )
    return rows


def exact_unit_rows(model, canonical_rows: list[dict]) -> list[dict]:
    """Attach the checkpoint-native readout row to each biological identity.

    Historical evaluation arrays are indexed by their own source position;
    model heads are indexed by the ordered ``cids`` stored in the dataset
    configuration.  Those coordinates are not interchangeable.  This is the
    single authoritative conversion between them.
    """
    rows: list[dict] = []
    for channel, source in enumerate(canonical_rows):
        dataset_index = int(source["model_readout_index"])
        configured = np.asarray(
            model.model.dataset_configs[dataset_index].get("cids", []),
            dtype=int,
        )
        if configured.ndim != 1 or len(np.unique(configured)) != len(configured):
            raise ValueError(
                f"configured CIDs for {source['session']!r} are not unique "
                "and one-dimensional"
            )
        matches = np.flatnonzero(configured == int(source["source_cid"]))
        row = dict(source)
        row["channel"] = int(channel)
        row["available"] = bool(len(matches) == 1)
        row["model_readout_row"] = int(matches[0]) if len(matches) == 1 else None
        rows.append(row)
    return rows


def audit_native_cid_mapping(model, unit_rows: list[dict]) -> dict:
    """Reject duplicate identities and any failed CID-to-head round trip."""
    identities = [(str(row["session"]), int(row["source_cid"])) for row in unit_rows]
    if len(set(identities)) != len(identities):
        raise RuntimeError("canonical biological (session, cid) identities are not unique")
    checked = 0
    for row in unit_rows:
        dataset_index = int(row["model_readout_index"])
        if str(model.names[dataset_index]) != str(row["session"]):
            raise RuntimeError("canonical session does not match the native model readout")
        if not bool(row["available"]):
            continue
        native = model.model.readouts[dataset_index]
        model_row = int(row["model_readout_row"])
        if model_row < 0 or model_row >= int(native.n_units):
            raise RuntimeError("canonical CID maps outside the native readout")
        configured_cid = int(
            model.model.dataset_configs[dataset_index]["cids"][model_row]
        )
        if configured_cid != int(row["source_cid"]):
            raise RuntimeError("native readout row does not map back to the requested CID")
        checked += 1
    if checked == 0:
        raise RuntimeError("no canonical biological CIDs are available in this checkpoint")
    return {
        "n_canonical_identities": int(len(unit_rows)),
        "n_available_native_readouts_checked": int(checked),
        "n_unavailable": int(len(unit_rows) - checked),
        "identity_key": "(session, cid)",
        "readout_path": "checkpoint-native per-session readout; no reconstructed population head",
        "passed": True,
    }
