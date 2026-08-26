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
