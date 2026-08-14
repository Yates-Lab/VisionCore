"""Frozen definitions shared by every run in the model-selection family.

This module exists to stop silent drift. The sweep runs over days or weeks,
and the failure mode is comparing a model trained under one split against a
model trained under another, or reporting a metric computed two different
ways. Every run manifest records PROTOCOL_HASH; `collect.py` refuses to pool
runs whose hashes disagree.

Change anything here and the hash changes, which invalidates every prior run
by design. That is the point. Do not add convenience knobs.
"""
from __future__ import annotations

import hashlib
import json
from typing import Final


# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------
# Three-way split by trial. The paper model was trained under a two-way 80/20
# train/val split, which meant checkpoint selection and reported performance
# drew on the same trials. That is tolerable for a single model but not for a
# family: selecting the best of N models on validation and then reporting
# validation numbers is optimistically biased in N. The test split is touched
# exactly once, to report the selected model.
TRAIN_SPLIT: Final = 0.70
VAL_SPLIT: Final = 0.15
TEST_SPLIT: Final = 0.15
SPLIT_SEED: Final = 1002  # matches the seed used for the paper model

assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Stimulus conditions
# ---------------------------------------------------------------------------
# The twin is fit only to free-viewing conditions. The fixated flashed-image
# condition ("fixrsvp") is withheld from fitting entirely and is the condition
# every figure evaluates on, which makes those evaluations out-of-domain
# generalization rather than within-distribution test scores. This asymmetry
# is a deliberate design property of the study, not an artifact of how
# `load_single_dataset` happens to append evaluation types.
TRAIN_TYPES: Final = ("backimage", "gaborium", "gratings")
HELD_OUT_TYPES: Final = ("fixrsvp",)

assert not set(TRAIN_TYPES) & set(HELD_OUT_TYPES)


# ---------------------------------------------------------------------------
# Unit and session inclusion
# ---------------------------------------------------------------------------
# Matched to the figure 2 / figure 3 population so that model-selection
# numbers are computed over the same neurons the paper reports on.
MIN_TOTAL_SPIKES: Final = 200      # fig3 neuron inclusion
MIN_PSTH_R2: Final = 0.10          # fig2/fig3 reliability floor
MIN_SESSION_UNITS: Final = 10      # drop sessions below this after filtering


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
DT: Final = 1 / 120                # seconds per bin; the twin's native rate
CCNORM_N_SPLITS: Final = 500       # split-half iterations, matches fig3
VALID_TIME_BINS: Final = 120
MIN_FIX_DUR: Final = 20            # bins

# Metrics reported for every run. BPS is the training objective expressed per
# spike and is the selection criterion; CC_norm is the cross-paper comparable
# number; single-trial r^2 is what figures 3 and 4 actually depend on.
METRICS: Final = ("bps", "ccnorm", "single_trial_r2")
SELECTION_METRIC: Final = "bps"
SELECTION_SPLIT: Final = "val"     # never "test"


# ---------------------------------------------------------------------------
# Protocol identity
# ---------------------------------------------------------------------------
def protocol_dict() -> dict:
    """The full protocol as a plain dict, for hashing and for manifests."""
    return {
        "train_split": TRAIN_SPLIT,
        "val_split": VAL_SPLIT,
        "test_split": TEST_SPLIT,
        "split_seed": SPLIT_SEED,
        "train_types": list(TRAIN_TYPES),
        "held_out_types": list(HELD_OUT_TYPES),
        "min_total_spikes": MIN_TOTAL_SPIKES,
        "min_psth_r2": MIN_PSTH_R2,
        "min_session_units": MIN_SESSION_UNITS,
        "dt": DT,
        "ccnorm_n_splits": CCNORM_N_SPLITS,
        "valid_time_bins": VALID_TIME_BINS,
        "min_fix_dur": MIN_FIX_DUR,
        "metrics": list(METRICS),
        "selection_metric": SELECTION_METRIC,
        "selection_split": SELECTION_SPLIT,
    }


def protocol_hash() -> str:
    """Short stable hash of the protocol. Stamped into every run manifest."""
    payload = json.dumps(protocol_dict(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


PROTOCOL_HASH: Final = protocol_hash()


def assert_same_protocol(manifest: dict, source: str = "run") -> None:
    """Raise if a run manifest was produced under a different protocol."""
    found = manifest.get("protocol_hash")
    if found != PROTOCOL_HASH:
        raise ValueError(
            f"{source} was produced under protocol {found!r} but the current "
            f"protocol is {PROTOCOL_HASH!r}. These runs are not comparable. "
            f"Either re-run under the current protocol or check out the "
            f"revision of protocol.py that produced {found!r}."
        )


if __name__ == "__main__":
    print(json.dumps(protocol_dict(), indent=2))
    print(f"\nPROTOCOL_HASH = {PROTOCOL_HASH}")
