from __future__ import annotations

import numpy as np
from pypdf import PdfReader
import pytest

from paper.model_selection.render_m77_fixrsvp_session_atlases import (
    render_session,
    validate_pair,
)


def synthetic_pair() -> tuple[dict, dict]:
    time = np.arange(4, dtype=float)[:, None]
    units = np.arange(3, dtype=float)[None, :]
    data = 0.02 + 0.01 * time + 0.005 * units
    twin_prediction = data * 0.9 + 0.002
    m77_prediction = data * 0.95 + 0.001
    native_data = np.repeat(data / 2.0, 2, axis=0)
    native_prediction = np.repeat(m77_prediction / 2.0, 2, axis=0)
    common = {
        "session": "Allen_2022-04-13",
        "neuron_mask": np.asarray([4, 9, 15]),
        "robs_mean": data,
        "ccnorm": np.asarray([0.5, 0.6, 0.7]),
        "ve_model": np.asarray([0.01, 0.02, 0.03]),
    }
    m77 = {
        **common,
        "rhat_mean": m77_prediction,
        "native240": {
            "robs_mean": native_data,
            "rhat_mean": native_prediction,
        },
    }
    twin = {**common, "rhat_mean": twin_prediction}
    return m77, twin


def test_session_atlases_render_all_units_and_label_native_caveat(tmp_path) -> None:
    m77, twin = synthetic_pair()
    output120 = tmp_path / "session_120.pdf"
    output240 = tmp_path / "session_240.pdf"
    pages120 = render_session(
        m77,
        twin,
        out_path=output120,
        rate=120,
        cells_per_page=2,
        columns=2,
    )
    pages240 = render_session(
        m77,
        twin,
        out_path=output240,
        rate=240,
        cells_per_page=2,
        columns=2,
    )
    assert pages120 == pages240 == 2
    assert len(PdfReader(output120).pages) == 2
    reader240 = PdfReader(output240)
    assert len(reader240.pages) == 2
    text120 = "\n".join(page.extract_text() or "" for page in PdfReader(output120).pages)
    text240 = "\n".join(page.extract_text() or "" for page in reader240.pages)
    assert "Twin" in text120 and "M77" in text120
    assert "unit 4" in text120 and "unit 15" in text120
    assert "not a native-240 estimate" in text240


def test_session_atlas_rejects_neuron_order_mismatch() -> None:
    m77, twin = synthetic_pair()
    twin["neuron_mask"] = np.asarray([4, 15, 9])
    with pytest.raises(ValueError, match="neuron order"):
        validate_pair(m77, twin)


def test_session_atlas_accepts_float32_cache_roundoff_but_not_changed_data() -> None:
    m77, twin = synthetic_pair()
    twin["robs_mean"] = np.asarray(twin["robs_mean"], dtype=np.float32)
    validate_pair(m77, twin)

    twin["robs_mean"] = np.asarray(twin["robs_mean"], dtype=float)
    twin["robs_mean"][1, 1] += 1e-4
    with pytest.raises(AssertionError, match="cached 120-Hz observations differ"):
        validate_pair(m77, twin)
