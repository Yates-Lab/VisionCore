import pytest
import yaml

from paper.model_selection.evaluate_real_grating_tuning import DEFAULT_MODEL_SPEC
from paper.model_selection.render_real_grating_tuning_comparison import (
    parse_model_labels,
)


def test_p240c1_is_a_pinned_real_grating_model() -> None:
    spec = yaml.safe_load(DEFAULT_MODEL_SPEC.read_text(encoding="utf-8"))
    assert spec["checkpoint"]["path"].endswith("epoch=03-val_bps_overall=0.6210.ckpt")
    assert spec["training"]["datasets"]["descriptive_all_gratings"]["path"].endswith(
        "multi_240_long_split3_dekel35_allgratings.yaml"
    )


def test_renderer_accepts_one_model_and_rejects_bad_label_lists() -> None:
    assert parse_model_labels(" P240C1 ") == ("P240C1",)
    assert parse_model_labels("M77,P240C1") == ("M77", "P240C1")
    with pytest.raises(ValueError, match="at least one"):
        parse_model_labels(" , ")
    with pytest.raises(ValueError, match="duplicate"):
        parse_model_labels("P240C1,P240C1")
