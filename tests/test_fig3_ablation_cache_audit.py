import numpy as np
import pytest

from paper.fig3.audit_ablation_cache import audit_ablation_payload


def _fixtures(schema=7):
    ccmax = np.asarray([0.5, 0.25, 0.0])
    unstable = np.asarray([False, False, True])
    ccabs = {
        "intact": np.asarray([0.25, 0.10, 0.0]),
        "zeroed": np.asarray([0.20, 0.05, 0.0]),
        "stabilized": np.asarray([0.15, 0.025, 0.0]),
    }
    ccnorm = {}
    for condition, values in ccabs.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = values / ccmax
        normalized[unstable] = np.nan
        ccnorm[condition] = normalized
    fem = {
        condition: {
            "B_obs": np.asarray([0.2, 0.3, 0.4]),
            "B_obs_uncl": np.asarray([0.2, 0.3, 0.4]),
            "B_model": np.asarray([0.1, 0.2, 0.3]),
            "B_model_uncl": np.asarray([0.1, 0.2, 0.3]),
        }
        for condition in ccabs
    }
    result = {
        "session": "s0",
        "neuron_mask": np.asarray([3, 5, 8]),
        "ccmax": ccmax,
        "ccnorm_unstable": unstable,
        "ccabs": ccabs,
        "ccnorm": ccnorm,
    }
    if schema >= 7:
        result["femfraction"] = fem
    payload = {
        "schema_version": schema,
        "femfraction_count_bins": 3 if schema >= 7 else None,
        "results": [result],
    }
    intact = [
        {
            "session": "s0",
            "neuron_mask": result["neuron_mask"].copy(),
            "ccmax": ccmax.copy(),
            "ccnorm_unstable": unstable.copy(),
            "ccabs": ccabs["intact"].copy(),
            "ccnorm": ccnorm["intact"].copy(),
        }
    ]
    return payload, intact


def test_ablation_audit_proves_shared_anchor_identity_and_fem_invariance():
    report = audit_ablation_payload(*_fixtures())
    assert report["shared_anchor_exact"] is True
    assert report["ccnorm_identity_max_abs_error"] == 0.0
    assert report["fem_observed_exact_across_conditions"] is True
    assert report["n_units"] == 3


def test_ablation_audit_derives_stability_mask_from_legacy_intact_ccnorm():
    payload, intact = _fixtures()
    intact[0].pop("ccnorm_unstable")
    report = audit_ablation_payload(payload, intact)
    assert report["shared_anchor_exact"] is True
    assert report["sessions"][0]["stability_mask_exact"] is True


def test_ablation_audit_rejects_wrong_mask_against_legacy_intact_ccnorm():
    payload, intact = _fixtures()
    intact[0].pop("ccnorm_unstable")
    payload["results"][0]["ccnorm_unstable"][0] = True
    with pytest.raises(AssertionError, match="exact intact metric anchor"):
        audit_ablation_payload(payload, intact)


def test_ablation_audit_rejects_condition_specific_noise_ceiling_identity():
    payload, intact = _fixtures()
    payload["results"][0]["ccnorm"]["zeroed"][0] += 0.01
    with pytest.raises(AssertionError, match="CCnorm identity failed"):
        audit_ablation_payload(payload, intact)


def test_ablation_audit_rejects_changed_data_only_fem_term():
    payload, intact = _fixtures()
    payload["results"][0]["femfraction"]["zeroed"]["B_obs"][0] += 0.01
    with pytest.raises(AssertionError, match="data-only FEM"):
        audit_ablation_payload(payload, intact)


def test_ablation_audit_can_screen_legacy_cache_without_claiming_fem():
    payload, intact = _fixtures(schema=6)
    report = audit_ablation_payload(payload, intact, require_femfraction=False)
    assert report["fem_observed_exact_across_conditions"] is None


def test_ablation_audit_rejects_duplicate_sessions():
    payload, intact = _fixtures()
    payload["results"].append(payload["results"][0].copy())
    with pytest.raises(ValueError, match="duplicate sessions"):
        audit_ablation_payload(payload, intact)


def test_ablation_audit_rejects_metric_shape_mismatch():
    payload, intact = _fixtures()
    payload["results"][0]["ccabs"]["zeroed"] = np.asarray([0.2, 0.05])
    with pytest.raises(ValueError, match="CC metric shapes"):
        audit_ablation_payload(payload, intact)


def test_ablation_audit_rejects_wrong_fem_counting_window():
    payload, intact = _fixtures()
    payload["femfraction_count_bins"] = 1
    with pytest.raises(ValueError, match="counting window"):
        audit_ablation_payload(payload, intact)


def test_ablation_audit_rejects_resumable_partial_cache():
    payload, intact = _fixtures()
    payload["complete"] = False
    with pytest.raises(ValueError, match="resumable partial"):
        audit_ablation_payload(payload, intact)
