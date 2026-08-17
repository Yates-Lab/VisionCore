from __future__ import annotations

import inspect
import json

import pytest
import torch

from paper.fig4.mechanism_audit_v1.causal_low_rank import (
    common,
    evaluate_subspaces,
    optimize_subspaces,
)
from paper.fig4.mechanism_audit_v1.causal_low_rank.common import (
    intervention_preactivations,
    normalize_rate,
    projected_delta_preactivation,
    project_channel_delta,
    rate_map_components,
    readout_preactivation,
    softplus_rate,
)


def synthetic_problem(seed: int = 17):
    generator = torch.Generator().manual_seed(seed)
    batch, channels, units, spatial, kernel = 3, 7, 5, 9, 3
    h_a = torch.randn(batch, channels, spatial, spatial, generator=generator)
    h_b = torch.randn(batch, channels, spatial, spatial, generator=generator)
    feature = torch.randn(units, channels, generator=generator)
    bias = torch.randn(units, generator=generator)
    space = torch.randn(units, kernel, kernel, generator=generator)
    return h_a, h_b, feature, bias, space


def test_rank_zero_and_identity_reproduce_exact_intervention_endpoints():
    h_a, h_b, feature, bias, space = synthetic_problem()
    delta = h_b - h_a
    zero = torch.empty(h_a.shape[1], 0)
    identity = torch.eye(h_a.shape[1])

    z_a = readout_preactivation(h_a, feature, bias, space)
    z_b = readout_preactivation(h_b, feature, bias, space)
    for basis, expected_suff, expected_nec in (
        (zero, z_a, z_b),
        (identity, z_b, z_a),
    ):
        projected = project_channel_delta(delta, basis)
        z_suff = readout_preactivation(h_a + projected, feature, bias, space)
        z_nec = readout_preactivation(h_b - projected, feature, bias, space)
        torch.testing.assert_close(z_suff, expected_suff, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(z_nec, expected_nec, atol=2e-5, rtol=2e-5)

        rate_suff = softplus_rate(z_suff)
        rate_nec = softplus_rate(z_nec)
        expected_rate_suff = softplus_rate(expected_suff)
        expected_rate_nec = softplus_rate(expected_nec)
        torch.testing.assert_close(rate_suff, expected_rate_suff, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(rate_nec, expected_rate_nec, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(normalize_rate(rate_suff), normalize_rate(expected_rate_suff))
        torch.testing.assert_close(normalize_rate(rate_nec), normalize_rate(expected_rate_nec))
        torch.testing.assert_close(
            rate_map_components(rate_suff)["ssi"], rate_map_components(expected_rate_suff)["ssi"]
        )
        torch.testing.assert_close(
            rate_map_components(rate_nec)["ssi"], rate_map_components(expected_rate_nec)["ssi"]
        )


def test_fast_projected_readout_delta_matches_literal_projected_state():
    h_a, h_b, feature, bias, space = synthetic_problem(seed=23)
    delta = h_b - h_a
    basis, _ = torch.linalg.qr(torch.randn(h_a.shape[1], 3, generator=torch.Generator().manual_seed(29)))
    literal = (
        readout_preactivation(project_channel_delta(delta, basis), feature, torch.zeros_like(bias), space)
    )
    fast = projected_delta_preactivation(delta, basis, feature, space)
    torch.testing.assert_close(fast, literal, atol=1e-5, rtol=1e-5)

    z_a = readout_preactivation(h_a, feature, bias, space)
    z_b = readout_preactivation(h_b, feature, bias, space)
    literal_suff = readout_preactivation(h_a + project_channel_delta(delta, basis), feature, bias, space)
    literal_nec = readout_preactivation(h_b - project_channel_delta(delta, basis), feature, bias, space)
    torch.testing.assert_close(z_a + fast, literal_suff, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(z_b - fast, literal_nec, atol=1e-5, rtol=1e-5)


def test_canonical_intervention_runs_readout_on_literal_patched_states():
    h_a, h_b, feature, bias, space = synthetic_problem(seed=31)
    basis, _ = torch.linalg.qr(
        torch.randn(h_a.shape[1], 3, generator=torch.Generator().manual_seed(37))
    )
    projected = project_channel_delta(h_b - h_a, basis)
    expected_suff = readout_preactivation(h_a + projected, feature, bias, space)
    expected_nec = readout_preactivation(h_b - projected, feature, bias, space)
    observed_suff, observed_nec = intervention_preactivations(
        h_a, h_b, basis, feature, bias, space
    )
    torch.testing.assert_close(observed_suff, expected_suff, atol=0.0, rtol=0.0)
    torch.testing.assert_close(observed_nec, expected_nec, atol=0.0, rtol=0.0)


def test_optimizer_and_evaluator_do_not_use_split_readout_path():
    for module in (optimize_subspaces, evaluate_subspaces):
        source = inspect.getsource(module)
        assert "intervention_preactivations" in source
        assert "projected_delta_preactivation" not in source


def test_normalization_is_per_sample_and_unit_over_space_only():
    rate = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[10.0, 20.0], [30.0, 40.0]],
            ],
            [
                [[2.0, 2.0], [2.0, 2.0]],
                [[4.0, 8.0], [12.0, 16.0]],
            ],
        ]
    )
    normalized = normalize_rate(rate)
    explicit = rate / rate.mean(dim=(-2, -1), keepdim=True)
    torch.testing.assert_close(normalized, explicit)
    torch.testing.assert_close(normalized.mean(dim=(-2, -1)), torch.ones(2, 2))
    torch.testing.assert_close(rate_map_components(rate)["gain"], normalized)


def test_global_gpu_budget_migrates_and_accumulates_all_gpu_stages(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "cache_generation_manifest.json").write_text(
        json.dumps({"accelerator_forward_hours": 0.25}), encoding="utf-8"
    )
    ledger = tmp_path / "gpu_budget.json"
    ledger.write_text(json.dumps({"optimization_wall_hours": 0.5}), encoding="utf-8")
    monkeypatch.setattr(common, "CACHE", cache)
    monkeypatch.setattr(common, "GLOBAL_GPU_BUDGET", ledger)
    monkeypatch.setattr(common, "ensure_output_dirs", lambda: None)

    migrated = common.load_global_gpu_budget(4.0)
    assert migrated["cache_accelerator_hours"] == pytest.approx(0.25)
    assert migrated["gpu_stage_wall_hours"] == pytest.approx(0.5)
    assert migrated["total_conservative_gpu_hours"] == pytest.approx(0.75)

    updated = common.record_global_gpu_time(
        "evaluation:crossval", 900.0, hard_limit_hours=4.0
    )
    assert updated["gpu_stage_wall_hours"] == pytest.approx(0.75)
    assert updated["total_conservative_gpu_hours"] == pytest.approx(1.0)
    assert updated["events"][-1]["stage"] == "evaluation:crossval"


def test_global_gpu_budget_fails_closed_on_corrupt_provenance(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "cache_generation_manifest.json").write_text(
        json.dumps({"accelerator_forward_hours": 0.0}), encoding="utf-8"
    )
    ledger = tmp_path / "gpu_budget.json"
    ledger.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(common, "CACHE", cache)
    monkeypatch.setattr(common, "GLOBAL_GPU_BUDGET", ledger)
    with pytest.raises(RuntimeError, match="Cannot read global GPU budget ledger"):
        common.load_global_gpu_budget(4.0)
