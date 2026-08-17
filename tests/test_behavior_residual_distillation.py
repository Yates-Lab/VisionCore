import torch

from paper.model_selection.distill_ryan_behavior_residual import (
    improves_behavior_residual,
    inverse_softplus,
    poisson_cross_entropy,
    residual_metrics,
)


def test_distillation_checkpoint_selection_uses_matched_improvement():
    best = {"adapted_nll": 0.20, "nll_improvement": 0.001}
    # A lower absolute NLL on an easier panel must not win if the adapter made
    # less improvement over that panel's identity baseline.
    easier_but_worse_residual = {
        "adapted_nll": 0.10,
        "nll_improvement": 0.0005,
    }
    better_residual = {
        "adapted_nll": 0.30,
        "nll_improvement": 0.002,
    }
    assert not improves_behavior_residual(easier_but_worse_residual, best)
    assert improves_behavior_residual(better_residual, best)


def test_inverse_softplus_roundtrip():
    rate = torch.logspace(-5, 2, 100)
    recovered = torch.nn.functional.softplus(inverse_softplus(rate))
    assert torch.allclose(recovered, rate, rtol=1e-5, atol=1e-7)


def test_poisson_cross_entropy_prefers_matching_rate():
    target = torch.tensor([[0.05, 0.2, 1.0, 3.0]])
    matched = poisson_cross_entropy(target, target)
    perturbed = poisson_cross_entropy(target * 1.7, target)
    assert matched < perturbed


def test_residual_metrics_identify_exact_adapter():
    zero = torch.tensor([[0.1, 0.5, 2.0], [0.2, 0.8, 1.2]])
    intact = torch.tensor([[0.2, 0.4, 2.5], [0.3, 1.1, 0.9]])
    metrics = residual_metrics(zero, intact, intact)
    assert metrics["adapted_nll"] < metrics["identity_nll"]
    assert metrics["residual_logit_r2"] > 0.999999
    assert metrics["residual_logit_correlation"] > 0.999999
