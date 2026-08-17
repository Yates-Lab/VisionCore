from __future__ import annotations

import torch

from paper.supp_model_replication._supp_inference import _rescale_affine_safely


def test_safe_affine_rescale_isolates_one_degenerate_neuron():
    robs = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [2.0, 3.0, 0.0]]
    )
    rhat = torch.tensor(
        [[1.0, 1.0, float("nan")], [2.0, 2.0, float("nan")], [3.0, 3.0, float("nan")]]
    )
    dfs = torch.tensor(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )

    def synthetic_production_rescale(y, x, d, *, mode):
        assert mode == "affine"
        # Mimic a joint LBFGS failure caused by the degenerate final column.
        if x.shape[1] > 1 or not torch.isfinite(x).all():
            raise FloatingPointError("synthetic joint calibration failure")
        return x + 0.5, object()

    calibrated, fallback_columns = _rescale_affine_safely(
        torch, synthetic_production_rescale, robs, rhat, dfs
    )

    assert fallback_columns == [2]
    assert torch.allclose(calibrated[:, :2], rhat[:, :2] + 0.5)
    assert torch.isfinite(calibrated).all()
    assert torch.all(calibrated[:, 2] > 0)
