"""A behavior-conditioned twin must not silently skip its modulator.

TWIN_IMPROVEMENTS item 2. `core_forward` used to run the modulator only when
`behavior is not None`. For a `concat` twin the recurrent stack is built for
`convnet_channels + modulator_dim` channels, so passing `behavior=None` fed it
the wrong channel count -- a crash, or silently wrong features. The 42-dim
behavior of the current twin is entirely eye-movement derived, so zeroing it is
a real ablation, never a neutral default.
"""
import pytest
import torch
import torch.nn as nn

from models.modules.models import require_behavior


class _Mod(nn.Module):
    def __init__(self, behavior_dim=42):
        super().__init__()
        self.behavior_dim = behavior_dim


def test_no_modulator_tolerates_absent_behavior():
    require_behavior(None, None, where="core_forward")


def test_no_modulator_tolerates_present_behavior():
    require_behavior(None, torch.zeros(2, 42), where="core_forward")


def test_modulator_with_behavior_passes():
    require_behavior(_Mod(), torch.zeros(2, 42), where="core_forward")


def test_modulator_without_behavior_raises():
    with pytest.raises(ValueError):
        require_behavior(_Mod(), None, where="core_forward")


def test_the_error_names_the_dim_to_pass_and_calls_zeroing_an_ablation():
    """The fix must be obvious from the message, and must not read as though
    zeros were a neutral default."""
    with pytest.raises(ValueError) as exc:
        require_behavior(_Mod(behavior_dim=42), None, where="core_forward")

    msg = str(exc.value)
    assert "42" in msg
    assert "core_forward" in msg
    assert "ablation" in msg.lower()


def test_the_error_survives_a_modulator_with_no_declared_dim():
    class _Bare(nn.Module):
        pass

    with pytest.raises(ValueError):
        require_behavior(_Bare(), None, where="core_forward")
