"""Tests for the experiment/run naming layer.

The property under test is that a label is *derived*, never declared: adding an
arm that varies a new knob has to widen the labels on its own, because the
alternative -- a hand-maintained axis list -- is a second place to state a fact
the run definitions already fix, and it drifts.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "paper" / "model_selection"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from experiments import (  # noqa: E402
    axes_for, format_value, label_map, run_label,
)


def spec(**over):
    base = dict(lr=1e-3, batch_size=256, effective_batch=256, accumulate=1,
                width=0.5, wd=1e-5, core_lr_scale=1.0, max_epochs=61,
                homogeneous=True, adapter=False, seed=201, name="x",
                note="", experiment="test")
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# format_value
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    (1e-3, "1e-3"),
    (3e-3, "3e-3"),
    (1e-2, "1e-2"),      # exactly at the threshold; the naive `< 1e-2` gave "0.01"
    (3e-4, "3e-4"),
    (1e-5, "1e-5"),
    (0.3, "0.3"),
    (1.0, "1"),
    (0.5, "0.5"),
    (2.0, "2"),
])
def test_numbers_render_the_way_they_are_read(value, expected):
    assert format_value("lr", value) == expected


def test_flags_render_as_states_not_booleans():
    assert format_value("homogeneous", True) == "homog"
    assert format_value("homogeneous", False) == "xsess"
    assert format_value("adapter", False) == "noadapter"


def test_paths_render_as_their_stem():
    assert format_value("model_config", "a/b/width0p5_noadapter.yaml") == \
        "width0p5_noadapter"


# ---------------------------------------------------------------------------
# axes_for
# ---------------------------------------------------------------------------
def test_constant_knobs_are_not_axes():
    specs = [spec(lr=1e-3), spec(lr=3e-3)]
    assert axes_for(specs) == ("lr",)


def test_derived_knobs_drop_out_in_favour_of_what_was_set():
    """Experiment 02's shape: batch size varies, three fields follow it."""
    specs = [spec(lr=3e-3, batch_size=256, effective_batch=256, max_epochs=61),
             spec(lr=3e-3, batch_size=128, effective_batch=128, max_epochs=122)]
    assert axes_for(specs) == ("batch_size",)


def test_effective_batch_survives_when_it_is_the_knob_being_set():
    """Experiment 01's shape: micro-batch fixed, accumulation varies.

    The mirror of the test above, and the reason the rule is a redundancy test
    rather than a list of derived field names.
    """
    specs = [spec(batch_size=256, effective_batch=256, accumulate=1),
             spec(batch_size=256, effective_batch=1024, accumulate=4),
             spec(batch_size=256, effective_batch=4096, accumulate=16)]
    assert axes_for(specs) == ("effective_batch",)


def test_two_independent_knobs_both_survive():
    specs = [spec(lr=3e-3, batch_size=256, effective_batch=256, max_epochs=61),
             spec(lr=1e-2, batch_size=256, effective_batch=256, max_epochs=61),
             spec(lr=3e-3, batch_size=128, effective_batch=128, max_epochs=122),
             spec(lr=1e-2, batch_size=128, effective_batch=128, max_epochs=122)]
    assert axes_for(specs) == ("lr", "batch_size")


def test_seed_is_never_an_axis():
    """Replicates are the same arm, so they must share a label."""
    specs = [spec(seed=201), spec(seed=202), spec(seed=203)]
    assert axes_for(specs) == ()
    assert {run_label(s, axes_for(specs)) for s in specs} == {"base"}


def test_a_new_knob_widens_the_labels_without_being_declared():
    two = [spec(lr=3e-3), spec(lr=1e-2)]
    assert axes_for(two) == ("lr",)
    three = two + [spec(lr=3e-3, wd=1e-4)]
    assert axes_for(three) == ("lr", "wd")


# ---------------------------------------------------------------------------
# run_label
# ---------------------------------------------------------------------------
def test_label_is_complete_over_axes_not_over_differences():
    """Every axis appears at its own value, including the baseline's.

    The point of the rule: `02_lr3e-3_bs256` states its batch even though 256
    is the baseline, so the label keeps its meaning if the baseline moves.
    """
    specs = [spec(lr=3e-3, batch_size=256, effective_batch=256, max_epochs=61),
             spec(lr=3e-3, batch_size=128, effective_batch=128, max_epochs=122)]
    axes = axes_for(specs)
    assert run_label(specs[0], axes) == "bs256"
    assert run_label(specs[1], axes) == "bs128"


def test_axis_order_is_stable_across_runs():
    """Axes render in AXIS_ABBREV order, not in whatever order they varied."""
    specs = [spec(lr=3e-3, batch_size=128, effective_batch=128, max_epochs=122),
             spec(lr=1e-2, batch_size=256, effective_batch=256, max_epochs=61),
             spec(lr=3e-3, batch_size=256, effective_batch=256, max_epochs=61),
             spec(lr=1e-2, batch_size=128, effective_batch=128, max_epochs=122)]
    axes = axes_for(specs)
    assert [run_label(s, axes) for s in specs] == [
        "lr3e-3_bs128", "lr1e-2_bs256", "lr3e-3_bs256", "lr1e-2_bs128"]


def test_a_fully_confounded_design_collapses_to_one_axis():
    """Two knobs that only ever move together cannot both be axes.

    Not a defect: with these two runs alone, nothing distinguishes "lr 1e-2"
    from "batch 256", and a label claiming to name both would imply a
    separation the design does not support. The full 2x2 above restores both.
    """
    specs = [spec(lr=3e-3, batch_size=128, effective_batch=128, max_epochs=122),
             spec(lr=1e-2, batch_size=256, effective_batch=256, max_epochs=61)]
    assert axes_for(specs) == ("lr",)


# ---------------------------------------------------------------------------
# Against the real arm definitions
# ---------------------------------------------------------------------------
def test_real_experiments_label_as_intended():
    from launch import FROZEN_RUNS, RUNS, resolve

    labels = label_map(resolve, list(FROZEN_RUNS) + list(RUNS))

    # Experiment 02: exactly the two swept knobs, nothing derived.
    assert labels["02_lr3e-3_bs256"] == "lr3e-3_bs256"
    assert labels["02_lr1e-2_bs128"] == "lr1e-2_bs128"

    # Experiment 01: effective batch is the knob, adapter is the control.
    assert labels["F2a"] == "lr1e-3_eb256_noadapter"
    assert labels["F0"] == "lr1e-3_eb1024_adapter"

    # Replicates share a label, which is what makes them replicates.
    assert labels["F1a"] == labels["F1b"] == labels["F1c"]


def test_new_run_directory_names_match_their_labels():
    """A new run's id is its prefix plus its label, or the scheme is a lie."""
    from launch import RUNS, resolve

    labels = label_map(resolve, list(RUNS))
    for name in [n for n in RUNS if n.startswith("02_")]:
        assert name == f"02_{labels[name]}"


def test_batch_size_change_counts_as_one_knob_not_three():
    """The comparison the final selection rests on must get a verdict.

    `02_lr1e-3_bs128` differs from F2a in batch_size, effective_batch and
    max_epochs, because at accumulate 1 and a fixed sample budget the first
    determines the other two. Counted raw, that is three knobs and
    `stability.py` files it as confounded -- refusing a verdict to the
    cleanest single-knob comparison in the sweep.
    """
    from launch import resolve
    from collect import spec_diff
    from stability import independent_knobs

    f2a, arm = resolve("F2a"), resolve("02_lr1e-3_bs128")
    assert sorted(spec_diff(f2a, arm)) == ["batch_size", "effective_batch",
                                           "max_epochs"]
    assert independent_knobs(f2a, arm) == ["batch_size"]
    assert f2a["accumulate"] == arm["accumulate"] == 1
    assert f2a["samples_actual"] == arm["samples_actual"]


def test_reduction_leaves_a_genuine_single_knob_alone():
    from launch import resolve
    from stability import independent_knobs

    assert independent_knobs(resolve("F2a"),
                             resolve("02_lr3e-3_bs256")) == ["lr"]


def test_experiment_02_holds_samples_fixed_at_accumulate_one():
    """The manipulation the experiment claims to make, checked directly."""
    from launch import RUNS, resolve

    arms = [resolve(n) for n in RUNS if n.startswith("02_")]
    assert {a["accumulate"] for a in arms} == {1}
    assert len({a["samples_actual"] for a in arms}) == 1
    by_batch = {a["batch_size"]: a["max_epochs"] for a in arms}
    assert by_batch[128] == 2 * by_batch[256]
