"""Descriptive names for experiments and the runs inside them.

`E1a`, `F3c` and "Stage 0b" carry no meaning: reading a result table means
holding `launch.py`'s arm definitions in your head, and a name that has to be
looked up is a name that gets misremembered. This module gives every run a
label that states what it *is* -- `lr3e-3_bs128` -- so the table and the
analysis say the same thing.

Two rules define a label.

**An experiment declares its members and its baseline; its axes are derived.**
The axes are the knobs that actually take more than one value across the
experiment's runs. Declaring them separately would be a second place to state
a fact that the run definitions already determine, and `config_signature` in
`collect.py` avoids exactly that mistake for replicate groups. Adding an arm
that varies a new knob widens the axes on its own.

**A label is complete over the axes, not over the differences.** Every axis
appears in every label in the experiment, including at its baseline value:
`02_lr3e-3_bs256` rather than `02_lr3e-3`. Naming only the differing knobs
makes the label relative to the baseline, so it silently changes meaning if
the baseline moves, and two runs in one sweep end up described at different
levels of detail.

Existing runs keep their directory names. The E/F ids are woven through the
notes, the frozen arm definitions, `BASELINE_RUN` and five commits of history,
and renaming the frozen record to improve the readability of runs nobody will
launch again is risk without return. They gain a *derived* label instead, so
`E3c` prints as `E3c (lr3e-3_bs256_eb1024_xsess)` and reads the same way a new
run does.
"""
from __future__ import annotations

# Which experiment each run belongs to, and which run is its baseline. This is
# the whole declaration: everything else about a label is derived from the
# specs themselves.
#
# `baseline` may name a run from an earlier experiment. 02's baseline is F2a,
# which already ran at lr 1e-3 / batch 256 / accumulate 1 -- re-running the
# same configuration under a new name would cost 3.3 GPU-hours to learn
# nothing.
EXPERIMENTS = {
    "00-batch-composition": {
        "baseline": "E1a",
        "summary": "Stage 0. Batch composition, effective batch and lr at "
                   "width 1.0 with the adapter, cross-session baseline.",
    },
    "01-adapter-and-optimizer": {
        "baseline": "F1a",
        "summary": "Stage 0b. Adapter control plus effective batch and lr at "
                   "width 0.5 under homogeneous batching.",
    },
    "02-lr-batch-landscape": {
        "baseline": "F2a",
        "summary": "lr x batch size at accumulate 1 and fixed 8M samples. "
                   "Maps where the optimization axis turns over.",
    },
    "05-lr-at-width-1": {
        "baseline": "04_fe4",
        "summary": "Is the width-1.0 null an lr artifact? lr 1e-3 was chosen "
                   "at width 0.5, and the val curve shows width 1.0 leading "
                   "early, trailing mid-training, converging as lr anneals.",
    },
    "04-frontend-bottleneck": {
        "baseline": "04_fe4",
        "summary": "Temporal frontend width at width 1.0, 32M samples. Does "
                   "the 4-channel retinal prior (midget/parasol x ON/OFF) "
                   "cost accuracy? 8 first; 16 only on a large gain.",
    },
    "06-capacity-ladder": {
        "baseline": "05_lr5e-4",
        "summary": "Does capacity keep paying once every rung gets the lr an "
                   "inverse-width rule prescribes (5e-4 / width, anchored at "
                   "width 1.0)? Widths 2.0 and 3.0, serial.",
    },
    "03-sample-budget": {
        "baseline": "02_lr1e-3_bs128",
        "summary": "Sample budget at the chosen batch and lr 1e-3. 8M is 1.12 "
                   "passes over the data, so saturation is unmeasured. 2x, "
                   "then 4x only if 2x improves.",
    },
}

# Knobs that can appear in a label, with the abbreviation used for each. Order
# here is the order in a label, so labels sort and compare readably. A knob
# absent from this table can still vary -- it just renders as `key=value`,
# which is ugly enough to prompt adding it here.
AXIS_ABBREV = (
    ("lr", "lr"),
    ("batch_size", "bs"),
    ("samples", "s"),
    ("frontend_channels", "fe"),
    ("effective_batch", "eb"),
    ("accumulate", "acc"),
    ("width", "w"),
    ("wd", "wd"),
    ("core_lr_scale", "clr"),
    ("max_epochs", "ep"),
    ("homogeneous", ""),
    ("adapter", ""),
    ("model_config", "cfg"),
    ("config", "data"),
)

# Knobs whose value is a flag rather than a number, and what each state is
# called. "xsess" and "noadapter" are the off states; naming them explicitly
# beats `homogeneous=False`, which is a fact about a keyword argument rather
# than about the run.
FLAG_NAMES = {
    "homogeneous": ("homog", "xsess"),
    "adapter": ("adapter", "noadapter"),
}


# Knobs counted in samples, rendered in millions: `s16M` reads at a glance
# where `s16000000` does not.
MILLIONS = frozenset({"samples"})


def format_value(key, value):
    """Render one axis value for a label."""
    if key in MILLIONS and value:
        return f"{value / 1e6:g}M"
    if key in FLAG_NAMES:
        on, off = FLAG_NAMES[key]
        return on if value else off
    if isinstance(value, float):
        # Learning rates and weight decays are read as powers of ten, and
        # `f"{0.003:g}"` gives "0.003", which is harder to compare across arms
        # than "3e-3". Whole numbers (width 2.0) stay plain.
        #
        # Two significant figures, not one: an inverse-width lr rule produces
        # 2.5e-4 and 1.67e-4, and at one figure *both* render "2e-4" -- two
        # distinct runs with the same label. Trailing zeros are stripped so
        # 1e-3 stays "1e-3" rather than becoming "1.00e-3".
        if value and (abs(value) <= 1e-2 or abs(value) >= 1e4):
            mantissa, exponent = f"{value:.2e}".split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            return f"{mantissa}e{int(exponent)}"
        return f"{value:g}"
    if isinstance(value, str):
        # Config paths are long and mostly shared prefix; the stem identifies.
        return value.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return str(value)


# Elimination order for redundant axes: the knobs `_finish` computes come
# first, so a label names what was *set* rather than what followed from it.
# Experiment 02 varies `batch_size`, and `effective_batch`, `accumulate` and
# `max_epochs` all move with it; experiment 01 varies `effective_batch`
# directly with the micro-batch fixed. Neither case is hard-coded -- the
# redundancy test below decides, and it gets both right.
#
# `width` sits *after* `lr` for the same reason. A capacity ladder gives every
# rung the lr an inverse-width rule prescribes, so width and lr are in exact
# correspondence and only one can survive. Width is the knob being set and lr
# is what the rule returns, so the ladder's labels should read `w2`, `w3` --
# eliminating lr, not width. Width is constant within experiments 00-05, so it
# is never a candidate there and this ordering changes nothing already run.
ELIMINATION_ORDER = (
    "model_config", "config", "accumulate", "max_epochs", "effective_batch",
    "wd", "core_lr_scale", "batch_size", "lr", "width", "homogeneous",
    "adapter",
)


def _determined_by(key, others, specs):
    """True if `others` fix `key` across every spec, making `key` redundant.

    A knob whose value is a function of the axes already in the label tells a
    reader nothing new: in experiment 02 the epoch count is 61 whenever the
    batch is 256 and 122 whenever it is 128, so `ep` is length without
    information.
    """
    seen = {}
    for spec in specs:
        signature = tuple(format_value(o, spec.get(o)) for o in others)
        value = format_value(key, spec.get(key))
        if seen.setdefault(signature, value) != value:
            return False
    return True


def axes_for(specs):
    """The knobs that vary independently across a group of specs.

    `specs` is the resolved spec of every run in the experiment. Two knobs are
    excluded: one held constant across the whole experiment -- width is 0.5
    throughout experiment 02, so `w0.5` in all four labels distinguishes
    nothing -- and one determined by the others, which is how the derived
    fields drop out.
    """
    keys = [k for k, _ in AXIS_ABBREV]
    keys += sorted({k for s in specs for k in s} - set(keys) - _IGNORED)
    axes = [k for k in keys
            if len({format_value(k, s.get(k)) for s in specs}) > 1]

    for key in ELIMINATION_ORDER:
        if key in axes and _determined_by(key, [a for a in axes if a != key],
                                          specs):
            axes.remove(key)
    return tuple(k for k in keys if k in axes)


# Spec fields that describe bookkeeping rather than configuration, and so can
# never be axes however much they vary.
# `samples` is deliberately absent: experiment 03 varies it, and it is the knob
# that is *set* while `max_epochs` and `samples_actual` merely follow from it.
_IGNORED = frozenset({"name", "note", "seed", "experiment", "samples_actual"})


def run_label(spec, axes):
    """The label for one run: every axis, in order, at this run's value.

    Returns "base" for an experiment with no axes at all, which is what a
    replicate-only group is.
    """
    abbrev = dict(AXIS_ABBREV)
    parts = [f"{abbrev.get(k, k + '=')}{format_value(k, spec.get(k))}"
             for k in axes]
    return "_".join(parts) if parts else "base"


def experiment_of(spec):
    """The experiment a spec belongs to, or None if it declares none."""
    return spec.get("experiment")


def label_map(resolve, run_names):
    """`{run_name: label}` for every run that declares an experiment.

    Takes `resolve` rather than importing it, because `launch.py` imports this
    module to name its own arms and the reverse import would be a cycle.
    """
    specs = {}
    for name in run_names:
        try:
            specs[name] = resolve(name)
        except SystemExit:
            continue

    labels = {}
    for experiment in {experiment_of(s) for s in specs.values()} - {None}:
        members = {n: s for n, s in specs.items()
                   if experiment_of(s) == experiment}

        # The baseline joins the axis computation even when it belongs to an
        # earlier experiment. An experiment whose baseline lives elsewhere and
        # which declares a single arm would otherwise have nothing varying
        # within it, and every such arm would label as "base" -- experiment 05
        # is exactly that shape. The knob is only visible against the point
        # being compared to, so that point belongs in the group.
        group = list(members.values())
        base_name = EXPERIMENTS.get(experiment, {}).get("baseline")
        if base_name and base_name not in members:
            try:
                group.append(resolve(base_name))
            except SystemExit:
                pass

        axes = axes_for(group)
        for name, spec in members.items():
            labels[name] = run_label(spec, axes)
    return labels


def prefix_of(experiment):
    """The numeric prefix a new run in this experiment is named with."""
    return experiment.split("-", 1)[0] if experiment else ""
