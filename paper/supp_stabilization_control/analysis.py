"""Paired comparison of history-, trial-, and global stabilization on Figure 3 C/D.

The replays use the production scoring code and population. They do not fit a
new encoding model or compute a new FEM variance decomposition.
"""
import hashlib
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_replay_provenance(control_path, run_manifest, reference, selection):
    """Keep model identity fixed across selection changes that leave the replays valid."""
    recorded = run_manifest.get("selection")
    protected = ("checkpoint_sha256", "model_spec_sha256", "final_manifest_sha256")
    if (not isinstance(recorded, dict)
            or any(key not in recorded or key not in selection or recorded[key] != selection[key]
                   for key in protected)
            or run_manifest.get("stabilization_reference", "history_endpoint") != reference
            or digest(control_path) != run_manifest.get("local_cache_sha256")):
        raise ValueError(f"{reference} replay provenance differs from selected model or completed cache")


def validate_trial_replay_acceptance(trial_cache, accepted_sha256):
    """Return the verified accepted digest, or None when no exception was requested."""
    if accepted_sha256 is None:
        return None
    actual = digest(trial_cache)
    if accepted_sha256 != actual:
        raise ValueError(
            f"Accepted trial replay SHA-256 does not match {trial_cache}: "
            f"expected {accepted_sha256}, actual {actual}"
        )
    return actual


def replay_check_record(
    reference, metric, condition, maximum, tolerance, accepted_trial_sha256
):
    """Record a replay check, failing unless this exact trial cache was accepted."""
    record = {"maximum_absolute_difference": maximum, "tolerance": tolerance}
    if maximum <= tolerance:
        return record
    if reference == "trial" and accepted_trial_sha256 is not None:
        return {
            **record,
            "status": "accepted",
            "accepted_trial_replay_sha256": accepted_trial_sha256,
        }
    raise AssertionError(
        f"Unchanged {reference} replay differs: {metric}/{condition} {maximum}"
    )


def assemble_control_rows(global_payload, history_payload, trial_payload):
    """Validate both controls and order them like the production cache."""
    controls = {
        "history_endpoint": history_payload,
        "trial_centroid": trial_payload,
    }
    for reference, payload in controls.items():
        if payload.get("schema_version") != global_payload.get("schema_version"):
            raise ValueError(f"{reference} and main-figure caches use different scoring schemas")
        if not payload.get("complete") or payload.get("checkpoint_path") != global_payload.get("checkpoint_path"):
            raise ValueError(f"Incomplete or mismatched {reference} control")

    global_rows = global_payload["results"]
    expected_sessions = {row["session"] for row in global_rows}
    ordered = {}
    checks = {}
    identity_keys = (
        "neuron_mask", "ccmax", "ccnorm_unstable", "matched_var_y",
        "matched_n_windows", "n_base_windows",
    )
    for reference, payload in controls.items():
        by_session = {row["session"]: row for row in payload["results"]}
        if len(by_session) != len(payload["results"]) or set(by_session) != expected_sessions:
            raise ValueError(f"{reference} does not cover the exact main-figure session set")
        rows = [by_session[row["session"]] for row in global_rows]
        checks[reference] = {}
        for global_row, control_row in zip(global_rows, rows):
            if control_row.get("stabilization_reference") != reference:
                raise ValueError(f"Unexpected {reference} stabilization reference")
            session_checks = {}
            for key in identity_keys:
                first = np.asarray(global_row[key])
                second = np.asarray(control_row[key])
                try:
                    same = np.array_equal(first, second, equal_nan=True)
                except TypeError:
                    same = np.array_equal(first, second)
                session_checks[key] = bool(same)
            if not all(session_checks.values()):
                raise AssertionError(
                    f"Data-only scoring quantities changed: {global_row['session']} "
                    f"{reference} {session_checks}"
                )
            checks[reference][global_row["session"]] = session_checks
        ordered[reference] = rows
    return ordered["history_endpoint"], ordered["trial_centroid"], checks


def paired_metrics(global_data, history_data, trial_data):
    """Per-unit scores for every condition, keyed in display order."""
    return {
        "ccnorm": {"full": global_data["ccnorm"]["intact"],
                   "retinal": global_data["ccnorm"]["zeroed"],
                   "global": global_data["ccnorm"]["stabilized"],
                   "trial": trial_data["ccnorm"]["stabilized"],
                   "history": history_data["ccnorm"]["stabilized"]},
        "fraction": {"psth": global_data["explainable_fraction"]["psth"],
                     "full": global_data["explainable_fraction"]["intact"],
                     "retinal": global_data["explainable_fraction"]["zeroed"],
                     "global": global_data["explainable_fraction"]["stabilized"],
                     "trial": trial_data["explainable_fraction"]["stabilized"],
                     "history": history_data["explainable_fraction"]["stabilized"]}}


def unchanged_input_replay_checks(global_data, controls, pop, accepted_trial_sha256):
    """Compare replayed Full and Retinal-only scores with the production cache."""
    # Replaying the unchanged inputs is a numerical reproducibility check.
    # Mixed precision may vary slightly across GPU/batch implementations; a
    # 0.005 absolute score tolerance is much smaller than the main effects.
    replay_checks = {}
    for reference, control_data in controls.items():
        for metric, tolerance in (("ccnorm", .002), ("explainable_fraction", .005)):
            for condition in ("intact", "zeroed"):
                a, b = global_data[metric][condition], control_data[metric][condition]
                valid = pop & np.isfinite(a) & np.isfinite(b)
                maximum = float(np.max(np.abs(a[valid] - b[valid])))
                key = f"{metric}/{condition}" if reference == "history" else f"trial/{metric}/{condition}"
                replay_checks[key] = replay_check_record(
                    reference,
                    metric,
                    condition,
                    maximum,
                    tolerance,
                    accepted_trial_sha256,
                )
    return replay_checks


def adjacent_scope_contrasts(values, sessions, mask, bootstrap):
    """Directly pair each neighboring retinal-stabilization scope per unit."""
    return {
        second + "_minus_" + first: bootstrap(
            values[first], values[second], sessions, mask
        )
        for first, second in (
            ("full", "history"),
            ("history", "trial"),
            ("trial", "global"),
        )
    }


def summarize(metrics, sessions, pop, bootstrap):
    """Medians and paired session-bootstrap contrasts, plus the scores they use."""
    reports, arrays = {}, {}
    # One population for every panel: units scored under all conditions on both metrics.
    mask = pop & np.logical_and.reduce(
        [np.isfinite(value) for values in metrics.values() for value in values.values()])
    for name, values in metrics.items():
        reports[name] = {"n_units": int(mask.sum()), "n_sessions": len(np.unique(sessions[mask])),
                         "medians": {key: float(np.median(values[key][mask])) for key in values},
                         "contrasts": {}}
        for key in values:
            arrays[name+"_"+key] = np.asarray(values[key])
        arrays[name+"_mask"] = mask
        # Compare the ablation costs within each unit before pooling. Their
        # difference is (Full - History) - (Full - Retinal) = Retinal - History;
        # subtracting two population medians would estimate a different quantity.
        for first, second in (("full", "history"), ("global", "history"),
                              ("history", "full"), ("retinal", "full"),
                              ("history", "retinal"),
                              # Panel D: every ablation minus Full.
                              ("full", "retinal"), ("full", "trial"), ("full", "global")):
            stat = bootstrap(values[first], values[second], sessions, mask)
            reports[name]["contrasts"][second+"_minus_"+first] = stat
        reports[name]["contrasts"].update(
            adjacent_scope_contrasts(values, sessions, mask, bootstrap)
        )
    return reports, arrays


def statistics_tex(metrics):
    macros = {}
    for name, record in metrics.items():
        prefix = "StabC" if name == "ccnorm" else "StabD"
        macros[prefix+"Units"] = str(record["n_units"])
        for condition, median in record["medians"].items():
            macros[prefix+condition.capitalize()] = f"{median:.3f}"
        for reference in ("full", "global"):
            stat = record["contrasts"]["history_minus_"+reference]
            macros[prefix+reference.capitalize()+"Delta"] = f"{stat['median']:+.3f}"
            macros[prefix+reference.capitalize()+"CI"] = f"[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}]"
        for suffix, contrast in (("MotionLoss", "full_minus_history"),
                                 ("ExtraLoss", "full_minus_retinal"),
                                 ("CostDifference", "retinal_minus_history"),
                                 ("FullToHistory", "history_minus_full"),
                                 ("FullToRetinal", "retinal_minus_full"),
                                 ("FullToTrial", "trial_minus_full"),
                                 ("FullToGlobal", "global_minus_full"),
                                 ("HistoryToTrial", "trial_minus_history"),
                                 ("TrialToGlobal", "global_minus_trial")):
            stat = record["contrasts"][contrast]
            macros[prefix+suffix] = f"{stat['median']:.3f}"
            macros[prefix+suffix+"CI"] = f"[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}]"
    return ("% Generated by render_stabilization_control.py from paired production-population scores.\n" +
            "".join(f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in sorted(macros.items())))
