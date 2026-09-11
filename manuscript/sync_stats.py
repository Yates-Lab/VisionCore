#!/usr/bin/env python3
"""Export manuscript numbers from the selected, completed production run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

# Also support bundle drivers loading this exporter with importlib.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_selection import SELECTION, selected_analysis
from figure4_selection import SPECTRUM_SELECTION, spectrum_update

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = Path(__file__).resolve().parent
SELECTED = selected_analysis()
BUNDLE = ROOT / SELECTED['bundle']
CHECKPOINT = SELECTED['checkpoint_sha256']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated_stats.tex is stale, without changing it")
    args = parser.parse_args()
    sources = {}
    if SELECTION.exists() and BUNDLE.resolve() == (ROOT / SELECTED['bundle']).resolve():
        sources[str(SELECTION.relative_to(ROOT))] = digest(SELECTION)

    def read(relative: str) -> dict:
        path = BUNDLE / relative
        sources[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())

    final = read("FINAL_MANIFEST.json")
    if final["status"] != "complete" or final["checkpoint_sha256"] != CHECKPOINT:
        raise ValueError("The selected reproduction is not complete or changed checkpoint")
    three = read("figure3/figures/figure3_manifest.json")
    four = read("figure4/production_figure4/figure/results_provenance.json")
    if three["model"]["checkpoint_sha256"] != CHECKPOINT:
        raise ValueError("Figure 3 checkpoint mismatch")
    if four["checkpoint_sha256"] != CHECKPOINT or not four["release_ready"]:
        raise ValueError("Figure 4 checkpoint mismatch or failed release")
    for record in four["sources"].values():
        path = Path(record["path"])
        actual = digest(path)
        if actual != record["sha256"]:
            raise ValueError(f"Figure 4 source changed: {path}")
        sources[str(path.relative_to(ROOT))] = actual

    values = {}

    def value(name: str, number: float, decimals: int = 3) -> None:
        values[name] = f"{number:.{decimals}f}"

    def probability(name: str, number: float) -> None:
        mantissa, exponent = f"{number:.1e}".split("e")
        values[name] = rf"{mantissa}\times10^{{{int(exponent)}}}"

    model_audit = read('audits/production_model.json')
    if model_audit['status'] != 'passed':
        raise ValueError('The selected model architecture audit failed')
    for key, name in (('total', 'Parameters'), ('visual_core', 'VisualParameters'),
                      ('behavior_modulator', 'BehaviorParameters'),
                      ('deep_readouts', 'ReadoutParameters'),
                      ('output_channels', 'OutputChannels'), ('sessions', 'Sessions')):
        values['Model' + name] = format(int(model_audit['production'][key]), ',')
    if 'readout_rank' in model_audit['production']:
        value('ModelReadoutRank', model_audit['production']['readout_rank'], 0)

    empirical_path = MANUSCRIPT / "analysis/empirical_stats.json"
    empirical = json.loads(empirical_path.read_text())
    if digest(Path(empirical["source"])) != empirical["sha256"]:
        raise ValueError("The empirical analysis cache changed; regenerate the Figure 2 statistics")
    sources[str(empirical_path.relative_to(ROOT))] = digest(empirical_path)
    sources[str(Path(empirical["source"]).relative_to(ROOT))] = empirical["sha256"]
    def interval(name, pair, decimals=3):
        values[name] = f"[{pair[0]:.{decimals}f}, {pair[1]:.{decimals}f}]"
    value("FigTwoFemMedian", empirical["alpha"]["median"], 2)
    value("FigTwoFemUnits", empirical["alpha"]["n"], 0)
    interval("FigTwoFemCI", empirical["alpha"]["median_ci"], 2)
    interval("FigTwoFemIQR", empirical["alpha"]["iqr"], 2)
    for key, label in (("unc", "Uncorrected"), ("cor", "Corrected")):
        value("FigTwoFano" + label, empirical["fano"][key]["mean"], 2)
    for key, label in (("r_u", "Uncorrected"), ("r_c", "Corrected"), ("dr", "Difference")):
        value("FigTwoNC" + label, empirical["noise_correlation"][key + "_mean"])
        interval("FigTwoNC" + label + "CI", empirical["noise_correlation"][key + "_ci"])
    interval("FigTwoNCNullCI", empirical["noise_correlation"]["null_dr_ci"])
    for key, label in (("x", "StimInFem"), ("y", "FemInStim")):
        entry = empirical["alignment"][key]
        values["FigTwo" + label] = rf"{entry['mean']:.2f}\pm{entry['sd']:.2f}"
    for key, label in (("fem", "Fem"), ("psth", "Stimulus"), ("resid", "Residual")):
        entry = empirical["participation_ratio"][key]
        values["FigTwoPR" + label] = rf"{entry['mean']:.1f}\pm{entry['sd']:.1f}"
    value("FigTwoPRSignP", empirical["participation_ratio"]["sign_p"], 2)

    c, d, e = (three[f"panel_{panel}_stats"] for panel in "cde")
    value("FigThreeCUnits", c["n_units"], 0)
    value("FigThreeSessions", c["n_sessions"], 0)
    for key, label in (("intact", "Full"), ("zeroed", "Retinal"), ("stabilized", "Stabilized")):
        value("FigThreeC" + label, c["medians"][key])
        value("FigThreeD" + label, d["conditions"][key]["median"])
        value("FigThreeE" + label, e["medians"][key])
    for key, label in (("zeroed_vs_intact", "Retinal"), ("stabilized_vs_intact", "Stabilized")):
        contrast = c["contrasts"][key]
        value("FigThreeC" + label + "Delta", contrast["median_difference"])
        value("FigThreeC" + label + "Reduction", contrast["percent_of_intact_median"], 1)
        probability("FigThreeC" + label + "P", contrast["wilcoxon_p"])
    value("FigThreeDUnits", d["n_units"], 0)
    value("FigThreeDPsth", d["conditions"]["psth"]["median"])
    for key, label in (("full_vs_psth", "FullGain"), ("ablated_vs_full", "RetinalLoss"), ("stabilized_vs_full", "StabilizedLoss")):
        contrast = d["contrasts"][key]
        value("FigThreeD" + label, contrast["median_unit_difference"])
        probability("FigThreeD" + label + "P", contrast["bootstrap"]["p_boot"])
        values["FigThreeD" + label + "CI"] = (
            f"[{contrast['bootstrap']['ci_low']:+.3f}, {contrast['bootstrap']['ci_high']:+.3f}]"
        )
    value("FigThreeDGainPercent", d["contrasts"]["full_vs_psth"]["percent_of_reference_median"], 0)
    value("FigThreeDLossPercent", abs(d["contrasts"]["stabilized_vs_full"]["percent_of_reference_median"]), 0)
    value("FigThreeDFullPositiveSessions", d["contrasts"]["full_vs_psth"]["session_sign_test"]["n_positive"], 0)
    value("FigThreeEEmpirical", e["medians"]["emp"])
    value("FigThreeEStabilizedDifference", e["stabilized_vs_empirical"]["median_empirical_minus_model"])
    probability("FigThreeEStabilizedP", e["stabilized_vs_empirical"]["wilcoxon_p"])
    for key, label in (("intact", "Full"), ("zeroed", "Retinal")):
        probability("FigThreeE" + label + "TostP", e["tost"][key]["p"])
    value('FigThreeEStabilizedTostP', e['tost']['stabilized']['p'])

    claims = four["claims"]
    b = claims["panel_b"]
    for key, label in (("n_units", "Units"), ("n_images", "Images"), ("n_traces", "Traces")):
        value("FigFour" + label, b[key], 0)
    value("FigFourPath", b["reference_path_bin"]["rate"]["x_median"], 1)
    for key, label in (("rate", "Rate"), ("SSI", "SSI")):
        result = b["reference_path_bin"][key]
        value("FigFour" + label + "Effect", result["effect_percent"], 1)
        values["FigFour" + label + "CI"] = f"[{result['ci_low']:.1f}, {result['ci_high']:.1f}]"
    value("FigFourValidated", claims["tuning"]["n_validated"], 0)
    value("FigFourUnvalidated", claims["tuning"]["n_acquired"] - claims["tuning"]["n_validated"], 0)
    for key, label in (("rate_percent", "Rate"), ("ssi_percent", "SSI")):
        result = claims["passband_vs_path_length"][key]
        value("FigFour" + label + "PassbandRho", result["median_within_unit_passband_spearman"])
        value("FigFour" + label + "PathRho", result["median_within_unit_path_length_spearman"])
        value("FigFour" + label + "RhoDifference", result["median_paired_difference"])
        lo, hi = result["paired_unit_bootstrap_ci95"]
        values["FigFour" + label + "RhoCI"] = f"[{lo:.3f}, {hi:.3f}]"
    trajectory = claims["gain_invariant_stage_trajectory"]
    for key, label in (("n_images", "Images"), ("n_traces", "Traces"), ("n_movies", "Movies")):
        value("FigFourStage" + label, trajectory[key], 0)
    figure = read("figure4/production_figure4/figure/summary.json")
    update = spectrum_update(BUNDLE)
    spectral_figure = figure
    if update:
        spectral_figure = json.loads((ROOT / update['figure_summary']).read_text())
        sources.update(update['source_sha256'])
        sources[str(SPECTRUM_SELECTION.relative_to(ROOT))] = digest(SPECTRUM_SELECTION)
    regimes = spectral_figure['panels']['C']['regimes']
    if [entry['name'] for entry in regimes] != ['drift', 'microsaccades']:
        raise ValueError('Drift/microsaccade counts require event-defined spectra')
    for code, name in enumerate(("Drift", "Microsaccade")):
        value("FigFour" + name + "Windows", regimes[code]["n_fixation_epochs"], 0)
    for key, label in (("rate_percent", "Rate"), ("ssi_percent", "SSI")):
        centers = figure["panels"]["G"][key]["binned_center"]
        value("FigFour" + label + "EngagementLow", centers[0], 1)
        value("FigFour" + label + "EngagementHigh", centers[-1], 1)
    value("FigFourPassbandPercent", 100 * figure["panels"]["E"]["passband_response_fraction"], 0)
    example = read('figure4/panel_a_exemplar_audit/summary.json')
    if example['checkpoint_sha256'] != CHECKPOINT:
        raise ValueError('Figure 4 example uses a different checkpoint')
    lag = example['model_peak_lag']['resolved_rounded_peak_lag_frames']
    value('FigFourExampleLagFrames', lag, 0)
    value('FigFourExampleLagMs', 1000 * lag / 240, 1)
    fit_path = BUNDLE / 'figure4/all_available_yu_tuning/all_yu_fits.csv'
    sources[str(fit_path.relative_to(ROOT))] = digest(fit_path)
    with fit_path.open() as stream:
        fits = list(csv.DictReader(stream))
    if len(fits) != int(b['n_units']):
        raise ValueError('Figure 4 fit count differs from the selected population')
    converged = sum(row['optimizer_success'].lower() == 'true' for row in fits)
    value('FigFourConvergedFits', converged, 0)
    value('FigFourNonconvergedFits', len(fits) - converged, 0)

    text = "% Generated by manuscript/sync_stats.py; edit the source analysis, not these values.\n"
    text += f"% Checkpoint SHA-256: {CHECKPOINT}\n"
    if update:
        text += r"\newif\ifFigFourSpectrumOnly" + "\n"
        text += r"\FigFourSpectrumOnlytrue" + "\n"
    text += "".join(rf"\newcommand{{\{name}}}{{{entry}}}" + "\n" for name, entry in sorted(values.items()))
    target = MANUSCRIPT / "generated_stats.tex"
    if args.check:
        if not target.exists() or target.read_text() != text:
            raise ValueError("Generated statistics are stale; run make -C manuscript stats")
    else:
        target.write_text(text)
    audit = MANUSCRIPT / "build" / "stats_sources.json"
    audit.parent.mkdir(exist_ok=True)
    audit.write_text(json.dumps({"checkpoint_sha256": CHECKPOINT, "sources": sources, "macros": values}, indent=2) + "\n")
    print(f"{'Verified' if args.check else 'Wrote'} {len(values)} statistics in {target}")


if __name__ == "__main__":
    main()
