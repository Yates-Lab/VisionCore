"""Ideal-observer localization from translated model readouts.

The image and eye trajectory are known to the observer. Independent Poisson
spike counts are simulated from the model's native 240-Hz rate maps. Candidate
positions are a central 7x7 grid of exact model output positions. This is a
model-based localization assay, not decoding from recorded spikes or image
reconstruction. A second assay matches mean expected total spikes to ten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jake.eye_movement_routing.analyze import SEED, sha256, style, COLORS, LABELS


def ideal_decode(templates, rng, *, repeats=32, expected_spikes=None):
    """Uniform candidate prior and Poisson population likelihood, factorial cancels."""
    templates = np.asarray(templates, dtype=np.float64)
    if templates.ndim != 2 or not np.isfinite(templates).all() or (templates < 0).any():
        raise ValueError("Expected finite nonnegative [position, unit] counts")
    if expected_spikes is not None:
        templates = templates * float(expected_spikes) / max(templates.sum(1).mean(), 1e-12)
    templates = np.maximum(templates, 1e-12)
    truth = np.repeat(np.arange(len(templates)), repeats)
    counts = rng.poisson(templates[truth])
    likelihood = counts @ np.log(templates).T - templates.sum(1)[None]
    logposterior = likelihood - logsumexp(likelihood, axis=1)[:, None]
    correct_logp = logposterior[np.arange(len(truth)), truth]
    prediction = np.argmax(logposterior, axis=1)
    information = np.log2(len(templates)) + np.mean(correct_logp) / np.log(2)
    side = int(np.sqrt(len(templates)))
    error = np.sqrt((prediction // side - truth // side)**2 + (prediction % side - truth % side)**2)
    return {"information_bits": float(information), "accuracy": float(np.mean(prediction == truth)),
            "mean_error_grid_steps": float(error.mean()),
            "expected_total_spikes": float(templates.sum(1).mean())}


def check_decoder():
    rng = np.random.default_rng(1)
    flat = ideal_decode(np.ones((49, 10)) * .1, rng)
    assert abs(flat["information_bits"]) < 1e-12
    assert abs(flat["accuracy"] - 1/49) < 1e-12
    separated = ideal_decode(np.eye(49) * 15, rng)
    assert separated["accuracy"] > .99 and separated["information_bits"] > 5.5
    templates = np.random.default_rng(23).uniform(.05, .8, (49, 12))
    matched = ideal_decode(templates, np.random.default_rng(24), expected_spikes=10.)
    scaled = ideal_decode(templates*7, np.random.default_rng(24), expected_spikes=10.)
    assert abs(matched['expected_total_spikes']-10) < 1e-10
    assert abs(matched['information_bits']-scaled['information_bits']) < 1e-10
    # For stationary rates, the same simulated spikes must yield identical
    # posteriors with or without timestamps (counts are sufficient).
    stationary = np.random.default_rng(17).uniform(.01,2,(49,8))
    times = np.repeat(stationary[:,None,:]/60,60,axis=1)
    counts = np.random.default_rng(18).poisson(times[[0,8,16,24,32,40,48]])
    timed = counts.reshape(7,-1)@np.log(times.reshape(49,-1)).T-times.sum((1,2))
    pooled = counts.sum(1)@np.log(stationary).T-stationary.sum(1)
    timed -= logsumexp(timed,axis=1)[:,None]
    pooled -= logsumexp(pooled,axis=1)[:,None]
    assert np.allclose(timed,pooled,atol=1e-10)
    return {"identical_templates_information": flat["information_bits"],
            "identical_templates_accuracy": flat["accuracy"],
            "separated_templates_accuracy": separated["accuracy"],
            "matched_count_scalar_gain_invariant": True,
            "stationary_timed_and_count_posteriors_identical": True}


def replay(args, design):
    import torch
    from paper.fig4.spatiotemporal_tuning.retinal_replay import causal_histories
    from paper.fig4.upstream.real_trace_matrix.core import extract_patch
    from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer

    bundle = ROOT / "outputs/no_phase_readout_comparison_20260910/rank1/figure4"
    source = json.loads((bundle / "top_passband_stage_trajectory_10img_x_10fix/summary.json").read_text())
    selected = json.loads((ROOT / "manuscript/analysis/selected_model_bundle.json").read_text())
    if sha256(source["checkpoint"]) != selected["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    if not np.array_equal(np.sort(np.r_[design['selection_images'], design['evaluation_images']]), np.arange(40)):
        raise ValueError("Selection/evaluation image partition is not disjoint")
    image_table = pd.read_csv(source["inputs"]["image_table"])
    traces = np.load(source["inputs"]["trace_array"])
    scorer = RealTraceMatrixScorer.load(checkpoint_path=Path(source["checkpoint"]),
        dataset_configs=Path(source["dataset_config"]),
        population_spec_dir=bundle / "all_available_population_spec",
        population_version=source["population_version"], device=args.device, strict=True)
    scorer.model.model.eval()
    scorer.readout.eval()
    torch.set_num_threads(4)
    selected_units = design["groups"].reshape(-1)
    idx = torch.as_tensor(selected_units, device=args.device)
    frames_expected = 60
    canvas_cache = {}
    image_rows = design["decode_images"]
    trace_rows = np.r_[-1, design["decode_traces"]]
    maps_dir = args.out_dir / "rate_maps"
    maps_dir.mkdir(exist_ok=True)
    cached_rate = np.load(bundle / "response_matrix_40img_x_200fix/merged/mean_rate_matrix.npy", mmap_mode="r")
    if cached_rate.shape != (len(image_table)*len(traces), scorer.n_units):
        raise ValueError("Unexpected response matrix axes")
    cached_rate = cached_rate.reshape(len(image_table), len(traces), scorer.n_units)
    cached_stable = np.load(bundle / "response_matrix_40img_x_200fix/merged/stabilized_mean_rate_by_image.npy", mmap_mode="r")
    start_time = time.monotonic()
    for ii, image_row in enumerate(image_rows):
        patch, _ = extract_patch(image_table.iloc[int(image_row)], canvas_cache=canvas_cache, patch_size_px=540)
        for jj, trace_row in enumerate(trace_rows):
            path = maps_dir / f"image_{image_row:03d}_trace_{trace_row:03d}.npz"
            if path.exists():
                cached = np.load(path)
                if not np.array_equal(cached['unit_indices'], selected_units):
                    raise ValueError("Cached map unit identity changed")
                if str(cached['checkpoint_sha256']) != selected['checkpoint_sha256']:
                    raise ValueError("Cached map checkpoint identity changed")
                if 'temporal_counts' in cached.files:
                    continue
            trace = np.zeros_like(traces[0]) if trace_row == -1 else traces[int(trace_row)]
            histories = causal_histories(patch, trace, n_lags=scorer.n_lags,
                out_size=scorer.out_size, temporal_factor=scorer.temporal_factor,
                supervision_phase=scorer.supervision_phase)
            if len(histories) == frames_expected + 1:
                histories = histories[1:]
            if len(histories) != frames_expected:
                raise ValueError("Unexpected replay time grid")
            integrated = None
            temporal = []
            with torch.no_grad():
                for begin in range(0, len(histories), args.batch_size):
                    batch = histories[begin:begin+args.batch_size].to(args.device)
                    full = scorer._compute_rate_map(batch)
                    current = scorer.apply_population_view(full, scorer.population_view).index_select(1, idx).clamp_min(0)
                    cy, cx = np.array(current.shape[-2:])//2
                    temporal.append(current[:,:,cy-3:cy+4,cx-3:cx+4].cpu().numpy())
                    value = current.double().sum(dim=0).cpu().numpy()
                    integrated = value if integrated is None else integrated + value
                    del batch, full, current
            expected = cached_stable[int(image_row), selected_units] if trace_row == -1 else cached_rate[int(image_row), int(trace_row), selected_units]
            observed = integrated.mean(axis=(1, 2)) / .25
            error = float(np.max(np.abs(observed - expected)))
            if not np.allclose(observed, expected, atol=1e-3, rtol=2e-4):
                raise ValueError(f"Replay/cached rates disagree: {error}")
            np.savez_compressed(path, integrated_counts=integrated.astype(np.float32),
                temporal_counts=np.concatenate(temporal,axis=0),
                unit_indices=selected_units, image_row=image_row, trace_row=trace_row,
                max_rate_error_hz=error, checkpoint_sha256=selected['checkpoint_sha256'])
            print(f"Replay {ii*len(trace_rows)+jj+1}/{len(image_rows)*len(trace_rows)}; elapsed {time.monotonic()-start_time:.1f}s; max rate error {error:.3g} Hz", flush=True)
    return maps_dir


def decode_maps(args, design, maps_dir):
    rng = np.random.default_rng(SEED+5)
    rows = []
    n = len(design['groups'][0])
    for image_row in design['decode_images']:
        for trace_row in np.r_[-1, design['decode_traces']]:
            z = np.load(maps_dir / f"image_{image_row:03d}_trace_{trace_row:03d}.npz")
            if args.readout == 'time_resolved':
                # The likelihood uses every unit/time bin, preserving when
                # each cell fired. Positions remain identical to count-only.
                maps = z['temporal_counts'].transpose(1,0,2,3)
            else:
                maps = z['integrated_counts']
                cy, cx = np.array(maps.shape[-2:])//2
                maps = maps[:,cy-3:cy+4,cx-3:cx+4]
            for g in range(2):
                templates = maps[g*n:(g+1)*n].reshape(-1, 49).T
                for mode, budget in [('natural_counts', None), ('matched_10_spikes', 10.)]:
                    result = ideal_decode(templates, rng, repeats=args.repeats, expected_spikes=budget)
                    rows.append({'image_row':int(image_row), 'trace_row':int(trace_row), 'group':g, 'mode':mode, **result})
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / 'decoding_trials_summary.csv', index=False)
    return table


def summarize(args, design, table, checks):
    import matplotlib.pyplot as plt
    image_rows = design['decode_images']
    traces = design['decode_traces'].reshape(5,4)
    metrics = ['information_bits', 'accuracy', 'mean_error_grid_steps']
    modes = ['natural_counts', 'matched_10_spikes']
    effects = np.empty((2,3,len(image_rows),5,4,2))
    for m, mode in enumerate(modes):
        q = table.loc[table['mode'].eq(mode)].set_index(['image_row','trace_row','group'])
        for k, metric in enumerate(metrics):
            for i, image in enumerate(image_rows):
                for b in range(5):
                    for j, trace in enumerate(traces[b]):
                        for g in range(2):
                            effects[m,k,i,b,j,g] = q.loc[(image,trace,g),metric]-q.loc[(image,-1,g),metric]
    center = effects.mean(axis=(2,4))  # mode, metric, bin, group
    rng = np.random.default_rng(SEED+8)
    boot = []
    for _ in range(2000):
        ids = rng.integers(len(image_rows), size=len(image_rows))
        boot.append(np.stack([effects[:,:,ids,b][:,:, :,rng.integers(4,size=4),:].mean(axis=(2,3)) for b in range(5)],axis=2))
    boot = np.asarray(boot)
    low, high = np.quantile(boot,[.025,.975],axis=0)
    report = {'status':'exploratory model localization, not recorded-spike decoding',
        'readout':args.readout, 'time_bins':60 if args.readout=='time_resolved' else 1,
        'decoder':'ideal independent-Poisson likelihood with uniform position prior; image and trajectory known',
        'population':'translated checkpoint readouts, 48 tuning-defined model units per group',
        'positions':'central 7x7 exact output positions; four input pixels between positions',
        'position_step_arcmin':4/37.50476617*60, 'duration_ms':250,
        'images':image_rows.tolist(), 'traces':traces.tolist(), 'repeats_per_position':args.repeats,
        'mean_gain':center.tolist(), 'ci_low':low.tolist(), 'ci_high':high.tolist(),
        'faster_minus_slower_group_gain':(center[:,:,:,1]-center[:,:,:,0]).tolist(),
        'group_difference_ci95':np.quantile(boot[:,:,:,:,1]-boot[:,:,:,:,0],[.025,.975],axis=0).tolist(),
        'group_by_movement_interaction_ci95':np.quantile(
            boot[:,:,:,4,1]-boot[:,:,:,4,0]-boot[:,:,:,0,1]+boot[:,:,:,0,0],[.025,.975],axis=0).tolist(),
        'axes':['mode','metric','movement_quintile','group'], 'modes':modes, 'metrics':metrics,
        'checks':checks, 'interval':'crossed image and within-quintile trajectory bootstrap, fixed unit groups',
        'limits':['Spike independence is an assumption; biological noise correlations are not measured here.',
                  'The observer knows the image and movement, so this measures available localization information, not a fixed downstream decoder.',
                  'This does not demonstrate attentional control or reconstruction of arbitrary image content.']}
    errors = [float(np.load(path)['max_rate_error_hz']) for path in (args.out_dir/'rate_maps').glob('*.npz')]
    report['replay_vs_figure4_cache_max_rate_error_hz'] = max(errors)
    (args.out_dir/'decoding_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    style()
    fig, axes = plt.subplots(1,2,figsize=(10.5,4.5))
    for m,ax in enumerate(axes):
        for g in range(2):
            ax.fill_between(range(5),low[m,0,:,g],high[m,0,:,g],color=COLORS[g],alpha=.14,lw=0)
            ax.plot(range(5),center[m,0,:,g],'o-',color=COLORS[g],lw=2.3,label=LABELS[g])
        ax.axhline(0,color='.65',lw=.8)
        ax.set_xticks(range(5),range(1,6))
        ax.set_xlabel('Movement spectral balance (quintile)\n← slower-tuned drive     faster-tuned drive →')
        ax.set_ylabel('Localization information gain\nrelative to stabilized (bits / trial)')
        ax.set_title(['A   Natural spike counts','B   Matched expected spike count'][m],loc='left',fontweight='bold',pad=15)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Spatial decoding from the two model populations',fontsize=13,y=.99)
    fig.text(.5,.015,'Ideal Poisson observer · known image and movement · 10 image patches × 20 trajectories · matched condition: 10 mean spikes/trial',ha='center',fontsize=8,color='.4')
    fig.tight_layout(rect=[0,.07,1,.95])
    for ext in ['png','pdf','svg']:fig.savefig(args.out_dir/f'decoding_two_panels.{ext}',dpi=180)
    print(json.dumps({'mean_information_gain':center[:,0].tolist()},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,default=ROOT/'outputs/eye_movement_routing_20260914')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--repeats',type=int,default=64)
    p.add_argument('--decode-only',action='store_true')
    p.add_argument('--readout',choices=['time_resolved','count_only'],default='time_resolved')
    args=p.parse_args()
    checks=check_decoder()
    design=np.load(args.out_dir/'routing_analysis.npz')
    maps_dir=args.out_dir/'rate_maps' if args.decode_only else replay(args,design)
    table=decode_maps(args,design,maps_dir)
    summarize(args,design,table,checks)


if __name__=='__main__':
    main()
