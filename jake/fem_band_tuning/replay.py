"""Run endpoint tuning sweeps through the pinned Figure 4 nonlinear twin."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from jake.fem_band_tuning.common import render_basis, center_rates, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--max-scenes', type=int)
    p.add_argument('--max-traces', type=int)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    out = args.out_dir
    design = json.loads((out/'design.json').read_text())
    digest = sha256(out/'design.json')
    if sha256(design['checkpoint']) != design['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer
    from paper.fig4.spatiotemporal_tuning.activation_map_metrics import model_mean_peak_lag
    scorer = RealTraceMatrixScorer.load(checkpoint_path=Path(design['checkpoint']),
        dataset_configs=Path(design['dataset_config']),
        population_spec_dir=Path(design['population_spec_dir']),
        population_version=design['population_version'], device=args.device, strict=True)
    scorer.model.model.eval()
    scorer.readout.eval()
    # Model-loading utilities may change global precision preferences.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if (scorer.n_lags, scorer.input_rate_hz, scorer.output_rate_hz, scorer.n_units) != (60, 240, 240, 725):
        raise ValueError('Model temporal/population contract changed')
    target = out / ('smoke' if args.smoke else 'responses')
    target.mkdir(exist_ok=True)
    write_json(target/f'model_shard_{args.shard}.json', {'design_sha256': digest,
        'provenance': scorer.provenance, 'model_peak_lag': model_mean_peak_lag(scorer.model.model.convnet),
        'device': args.device, 'torch': torch.__version__, 'unit_rows': scorer.unit_rows})
    traces = np.load(out/'traces_endpoint_aligned.npy')
    offsets = np.load(out/'coefficient_offsets.npy')
    coefficient = torch.as_tensor(np.concatenate([np.ones((len(offsets),1)), offsets], axis=1),
        dtype=torch.float32, device=args.device)
    trace_rows = np.r_[-1, np.arange(len(traces))]
    if args.max_traces:
        # Smoke includes both event types, selected independently of responses.
        table = pd.read_csv(out/'traces.csv')
        d = table.index[table.event_class.eq('drift')].to_numpy()
        m = table.index[table.event_class.eq('microsaccade')].to_numpy()
        chosen = np.r_[d[:args.max_traces//2], m[:args.max_traces-args.max_traces//2]]
        trace_rows = np.r_[-1, chosen]
    scenes = np.arange(design['n_scenes'])
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    start = time.monotonic()
    calls, max_error = 0, 0.0
    with torch.inference_mode():
        for scene in scenes[args.shard::args.shards]:
            dest = target / f'scene_{scene:03d}.npz'
            if dest.exists():
                with np.load(dest) as z:
                    if str(z['design_sha256']) != digest or not np.array_equal(z['trace_rows'], trace_rows):
                        raise ValueError('Resume cache has different design/trace axes')
                print(f'Validated existing {dest.name}', flush=True)
                continue
            source = np.load(out/'scenes'/f'scene_{scene:03d}.npz')
            fields = source['fields'].copy()
            contrast = design['normalization']['global_contrast_factor']
            fields[0] = 127.5 + contrast * (fields[0]-127.5)
            fields[1:] *= contrast
            rates = np.empty((len(trace_rows), len(offsets), scorer.n_units), dtype=np.float32)
            for t, trace_row in enumerate(trace_rows):
                trace = np.zeros_like(traces[0]) if trace_row == -1 else traces[trace_row]
                basis = render_basis(fields, trace, device=args.device)
                if not torch.isfinite(basis).all():
                    raise ValueError('Nonfinite rendered history')
                for b in range(0, len(coefficient), args.batch_size):
                    history = torch.einsum('bc,cltyx->bltyx', coefficient[b:b+args.batch_size], basis)
                    # A fixed global affine scale keeps all interventions in gamut.
                    if history.min() < -127/255-1e-5 or history.max() > 128/255+1e-5:
                        raise ValueError('Out-of-range stimulus; refusing nonlinear clipping')
                    value, error = center_rates(scorer, history, audit=(calls == 0))
                    if error is not None:
                        max_error = max(max_error, error)
                    rates[t,b:b+len(value)] = value.cpu().numpy()
                    calls += len(value)
                if t % 10 == 0 or t == len(trace_rows)-1:
                    seconds = time.monotonic()-start
                    print(f'shard {args.shard} scene {scene:02d} trace {t+1}/{len(trace_rows)}; '
                        f'{calls} endpoint outputs; {seconds:.1f}s; {calls/max(seconds,1e-9):.2f}/s', flush=True)
            tmp = dest.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, rates_hz=rates, trace_rows=trace_rows,
                scene=int(scene), design_sha256=digest,
                center_readout_max_error_counts=max_error,
                elapsed_seconds=time.monotonic()-start)
            tmp.replace(dest)
    write_json(target/f'completed_shard_{args.shard}.json', {'design_sha256': digest,
        'scene_rows': scenes[args.shard::args.shards].tolist(), 'trace_rows': trace_rows.tolist(),
        'endpoint_outputs_computed': calls, 'elapsed_seconds': time.monotonic()-start,
        'center_readout_max_error_counts': max_error})


if __name__ == '__main__':
    main()
