"""Fail-closed audit of the complete FEM band-tuning analysis and artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from jake.fem_band_tuning.common import (
    ROOT, sha256, write_json, sweep_indices, gain_statistics, render_basis, center_rates)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--model',action='store_true')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args(); out=args.out_dir
    d=json.loads((out/'design.json').read_text()); digest=sha256(out/'design.json')
    checks={}
    def check(name,condition,detail=None):
        checks[name]={'passed':bool(condition),'detail':detail}
        if not condition:
            write_json(out/'audit.json',{'passed':False,'checks':checks})
            raise AssertionError(name+': '+str(detail))
    selected=json.loads((ROOT/'manuscript/analysis/selected_model_bundle.json').read_text())
    check('current_figure4_checkpoint',sha256(d['checkpoint'])==d['checkpoint_sha256']==selected['checkpoint_sha256'])
    for name,entry in d['source_files'].items():
        check('source_'+name,sha256(entry['path'])==entry['sha256'])
    for name,expected in d['input_files'].items():
        check('input_'+name,sha256(out/name)==expected)
    traces=np.load(out/'traces_endpoint_aligned.npy')
    check('same_endpoint_for_every_trace',np.array_equal(traces[:,-1],np.zeros((200,2))))
    check('native_temporal_contract',traces.shape==(200,60,2) and d['output_bins_per_history']==1)
    check('pyramid_reconstruction',d['pyramid']['max_reconstruction_error_uint']<2e-4,
        d['pyramid']['max_reconstruction_error_uint'])
    table=pd.read_csv(out/'traces.csv')
    geometry=json.loads((out/'input_geometry_audit.json').read_text())
    check('no_retinal_sampling_of_crop_padding',geometry['minimum_crop_margin_px']>0,
        geometry['minimum_crop_margin_px'])
    check('all_audited_event_classes',table.event_class.value_counts().to_dict()=={'drift':146,'microsaccade':54})
    offsets=np.load(out/'coefficient_offsets.npy'); indices=sweep_indices(d['interventions'])
    check('intervention_coverage',offsets.shape==(33,6) and np.count_nonzero(offsets[0])==0)
    scenes=sorted((out/'responses').glob('scene_*.npz'))
    check('complete_crossed_dataset',len(scenes)==40,len(scenes))
    max_head_error=0.0; maximum_rate=0.0
    arrays=np.load(out/'analysis_arrays.npz')
    summary=json.loads((out/'summary.json').read_text())
    check('analysis_design_identity',summary['design_sha256']==digest)
    check('screen_edge_sensitivity_control',summary['interior_scene_control']['excluded_screen_edge_scene_rows']==
        geometry['scenes_whose_swept_field_may_cross_screen_edge'] and len(summary['interior_scene_control']['scene_rows'])==38)
    for j,path in enumerate(scenes):
        z=np.load(path); r=z['rates_hz']
        check(f'response_scene_{j}',int(z['scene'])==j and str(z['design_sha256'])==digest
            and r.shape==(201,33,725) and np.array_equal(z['trace_rows'],np.arange(-1,200))
            and np.isfinite(r).all() and np.min(r)>=0)
        check(f'response_digest_{j}',sha256(path)==summary['response_files'][str(path.relative_to(out))])
        gain,_,_=gain_statistics(r[:,indices,:])
        check(f'gain_derivation_{j}',np.array_equal(gain[1:],arrays['gain_hz'][j]))
        check(f'original_baseline_shared_{j}',np.array_equal(r[:,0],r[:,indices[0,2]]))
        max_head_error=max(max_head_error,float(z['center_readout_max_error_counts']))
        maximum_rate=max(maximum_rate,float(r.max()))
    check('central_native_readout_equivalence',max_head_error<3e-6,max_head_error)
    unit_table=pd.read_csv(d['tuning_table']).sort_values('unit_index')
    model=json.loads((out/'responses/model_shard_0.json').read_text())
    check('all_725_exact_unit_rows',len(model['unit_rows'])==725 and len(unit_table)==725)
    # Exact canonical channel is preserved by the fixed population view.
    if all('canonical_channel' in row for row in model['unit_rows']):
        check('canonical_unit_order',np.array_equal([r['canonical_channel'] for r in model['unit_rows']],
            unit_table.canonical_channel.to_numpy()))
    check('exact_sessions_and_cids',np.array_equal([r['canonical_session'] for r in model['unit_rows']],
        unit_table.session.to_numpy()) and np.array_equal([r['canonical_source_cid'] for r in model['unit_rows']],
        unit_table.cid.to_numpy()))
    for name in ['scene_and_neuron_tuning','population_interaction','example_tuning_curves','neuronal_subpopulations']:
        for suffix in ['png','pdf','svg']:
            path=out/f'{name}.{suffix}'
            check('artifact_'+path.name,path.exists() and path.stat().st_size>1000)
    check('complete_scene_atlas',(out/'all_scene_atlas.pdf').exists()
        and json.loads((out/'figure_manifest.json').read_text())['atlas_pages']==40)
    pdfinfo=subprocess.run(['pdfinfo',str(out/'all_scene_atlas.pdf')],check=True,capture_output=True,text=True).stdout
    pages=int(next(line.split(':',1)[1] for line in pdfinfo.splitlines() if line.startswith('Pages:')))
    check('actual_pdf_page_count',pages==40,pages)
    rerun_error=None; geometry_error=None
    if args.model:
        import torch
        from paper.fig4.upstream.real_trace_matrix.model import RealTraceMatrixScorer,_standardize_uint_like
        from paper.fig4.spatiotemporal_tuning.retinal_replay import causal_histories,render_movies
        torch.set_num_threads(4)
        scorer=RealTraceMatrixScorer.load(checkpoint_path=Path(d['checkpoint']),
            dataset_configs=Path(d['dataset_config']),population_spec_dir=Path(d['population_spec_dir']),
            population_version=d['population_version'],device=args.device,strict=True)
        scorer.model.model.eval(); scorer.readout.eval()
        torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        rerun_error=0.; geometry_error=0.
        with torch.inference_mode():
            for s in [0,19,39]:
                archive=np.load(out/'scenes'/f'scene_{s:03d}.npz')
                fields=archive['fields'].copy()
                contrast=d['normalization']['global_contrast_factor']
                fields[0]=127.5+contrast*(fields[0]-127.5); fields[1:]*=contrast
                cached=np.load(out/'responses'/f'scene_{s:03d}.npz')['rates_hz']
                for e in [-1,0,100,199]:
                    trace=np.zeros_like(traces[0]) if e==-1 else traces[e]
                    basis=render_basis(fields,trace,device=args.device)
                    # Independent helper path verifies conventions and newest-first time order.
                    original=archive['original_patch']
                    direct_cpu=render_basis(_standardize_uint_like(original)[None],trace,device='cpu')[0]
                    reference=causal_histories(original,trace,n_lags=60,out_size=(151,151),
                        temporal_factor=1,supervision_phase=0)[-1]
                    error=float((direct_cpu-reference).abs().max())
                    geometry_error=max(geometry_error,error)
                    check(f'figure4_cpu_geometry_scene{s}_trace{e}',torch.allclose(direct_cpu,reference,atol=2e-6,rtol=1e-5),error)
                    # Compare CUDA to the existing CUDA renderer separately.
                    # Mixing CPU and CUDA grid_sample introduces harmless FMA
                    # roundoff in subpixel coordinates at large displacements.
                    direct_gpu=render_basis(_standardize_uint_like(original)[None],trace,device=args.device)[0]
                    movie=render_movies(original,trace[None],device=args.device)[0]
                    reference_gpu=torch.as_tensor(movie.copy(),device=args.device).flip(0).unsqueeze(0)
                    reference_gpu=(reference_gpu-127)/255
                    error=float((direct_gpu-reference_gpu).abs().max())
                    check(f'figure4_gpu_geometry_scene{s}_trace{e}',torch.allclose(direct_gpu,reference_gpu,atol=2e-6,rtol=1e-5),error)
                    probes=[0,1,4,13,24,32]
                    coefficient=torch.tensor(np.c_[np.ones(len(probes)),offsets[probes]],dtype=torch.float32,device=args.device)
                    history=torch.einsum('bc,cltyx->bltyx',coefficient,basis)
                    fresh,head_error=center_rates(scorer,history,audit=True)
                    error=float(np.max(np.abs(fresh.cpu().numpy()-cached[e+1,probes])))
                    rerun_error=max(rerun_error,error)
                    check(f'fresh_model_scene{s}_trace{e}',np.allclose(fresh.cpu().numpy(),cached[e+1,probes],atol=.002,rtol=4e-5),error)
    local_sources={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'jake/fem_band_tuning').glob('*.py'))}
    for relative in ['paper/fig4/upstream/real_trace_matrix/model.py',
        'paper/fig4/upstream/real_trace_matrix/core.py',
        'paper/fig4/spatiotemporal_tuning/retinal_replay.py',
        'paper/fig4/fixation_stats/backimage_canvas.py']:
        local_sources[relative]=sha256(ROOT/relative)
    write_json(out/'audit.json',{'passed':True,'design_sha256':digest,'checks':checks,
        'fresh_model_replay_performed':args.model,'fresh_model_max_abs_error_hz':rerun_error,
        'legacy_renderer_max_abs_error_normalized':geometry_error,
        'maximum_predicted_rate_hz':maximum_rate,'source_code_sha256':local_sources,
        'scientific_tests':'python -m unittest jake.fem_band_tuning.test_analysis -v'})
    print(json.dumps({'passed':True,'n_checks':len(checks),'fresh_model_error_hz':rerun_error,
        'geometry_error':geometry_error},indent=2),flush=True)


if __name__=='__main__':
    main()
