"""Freeze all scene, trace, normalization, and intervention inputs before replay."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from jake.fem_band_tuning.common import (
    ROOT, LEVELS, STRENGTHS, interventions, laplacian_bands, selected_sources,
    sha256, write_json, SEED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out/'scenes').mkdir(exist_ok=True)
    selected, source, bundle, selected_path, source_path = selected_sources()
    if sha256(source['checkpoint']) != selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint digest mismatch')
    # Import the existing loader first to establish the repository's data paths.
    from paper.fig4.upstream.real_trace_matrix import model
    from paper.fig4.upstream.real_trace_matrix.core import extract_patch
    from fixation_stats.backimage_canvas import _clip_patch
    images = pd.read_csv(source['inputs']['image_table'])
    table = pd.read_csv(source['inputs']['trace_table'])
    traces = np.load(source['inputs']['trace_array'])
    if traces.shape != (200, 60, 2) or not table.event_class.isin(['drift', 'microsaccade']).all():
        raise ValueError('Expected the audited 200 native FEM histories')
    # This new endpoint assay explicitly aligns the latest frame, unlike the
    # mean-peak-lag anchor used in the published temporal-average SSI assay.
    traces = traces - traces[:, -1:, :]
    np.save(out/'traces_endpoint_aligned.npy', traces)
    table.insert(0, 'analysis_trace', np.arange(len(table)))
    table.to_csv(out/'traces.csv', index=False)
    images.to_csv(out/'images.csv', index=False)
    offsets, rows = interventions()
    np.save(out/'coefficient_offsets.npy', offsets)
    cache, metas = {}, []
    maximum_deviation = 0.0
    max_reconstruction = 0.0
    for j, row in images.iterrows():
        original_patch, meta = extract_patch(row, canvas_cache=cache, patch_size_px=540)
        canvas, ppd, _ = cache[(str(row.session), int(row.trial_idx))]
        lo, hi = np.percentile(original_patch, [0.5, 99.5])
        if hi <= lo:
            raise ValueError('Constant scene')
        full = np.clip((canvas-lo)/(hi-lo), 0, 1).astype(np.float32)*255
        bands, dc = laplacian_bands(full)
        error = float(np.max(np.abs(bands.sum(axis=0)+dc-full)))
        max_reconstruction = max(max_reconstruction, error)
        center = (meta['patch_center_x_px'], meta['patch_center_y_px'])
        patch = _clip_patch(full, center, 540)
        patch_bands = np.stack([_clip_patch(b, center, 540) for b in bands])
        for d in offsets:
            variant = patch + np.einsum('k,kyx->yx', d, patch_bands)
            maximum_deviation = max(maximum_deviation, float(np.abs(variant-127.5).max()))
        fields = np.concatenate([patch[None], patch_bands])
        np.savez_compressed(out/'scenes'/f'scene_{j:03d}.npz', fields=fields,
            original_patch=original_patch, original_full_preview=full[::2, ::2],
            full_canvas_shape=np.array(full.shape), normalization_percentiles=np.array([lo, hi]),
            dc=dc, reconstruction_error=error, source_ppd=ppd)
        metas.append({'scene': int(j), **meta, 'source_canvas_sha256':
            __import__('hashlib').sha256(canvas.tobytes()).hexdigest(),
            'pyramid_reconstruction_max_error_uint': error})
        print(f'Prepared scene {j+1}/{len(images)}', flush=True)
    # One global affine contrast factor, fixed for every scene/trace/probe.
    # This leaves headroom without clipping any intervention or renormalizing it.
    contrast = min(1.0, 126.0/maximum_deviation)
    source_files = {name: {'path': str(path), 'sha256': sha256(path)} for name, path in
        [('selected_bundle', selected_path), ('figure4_summary', source_path),
         ('dataset_config', source['dataset_config']), *source['inputs'].items()]
        if isinstance(path, (str, Path))}
    manifest = {'analysis': 'Laplacian scene coefficient tuning conditional on native FEM histories',
        'checkpoint': source['checkpoint'], 'checkpoint_sha256': selected['checkpoint_sha256'],
        'dataset_config': source['dataset_config'], 'population_version': source['population_version'],
        'population_spec_dir': str(bundle/'all_available_population_spec'),
        'tuning_table': str(bundle/'all_available_yu_tuning/tuning_summary.csv'),
        'n_units': 725, 'n_scenes': len(images), 'n_traces': len(traces),
        'event_counts': table.event_class.value_counts().to_dict(), 'history_frames': 60,
        'input_rate_hz': 240, 'output_rate_hz': 240, 'output_bins_per_history': 1,
        'eye_alignment': 'subtract final eye position; latest retinal frame identical across trajectories',
        'stationary_control': 'all 60 eye positions zero, sharing the endpoint image',
        'pyramid': {'kind': 'OpenCV binomial Gaussian reduce/expand Laplacian pyramid',
            'detail_levels': LEVELS, 'n_bands': LEVELS+1, 'order': 'finest to coarsest residual',
            'decomposition_domain': 'entire recorded full-screen luminance canvas before cropping',
            'dc': 'fixed; every expanded full-canvas band is mean-subtracted',
            'max_reconstruction_error_uint': max_reconstruction},
        'strengths': STRENGTHS.tolist(), 'interventions': rows,
        'normalization': {'initial': 'original 540px patch 0.5/99.5 percentiles applied once to full canvas',
            'global_contrast_factor': contrast, 'center_uint': 127.5,
            'model_input': '(fixed reference or modified image - 127)/255',
            'no_per_probe_normalization': True, 'no_per_probe_clipping': True,
            'maximum_pre_headroom_deviation': maximum_deviation},
        'response': 'Hz per exact neuron at central translated Figure4 readout; one endpoint bin',
        'population_scope': '725 exact-CID model readouts pooled across sessions; not simultaneous recorded population',
        'behavior': 'zero, identical to Figure 4 counterfactual replay',
        'seed': SEED, 'source_files': source_files, 'scenes': metas}
    manifest['input_files'] = {str(p.relative_to(out)): sha256(p) for p in
        [out/'traces_endpoint_aligned.npy', out/'coefficient_offsets.npy', out/'images.csv', out/'traces.csv',
         *sorted((out/'scenes').glob('*.npz'))]}
    write_json(out/'design.json', manifest)
    print(f'Global fixed contrast factor: {contrast:.6f}; pyramid error {max_reconstruction:.3g}', flush=True)


if __name__ == '__main__':
    main()
