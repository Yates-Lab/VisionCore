"""Illustrated feedforward twin, with textures from the selected model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from matplotlib.transforms import Affine2D, Bbox

INK = '#303b45'
MUTED = '#617080'
STAGES = ('#5791ad', '#5679a4', '#8174a2')
BEHAVIOR = '#398777'
PHASE = '#ba8741'


def _model_maps(assets, *, no_phase_preview=False):
    """Evaluate one cached retinal history; all displayed maps are real outputs."""
    import torch
    from models.modules.dekel import DekelCore
    from models.data.transforms import _make_pixelnorm
    from _fig3_data import CHECKPOINT_PATH, FIG_DIR
    from _fig3a_data import PANEL_A_CACHE_PATH
    from generate_figure3 import INTACT_COLOR, ABLATED_COLOR, STABILIZED_COLOR

    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    model_config = checkpoint['hyper_parameters']['model_config_dict']
    config = dict(model_config['convnet']['params'])
    config['initial_channels'] = 1
    core = DekelCore(config).eval()
    prefix = 'model.convnet.'
    core.load_state_dict({k[len(prefix):]: v for k, v in checkpoint['state_dict'].items()
                          if k.startswith(prefix)}, strict=True)
    # The cached visualization is chronological; the model uses lag 0 first.
    history = np.ascontiguousarray(assets.lag_cube[::-1], dtype=np.float32)
    stabilized_history = np.ascontiguousarray(assets.stab_lag_cube[::-1], dtype=np.float32)
    if history.shape != (60, 35, 35) or stabilized_history.shape != history.shape:
        raise ValueError('The manuscript requires paired 60-frame, 35-pixel FixRSVP histories')
    with torch.inference_mode():
        stimulus = _make_pixelnorm({})(torch.from_numpy(history))[None, None]
        stages = core.forward_stages(stimulus)
        scaffold = [core._to_scaffold_size(stage)[0].numpy() for stage in stages]
        weights = core.effective_temporal_weight().numpy()[:, 0]
    maps = [stage[0].numpy() for stage in stages]
    indices = [np.argsort(v.var(axis=(1, 2)), kind='stable')[-4:] for v in maps]
    selected = [v[i] for v, i in zip(maps, indices)]
    pooled = [v[i[-1]] for v, i in zip(scaffold, indices)]
    peak = np.argmax(np.square(weights).sum(axis=(2, 3)), axis=1)
    filters = weights[np.arange(len(weights)), peak]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIG_DIR/'architecture_maps.npz', history=history,
                        stabilized_history=stabilized_history,
                        **{f'stage{i+1}': v for i, v in enumerate(selected)},
                        filters=filters)
    record = {
        'checkpoint_sha256': hashlib.sha256(Path(CHECKPOINT_PATH).read_bytes()).hexdigest(),
        'history_sha256': hashlib.sha256(history.tobytes()).hexdigest(),
        'history_order': 'lag 0 first, reversed from cached chronological FixRSVP lag_cube',
        'history_source': {
            'condition': 'fixrsvp', 'session': assets.session,
            'cache': str(PANEL_A_CACHE_PATH),
            'cache_sha256': hashlib.sha256(PANEL_A_CACHE_PATH.read_bytes()).hexdigest(),
            'display_alignment': 'paired cached illustrations share a most-recent frame aligned to the test-screen ROI',
            'stabilized_history_sha256': hashlib.sha256(stabilized_history.tobytes()).hexdigest(),
        },
        'behavior_source': 'separate cached FixRSVP eye-position and speed example from the same session',
        'preprocessing': '35-pixel cached crop; production pixelnorm: (pixel - 127) / 255',
        'feature_channels': [i.tolist() for i in indices],
        'filter_channels': [0, 5, 10],
        'selection': 'four channels with greatest spatial variance at each stage',
        'display': 'each displayed feature map independently contrast-normalized',
        'readout_glyphs': 'schematic spatial and feature weights, not fitted coefficients',
        'example_psth': {key: assets.psth_neurons[0][key] for key in ('session', 'neuron_id')},
        'architecture': {
            'stage_shapes': [list(value.shape) for value in maps],
            'readout_rank': model_config['readout']['params'].get('rank', 1),
            'phase_readout_rank': (model_config.get('phase_readout') or {}).get('params', {}).get('rank', 0),
        },
        'hypothesis_tests': {
            'weights': 'same fitted weights for all three input conditions',
            'full': {'color': INTACT_COLOR, 'retinal': 'measured motion',
                     'extraretinal': 'retained'},
            'retinal_only': {'color': ABLATED_COLOR, 'retinal': 'measured motion',
                            'extraretinal': 'input zeroed before the behavioral MLP'},
            'stabilized': {'color': STABILIZED_COLOR, 'retinal': 'gaze-induced motion removed',
                           'extraretinal': 'retained'},
            'stabilized_cube': 'paired FixRSVP history with image flashes preserved and ROI held at the example trial medoid; quantified ablation uses the session-global gaze centroid',
            'shared_input_arrows': 'one neutral retinal arrow and one behavioral arrow; colors mark condition labels and the two input interventions',
        },
    }
    record['schematic_architecture'] = dict(record['architecture'])
    if no_phase_preview:
        record['schematic_architecture'].update(
            phase_readout_rank=0, readout_rank=None, readout_rank_options=[1, 2])
        record['draft_note'] = (
            'Revised architecture preview. Feature-map textures and the example '
            'prediction retain the recorded source checkpoint; the rank-1 and '
            'rank-2 models are being retrained. Numerical results are unchanged.')
    (FIG_DIR/'architecture_maps.json').write_text(json.dumps(record, indent=2)+'\n')
    return selected, pooled, filters, record['schematic_architecture']


def _label(ax, x, y, text, *, size=8.5, color=INK, weight='normal', ha='center', va='center'):
    return ax.text(x, y, text, fontsize=size, color=color, weight=weight,
                   ha=ha, va=va, linespacing=1.25, zorder=20)


def _arrow(ax, points, color=INK, lw=.9, style='-'):
    points = np.asarray(points)
    if len(points) > 2:
        ax.plot(points[:-1, 0], points[:-1, 1], color=color, lw=lw,
                solid_joinstyle='round', linestyle=style, zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                mutation_scale=7, lw=lw, color=color,
                                linestyle=style, shrinkA=0, shrinkB=0, zorder=2))


def _sheet(ax, values, x, y, w, h, color, *, signed=False, z=5):
    """A textured feature plane with the same oblique projection throughout."""
    transform = Affine2D.from_values(w, .24*w, 0, h, x, y) + ax.transData
    quad = np.array([[x, y], [x+w, y+.24*w], [x+w, y+h+.24*w], [x, y+h]])
    border = Polygon(quad, facecolor='white', edgecolor=color, lw=.65, zorder=z)
    ax.add_patch(border)
    values = np.asarray(values)
    high = max(float(np.max(np.abs(values))), 1e-8)
    if signed:
        cmap = 'RdBu_r'; low = -high
    else:
        cmap = LinearSegmentedColormap.from_list('feature', ['#f5f8fa', color, '#273e53'])
        low = 0
    im = ax.imshow(values, cmap=cmap, vmin=low, vmax=high, origin='upper',
                   extent=(0, 1, 0, 1), transform=transform, interpolation='nearest',
                   zorder=z+.1, aspect='auto')
    im.set_clip_path(border)
    ax.add_patch(Polygon(quad, fill=False, edgecolor=color, lw=.65, zorder=z+.2))


def _stack(ax, maps, x, y, w, h, colors, *, signed=False):
    for i, values in enumerate(maps):
        depth = len(maps)-i-1
        _sheet(ax, values, x-.14*depth, y+.13*depth, w, h,
               colors[i] if not isinstance(colors, str) else colors,
               signed=signed, z=5+i)


def _readout(ax, x, y, color, rank=1):
    """Schematic factorization: spatial weighting × feature weighting."""
    yy, xx = np.mgrid[-1:1:9j, -1:1:9j]
    spatial = np.exp(-((xx-.1)**2+(yy+.12)**2)/.22)
    _stack(ax, [spatial]*rank, x, y, .65, .65, color)
    _label(ax, x+.95, y+.40, '×', size=11, color=color)
    for k, alpha in enumerate((.15, .65, .25, .9, .4, .12)):
        ax.add_patch(Polygon([[x+1.2,y+k*.14],[x+1.42,y+k*.14],
                              [x+1.42,y+(k+1)*.14],[x+1.2,y+(k+1)*.14]],
                             facecolor=color, alpha=alpha, edgecolor='white', lw=.3, zorder=8))


def _draw_model(ax, assets, *, no_phase_preview=False):
    from generate_fig3a import _draw_lag_cube, box_corners_3d
    from generate_figure3 import INTACT_COLOR, ABLATED_COLOR, STABILIZED_COLOR
    maps, pooled, filters, architecture = _model_maps(assets, no_phase_preview=no_phase_preview)
    phase_rank = architecture['phase_readout_rank']
    readout_rank = architecture['readout_rank']
    ax.set_xlim(-.35, 21.2)
    ax.set_ylim(.25, 10.3 if phase_rank else 9.05)
    ax.set_axis_off()
    _label(ax, -.25, 10.0 if phase_rank else 8.98, 'B', size=11,
           weight='bold', ha='left', va='top')
    # Align the bottoms of all column labels above the tallest feature stack.
    heading_y = 8.15

    # Main visual stream. Depth depicts channels; spatial resolution decreases.
    corners = box_corners_3d((1.15, 6.5, 0), (1.35, 1.35, .95),
                             yaw_deg=-25, pitch_deg=0, roll_deg=0)
    cube = np.asarray(assets.lag_cube)
    _draw_lag_cube(ax, cube[::-1], corners, outline=INTACT_COLOR, edge_width=.9)
    _label(ax, 1.0, heading_y, 'Retinal\nhistory', weight='bold', va='bottom')
    _label(ax, 1.0, 5.25, 'Measured\nmotion', color=INTACT_COLOR)
    # Both Full and Retinal only retain the measured retinal sequence.
    _arrow(ax, [(1.94, 6.5), (2.62, 6.5)])
    # A counterfactual replacement at the retinal input, not a separate core.
    stabilized = np.asarray(assets.stab_lag_cube)[::-1]
    corners = box_corners_3d((1.15, 3.0, 0), (1.35, 1.35, .95),
                             yaw_deg=-25, pitch_deg=0, roll_deg=0)
    _draw_lag_cube(ax, stabilized, corners, outline=STABILIZED_COLOR, edge_width=.9)
    _label(ax, 1.0, 4.48, 'Stabilized', color=STABILIZED_COLOR, weight='bold')
    _label(ax, 1.0, 1.47, 'Retinal motion\nremoved', color=STABILIZED_COLOR)
    _arrow(ax, [(1.94, 3.0), (2.17, 4.10), (2.17, 5.75), (2.62, 6.15)],
           STABILIZED_COLOR, style='--')
    _stack(ax, filters[[0, 5, 10]], 3.02, 5.9, .78, .94, '#677a8a', signed=True)
    _label(ax, 3.4, heading_y, 'Spatiotemporal\nfilters', weight='bold', va='bottom')
    _label(ax, 3.4, 5.35, '14 filters\n60 × 7 × 7', color=MUTED)
    _arrow(ax, [(3.96, 6.5), (4.7, 6.5)])
    starts = (5.25, 7.6, 9.95)
    widths = (1.15, 1.0, .85)
    for i, (x, width, color) in enumerate(zip(starts, widths, STAGES)):
        _stack(ax, maps[i], x, 5.85, width, width*1.15, color)
        _label(ax, x+width/2-.1, heading_y, f'Stage {i+1}', weight='bold', color=color, va='bottom')
        _label(ax, x+width/2-.1, 5.25,
               f'{maps[i].shape[-2]} × {maps[i].shape[-1]}\n168 ch', color=MUTED)
        if i < 2:
            _arrow(ax, [(x+width+.1, 6.5), (starts[i+1]-.55, 6.5)])
            _label(ax, x+width+.48, 6.98, '↓2', color=MUTED)

    # Each stage is resampled and concatenated; three colors retain its origin.
    for i, x in enumerate(starts):
        _arrow(ax, [(x-.48, 5.7), (x-.48, 4.1-.20*i),
                    (12.7+.2*i, 4.1-.20*i), (12.7+.2*i, 5.60)], STAGES[i], lw=.8)
        ax.plot([x-.30,x-.48],[6.0,5.7], color=STAGES[i], lw=.8, zorder=1)
    _stack(ax, pooled, 12.55, 5.85, 1.16, 1.3, STAGES)
    _label(ax, 13.0, heading_y, 'Multiscale\nfeatures', weight='bold', va='bottom')
    _label(ax, 12.9, 3.02, '504 ch · 9 × 9', color=MUTED)

    # Behavioral modulation is a distinct lower path, feeding only this stream.
    _arrow(ax, [(13.9,6.5),(14.02,6.5),(14.02,7.18),(15.0,7.18),(15.0,6.77)])
    for y, symbol in ((6.5,'×'),(5.7,'∥')):
        ax.add_patch(Circle((15.0,y), .24, facecolor='white', edgecolor=BEHAVIOR, lw=.9, zorder=8))
        _label(ax,15.0,y,symbol,size=10,color=BEHAVIOR)
    _arrow(ax,[(15.0,6.24),(15.0,5.97)],BEHAVIOR)
    _label(ax,15.1,heading_y,'Behavioral\nmodulation',weight='bold',color=BEHAVIOR,va='bottom')
    _label(ax,15.42,6.90,'gain',ha='left',color=BEHAVIOR)
    _label(ax,15.42,5.10,'append 64',ha='left',color=BEHAVIOR)
    _arrow(ax,[(15.25,5.7),(16.75,5.7),(16.75,6.5),(17.25,6.5)])

    _readout(ax,17.55,6.05,INK,rank=readout_rank or 1)
    readout_label = 'Unit readout' if readout_rank is None else f'Unit readout\nrank {readout_rank}'
    _label(ax,18.3,heading_y,readout_label,weight='bold',va='bottom')
    _label(ax,18.3,5.6,'spatial × feature',color=MUTED)
    _arrow(ax,[(19.12,6.5),(19.8 if phase_rank else 20.05,6.5)])
    if phase_rank:
        ax.add_patch(Circle((20.05,6.5),.23,facecolor='white',edgecolor=INK,lw=.9,zorder=8))
        _label(ax,20.05,6.5,'+',size=11)
        _arrow(ax,[(6.37,7.25),(6.8,7.5),(6.8,9.05),(17.0,9.05)],PHASE)
        _label(ax,11.9,9.39,f'High-resolution stage-1 readout (rank {phase_rank})',color=PHASE)
        _readout(ax,17.55,8.66,PHASE,rank=phase_rank)
        _arrow(ax,[(19.12,9.05),(20.05,9.05),(20.05,6.77)],PHASE)

    # Measured eye traces and an MLP glyph, without a box around either.
    _label(ax,4.65,3.55,'Extraretinal input\nposition & velocity',weight='bold',color=BEHAVIOR)
    time = np.asarray(assets.behavior_t)
    traces = [np.asarray(assets.behavior_eyepos)[:,0],
              np.asarray(assets.behavior_eyepos)[:,1],
              np.asarray(assets.behavior_speed)]
    for i, values in enumerate(traces):
        values=(values-values.min())/max(float(np.ptp(values)),1e-8)
        ax.plot(3.15+(time-time[0])/max(float(time[-1]-time[0]),1e-8)*2.9,
                2.58-i*.47+values*.35,color=BEHAVIOR,lw=.75)
    # Full and Stabilized retain the measured extraretinal input. Retinal only
    # replaces that input with zero *before* the MLP, preserving its biases.
    _arrow(ax,[(6.18,2.32),(7.23,2.32)],BEHAVIOR)
    ax.add_patch(Circle((7.4,2.32),.16,facecolor='white',edgecolor=INK,lw=.7,zorder=8))
    _arrow(ax,[(7.56,2.32),(8.72,2.32)],BEHAVIOR)
    _label(ax,7.4,2.94,'Full',color=INTACT_COLOR,weight='bold')
    _arrow(ax,[(7.4,1.57),(7.4,2.14)],ABLATED_COLOR)
    _label(ax,7.4,1.16,'Retinal only',color=ABLATED_COLOR,weight='bold')
    _label(ax,7.4,.72,'input → 0',color=ABLATED_COLOR)
    layers=[[(x,y) for y in np.linspace(1.65,2.95,n)]
            for x,n in ((9.0,3),(9.69,5),(10.38,3))]
    for before,after in zip(layers[:-1],layers[1:]):
        for a in before:
            for b in after: ax.plot([a[0],b[0]],[a[1],b[1]],color='#bad5cb',lw=.35,zorder=1)
    for layer in layers:
        for pos in layer:ax.add_patch(Circle(pos,.075,facecolor='white',edgecolor=BEHAVIOR,lw=.6,zorder=3))
    _label(ax,9.69,3.35,'Behavioral MLP',weight='bold',color=BEHAVIOR)
    _label(ax,10.12,1.22,'42 → 128 → 64',color=BEHAVIOR)
    _arrow(ax,[(10.54,2.32),(14.4,2.32),(14.4,6.5),(14.73,6.5)],BEHAVIOR)
    _arrow(ax,[(14.4,5.7),(14.73,5.7)],BEHAVIOR)
    _label(ax,11.25,.60,'Same weights in all conditions',color=MUTED)

    # Softplus and one real held-out prediction illustrate the output.
    _arrow(ax,[(20.05,6.23 if phase_rank else 6.5),(20.05,5.0)])
    x=np.linspace(-3,3,70);y=np.logaddexp(0,x)
    ax.plot(19.50+(x+3)/6*1.0,4.40+y/3.05*.64,color=INK,lw=1.1)
    _label(ax,19.1,4.68,'Softplus',ha='right')
    _arrow(ax,[(20.05,4.35),(20.05,3.8)])
    _label(ax,18.40,3.42,'Example test prediction',weight='bold')
    p=assets.psth_neurons[0]
    t=np.asarray(p['t']); obs=np.asarray(p['robs_rate']);pred=np.asarray(p['rhat_rate'])
    maximum=max(float(obs.max()),float(pred.max()),1e-8)
    xx=16.5+(t-t[0])/(t[-1]-t[0])*4.1
    for values,color,lw in ((obs,'#777777',.7),(pred,INTACT_COLOR,1.0)):
        ax.plot(xx,1.22+values/maximum*1.45,color=color,lw=lw)
    ax.plot([16.5,16.5+.1/(t[-1]-t[0])*4.1],[.98,.98],color=INK,lw=1)
    _label(ax,16.9,.65,'100 ms',color=MUTED)
    _label(ax,18.6,.96,'recorded',color='#777777')
    _label(ax,20.1,.96,'Full',color=INTACT_COLOR)


def plot_panel_ab(ax, assets, *, no_phase_preview=False):
    from generate_fig3a import _draw_top_row, _fit_one_axes_in_rect, _data_aspect
    if assets.arch['model_family'] != 'dekel' or assets.arch['frontend_k'] != 60:
        raise ValueError('The manuscript schematic requires the 60-frame Dekel twin')
    fig=ax.figure; rect=ax.get_position(); ax.remove()
    top_rect=Bbox.from_bounds(rect.x0,rect.y0+.57*rect.height,rect.width,.43*rect.height)
    top=fig.add_axes(top_rect.bounds)
    bounds=_draw_top_row(top,assets,row_cy=0)
    top.set_xlim(bounds['x_left']-.3,bounds['x_right']+.3)
    top.set_ylim(bounds['y_bottom']-.5,bounds['y_top']+.5)
    top.set_aspect('equal');top.set_axis_off()
    _fit_one_axes_in_rect(fig,top,top_rect,_data_aspect(top))
    fig.text(rect.x0+.004,top_rect.y1-.006,'A',weight='bold',fontsize=11,va='top')
    b=fig.add_axes([rect.x0,rect.y0,rect.width,.55*rect.height])
    _draw_model(b,assets,no_phase_preview=no_phase_preview)
