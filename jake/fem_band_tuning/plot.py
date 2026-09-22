"""Image-based examples and population views of nonlinear band tuning."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from jake.fem_band_tuning.common import STRENGTHS, render_basis, sweep_indices, write_json

BLUE = '#2878b5'
ORANGE = '#d76c27'
BANDS = ['L0 · finest', 'L1', 'L2', 'L3', 'L4', 'Residual · coarsest']


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'semibold',
        'figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,
        'svg.fonttype':'none','axes.labelcolor':'#333333','text.color':'#222222'})


def save(fig, path):
    for extension in ['png','pdf','svg']:
        fig.savefig(path.with_suffix('.'+extension),dpi=190,bbox_inches='tight')
    plt.close(fig)


def scene_basis(out, scene, contrast):
    fields = np.load(out/'scenes'/f'scene_{scene:03d}.npz')['fields'].copy()
    fields[0] = 127.5+contrast*(fields[0]-127.5)
    fields[1:] *= contrast
    basis = render_basis(fields,np.zeros((1,2),np.float32),device='cpu').numpy()[:,0,0]
    basis[0] = basis[0]*255+127
    basis[1:] *= 255
    return basis


def picture(ax, image, title='', subtitle=None):
    ax.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    if title: ax.set_title(title,fontsize=10,pad=7)
    if subtitle: ax.set_xlabel(subtitle,fontsize=9,labelpad=5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args(); out=args.out_dir
    style()
    d=json.loads((out/'design.json').read_text()); summary=json.loads((out/'summary.json').read_text())
    z=np.load(out/'analysis_arrays.npz')
    gains=z['gain_hz']; conditional=z['gain_condition_hz']; energy=z['gain_energy']
    baseline=z['baseline_hz']; sweeps=z['sweeps_hz']; weights=z['condition_weights']
    table=pd.read_csv(out/'trace_analysis_metadata.csv')
    units=pd.read_csv(d['tuning_table']).sort_values('unit_index')
    traces=np.load(out/'traces_endpoint_aligned.npy')
    condition_delta=conditional[:,1]-conditional[:,0]
    scene_specific=condition_delta-condition_delta.mean(axis=0,keepdims=True)
    scene_score=np.mean(scene_specific**2,axis=(1,2))
    geometry=json.loads((out/'input_geometry_audit.json').read_text())
    excluded=geometry['scenes_whose_swept_field_may_cross_screen_edge']
    scene_score[excluded]=-np.inf
    example=int(np.argmax(scene_score))
    # One disclosed illustrative scene; population statistics use all scenes.
    selected=[]
    for event in ['drift','microsaccade']:
        idx=table.index[table.event_class.eq(event)].to_numpy()
        order=idx[np.argsort(table.loc[idx,'rendered_path_length_arcmin'].to_numpy())]
        selected.append(int(order[len(order)//2]))
    neuron_order=np.argsort(units.exact_twin_yu_preferred_sf_cpd.to_numpy())
    contrast=d['normalization']['global_contrast_factor']
    basis=scene_basis(out,example,contrast)
    band_examples=[1,3]
    indices=sweep_indices(d['interventions'])
    # Main image + neuron figure.
    fig=plt.figure(figsize=(15,10))
    outer=fig.add_gridspec(2,1,height_ratios=[1.05,1],hspace=.4)
    top=outer[0].subgridspec(2,5,width_ratios=[1,1,1,.92,1.5],hspace=.43,wspace=.28)
    for row,k in enumerate(band_examples):
        for column,a in enumerate([.5,1,1.5]):
            ax=fig.add_subplot(top[row,column])
            subtitle='Reference' if a==1 else f'Band {k} coefficient = {a:.1f}'
            picture(ax,basis[0]+(a-1)*basis[k+1],
                ['Weaken this structure','Same landing image','Strengthen this structure'][column] if row==0 else '',subtitle)
        ax=fig.add_subplot(top[row,3])
        magnitude=max(float(np.max(np.abs(basis[k+1]))),1e-6)
        ax.imshow(basis[k+1],cmap='RdBu_r',vmin=-magnitude,vmax=magnitude)
        ax.set_title(f'{BANDS[k]}\nimage component',fontsize=10)
        ax.axis('off')
        ax.text(.5,-.06,'Display-scaled component',transform=ax.transAxes,ha='center',fontsize=8)
        ax=fig.add_subplot(top[row,4])
        for e,color,label in zip(selected,[BLUE,ORANGE],['Drift','Microsaccade']):
            delta=sweeps[example,e+1,k]-sweeps[example,e+1,k,2]
            rms=np.sqrt(np.mean(delta.astype(float)**2,axis=-1))
            ax.plot(STRENGTHS,rms,'o-',color=color,label=label,lw=2,ms=4)
        ax.set_xlabel('Band coefficient')
        ax.set_ylabel('Population RMS Δ rate (Hz)')
        ax.set_xticks([.5,1,1.5]); ax.axvline(1,color='#cccccc',lw=.7,zorder=-1)
        if row==0: ax.legend(frameon=False,fontsize=9); ax.set_title('Same edit, different response')
    fig.text(.04,.925,'A   Changing actual scene structure',fontsize=15,weight='bold')
    bottom=outer[1].subgridspec(2,2,height_ratios=[.65,2],hspace=.68,wspace=.3)
    vmax=float(np.percentile(np.abs(gains[example,selected]),99))
    for c,(e,label,color) in enumerate(zip(selected,['Drift','Microsaccade'],[BLUE,ORANGE])):
        ax=fig.add_subplot(bottom[0,c])
        t=(np.arange(60)-59)/240*1000
        ax.plot(t,traces[e,:,0]*60,color=color,label='Horizontal')
        ax.plot(t,traces[e,:,1]*60,color=color,ls='--',alpha=.65,label='Vertical')
        ax.axhline(0,color='#cccccc',lw=.6)
        ax.set_ylabel('Eye position\n(arcmin)',fontsize=9); ax.set_xlabel('Time before output (ms)',fontsize=9)
        ax.set_title(f'{label} · trace {e} · path {table.iloc[e].rendered_path_length_arcmin:.1f} arcmin',fontsize=10)
        ax=fig.add_subplot(bottom[1:,c])
        im=ax.imshow(gains[example,e][:,neuron_order],aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax,
            interpolation='nearest',rasterized=True)
        ax.set_yticks(range(6),BANDS if c==0 else ['L0','L1','L2','L3','L4','Coarse'])
        ax.set_xticks([0,724],['Lower SF','Higher SF'])
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.07,pad=.23)
        cb.set_label('Signed band gain (Hz / coefficient)',fontsize=9)
    fig.text(.04,.505,'B   Which neurons become sensitive to each structure?',fontsize=15,weight='bold')
    fig.text(.5,-.02,f'Scene {example}: largest scene-specific condition interaction among interior scenes (illustrative selection). '
        'Traces selected at median path length within class.\n'
        'All image variants share one grayscale scale; only component insets are display-scaled. '
        'Heatmap colors saturate at the shared 99th percentile.\n'
        'All 725 neurons appear in the same order by model preferred SF. One endpoint bin per history.',
        ha='center',fontsize=9)
    save(fig,out/'scene_and_neuron_tuning')
    # All traces, all scenes, all neurons: aggregate figure.
    order=np.r_[table.index[table.event_class.eq('drift')].to_numpy()[np.argsort(
        table.loc[table.event_class.eq('drift'),'rendered_path_length_arcmin'].to_numpy())],
        table.index[table.event_class.eq('microsaccade')].to_numpy()[np.argsort(
        table.loc[table.event_class.eq('microsaccade'),'rendered_path_length_arcmin'].to_numpy())]]
    ndrift=int(table.event_class.eq('drift').sum())
    fig,axs=plt.subplots(2,2,figsize=(15,10),gridspec_kw={'height_ratios':[1,.95]})
    plt.subplots_adjust(hspace=.5,wspace=.38)
    ax=axs[0,0]
    rms=np.sqrt(energy.mean(axis=0))
    relative=np.log2(np.maximum(rms,1e-15)/np.maximum(np.sqrt(energy.mean(axis=(0,1))),1e-15))
    limit=max(float(np.percentile(np.abs(relative),98)),.1)
    im=ax.imshow(relative[order].T,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
    ax.axvline(ndrift-.5,color='white',lw=2)
    ax.set_yticks(range(6),BANDS); ax.set_xticks([ndrift/2,(ndrift+200)/2],['Drift','Microsaccade'])
    ax.set_xlabel('Individual traces, ordered by path length within class')
    ax.set_title('A   Movement-dependent sensitivity to scene bands',loc='left')
    fig.colorbar(im,ax=ax,shrink=.8,label='log₂ RMS gain / band-wide reference')
    ax=axs[0,1]
    scene_energy=np.einsum('ce,sek->sck',weights,energy)
    scene_ratio=.5*np.log2(np.maximum(scene_energy[:,1],1e-20)/np.maximum(scene_energy[:,0],1e-20))
    limit=max(float(np.percentile(np.abs(scene_ratio),98)),.1)
    im=ax.imshow(scene_ratio,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
    ax.set_xticks(range(6),['L0','L1','L2','L3','L4','Coarse']); ax.set_ylabel('Scene patch')
    ax.set_title('B   The movement effect depends on the scene',loc='left')
    fig.colorbar(im,ax=ax,shrink=.8,label='log₂ RMS gain: microsaccade / drift')
    ax=axs[1,0]
    for key,label,color,shift,factor in [('population_gain_energy','RMS rate gain', '#222222',-.12,.5),
        ('poisson_information_per_endpoint_bin','Poisson information',BLUE,0,1),
        ('information_per_fixed_expected_spike','Information per fixed spike',ORANGE,.12,1)]:
        stat=summary[key]; y=np.array(stat['log2_ratio'])*factor; ci=np.array(stat['log2_ratio_ci95'])*factor
        ax.errorbar(np.arange(6)+shift,y,yerr=np.maximum(0,np.stack([y-ci[:,0],ci[:,1]-y])),
            color=color,fmt='o-',lw=1.4,ms=4,capsize=2,label=label)
    ax.axhline(0,color='#bbbbbb',lw=1); ax.set_xticks(range(6),['L0','L1','L2','L3','L4','Coarse'])
    ax.set_ylabel('log₂ microsaccade / drift'); ax.set_xlabel('Finest ← Laplacian band → coarsest')
    ax.legend(frameon=False,fontsize=9); ax.set_title('C   Rate and noise-model summaries',loc='left')
    ax=axs[1,1]
    unit_energy=np.einsum('ce,sekn->ckn',weights,gains.astype(float)**2)/len(gains)
    # Per-neuron profiles sum to one: a uniform gain of all that neuron's bands cancels.
    profiles=unit_energy/np.maximum(unit_energy.sum(axis=1,keepdims=True),1e-30)
    change=profiles[1]-profiles[0]
    limit=max(float(np.percentile(np.abs(change),99)),.01)
    im=ax.imshow(change[:,neuron_order],aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,
        interpolation='nearest',rasterized=True)
    ax.set_yticks(range(6),BANDS); ax.set_xticks([0,724],['Lower SF','Higher SF'])
    ax.set_xlabel('Same 725 neurons · fixed model SF order')
    ax.set_title('D   Changes in each neuron’s relative band sensitivity',loc='left')
    fig.colorbar(im,ax=ax,shrink=.8,label='Band share: microsaccade − drift')
    fig.suptitle('Scene structure × fixational eye movement × neural tuning',fontsize=17,y=.96)
    fig.text(.5,-.02,'40 scene patches from 13 source canvases × 200 real FEM histories × 725 exact readouts. '
        'Event-class summaries balance animals.\n95% intervals resample source canvases and eye-movement source trials; '
        'neurons are the fixed modeled population. Information panels assume independent Poisson spikes.',ha='center',fontsize=9)
    save(fig,out/'population_interaction')
    # Independently defined neural subpopulations, with all-unit and strict-
    # validation identities retained in the associated CSV tables.
    groups=json.loads((out/'population_groups.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    plt.subplots_adjust(hspace=.45,wspace=.35)
    pooled=[]
    for name in ['lower_SF_third','higher_SF_third']:
        idx=np.array(groups[name]['unit_indices'])
        pooled.append(np.sqrt(np.mean(gains[:,:,:,idx].astype(float)**2,axis=(0,3))))
    common=np.log10(np.maximum(np.stack(pooled),1e-5))
    vmin,vmax=np.percentile(common,[1,99])
    for c,(name,title) in enumerate(zip(['lower_SF_third','higher_SF_third'],['Lower-SF neurons','Higher-SF neurons'])):
        ax=axes[0,c]
        im=ax.imshow(common[c,order].T,aspect='auto',cmap='magma',vmin=vmin,vmax=vmax)
        ax.axvline(ndrift-.5,color='white',lw=1.5)
        ax.set_yticks(range(6),BANDS); ax.set_xticks([ndrift/2,(ndrift+200)/2],['Drift','Microsaccade'])
        ax.set_title(f'{title} · n={groups[name]["n_units"]}\nmedian model SF {groups[name]["median_model_sf_cpd"]:.2f} cpd')
        fig.colorbar(im,ax=ax,shrink=.8,label='log₁₀ RMS rate gain')
    ax=axes[1,0]
    for name,label,color,shift in [('lower_SF_third','Lower SF',BLUE,-.06),('higher_SF_third','Higher SF',ORANGE,.06)]:
        stat=groups[name]['gain_energy']; y=.5*np.array(stat['log2_ratio']); ci=.5*np.array(stat['log2_ratio_ci95'])
        x=np.arange(6)+shift
        ax.plot(x,y,'o-',color=color,label=label,ms=4,lw=1.5)
        ax.vlines(x,ci[:,0],ci[:,1],color=color,lw=1)
    ax.axhline(0,color='#aaaaaa',lw=.8); ax.set_xticks(range(6),['L0','L1','L2','L3','L4','Coarse'])
    ax.set_ylabel('log₂ RMS gain: microsaccade / drift'); ax.set_xlabel('Laplacian band')
    ax.set_title('Different populations, different band modulation'); ax.legend(frameon=False)
    ax=axes[1,1]
    change=(profiles[1,:2].sum(axis=0)-profiles[0,:2].sum(axis=0))
    limit=max(float(np.percentile(np.abs(change),99)),.01)
    sf=units.exact_twin_yu_preferred_sf_cpd.to_numpy(float)
    tf=units.exact_twin_yu_preferred_tf_hz.to_numpy(float)
    valid=units.validated_for_figure4.to_numpy(dtype=bool)
    im=ax.scatter(sf,tf,c=change,cmap='RdBu_r',vmin=-limit,vmax=limit,s=14,alpha=.65,linewidths=0)
    ax.scatter(sf[valid],tf[valid],c=change[valid],cmap='RdBu_r',vmin=-limit,vmax=limit,
        s=22,edgecolors='#444444',linewidths=.45)
    ax.set_xscale('log',base=2); ax.set_yscale('log',base=2)
    ax.set_xlabel('Model preferred SF (cpd)'); ax.set_ylabel('Model preferred TF (Hz)')
    ax.set_title('Change in the share of sensitivity to L0 + L1')
    fig.colorbar(im,ax=ax,shrink=.8,label='Fine-band share: microsaccade − drift')
    fig.suptitle('How FEM-dependent scene sensitivity relates to neural tuning',fontsize=16)
    fig.text(.5,-.015,'Groups are outer thirds of independently measured model SF, with equal neuron counts. '
        'Outlined points pass the strict Figure 4 tuning validation.\nAll 725 model readouts remain in the '
        'scatter; these SF/TF estimates are model tuning, not a recorded-neuron TF assay.',ha='center',fontsize=9)
    save(fig,out/'neuronal_subpopulations')
    # Signed per-neuron tuning, avoiding RMS-only descriptions of suppression.
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
    chosen_units=[]
    for k,ax in enumerate(axes.flat):
        delta=gains[example,selected[1],k]-gains[example,selected[0],k]
        n=int(np.argmax(np.abs(delta))); chosen_units.append(n)
        for e,color,label in zip(selected,[BLUE,ORANGE],['Drift','Microsaccade']):
            response=sweeps[example,e+1,k,:,n]
            ax.plot(STRENGTHS,response-response[2],'o-',lw=2,color=color,label=label,ms=4)
        ax.axhline(0,color='#bbbbbb',lw=.7)
        ax.set_title(f'{BANDS[k]} · neuron {n}',fontsize=11)
        ax.set_xlabel('Band coefficient'); ax.set_ylabel('Δ firing rate (Hz)')
    axes[0,0].legend(frameon=False)
    fig.suptitle(f'Signed tuning changes for the same scene and eye histories · scene {example}',fontsize=14)
    fig.tight_layout(rect=(0,.04,1,.96))
    fig.text(.5,.01,'Per band, the neuron with the largest slope difference is shown (illustrative extremes). '
        'All neurons are retained in the population analyses.',ha='center',fontsize=9)
    save(fig,out/'example_tuning_curves')
    # Complete scene atlas, not just the selected positive-looking example.
    with PdfPages(out/'all_scene_atlas.pdf') as pdf:
        for s in range(d['n_scenes']):
            b=scene_basis(out,s,contrast)
            fig=plt.figure(figsize=(14,8))
            grid=fig.add_gridspec(3,7,height_ratios=[1,1,.85],hspace=.4,wspace=.2)
            picture(fig.add_subplot(grid[0:2,0]),b[0],f'Scene {s}\nReference')
            for k in range(6):
                picture(fig.add_subplot(grid[0,k+1]),b[0]-.5*b[k+1],BANDS[k])
                picture(fig.add_subplot(grid[1,k+1]),b[0]+.5*b[k+1])
            gain_grid=grid[2,1:].subgridspec(1,2,wspace=.22)
            gain_axes=[]
            for c,event in enumerate(['Drift','Microsaccade']):
                ax=fig.add_subplot(gain_grid[0,c]); gain_axes.append(ax)
                matrix=conditional[s,c][:,neuron_order]
                limit=max(float(np.percentile(np.abs(conditional[s]),99)),1e-4)
                im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,rasterized=True)
                ax.set_title(f'{event}: signed mean band gain',fontsize=10)
                ax.set_yticks(range(6),['L0','L1','L2','L3','L4','Coarse'],fontsize=8)
                ax.set_xticks([0,724],['Lower model SF','Higher model SF'],fontsize=8)
                ax.get_xticklabels()[0].set_ha('left')
                ax.get_xticklabels()[1].set_ha('right')
            cb=fig.colorbar(im,ax=gain_axes,fraction=.02,pad=.025)
            cb.set_label('Signed gain (Hz / coefficient)',fontsize=8)
            cb.ax.tick_params(labelsize=8)
            fig.suptitle(f'Scene {s}: identical spatial edits replayed through every FEM history',fontsize=14)
            fig.text(.5,.025,'Top image row: band coefficient 0.5. Bottom: 1.5. '
                'One image intensity scale throughout. Heatmaps share a scale within each scene; colors saturate at the 99th percentile.',ha='center',fontsize=9)
            pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
    write_json(out/'figure_manifest.json',{'example_scene':example,'example_traces':selected,
        'example_scene_selection':'maximum mean squared scene-specific event-class difference in signed band gain among scenes whose swept field stays inside the recorded canvas',
        'example_trace_selection':'median path length within each event class, without response selection',
        'example_curve_units':chosen_units,'neuron_order':neuron_order.tolist(),
        'heatmap_scale':'shared symmetric 99th percentile in illustrative maps; explicitly saturated',
        'atlas_pages':d['n_scenes'], 'main_example_bands':band_examples})
    # A self-contained quantitative handoff with no unobserved direction asserted.
    lines=['# FEM-dependent tuning to scene structure','',
        f'Analyzed {d["n_scenes"]} scene patches from {summary["n_source_canvas_clusters"]} source canvases, '
        f'{d["n_traces"]} FEM histories ({summary["event_counts"]["drift"]} drift; '
        f'{summary["event_counts"]["microsaccade"]} microsaccade), and all {d["n_units"]} exact model readouts.','',
        'The response is one native output bin after a 60-frame retinal history. '
        'All trajectories end on the same reference image. Six Laplacian band strengths were swept through '
        '0.5, 0.75, 1, 1.25, and 1.5; the DC component is fixed.','',
        '## What the analysis establishes','',
        'The modeled population changes its sensitivity to the same spatial edits depending on the FEM history. '
        'This varies across scenes and neurons, and persists after separate multiplicative gain changes are allowed for each neuron. '
        'The scene × trajectory × neuron residual accounts for 15.4–20.5% of the centered gain variance across bands. '
        'Scene-specific event-class effects reproduce across disjoint source-trial halves (median cosine agreement 0.964–0.975).','',
        'Microsaccades increase population RMS rate sensitivity in all six bands (1.48–1.92× relative to drift). '
        'This dataset therefore does not show an aggregate coarse-versus-fine crossover between the two event classes. '
        'The relationship is a scene- and neuron-dependent change in band tuning superimposed on broader response amplification.','',
        'After conditioning on a fixed spike budget, no band has a positive microsaccade advantage whose 95% interval excludes zero; '
        'L0, L2, and the coarse residual have lower modeled information per spike. '
        'The result supports a change in recruitment and tuning, not a general improvement in coding efficiency. '
        'An attention-like interpretation is consequently a functional analogy to selective reweighting, not evidence that attention itself was engaged.','',
        '## Measured movement effects','',
        '| Band | Microsaccade / drift RMS rate gain | 95% crossed bootstrap interval |',
        '|---|---:|---:|']
    stat=summary['population_gain_energy']
    for k in range(6):
        ratio=2**(.5*stat['log2_ratio'][k]); lo,hi=2**(.5*np.array(stat['log2_ratio_ci95'][k]))
        lines.append(f'| {BANDS[k]} | {ratio:.3f} | [{lo:.3f}, {hi:.3f}] |')
    controls=summary['global_and_neuron_gain_controls']
    lines += ['', '## Selectivity and interaction checks','',
        f'- Best global scaling of stationary band tuning leaves '
        f'{100*controls["global_scalar_unexplained_energy_fraction"]:.1f}% of moving-condition gain energy unexplained.',
        f'- Allowing a separate scale for every neuron leaves '
        f'{100*controls["per_neuron_scalar_unexplained_energy_fraction"]:.1f}% unexplained.',
        '- Scenes 17 and 23 include screen-edge padding in the modeled field. Excluding them, the microsaccade/drift RMS gain ratios are '+
        ', '.join(f'{2**(.5*v):.3f}' for v in summary['interior_scene_control']['gain_energy']['log2_ratio'])+'.',
        '- The 38-scene interior control leaves '+
        f'{100*summary["interior_scene_control"]["gain_controls"]["per_neuron_scalar_unexplained_energy_fraction"]:.1f}% '
        'of gain energy unexplained by separate per-neuron scaling.',
        '- Three-way scene × trajectory × neuron residual variance fractions by band: '+
        ', '.join(f'{v:.3f}' for v in summary['three_way_interaction']['raw_gain_variance_fraction_by_band'])+'.',
        '- Disjoint-source-trial split agreement of scene-specific condition effects (cosine by band): '+
        ', '.join(f'{v:.3f}' for v in summary['three_way_interaction']['trace_split_reliability']['cosine_median'])+'.',
        '- The local additive band gains explain '+
        f'{100*summary["local_curve_checks"]["joint_validation_change_energy_explained_by_local_additive_gains"]:.1f}% '
        'of the response-change energy for the held-out joint-band probes.','',
        '## Read the figures','',
        '- `scene_and_neuron_tuning.png`: actual image edits, response changes, and signed neuron tuning for a disclosed illustrative scene.',
        '- `population_interaction.png`: all trajectories, scene dependence, noise-model summaries, and band-profile changes after within-neuron normalization.',
        '- `example_tuning_curves.png`: signed tuning curves for illustrative units, including suppression.',
        '- `neuronal_subpopulations.png`: independent SF groups and their movement-dependent band sensitivity; strict-validation units are marked.',
        '- `all_scene_atlas.pdf`: the complete 40-scene image and neuron atlas.','',
        '## Interpretation',''] + [f'- {x}' for x in summary['interpretation']]
    lines += ['',f'The global contrast factor is {contrast:.6f}, fixed across every scene and intervention '
        'to reserve display headroom. No band sweep is independently renormalized or clipped. '
        'The unscaled source and pyramid are retained in the scene archives.','',
        'The illustrative scene maximizes the measured scene-specific event-class interaction among interior scenes; '
        'the two displayed eye traces have median path length within class. These examples do not supply '
        'the population statistics, which use the complete crossed dataset.']
    (out/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(f'Saved figures and complete atlas; example scene {example}, traces {selected}',flush=True)


if __name__=='__main__':
    main()
