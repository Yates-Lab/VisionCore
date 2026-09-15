#!/usr/bin/env python3
"""Audit figure fonts in the final LaTeX PDF and check reproduced statistics."""
from pathlib import Path
import hashlib
import json
import re
import pymupdf as fitz
from figure4_selection import SPECTRUM_SELECTION, spectrum_update
from analysis_selection import SELECTION, selected_analysis

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SELECTED=selected_analysis()
BUNDLE=ROOT/SELECTED['bundle']
FIGURES = [HERE/'figures'/f'figure{i}.pdf' for i in range(1,5)] + [
    HERE/'figures/supplement1.pdf', HERE/'old_figures/extended_fig2.pdf',
    HERE/'figures/stabilization_control.pdf']
RENDER_SOURCES = [
    'VisionCore/figure_typography.py',
    'paper/fig1/generate_fig1.py', 'paper/fig1/fig1a.svg',
    'paper/fig2/generate_figure2.py', 'paper/fig2/generate_panel_example.py',
    'paper/fig3/generate_fig3a.py', 'paper/fig3/generate_figure3.py',
    'paper/fig3/manuscript_schematic.py',
    'paper/fig3/manuscript_examples.py', 'paper/fig3/history_stabilization.py',
    'paper/fig3/run_history_stabilization.py', 'paper/fig3/_fig3_ablation_data.py',
    'paper/fig3/_fig3_femfraction.py',
    'manuscript/render_stabilization_control.py',
    'models/modules/dekel.py', 'models/data/transforms.py',
    'paper/fig4/spatiotemporal_tuning/_figure4_renderer.py',
    'paper/fig4/spatiotemporal_tuning/_figure4_rendering.py',
    'paper/fig4/spatiotemporal_tuning/audit_panel_a_exemplars.py',
    'manuscript/render_support.py', 'manuscript/render_figures.py',
    'manuscript/sync_stats.py', 'manuscript/export_empirical_stats.py',
    'manuscript/audit_figures.py', 'manuscript/Makefile',
    'manuscript/figure4_selection.py',
    'manuscript/analysis_selection.py',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spans(page):
    return [s for b in page.get_text('dict')['blocks'] if b['type']==0
            for line in b['lines'] for s in line['spans'] if s['text'].strip()]


def main():
    font_names=set()
    sources=[]
    errors=[]
    schematic_overlaps=[]
    schematic_image_overlaps=[]
    for path in FIGURES:
        doc=fitz.open(path)
        text=[s for page in doc for s in spans(page)]
        if not text:errors.append(f'No measurable figure text: {path}')
        font_names.update(s['font'] for s in text)
        for page in doc:
            for s in spans(page):
                # Font ascenders occasionally extend slightly beyond the ink.
                if not (page.rect + (-1,-1,1,1)).contains(fitz.Rect(s['bbox'])):
                    errors.append(f'Text outside source page in {path.name}: {s["text"]}')
        sources.append({'file':str(path.relative_to(HERE)),
                        'sha256':digest(path),
                        'text_spans':len(text), 'minimum_source_pt':min((s['size'] for s in text),default=None)})
        if path.name=='figure3.pdf':
            top=next(s['bbox'][1] for s in text if s['text'].strip()=='B')-5
            bottom=doc[0].search_for('Ablations modestly reduce')[0].y0-17
            labels=[s for s in text if s['bbox'][1]>=top and s['bbox'][3]<=bottom]
            images=[fitz.Rect(item['bbox']) for item in doc[0].get_image_info()
                    if item['bbox'][1]>=top and item['bbox'][3]<=bottom]
            for label in labels:
                if any((fitz.Rect(label['bbox']) & rect).width>1
                       and (fitz.Rect(label['bbox']) & rect).height>1 for rect in images):
                    schematic_image_overlaps.append(label['text'])
            if schematic_image_overlaps:errors.append('Figure 3 schematic labels overlap image content')
            if any('A feedforward model' in s['text'] or 'Δ=' in s['text'] or '% of ' in s['text'] for s in text):
                errors.append('Figure 3 retains a removed heading or in-panel effect-size annotation')
            for i,first in enumerate(labels):
                for second in labels[i+1:]:
                    overlap=fitz.Rect(first['bbox']) & fitz.Rect(second['bbox'])
                    if overlap.width>1 and overlap.height>1:
                        schematic_overlaps.append([first['text'],second['text']])
            if schematic_overlaps:errors.append('Overlapping Figure 3 schematic labels')
    installed={}
    for i in range(1,5):
        directory='fig1' if i==1 else 'fig2' if i==2 else f'figure{i}'
        installed[str(i)]=digest(HERE/'build'/directory/f'figure{i}.pdf')==digest(FIGURES[i-1])
    if not all(installed.values()):errors.append('Installed figures differ from the audited render outputs')
    doc=fitz.open(HERE/'build/main.pdf')
    pages=[]
    preview=HERE/'build/page_previews';preview.mkdir(exist_ok=True)
    for i,page in enumerate(doc):
        text=[s for s in spans(page) if s['font'] in font_names]
        if not text:continue
        small=[s for s in text if s['size'] < 6.0 - .001]
        if small:errors.extend(f'Page {i+1}: {s["text"]!r} is {s["size"]:.3f} pt' for s in small)
        pages.append({'pdf_page':i+1,'minimum_figure_font_pt':min(s['size'] for s in text),'text_spans':len(text)})
        page.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(preview/f'page_{i+1:02d}.png')
    if len(pages)!=len(FIGURES):errors.append(f'Expected {len(FIGURES)} figure pages, found {len(pages)}')
    if sum(p['text_spans'] for p in pages)!=sum(s['text_spans'] for s in sources):
        errors.append('Compiled figure text span count differs from the source PDFs')
    base=json.loads((BUNDLE/'figure3/figures/figure3_manifest.json').read_text())
    new=json.loads((HERE/'build/figure3/figure3_manifest.json').read_text())
    stats3={key:base[key]==new[key] for key in ('panel_c_stats','panel_d_stats','panel_e_stats')}
    if not all(stats3.values()):errors.append('Figure 3 numerical results changed')
    base4=json.loads((BUNDLE/'figure4/production_figure4/figure/summary.json').read_text())
    new4=json.loads((HERE/'build/figure4/summary.json').read_text())
    update=spectrum_update(BUNDLE)
    if update:
        spectral_figure=json.loads((ROOT/update['figure_summary']).read_text())
        for key in update['updated_panels']:
            base4['panels'][key]=spectral_figure['panels'][key]
    # The manuscript removes the example heatmaps and draws their unchanged
    # fitted contours on C. Only this display flag differs from source metrics.
    expected_c = dict(base4['panels']['C'], passband_contours_drawn=True)
    stats4={key:(expected_c if key == 'C' else base4['panels'][key])==new4['panels'][key]
            for key in 'BCDEFGH'}
    if new4.get('display_panel_letters') != {'A':'A','B':'B','C':'C','E':'D','F':'E','G':'F','H':'G'}:
        errors.append('Figure 4 manuscript panel mapping differs from the caption')
    if not all(stats4.values()):errors.append('Rendered Figure 4 results differ from the selected analysis')
    selection=json.loads((HERE/'analysis/panel_a_selection.json').read_text())
    example=json.loads((HERE/'build/panel_a_exemplar_audit/summary.json').read_text())
    if selection!=example:errors.append('Saved Figure 4A selection differs from the replay summary')
    stats_sources=json.loads((HERE/'build/stats_sources.json').read_text())
    if selection['checkpoint_sha256']!=stats_sources['checkpoint_sha256']:
        errors.append('Figure 4A and manuscript statistics use different checkpoints')
    schematic=json.loads((HERE/'analysis/figure3_schematic.json').read_text())
    if schematic!=json.loads((HERE/'build/figure3/architecture_maps.json').read_text()):
        errors.append('Saved Figure 3 schematic provenance differs from the render')
    if schematic['checkpoint_sha256']!=stats_sources['checkpoint_sha256']:
        errors.append('Figure 3 schematic and manuscript statistics use different checkpoints')
    if schematic['example_selection']['render_audit']['anchor']!='session_global':
        errors.append('Figure 3 schematic must depict the global stabilization quantified in C--E')
    schematic_architecture=schematic.get('schematic_architecture',{})
    draft_architecture=(new.get('schematic_no_phase_preview') is True
                        and schematic_architecture.get('phase_readout_rank')==0
                        and schematic_architecture.get('readout_rank_options')==[1,2]
                        and bool(schematic.get('draft_note')))
    if SELECTED.get('schematic_no_phase_preview',False):
        if not draft_architecture:
            errors.append('Figure 3 must declare its architecture preview and retained source results')
    else:
        model_audit=json.loads((BUNDLE/'audits/production_model.json').read_text())
        if (model_audit['status']!='passed' or new.get('schematic_no_phase_preview')
                or schematic.get('draft_note')
                or schematic_architecture!=schematic['architecture']
                or any(schematic_architecture[key]!=model_audit['production'][key]
                       for key in ('readout_rank','phase_readout_rank'))):
            errors.append('Figure 3 architecture does not match the selected fitted model')
        pdf_text=re.sub(r'\s+', ' ', ' '.join(page.get_text().lower() for page in doc))
        if any(phrase in pdf_text for phrase in ('draft status','draft update','preceding model','during retraining')):
            errors.append('Completed manuscript retains a draft-model or pending-analysis note')
        if update:
            errors.append('Completed manuscript retains an interim Figure 4 spectrum override')
    if stats_sources['checkpoint_sha256']!=SELECTED['checkpoint_sha256']:
        errors.append('Manuscript statistics differ from the selected checkpoint')
    from render_stabilization_control import statistics_tex
    control_binding=json.loads((HERE/'analysis/stabilization_control.json').read_text())
    control_path=ROOT/control_binding['summary']
    control=json.loads(control_path.read_text())
    if digest(control_path)!=control_binding['summary_sha256']:
        errors.append('Stabilization-control summary differs from its binding')
    if control['checkpoint_sha256']!=SELECTED['checkpoint_sha256']:
        errors.append('Stabilization control uses a different model')
    for key in ('local_cache','global_cache'):
        if digest(ROOT/control[key])!=control[key+'_sha256']:
            errors.append(f'Stabilization control {key} source changed')
    if digest(HERE/'analysis/stabilization_control/paired_scores.npz')!=control['paired_scores_sha256']:
        errors.append('Stabilization-control paired scores changed')
    if (HERE/'stabilization_stats.tex').read_text()!=statistics_tex(control['metrics']):
        errors.append('Stabilization-control manuscript numbers are stale')
    if digest(HERE/'figures/stabilization_control.pdf')!=digest(HERE/'build/stabilization_control/stabilization_control.pdf'):
        errors.append('Installed stabilization supplement differs from its render')
    if not all(all(checks.values()) for checks in control['data_identity_checks'].values()):
        errors.append('Stabilization controls do not share data-only scoring quantities')
    out={'passed':not errors,'errors':errors,'figure_sources':sources,'compiled_pages':pages,
         'compiled_pdf':{'sha256':digest(HERE/'build/main.pdf'),'pages':len(doc)},
         'installed_figures_match_rendered':installed,
         'figure3_statistics_unchanged':stats3,'figure4_matches_selected_analysis':stats4,
         'stabilization_control':control_binding,
         'selected_analysis_bundle':str(BUNDLE),
         'figure4_interim_spectrum_selection':update,
         'figure4_interim_selection_sha256':digest(SPECTRUM_SELECTION) if update else None,
         'figure3_schematic_text_overlaps':schematic_overlaps,
         'figure3_schematic_image_overlaps':schematic_image_overlaps,
         'revised_architecture_with_retained_results':draft_architecture,
         'statistics_provenance':stats_sources,
         'source_sha256':{path:digest(ROOT/path) for path in RENDER_SOURCES},
         'manuscript_sha256':{name:digest(HERE/name) for name in
             ('main.tex','refs.bib','lite.cls','generated_stats.tex','analysis/panel_a_selection.json',
              'analysis/figure3_schematic.json','stabilization_stats.tex')},
         'outlined_text_note':'Figure 1 schematic labels were restored from SVG aria-label paths; all other main-figure text exported as measurable fonts. Supplement 1 vector text re-typeset at the original baselines.'}
    (HERE/'analysis/figure_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    raise SystemExit(bool(errors))

if __name__=='__main__':main()
