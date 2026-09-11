#!/usr/bin/env python3
"""Render manuscript figures from the pinned analysis, then install the PDFs.

Run in the yatesfv environment. The original completed bundle is read-only.
Use --replay-example to regenerate Figure 4A on the local GPU.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from figure4_selection import spectrum_update
from analysis_selection import SELECTION, selected_analysis

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SELECTED=selected_analysis()
BUNDLE=ROOT/SELECTED['bundle']
BUILD=HERE/'build'


def run(command, env):
    print(shlex.join(map(str,command)),flush=True)
    subprocess.run(list(map(str,command)),cwd=ROOT,env=env,check=True)


def replay_example(env, device):
    previous=json.loads((BUNDLE/'figure4/panel_a_exemplar_audit/summary.json').read_text())
    spectrum=json.loads((BUNDLE/'figure4/kuang_rucci_ensemble/summary.json').read_text())
    args={'image-table':BUNDLE/'inputs/natural_image_table_100.csv',
          'trace-bank':Path(spectrum['trace_source']).parent,
          'rucci-ensemble':previous['rucci_ensemble_provenance']['path'],
          'checkpoint':previous['checkpoint'],
          'dataset-config':ROOT/'paper/model_selection/configs/multi_240_long_split3_dekel35_allgratings.yaml',
          'population-spec-dir':BUNDLE/'figure4/all_available_population_spec',
          'population-version':previous['population_version'],
          'mcfarland-outputs':ROOT/'scripts/mcfarland_outputs_mono.pkl',
          'population-matrix-dir':BUNDLE/'figure4/response_matrix_40img_x_200fix/merged',
          'out-dir':BUILD/'panel_a_exemplar_audit','model-label':previous['model_label'],
          'device':device,'image-selection':'central_detail'}
    command=[sys.executable,ROOT/'paper/fig4/spatiotemporal_tuning/audit_panel_a_exemplars.py']
    for key,value in args.items():command.extend(['--'+key,value])
    run(command,env)
    shutil.copy2(BUILD/'panel_a_exemplar_audit/summary.json',HERE/'analysis/panel_a_selection.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choices=('1','2','3','4','supplement')
    parser.add_argument('figures',nargs='*',metavar='FIGURE',help='Select 1, 2, 3, 4, or supplement (default: all)')
    parser.add_argument('--replay-example',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if any(which not in choices for which in args.figures):
        parser.error('FIGURE must be 1, 2, 3, 4, or supplement')
    env=os.environ.copy()
    env.update(MPLCONFIGDIR=str(BUILD/'mpl'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    BUILD.mkdir(exist_ok=True);(BUILD/'mpl').mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(BUILD/'mpl'))
    if args.replay_example:replay_example(env,args.device)
    for which in args.figures or choices:
        current=env.copy()
        out=BUILD/f'figure{which}';out.mkdir(exist_ok=True)
        if which=='1':
            current.update(VISIONCORE_MIN_FIGURE_FONT_PT='7.1')
            command=[sys.executable,'-c',
                "import sys;from pathlib import Path;sys.path[:0]=['manuscript','paper/fig1'];"
                "import generate_fig1 as g;from render_support import restore_svg_labels;"
                f"out=Path({str(BUILD/'fig1')!r});out.mkdir(parents=True,exist_ok=True);"
                "restore_svg_labels(g.HERE/'fig1a.svg',out/'fig1a.svg');"
                "g.FIG_DIR=out;g.HERE=out;g.compose(out_stem='figure1')"]
            source=BUILD/'fig1/figure1.pdf'
        elif which=='2':
            current.update(VISIONCORE_CACHE_DIR=str(BUILD/'figure2_cache'),
                           VISIONCORE_FIGURES_DIR=str(BUILD),VISIONCORE_STATS_DIR=str(out/'stats'),
                           VISIONCORE_MIN_FIGURE_FONT_PT='8.8')
            cache=BUILD/'figure2_cache/fig2_lead_pair_scan_Allen_2022-04-08.pkl'
            # This small example cache can be built independently of the pinned
            # population decomposition; its own loader determines reuse.
            if not cache.exists():
                run([sys.executable,ROOT/'paper/fig2/build_lead_scan_caches.py',
                     '--sessions','Allen_2022-04-08'],current)
            command=[sys.executable,'-c',
                "import sys,runpy;from pathlib import Path;sys.path[:0]=['paper/fig2','paper/covariance_decomposition'];"
                "import derive;derive.DERIVED_CACHE=Path('outputs/dekel240_paper/m77_epoch279/production_figure3/cache/covdecomp_derived.pkl');"
                "runpy.run_path('paper/fig2/generate_figure2.py',run_name='__main__')"]
            source=BUILD/'fig2/figure2.pdf'
        elif which=='3':
            manifest=json.loads((BUNDLE/'figure3/run_manifest.json').read_text())
            current.update(manifest['environment'])
            current.update(FIG3_FIG_DIR=str(out),FIG3_STAT_DIR=str(out/'stats'),
                           FIG3_REUSE_EXISTING_CACHES='1',VISIONCORE_MIN_FIGURE_FONT_PT='8.5')
            command=[sys.executable,ROOT/'paper/fig3/generate_figure3.py',
                     '--out-dir',out,'--layout','manuscript']
            if SELECTED.get('schematic_no_phase_preview',False):
                command.append('--schematic-no-phase')
            source=out/'figure3.pdf'
        elif which=='4':
            example=BUILD/'panel_a_exemplar_audit'
            if SELECTION.exists():
                example=BUNDLE/'figure4/panel_a_exemplar_audit'
                # Install the exact selected analysis alongside the manuscript
                # render; no new example selection occurs during typesetting.
                shutil.copytree(example, BUILD/'panel_a_exemplar_audit', dirs_exist_ok=True)
                shutil.copy2(example/'summary.json', HERE/'analysis/panel_a_selection.json')
            if not (example/'selected_example.npz').exists():
                raise FileNotFoundError('Run with --replay-example to create the current Figure 4A example')
            manifest=json.loads((BUNDLE/'figure4/production_figure4/run_manifest.json').read_text())
            command=shlex.split(manifest['commands'][0]);command[0]=sys.executable
            command[command.index('--out-dir')+1]=str(out)
            command[command.index('--panel-a-audit')+1]=str(example)
            update=spectrum_update(BUNDLE)
            if update:
                command[command.index('--rucci-ensemble')+1]=str(ROOT/update['rucci_ensemble'])
            command.extend(['--layout','manuscript'])
            current['VISIONCORE_MIN_FIGURE_FONT_PT']='6.1'
            source=out/'figure4.pdf'
        else:
            from render_support import enlarge_pdf_text
            enlarge_pdf_text(HERE/'old_figures/fig1d_extended_single_cell_examples.pdf',
                             HERE/'figures/supplement1.pdf',7.0)
            continue
        run(command,current)
        shutil.copy2(source,HERE/'figures'/f'figure{which}.pdf')
        if which=='3':
            shutil.copy2(out/'architecture_maps.json',HERE/'analysis/figure3_schematic.json')

if __name__=='__main__':main()
