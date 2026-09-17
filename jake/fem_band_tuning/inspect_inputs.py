"""Review the actual input scenes, band amplitudes, and rendering margins."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jake.fem_band_tuning.plot import scene_basis,picture,style
from jake.fem_band_tuning.common import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args(); out=args.out_dir
    d=json.loads((out/'design.json').read_text()); style()
    rows=[]; crop_margins=[]; full_margins=[]
    trace=np.load(out/'traces_endpoint_aligned.npy')
    maximum=np.max(np.abs(trace),axis=(0,1))*37.50476617
    fig,axes=plt.subplots(5,8,figsize=(18,12))
    for s,ax in enumerate(axes.flat):
        basis=scene_basis(out,s,d['normalization']['global_contrast_factor'])
        picture(ax,basis[0],f'Scene {s}')
        for k in range(6):
            component=basis[k+1]/255
            rows.append({'scene':s,'band':k,'endpoint_band_rms_contrast':float(np.sqrt(np.mean(component**2))),
                'endpoint_band_mean_contrast':float(component.mean()),
                'endpoint_band_std_contrast':float(component.std())})
        source=np.load(out/'scenes'/f'scene_{s:03d}.npz')
        h,w=source['full_canvas_shape']; meta=d['scenes'][s]
        x,y=meta['patch_center_x_px'],meta['patch_center_y_px']
        full_margins.append(float(min(x-76-maximum[0],w-x-76-maximum[0],
            y-76-maximum[1],h-y-76-maximum[1])))
        crop_margins.append(float(270-76-maximum.max()))
    fig.suptitle('All 40 reference scenes · one shared display scale and contrast factor',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(out/'input_scene_contact_sheet.png',dpi=150); plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'input_band_amplitudes.csv',index=False)
    write_json(out/'input_geometry_audit.json',{'minimum_crop_margin_px':min(crop_margins),
        'minimum_full_canvas_margin_px':min(full_margins),
        'scenes_whose_swept_field_may_cross_screen_edge':[i for i,m in enumerate(full_margins) if m<0],
        'note':'Conservative bound includes the full 151px model input at the largest displacement across all traces, not only the central neuron RF.'})
    print(json.dumps({'minimum_crop_margin_px':min(crop_margins),
        'minimum_full_canvas_margin_px':min(full_margins)},indent=2),flush=True)


if __name__=='__main__':
    main()
