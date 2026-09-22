"""Compose the two-panel routing/decoding prototype from audited summaries."""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jake.eye_movement_routing.analyze import style, draw_curves


def main():
    out = ROOT / 'outputs/eye_movement_routing_20260914'
    routing = np.load(out/'routing_analysis.npz')
    decoding = json.loads((out/'decoding_summary.json').read_text())
    center, low, high = [np.asarray(decoding[k]) for k in ['mean_gain','ci_low','ci_high']]
    style()
    for mode in range(2):
        fig, axes = plt.subplots(1,2,figsize=(10.5,4.7))
        draw_curves(axes[0], routing['center'][1], routing['ci_low'][1], routing['ci_high'][1],
                    'Spatial information gain (%)')
        draw_curves(axes[1], center[mode,0].T, low[mode,0].T, high[mode,0].T,
                    'Localization information gain\n(bits / observation)')
        axes[0].set_title('A   Population-specific information gain',loc='left',fontweight='bold',pad=15)
        title = 'B   Spatial decoding from spike timing' if mode==0 else 'B   Spatial decoding at matched spike counts'
        axes[1].set_title(title,loc='left',fontweight='bold',pad=15)
        axes[0].legend(frameon=False,fontsize=9,loc='upper left')
        fig.suptitle('Eye movements change the information available in different model populations',fontsize=12,y=.995)
        fig.text(.5,.075,'Changes relative to stabilization · 48 tuning-defined units per group · bands: 95% image/trajectory bootstrap',ha='center',fontsize=8,color='.35')
        budget = 'model-predicted spike counts' if mode==0 else '10 mean spikes/observation'
        fig.text(.5,.045,f'A: 20 image patches × 200 trajectories.  B: 10 patches × 20 trajectories; ideal Poisson observer, {budget}.',ha='center',fontsize=8,color='.35')
        fig.text(.5,.015,'Exploratory model result. Image and movement are known to the decoder; movement quintiles are not matched for path length.',ha='center',fontsize=8,color='.35')
        fig.tight_layout(rect=[0,.13,1,.95])
        suffix = '' if mode==0 else '_matched_counts'
        for ext in ['png','pdf','svg']:
            fig.savefig(out/f'routing_and_decoding{suffix}.{ext}',dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()
