#!/usr/bin/env python3
"""
Classify layers. Assumes you have an all_results in your workspace. We'll turn this into a function later.
"""

#%% Setup and Imports
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import sys
from pathlib import Path
sys.path.append('.')

import numpy as np
from eval_stack_multidataset import evaluate_model_multidataset
from DataYatesV1 import enable_autoreload

import matplotlib.pyplot as plt

from eval_stack_utils import argmin_subpixel, argmax_subpixel

enable_autoreload()

#%% plot summary of laminar-linked data
from DataYatesV1 import get_session, get_complete_sessions

sessions = get_complete_sessions()
# dummy_layer_split = {'exclusion_bottom': [np.nan, np.nan], '5/6': [np.nan, np.nan], '4/5': [np.nan, np.nan], '4C': [np.nan, np.nan], '3/2': [np.nan, np.nan], 'exclusion_top': [np.nan, np.nan]}
dummy_layer_split = {'exclusion_bottom': np.nan, '5/6': np.nan, '4/5': np.nan, '4C': np.nan, '3/2': np.nan, 'exclusion_top': np.nan}
layer_splits = {sess.sess_dir.name: dummy_layer_split.copy() for sess in sessions}

collapse_shanks = True

if collapse_shanks:
    print('collapsing shanks')
    layer_splits['standard'] = dummy_layer_split.copy()
    layer_splits['standard']['exclusion_bottom'] = 0
    layer_splits['standard']['5/6'] = 200
    layer_splits['standard']['4/5'] = 364
    layer_splits['standard']['4C'] = 626
    layer_splits['standard']['3/2'] = 868
    layer_splits['standard']['exclusion_top'] = 1244

    layer_splits['Allen_2022-02-16']['exclusion_bottom'] = 0
    layer_splits['Allen_2022-02-16']['5/6'] = 300
    layer_splits['Allen_2022-02-16']['4/5'] = 500
    layer_splits['Allen_2022-02-16']['4C'] = 700
    layer_splits['Allen_2022-02-16']['3/2'] = 850
    layer_splits['Allen_2022-02-16']['exclusion_top'] = 1100


    layer_splits['Allen_2022-02-18']['exclusion_bottom'] = 400
    layer_splits['Allen_2022-02-18']['5/6'] = 640
    layer_splits['Allen_2022-02-18']['4/5'] = 760
    layer_splits['Allen_2022-02-18']['4C'] = 1000
    layer_splits['Allen_2022-02-18']['3/2'] = np.nan
    layer_splits['Allen_2022-02-18']['exclusion_top'] = np.nan

    layer_splits['Allen_2022-02-24']['exclusion_bottom'] = 0
    layer_splits['Allen_2022-02-24']['5/6'] = 300
    layer_splits['Allen_2022-02-24']['4/5'] = 400
    layer_splits['Allen_2022-02-24']['4C'] = 650
    layer_splits['Allen_2022-02-24']['3/2'] = 800
    layer_splits['Allen_2022-02-24']['exclusion_top'] = 1000

    layer_splits['Allen_2022-03-02']['exclusion_bottom'] = 200
    layer_splits['Allen_2022-03-02']['5/6'] = 500
    layer_splits['Allen_2022-03-02']['4/5'] = 600
    layer_splits['Allen_2022-03-02']['4C'] = 880
    layer_splits['Allen_2022-03-02']['3/2'] = np.nan
    layer_splits['Allen_2022-03-02']['exclusion_top'] = np.nan

    layer_splits['Allen_2022-03-04']['exclusion_bottom'] = 200
    layer_splits['Allen_2022-03-04']['5/6'] = 450
    layer_splits['Allen_2022-03-04']['4/5'] = 550
    layer_splits['Allen_2022-03-04']['4C'] = 750
    layer_splits['Allen_2022-03-04']['3/2'] = 900
    layer_splits['Allen_2022-03-04']['exclusion_top'] = np.nan

    layer_splits['Allen_2022-03-30']['exclusion_bottom'] = 0
    layer_splits['Allen_2022-03-30']['5/6'] = 190
    layer_splits['Allen_2022-03-30']['4/5'] = 325
    layer_splits['Allen_2022-03-30']['4C'] = 510
    layer_splits['Allen_2022-03-30']['3/2'] = 750
    layer_splits['Allen_2022-03-30']['exclusion_top'] = 900

    layer_splits['Allen_2022-04-01']['exclusion_bottom'] = 150
    layer_splits['Allen_2022-04-01']['5/6'] = 350
    layer_splits['Allen_2022-04-01']['4/5'] = 525
    layer_splits['Allen_2022-04-01']['4C'] = 800
    layer_splits['Allen_2022-04-01']['3/2'] = np.nan
    layer_splits['Allen_2022-04-01']['exclusion_top'] = np.nan

    layer_splits['Allen_2022-04-06']['exclusion_bottom'] = 0
    layer_splits['Allen_2022-04-06']['5/6'] = 250
    layer_splits['Allen_2022-04-06']['4/5'] = 375
    layer_splits['Allen_2022-04-06']['4C'] = 600
    layer_splits['Allen_2022-04-06']['3/2'] = 830
    layer_splits['Allen_2022-04-06']['exclusion_top'] = np.nan

    layer_splits['Allen_2022-04-08']['exclusion_bottom'] = 50
    layer_splits['Allen_2022-04-08']['5/6'] = 250
    layer_splits['Allen_2022-04-08']['4/5'] = 400
    layer_splits['Allen_2022-04-08']['4C'] = 650
    layer_splits['Allen_2022-04-08']['3/2'] = 900
    layer_splits['Allen_2022-04-08']['exclusion_top'] = np.nan

    layer_splits['Allen_2022-04-13']['exclusion_bottom'] = 200
    layer_splits['Allen_2022-04-13']['5/6'] = 500
    layer_splits['Allen_2022-04-13']['4/5'] = 650
    layer_splits['Allen_2022-04-13']['4C'] = 900
    layer_splits['Allen_2022-04-13']['3/2'] = np.nan
    layer_splits['Allen_2022-04-13']['exclusion_top'] = np.nan

    layer_splits['Allen_2022-04-15']['exclusion_bottom'] = 0
    layer_splits['Allen_2022-04-15']['5/6'] = 250
    layer_splits['Allen_2022-04-15']['4/5'] = 400
    layer_splits['Allen_2022-04-15']['4C'] = 700
    layer_splits['Allen_2022-04-15']['3/2'] = 900
    layer_splits['Allen_2022-04-15']['exclusion_top'] = np.nan

    layer_splits['Allen_2022-06-01']['exclusion_bottom'] = np.nan
    layer_splits['Allen_2022-06-01']['5/6'] = 300
    layer_splits['Allen_2022-06-01']['4/5'] = 400
    layer_splits['Allen_2022-06-01']['4C'] = 600
    layer_splits['Allen_2022-06-01']['3/2'] = 800
    layer_splits['Allen_2022-06-01']['exclusion_top'] = 1000

    layer_splits['Allen_2022-06-10']['exclusion_bottom'] = 0
    layer_splits['Allen_2022-06-10']['5/6'] = 215
    layer_splits['Allen_2022-06-10']['4/5'] = 350
    layer_splits['Allen_2022-06-10']['4C'] = 640
    layer_splits['Allen_2022-06-10']['3/2'] = 850
    layer_splits['Allen_2022-06-10']['exclusion_top'] = np.nan

    layer_splits['Allen_2022-08-05']['exclusion_bottom'] = 100
    layer_splits['Allen_2022-08-05']['5/6'] = 300
    layer_splits['Allen_2022-08-05']['4/5'] = 410
    layer_splits['Allen_2022-08-05']['4C'] = 750
    layer_splits['Allen_2022-08-05']['3/2'] = np.nan
    layer_splits['Allen_2022-08-05']['exclusion_top'] = np.nan

    layer_splits['Logan_2019-12-20']['exclusion_bottom'] = np.nan
    layer_splits['Logan_2019-12-20']['5/6'] = np.nan
    layer_splits['Logan_2019-12-20']['4/5'] = 0
    layer_splits['Logan_2019-12-20']['4C'] = 200
    layer_splits['Logan_2019-12-20']['3/2'] = 500
    layer_splits['Logan_2019-12-20']['exclusion_top'] = 500

    layer_splits['Logan_2019-12-23']['exclusion_bottom'] = np.nan
    layer_splits['Logan_2019-12-23']['5/6'] = np.nan
    layer_splits['Logan_2019-12-23']['4/5'] = 100
    layer_splits['Logan_2019-12-23']['4C'] = 400
    layer_splits['Logan_2019-12-23']['3/2'] = 600
    layer_splits['Logan_2019-12-23']['exclusion_top'] = 800

    layer_splits['Logan_2020-02-29']['exclusion_bottom'] = 100
    layer_splits['Logan_2020-02-29']['5/6'] = 250
    layer_splits['Logan_2020-02-29']['4/5'] = 350
    layer_splits['Logan_2020-02-29']['4C'] = 700
    layer_splits['Logan_2020-02-29']['3/2'] = np.nan
    layer_splits['Logan_2020-02-29']['exclusion_top'] = np.nan


    layer_splits['Logan_2020-03-02']['exclusion_bottom'] = 100
    layer_splits['Logan_2020-03-02']['5/6'] = 250
    layer_splits['Logan_2020-03-02']['4/5'] = 350
    layer_splits['Logan_2020-03-02']['4C'] = 600
    layer_splits['Logan_2020-03-02']['3/2'] = np.nan
    layer_splits['Logan_2020-03-02']['exclusion_top'] = np.nan


    layer_splits['Logan_2020-03-04']['exclusion_bottom'] = 400
    layer_splits['Logan_2020-03-04']['5/6'] = 700
    layer_splits['Logan_2020-03-04']['4/5'] = 770
    layer_splits['Logan_2020-03-04']['4C'] = 900
    layer_splits['Logan_2020-03-04']['3/2'] = np.nan
    layer_splits['Logan_2020-03-04']['exclusion_top'] = np.nan

    layer_splits['Logan_2020-03-06']['exclusion_bottom'] = 500
    layer_splits['Logan_2020-03-06']['5/6'] = 800
    layer_splits['Logan_2020-03-06']['4/5'] = 900
    layer_splits['Logan_2020-03-06']['4C'] = 1050
    layer_splits['Logan_2020-03-06']['3/2'] = np.nan
    layer_splits['Logan_2020-03-06']['exclusion_top'] = np.nan
else:
    print('NOT collapsing shanks')
    layer_splits['standard'] = dummy_layer_split.copy()
    for key in layer_splits['standard'].keys():
        layer_splits['standard'][key] = [layer_splits['standard'][key], layer_splits['standard'][key]]
            
    layer_splits['standard']['exclusion_bottom'] = [0, 0]
    layer_splits['standard']['5/6'] = [200, 200]
    layer_splits['standard']['4/5'] = [364, 364]
    layer_splits['standard']['4C'] = [626, 626]
    layer_splits['standard']['3/2'] = [868, 868]
    layer_splits['standard']['exclusion_top'] = [1244, 1244]

    layer_splits['Allen_2022-02-16']['exclusion_bottom'] = [0, 100]
    layer_splits['Allen_2022-02-16']['5/6'] = [350, 300]
    layer_splits['Allen_2022-02-16']['4/5'] = [500, 450]
    layer_splits['Allen_2022-02-16']['4C'] = [700, 650]
    layer_splits['Allen_2022-02-16']['3/2'] = [880, 800]
    layer_splits['Allen_2022-02-16']['exclusion_top'] = [1200, 1100]

    layer_splits['Allen_2022-02-18']['exclusion_bottom'] = [400, 400]
    layer_splits['Allen_2022-02-18']['5/6'] = [610, 610]
    layer_splits['Allen_2022-02-18']['4/5'] = [790, 790]
    layer_splits['Allen_2022-02-18']['4C'] = [950, 950]
    layer_splits['Allen_2022-02-18']['3/2'] = [np.nan, np.nan]
    layer_splits['Allen_2022-02-18']['exclusion_top'] = [np.nan, np.nan]

    layer_splits['Allen_2022-02-24']['exclusion_bottom'] = [0, 0]
    layer_splits['Allen_2022-02-24']['5/6'] = [300, 300]
    layer_splits['Allen_2022-02-24']['4/5'] = [410, 410]
    layer_splits['Allen_2022-02-24']['4C'] = [650, 650]
    layer_splits['Allen_2022-02-24']['3/2'] = [800, 800]
    layer_splits['Allen_2022-02-24']['exclusion_top'] = [1000, 1000]

    layer_splits['Allen_2022-03-02']['exclusion_bottom'] = [200, 200]
    layer_splits['Allen_2022-03-02']['5/6'] = [500, 500]
    layer_splits['Allen_2022-03-02']['4/5'] = [600, 600]
    layer_splits['Allen_2022-03-02']['4C'] = [880, 880]
    layer_splits['Allen_2022-03-02']['3/2'] = [np.nan, np.nan]
    layer_splits['Allen_2022-03-02']['exclusion_top'] = [np.nan, np.nan]


# ATTEMPT 2
# layer_splits['Allen_2022-02-16']['exclusion_bottom'] = 0
# layer_splits['Allen_2022-02-16']['5/6'] = 290
# layer_splits['Allen_2022-02-16']['4/5'] = 450
# layer_splits['Allen_2022-02-16']['4C'] = 750
# layer_splits['Allen_2022-02-16']['3/2'] = 900
# layer_splits['Allen_2022-02-16']['exclusion_top'] = 1100

# layer_splits['Allen_2022-02-18']['exclusion_bottom'] = 400
# layer_splits['Allen_2022-02-18']['5/6'] = 660
# layer_splits['Allen_2022-02-18']['4/5'] = 800
# layer_splits['Allen_2022-02-18']['4C'] = 1000
# layer_splits['Allen_2022-02-18']['3/2'] = np.nan
# layer_splits['Allen_2022-02-18']['exclusion_top'] = np.nan

# layer_splits['Allen_2022-02-24']['exclusion_bottom'] = 0
# layer_splits['Allen_2022-02-24']['5/6'] = 150
# layer_splits['Allen_2022-02-24']['4/5'] = 365
# layer_splits['Allen_2022-02-24']['4C'] = 700
# layer_splits['Allen_2022-02-24']['3/2'] = 800
# layer_splits['Allen_2022-02-24']['exclusion_top'] = np.nan

# layer_splits['Allen_2022-03-02']['exclusion_bottom'] = 180
# layer_splits['Allen_2022-03-02']['5/6'] = 450
# layer_splits['Allen_2022-03-02']['4/5'] = 700
# layer_splits['Allen_2022-03-02']['4C'] = 1000
# layer_splits['Allen_2022-03-02']['3/2'] = np.nan
# layer_splits['Allen_2022-03-02']['exclusion_top'] = np.nan

# layer_splits['Allen_2022-03-04']['exclusion_bottom'] = 0
# layer_splits['Allen_2022-03-04']['5/6'] = 400
# layer_splits['Allen_2022-03-04']['4/5'] = 550
# layer_splits['Allen_2022-03-04']['4C'] = 800
# layer_splits['Allen_2022-03-04']['3/2'] = 1000
# layer_splits['Allen_2022-03-04']['exclusion_top'] = np.nan

# layer_splits['Allen_2022-03-30']['exclusion_bottom'] = np.nan
# layer_splits['Allen_2022-03-30']['5/6'] = 50
# layer_splits['Allen_2022-03-30']['4/5'] = 190
# layer_splits['Allen_2022-03-30']['4C'] = 480
# layer_splits['Allen_2022-03-30']['3/2'] = 800
# layer_splits['Allen_2022-03-30']['exclusion_top'] = np.nan

# layer_splits['Allen_2022-04-01']['exclusion_bottom'] = np.nan
# layer_splits['Allen_2022-04-01']['5/6'] = 300
# layer_splits['Allen_2022-04-01']['4/5'] = 525
# layer_splits['Allen_2022-04-01']['4C'] = 800
# layer_splits['Allen_2022-04-01']['3/2'] = 1050
# layer_splits['Allen_2022-04-01']['exclusion_top'] = np.nan

# layer_splits['Allen_2022-04-06']['exclusion_bottom'] = np.nan
# layer_splits['Allen_2022-04-06']['5/6'] = 125
# layer_splits['Allen_2022-04-06']['4/5'] = 300
# layer_splits['Allen_2022-04-06']['4C'] = 600
# layer_splits['Allen_2022-04-06']['3/2'] = 850
# layer_splits['Allen_2022-04-06']['exclusion_top'] = np.nan

# layer_splits['Allen_2022-04-08']['exclusion_bottom'] = np.nan
# layer_splits['Allen_2022-04-08']['5/6'] = 150
# layer_splits['Allen_2022-04-08']['4/5'] = 365
# layer_splits['Allen_2022-04-08']['4C'] = 650
# layer_splits['Allen_2022-04-08']['3/2'] = 800
# layer_splits['Allen_2022-04-08']['exclusion_top'] = np.nan

# layer_splits['Allen_2022-04-13']['exclusion_bottom'] = np.nan
# layer_splits['Allen_2022-04-13']['5/6'] = 350
# layer_splits['Allen_2022-04-13']['4/5'] = 700
# layer_splits['Allen_2022-04-13']['4C'] = 1050
# layer_splits['Allen_2022-04-13']['3/2'] = np.nan
# layer_splits['Allen_2022-04-13']['exclusion_top'] = np.nan

# layer_splits['Allen_2022-04-15']['exclusion_bottom'] = np.nan
# layer_splits['Allen_2022-04-15']['5/6'] = 190
# layer_splits['Allen_2022-04-15']['4/5'] = 390
# layer_splits['Allen_2022-04-15']['4C'] = 710
# layer_splits['Allen_2022-04-15']['3/2'] = 950
# layer_splits['Allen_2022-04-15']['exclusion_top'] = np.nan

# layer_splits['Allen_2022-06-01']['exclusion_bottom'] = np.nan
# layer_splits['Allen_2022-06-01']['5/6'] = 250
# layer_splits['Allen_2022-06-01']['4/5'] = 390
# layer_splits['Allen_2022-06-01']['4C'] = 600
# layer_splits['Allen_2022-06-01']['3/2'] = 850
# layer_splits['Allen_2022-06-01']['exclusion_top'] = np.nan

# layer_splits['Allen_2022-06-10']['exclusion_bottom'] = 0
# layer_splits['Allen_2022-06-10']['5/6'] = 180
# layer_splits['Allen_2022-06-10']['4/5'] = 340
# layer_splits['Allen_2022-06-10']['4C'] = 600
# layer_splits['Allen_2022-06-10']['3/2'] = 800
# layer_splits['Allen_2022-06-10']['exclusion_top'] = np.nan

# layer_splits['Allen_2022-08-05']['exclusion_bottom'] = np.nan
# layer_splits['Allen_2022-08-05']['5/6'] = 150
# layer_splits['Allen_2022-08-05']['4/5'] = 400
# layer_splits['Allen_2022-08-05']['4C'] = 700
# layer_splits['Allen_2022-08-05']['3/2'] = 900
# layer_splits['Allen_2022-08-05']['exclusion_top'] = np.nan

# layer_splits['Logan_2019-12-20']['exclusion_bottom'] = np.nan
# layer_splits['Logan_2019-12-20']['5/6'] = np.nan
# layer_splits['Logan_2019-12-20']['4/5'] = np.nan
# layer_splits['Logan_2019-12-20']['4C'] = 150
# layer_splits['Logan_2019-12-20']['3/2'] = 300
# layer_splits['Logan_2019-12-20']['exclusion_top'] = 500

# layer_splits['Logan_2019-12-23']['exclusion_bottom'] = np.nan
# layer_splits['Logan_2019-12-23']['5/6'] = np.nan
# layer_splits['Logan_2019-12-23']['4/5'] = 0
# layer_splits['Logan_2019-12-23']['4C'] = 250
# layer_splits['Logan_2019-12-23']['3/2'] = 480
# layer_splits['Logan_2019-12-23']['exclusion_top'] = 650

# layer_splits['Logan_2020-02-29']['exclusion_bottom'] = np.nan
# layer_splits['Logan_2020-02-29']['5/6'] = 200
# layer_splits['Logan_2020-02-29']['4/5'] = 350
# layer_splits['Logan_2020-02-29']['4C'] = 550
# layer_splits['Logan_2020-02-29']['3/2'] = 700
# layer_splits['Logan_2020-02-29']['exclusion_top'] = np.nan


# layer_splits['Logan_2020-03-02']['exclusion_bottom'] = np.nan
# layer_splits['Logan_2020-03-02']['5/6'] = 150
# layer_splits['Logan_2020-03-02']['4/5'] = 350
# layer_splits['Logan_2020-03-02']['4C'] = 550
# layer_splits['Logan_2020-03-02']['3/2'] = 740
# layer_splits['Logan_2020-03-02']['exclusion_top'] = np.nan


# layer_splits['Logan_2020-03-04']['exclusion_bottom'] = 350
# layer_splits['Logan_2020-03-04']['5/6'] = 650
# layer_splits['Logan_2020-03-04']['4/5'] = 770
# layer_splits['Logan_2020-03-04']['4C'] = 950
# layer_splits['Logan_2020-03-04']['3/2'] = np.nan
# layer_splits['Logan_2020-03-04']['exclusion_top'] = np.nan


# layer_splits['Logan_2020-03-06']['exclusion_bottom'] = 500
# layer_splits['Logan_2020-03-06']['5/6'] = 800
# layer_splits['Logan_2020-03-06']['4/5'] = 950
# layer_splits['Logan_2020-03-06']['4C'] = 1150
# layer_splits['Logan_2020-03-06']['3/2'] = np.nan
# layer_splits['Logan_2020-03-06']['exclusion_top'] = np.nan


def plot_laminar_boundaries(ax, layer_splits):
    
    if isinstance(layer_splits['exclusion_bottom'], list):
        collapse_shanks = False
    else:
        collapse_shanks = True

    if collapse_shanks:    
        ax.axhline(layer_splits['exclusion_bottom'], color='k', linestyle='--')
        ax.axhline(layer_splits['5/6'], color='r', linestyle='--')
        ax.axhline(layer_splits['4/5'], color='g', linestyle='--')
        ax.axhline(layer_splits['4C'], color='g', linestyle='--')
        ax.axhline(layer_splits['3/2'], color='c', linestyle='--')
        ax.axhline(layer_splits['exclusion_top'], color='k', linestyle='--')
    else:
        xd = ax.get_xlim()
        ax.hlines(layer_splits['exclusion_bottom'], xd[0], xd[1]/2, color='k', linestyle='--')
        ax.hlines(layer_splits['5/6'], xd[0], xd[1]/2, color='r', linestyle='--')
        ax.hlines(layer_splits['4/5'], xd[0], xd[1]/2, color='g', linestyle='--')
        ax.hlines(layer_splits['4C'], xd[0], xd[1]/2, color='g', linestyle='--')
        ax.hlines(layer_splits['3/2'], xd[0], xd[1]/2, color='c', linestyle='--')
        ax.hlines(layer_splits['exclusion_top'], xd[0], xd[1]/2, color='k', linestyle='--')
        ax.hlines(layer_splits['exclusion_bottom'], xd[0]+xd[1]/2, xd[1], color='k', linestyle='--')
        ax.hlines(layer_splits['5/6'], xd[0]+xd[1]/2, xd[1], color='r', linestyle='--')
        ax.hlines(layer_splits['4/5'], xd[0]+xd[1]/2, xd[1], color='g', linestyle='--')
        ax.hlines(layer_splits['4C'], xd[0]+xd[1]/2, xd[1], color='g', linestyle='--')
        ax.hlines(layer_splits['3/2'], xd[0]+xd[1]/2, xd[1], color='c', linestyle='--')
        ax.hlines(layer_splits['exclusion_top'], xd[0]+xd[1]/2, xd[1], color='k', linestyle='--')



def mapDepthsLinear(depths, laminar_splits, target_splits, layer4c_only=False):
    """
    Linear-regression mapping from source depths to target depths using all
    shared (finite) control points. Depths outside source exclusion bounds
    are set to NaN before mapping.

    Parameters
    ----------
    depths : np.ndarray
        Array of depths (any shape). NaNs preserved.
    laminar_splits : dict
        Source session split levels (may contain NaNs).
    target_splits : dict
        Target session split levels (may contain NaNs).

    Returns
    -------
    new_depths : np.ndarray
        Mapped depths with same shape as `depths`.
        If <2 shared points, returns exclusion-masked depths unchanged.
    """
    d = np.asarray(depths, dtype=float).copy()
    out = np.full_like(d, np.nan)

    # Exclude outside source bounds BEFORE mapping
    lb = laminar_splits.get('exclusion_bottom', np.nan)
    ub = laminar_splits.get('exclusion_top', np.nan)
    valid = ~np.isnan(d)
    if np.isfinite(lb):
        valid &= (d >= lb)
    if np.isfinite(ub):
        valid &= (d <= ub)

    # Gather shared finite control points
    xs, ys = [], []
    for k, x in laminar_splits.items():
        if k not in target_splits or (layer4c_only and ~np.isin(k, ['4/5', '4C'])):
            continue
        y = target_splits[k]
        if np.isfinite(x) and np.isfinite(y):
            xs.append(float(x))
            ys.append(float(y))

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    # Need at least 2 points to fit a line
    if xs.size < 2:
        out[valid] = d[valid]
        return out

    # Fit y = a*x + b
    a, b = np.polyfit(xs, ys, 1)

    # Apply mapping
    out[valid] = a * d[valid] + b
    return out

def map_depths(depths, laminar_splits, target_splits=None):
    '''
    Map depths from their original laminar_splits to target_splits using
    piecewise-linear interpolation with linear extrapolation. Any depths
    outside the source exclusion bounds are set to NaN *before* mapping.

    Parameters
    ----------
    depths : np.ndarray
        Array of depths to be mapped (any shape). NaNs are preserved.
    laminar_splits : dict
        Source session split levels (may contain NaNs).
    target_splits : dict
        Target session split levels (may contain NaNs).

    Returns
    -------
    new_depths : np.ndarray
        Mapped depths with same shape as `depths`.
        If fewer than 2 shared control points exist, returns the input
        depths after exclusion masking (no mapping possible).
    '''

    if target_splits is None:
        target_splits = layer_splits['standard']

    order = ['exclusion_bottom', '5/6', '4/5', '4/3', '3/2', 'exclusion_top']

    d = np.asarray(depths, dtype=float).copy()
    new_depths = np.full_like(d, np.nan)

    # 1) Exclude outside source bounds BEFORE any interpolation
    lb = laminar_splits.get('exclusion_bottom', np.nan)
    ub = laminar_splits.get('exclusion_top', np.nan)

    valid_mask = ~np.isnan(d)
    if np.isfinite(lb):
        valid_mask &= (d >= lb)
    if np.isfinite(ub):
        valid_mask &= (d <= ub)

    # 2) Build control points present in BOTH source and target
    xs, ys = [], []
    for k in order:
        xk = laminar_splits.get(k, np.nan)
        yk = target_splits.get(k, np.nan)
        if np.isfinite(xk) and np.isfinite(yk):
            xs.append(float(xk))
            ys.append(float(yk))

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    # Need at least two control points to define a mapping
    if xs.size < 2:
        # Nothing to map; just return after exclusion masking
        new_depths[valid_mask] = d[valid_mask]
        return new_depths

    # Sort control points by source depth (monotonic assumed but we enforce)
    sort_idx = np.argsort(xs)
    xs = xs[sort_idx]
    ys = ys[sort_idx]

    # 3) Piecewise-linear interpolation with linear extrapolation
    dv = d[valid_mask]
    # indices of the segment to the left (right-open intervals)
    idx = np.searchsorted(xs, dv, side='right') - 1
    # clamp to valid segment indices [0, len(xs)-2] for extrapolation
    idx = np.clip(idx, 0, len(xs) - 2)

    x0 = xs[idx]
    x1 = xs[idx + 1]
    y0 = ys[idx]
    y1 = ys[idx + 1]

    denom = (x1 - x0)
    # guard against zero-length segments (shouldn't happen, but be safe)
    zero_seg = (denom == 0)
    denom[zero_seg] = 1.0

    t = (dv - x0) / denom
    yv = y0 + t * (y1 - y0)

    # put results back
    new_depths[valid_mask] = yv
    return new_depths



#%%
def plot_laminar_data_across_sessions(all_results, collapse_shanks=False, plot_latency_overlay=True):
    experiments = list(all_results.keys())
    experiment = experiments[0]
    model_types = list(all_results[experiment].keys())
    model_name = model_types[0]

    n_datasets = len(all_results[experiment][model_name]['dataset_names'])
    if plot_latency_overlay:
        n_cols = 3
    else:
        n_cols = 4
    
    fig, axs = plt.subplots(n_datasets, n_cols, figsize=(12, n_datasets*4), sharey=True)

    for dataset_idx in range(n_datasets):

        try: # laminar results
            sess = get_session(*all_results[experiment][model_name]['dataset_names'][dataset_idx].split('_'))

            laminar_results = np.load(sess.sess_dir / 'laminar' / 'laminar.npz', allow_pickle=True)

            psd_dict = laminar_results['psd'].item()
            csd_dict = laminar_results['csd'].item()
            psd_data = psd_dict['psd']  # List of arrays, one per shank
            csd_data = csd_dict['csd']  # List of arrays, one per shank
            depths = psd_dict['depths']
            ddepth = np.diff(depths[0]).mean()
            

            ax = axs[dataset_idx, 0]
            # ax.imshow(np.concatenate(psd_data, 1), aspect='auto', cmap='YlOrBr', extent=[psd_dict['freqs'][0], psd_dict['freqs'][-1], depths[-1][-1], depths[0][0]], origin='lower')
            if collapse_shanks:
                psd_data = psd_data[0] + psd_data[1]
                ax.imshow(psd_data.T, aspect='auto', cmap='YlOrBr', extent=[psd_dict['freqs'][0], psd_dict['freqs'][-1], depths[0][0], depths[-1][-1]], origin='lower')
            else:
                for j in range(len(psd_data)):
                    psd_data[j] = psd_data[j].T
                    psd_data[j] = (psd_data[j] - psd_data[j].min()) / np.ptp(psd_data[j])
                ax.imshow(np.concatenate(psd_data, 1), aspect='auto', cmap='YlOrBr', extent=[psd_dict['freqs'][0], psd_dict['freqs'][-1], depths[0][0], depths[-1][-1]], origin='lower')

            ax.set_title(sess.sess_dir.name)
            # ax.axvline(psd_data[0].shape[1], color='k', linestyle='--')
            plot_laminar_boundaries(ax, layer_splits[sess.sess_dir.name])

            ax = axs[dataset_idx, -1]
            # ax.imshow(np.concatenate(psd_data, 1), aspect='auto', cmap='YlOrBr', extent=[psd_dict['freqs'][0], psd_dict['freqs'][-1], depths[-1][-1], depths[0][0]], origin='lower')
            if collapse_shanks:
                csd_data = csd_data[0] + csd_data[1]
                ax.imshow(csd_data.T, aspect='auto', cmap='YlOrBr', extent=[csd_dict['times'][0], csd_dict['times'][-1], depths[0][0], depths[-1][-1]], origin='lower')
            else:
                for j in range(len(csd_data)):
                    csd_data[j] = csd_data[j].T
                    csd_data[j] = (csd_data[j] - csd_data[j].min()) / np.ptp(csd_data[j])
                ax.imshow(np.concatenate(csd_data, 1), aspect='auto', cmap='jet', extent=[csd_dict['times'][0], csd_dict['times'][-1], depths[0][0], depths[-1][-1]], origin='lower')

            ax.set_title(sess.sess_dir.name)
            ax.set_yticks(np.arange(0, 32*ddepth, 100))
            # ax.axvline(csd_data[0].shape[1], color='k', linestyle='--')
            plot_laminar_boundaries(ax, layer_splits[sess.sess_dir.name])

        except:
            print(f"Failed to load laminar results for {all_results[experiment][model_name]['dataset_names'][dataset_idx]}")
        
        ch_depth, _ = argmax_subpixel(all_results[experiment][model_name]['qc']['waveforms'][dataset_idx].var(1), 1)
        probe_1 = ch_depth < 32

        try: # grating tuning
            ax = axs[dataset_idx, 1]

            ori_tuning_robs = all_results[experiment][model_name]['gratings']['ori_tuning_robs'][dataset_idx]
            ori_snr = all_results[experiment][model_name]['gratings']['ori_snr_robs'][dataset_idx]
            oris = np.linspace(0, 180, ori_tuning_robs.shape[1])
            dori = oris[1] - oris[0]

            for i in np.where(ori_snr > 1)[0]:
                tuning = ori_tuning_robs[i]
                tuning = (tuning - np.min(tuning)) / np.ptp(tuning)
                tuning *= ddepth
                ori_pref = argmax_subpixel(tuning)

                alpha = np.minimum((ori_snr[i]-.5)/3, 1)
                
                if probe_1[i]:
                    ax.plot(oris, tuning + ch_depth[i]*ddepth, 'gray', alpha=alpha)
                    ax.plot(ori_pref[0]*dori, tuning.max() + ch_depth[i] * ddepth, 'bo', alpha=alpha)
                else:
                    ax.plot(oris+180, tuning + ddepth * ch_depth[i] - 32 * ddepth, 'gray', alpha=alpha)
                    ax.plot(ori_pref[0]*dori+180, tuning.max() + ch_depth[i] * ddepth - 32 * ddepth, 'ro', alpha=alpha)
            
            plot_laminar_boundaries(ax, layer_splits[sess.sess_dir.name])
            ax.set_title('Orientation Tuning')
            ax.set_ylim(depths[0][0], depths[-1][-1])
            ax.set_yticks(np.arange(0, 32*ddepth, 100))
            ax.axvline(180, color='k', linestyle='--')
        except:
            print(f"Failed to load grating tuning results for {all_results[experiment][model_name]['dataset_names'][dataset_idx]}")
        
        try: # gabor latency
            if plot_latency_overlay:
                ax = axs[dataset_idx, 0].twiny()
            else:
                ax = axs[dataset_idx, 2]
            
            dt = 1000/120
            latency = all_results[experiment][model_name]['sta']['peak_lag_subpixel_robs'][dataset_idx] * dt
            ste_snr = all_results[experiment][model_name]['sta']['snr_ste_robs'][dataset_idx]
            ste_snr[np.isnan(ste_snr)] = 0

            for cc in range(len(latency)):
                if probe_1[cc]:
                    color = 'b'
                    offset = 0
                    t_offset = 0
                else:
                    color = 'r'
                    offset = 32 * ddepth
                    if not collapse_shanks:
                        t_offset = 100
                    else:
                        t_offset = 0
                    
                ax.plot(latency[cc]+t_offset, ch_depth[cc]*ddepth - offset, '.', color=color, alpha=np.minimum(ste_snr[cc]/10, 1))
            
            plot_laminar_boundaries(ax, layer_splits[sess.sess_dir.name])
            # ax.plot(latency, ch_depth * ddepth, 'ko', alpha=np.minimum(ste_snr/10, 1))
            if not plot_latency_overlay:
                ax.set_title('Latency (ms)')
            ax.set_ylim(depths[0][0], depths[-1][-1])
            
            ax.axvline(30, color='k', linestyle='--')
            ax.set_yticks(np.arange(0, 32*ddepth, 100))
            ax.set_xlim(0, 80)
            if not collapse_shanks:
                ax.set_xlim(0, 180)
                ax.axvline(30+t_offset, color='k', linestyle='--')


        except:
            print(f"Failed to load STA results for {all_results[experiment][model_name]['dataset_names'][dataset_idx]}")
        

    # for ax in axs.flatten():
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['bottom'].set_visible(False)
    #     ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig('laminar_results.png')


def get_depths(all_results, target_session='standard', concatenate=True, map_linear=True, layer4c_only=False):
    '''
    Get original and mapped depths for all cells in all_results.
        
    Parameters
    ----------
    all_results : dict
        Results dictionary from evaluation pipeline
    target_session : str
        Name of target session for mapping depths

    Returns
    -------
    original_depth : np.ndarray
        Array of original depths for each cell
    new_depth : np.ndarray
        Array of mapped depths for each cell
    '''
    experiment = list(all_results.keys())[0]
    print(experiment)
    model_name = list(all_results[experiment].keys())[0]
    
    original_depth = []
    new_depth = []

    results = all_results[model_name]
    qc_data = results['qc']
    
    # Get basic cell info
    cids = qc_data['all_cids']
    dids = qc_data['all_datasets']

    unique_datasets = list(set(dids))

    for dataset_idx in range(len(unique_datasets)):
        print(f"Processing dataset {unique_datasets[dataset_idx]}")
        sess = get_session(*unique_datasets[dataset_idx].split('_'))
        try:
            laminar_results = np.load(sess.sess_dir / 'laminar' / 'laminar.npz', allow_pickle=True)

            psd_dict = laminar_results['psd'].item()
            
            depths = psd_dict['depths']
            ddepth = np.diff(depths[0]).mean()
            print(ddepth)
        except:
            ddepth = 35

        ch_depth, _ = argmax_subpixel(all_results[experiment][model_name]['qc']['waveforms'][dataset_idx].var(1), 1)
        ch_depth[ch_depth>32] -= 32

        original_depth.append(ch_depth * ddepth)
        print(layer_splits[sess.sess_dir.name])
        # mapDepthsLinear
        if map_linear:
            new_depth.append(mapDepthsLinear(ch_depth * ddepth, layer_splits[sess.sess_dir.name], layer_splits[target_session])) #, layer4c_only=layer4c_only)
        else:
            new_depth.append(map_depths(ch_depth * ddepth, layer_splits[sess.sess_dir.name], layer_splits[target_session]))
    
    if concatenate:
        original_depth = np.concatenate(original_depth)
        new_depth = np.concatenate(new_depth)
    return original_depth, new_depth
