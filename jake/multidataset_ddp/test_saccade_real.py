#!/usr/bin/env python
"""
Test script for updated saccade analysis functions with real model and data.

Based on model_analysis_devel.py to test the new saccade analysis functionality.
"""

#%% Setup and Imports
import torch._dynamo 
torch._dynamo.config.suppress_errors = True # suppress dynamo errors

import sys
sys.path.append('.')

import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from DataYatesV1 import enable_autoreload, get_free_device, get_session
from eval_stack_multidataset import load_model, run_saccade_analysis, load_single_dataset
from eval_stack_utils import get_saccade_eval, detect_saccades_from_session, get_stim_inds, evaluate_dataset

enable_autoreload()

#%% Test Configuration
device = get_free_device()
print(f"Using device: {device}")

checkpoint_dir = '/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120/checkpoints'

if not os.path.exists(checkpoint_dir):
    print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
    print("Please update the checkpoint_dir path")
    exit(1)

#%% Load Model
print("🔍 Loading model...")

model_type = 'learned_res_small_gru'
print(f"Loading model type: {model_type}")

model, model_info = load_model(
    model_type=model_type,
    model_index=None,  # none for best model
    checkpoint_path=None,
    checkpoint_dir=checkpoint_dir,
    device='cpu'
)

model.model.eval()
model.model.convnet.use_checkpointing = False 
model = model.to(device)

model_name = model_info.get('experiment', 'unknown')
print(f"✅ Model loaded successfully: {model_name}")

#%% Load Dataset
dataset_idx = 0  # Start with first dataset
batch_size = 64

print(f"Loading dataset {dataset_idx}...")
train_data, val_data, dataset_config = load_single_dataset(model, dataset_idx)
dataset_cids = dataset_config.get('cids', [])

print(f"✅ Dataset loaded: {len(dataset_cids)} cells")
print(f"Dataset name: {model.names[dataset_idx]}")

#%% Test Saccade Detection
dataset_name = model.names[dataset_idx]
print(f"Testing saccade detection for: {dataset_name}")

sess = get_session(*dataset_name.split('_'))
saccades = detect_saccades_from_session(sess)

print(f"✅ Saccade detection: Found {len(saccades)} saccades")

if len(saccades) == 0:
    print("❌ No saccades found - cannot test saccade analysis")
    exit(1)

# Test the structure of saccades
print(f"📊 First saccade keys: {list(saccades[0].keys())}")

#%% Test BPS Analysis (needed for saccade analysis)
print("Running BPS analysis (needed for saccade analysis)...")

# Get stimulus indices for backimage (commonly available)
stim_type = 'backimage'
stim_inds = get_stim_inds(stim_type, train_data, val_data)

print(f"Evaluating {stim_type} stimulus...")

# Run evaluation to get robs and pred
eval_results = evaluate_dataset(model, val_data, stim_inds, dataset_idx, batch_size=batch_size, desc=f"{stim_type} evaluation")

print(f"✅ BPS analysis completed")
print(f"   robs shape: {eval_results['robs'].shape}")
print(f"   pred shape: {eval_results['rhat'].shape}")

# Create eval_dict structure expected by get_saccade_eval
eval_dict = {stim_type: (eval_results['robs'], eval_results['rhat'], eval_results['bps'])}

#%% Test Updated get_saccade_eval Function
print("\n🧪 Testing updated get_saccade_eval function...")

sac_win = (-10, 100)  # Time window around saccades

try:
    sac_eval = get_saccade_eval(stim_type, train_data, val_data, eval_dict, saccades, win=sac_win)
    
    print("✅ get_saccade_eval completed successfully!")
    
    # Check the return structure
    expected_keys = ['robs', 'rhat', 'dfs', 'rbar', 'rbarhat', 'eyevel', 'saccade_info', 'win']
    actual_keys = list(sac_eval.keys())
    
    print(f"📋 Expected keys: {expected_keys}")
    print(f"📋 Actual keys: {actual_keys}")
    
    missing_keys = set(expected_keys) - set(actual_keys)
    extra_keys = set(actual_keys) - set(expected_keys)
    
    if missing_keys:
        print(f"❌ Missing keys: {missing_keys}")
    if extra_keys:
        print(f"⚠️  Extra keys: {extra_keys}")
    
    if not missing_keys:
        print("✅ All expected keys present!")
        
        # Check shapes
        print("\n📐 Data shapes:")
        for key, value in sac_eval.items():
            if isinstance(value, np.ndarray):
                print(f"   {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"   {key}: list of length {len(value)}")
            else:
                print(f"   {key}: {type(value)} - {value}")
        
        # Check saccade_info structure
        if len(sac_eval['saccade_info']) > 0:
            print(f"\n📝 First saccade_info keys: {list(sac_eval['saccade_info'][0].keys())}")
            
            # Check for time_previous and time_next
            if 'time_previous' in sac_eval['saccade_info'][0]:
                print("✅ time_previous field found in saccade_info")
            else:
                print("❌ time_previous field missing from saccade_info")
                
            if 'time_next' in sac_eval['saccade_info'][0]:
                print("✅ time_next field found in saccade_info")
            else:
                print("❌ time_next field missing from saccade_info")
        
        print(f"\n🎯 Found {len(sac_eval['saccade_info'])} valid saccades after filtering")
        
except Exception as e:
    print(f"❌ get_saccade_eval failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("🎉 Test completed!")
print("Ready to run full analysis if all tests passed.")
c = 0
# %%

time_next = np.array([s['time_next'] for s in sac_eval['saccade_info']])
ind = np.argsort(time_next)
c += 1
plt.imshow(np.sqrt(np.sum(sac_eval['eyevel'][ind,:,:]**2, -1)), aspect='auto', interpolation='none', cmap='gray_r')
plt.title('robs')
plt.show()
# %%
