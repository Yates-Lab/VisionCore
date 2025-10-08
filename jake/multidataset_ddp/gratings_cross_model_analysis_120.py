"""
Gratings Cross-Model Analysis for Dataset 120

This script performs gratings stimulus analysis across multiple neural network models,
comparing model predictions to observed neural responses. It extracts modulation indices
and generates comparison plots for each unit.

Main workflow:
1. Discover and load available trained models
2. Evaluate models on gratings datasets
3. Extract modulation indices from sine fits
4. Generate comparison plots and save to PDF
"""

#%% Setup and Imports
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import sys
from pathlib import Path
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

# Custom modules for model evaluation and gratings analysis
from eval_stack_multidataset import evaluate_model_multidataset, load_single_dataset, load_model
from eval_stack_utils import scan_checkpoints
from DataYatesV1 import enable_autoreload
from gratings_analysis import plot_gratings_comparison, gratings_comparison

enable_autoreload()

#%% Discover Available Models
print("🔍 Discovering available models...")
checkpoint_dir = '/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_smooth_120/checkpoints'
models_by_type = scan_checkpoints(checkpoint_dir)

print(f"Found {len(models_by_type)} model types:")
for model_type, models in models_by_type.items():
    if models:
        best_model = models[0]
        if best_model.get('metric_type') == 'bps' and best_model.get('val_bps') is not None:
            best_metric = f"best BPS: {best_model['val_bps']:.4f}"
        else:
            best_metric = f"best loss: {best_model['val_loss']:.4f}"
        print(f"  {model_type}: {len(models)} models ({best_metric})")
    else:
        print(f"  {model_type}: 0 models")

#%% Load Primary Model for Analysis
# Load the main model that will be used for gratings analysis
model, model_info = load_model(
    model_type='learned_res_small_gru',
    model_index=None,
    checkpoint_path=None,
    checkpoint_dir=checkpoint_dir,
    device='cpu'
)

model.eval()


#%% Load Multiple Models for Comparison
print("\n📊 Loading models for comparison...")

# Define models to compare (currently only using learned_res_small)
# Additional models can be uncommented: 'learned_res_small_gru', 'learned_res_small_pc', etc.
# models_to_compare = ['learned_res_small_gru', 'learned_res_small_none_gru', 'learned_res_small_none_gru_none_pool'] #'learned_res_small', 
models_to_compare = [
    'learned_res_small_gru', 
    'learned_res_small_none_gru_none_pool', 
    'learned_res_small_none_gru', 
    'learned_res2d_small_none_gru_none_pool', 
    'learned_res_small', 
    'learned_res2d_small'
]
available_models = [m for m in models_to_compare if m in models_by_type]

print(f"Comparing models: {available_models}")

# Evaluate each model and store results
all_results = {}
for model_type in available_models:
    print(f"\nLoading {model_type}...")

    # Run comprehensive evaluation including BPS, CCNORM, saccade, and STA analyses
    results = evaluate_model_multidataset(
        model_type=model_type,
        analyses=['bps', 'ccnorm', 'saccade', 'sta'],
        checkpoint_dir=checkpoint_dir,
        save_dir="/mnt/ssd/YatesMarmoV1/conv_model_fits/eval_stack_smooth_120",
        recalc=True,  # Use cached results if available
        batch_size=64
    )
    all_results.update(results)
    model_name = list(results.keys())[0]
    n_cells = len(results[model_name]['qc']['all_cids'])
    print(f"  ✅ {model_name}: {n_cells} cells")

    # Incremental save to prevent data loss during long evaluations
    save_path = Path('all_results_120_analysis_incremental.pkl')
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"  💾 Incremental save: {len(all_results)} models ({save_path.stat().st_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"  ⚠️ Incremental save failed: {e}")

print(f"\n✅ Loaded {len(all_results)} models for comparison")


#%%
# fpath = '/home/jake/repos/DataYatesV1/jake/multidataset_ddp/all_results_120_analysis.pkl'
# fpath = '/home/tejas/Documents/fixational-transients/DataYatesV1/jake/multidataset_ddp/all_results_120_analysis.pkl'
fpath = 'all_results_120_analysis_incremental.pkl'
with open(fpath, 'rb') as f:
    results = pickle.load(f)

#%% Gratings Analysis Across Datasets
# Process gratings stimuli across all 20 datasets
for i in range(len(results.keys())):
    gratings_results = {}

    # Process each of the 20 datasets to find and analyze gratings stimuli
    for dset_idx in range(20):
        train_data, val_data, dataset_config = load_single_dataset(model, dset_idx)
        session = dataset_config['session']

        # Check if this dataset contains gratings stimuli
        try:
            gratings_ind = int(np.where([d.metadata['name'] == 'gratings' for d in train_data.dsets])[0])
        except:
            print(f'No gratings dataset for {model.names[dset_idx]}')
            continue

        # Extract gratings dataset and validation indices
        gratings_dset = train_data.dsets[gratings_ind]
        train_inds = train_data.inds[train_data.inds[:,0] == gratings_ind][:,1]
        val_inds = val_data.inds[val_data.inds[:,0] == gratings_ind][:,1]

        # Use validation indices for analysis (more reliable for model evaluation)
        inds = val_inds
        print(f'Processing dataset {dset_idx}: {len(val_inds)} validation samples')

        # Extract stimulus parameters and neural responses
        n_lags = dataset_config['keys_lags']['stim'][-1]
        robs = gratings_dset['robs'][inds].numpy()  # Observed neural responses
        rhat = results[list(results.keys())[i]]['bps']['gratings']['rhat'][dset_idx].numpy()  # Model predictions

        print(f'R_obs: {robs.shape}, R_hat: {rhat.shape}')

        # Extract stimulus properties
        sf = gratings_dset['sf'][inds]  # Spatial frequency
        ori = gratings_dset['ori'][inds]  # Orientation
        phases = gratings_dset['stim_phase'][inds]
        phases = phases[:,phases.shape[1]//2, phases.shape[2]//2]  # Center pixel phase
        dt = 1/dataset_config['sampling']['target_rate']  # Time step
        dfs = gratings_dset['dfs'].numpy().squeeze()[inds]  # Drift frequency

        # Perform gratings comparison analysis (sine fitting, modulation indices, etc.)
        r = gratings_comparison(
            robs=robs,
            rhat=rhat,
            sf=sf,
            ori=ori,
            phases=phases,
            dt=dt,
            n_lags=n_lags,
            dfs=dfs,
            min_spikes=30  # Minimum spike count threshold for analysis
        )
        gratings_results[session] = r

        print('\n------------------------------------\n')

    # save gratings_results to a file that can be laoded on another machine

    # import pickle

    # fname = 'gratings_results_120.pkl'
    # with open(fname, 'wb') as f:
    #     pickle.dump(gratings_results, f)


    # Extract and Compare Modulation Indices
    # Collect modulation indices from sine fits for both observed and predicted responses
    robs_modulation_index = []
    rhat_modulation_index = []

    for r in gratings_results.values():
        # Extract modulation indices from observed responses
        for sine_fit in r['robs']['sine_fit_results']:
            if sine_fit is not None:
                robs_modulation_index.append(sine_fit['modulation_index'])
            else:
                robs_modulation_index.append(np.nan)

        # Extract modulation indices from model predictions
        for sine_fit in r['rhat']['sine_fit_results']:
            if sine_fit is not None:
                rhat_modulation_index.append(sine_fit['modulation_index'])
            else:
                rhat_modulation_index.append(np.nan)

    # Convert to arrays and count valid units
    robs_modulation_index = np.array(robs_modulation_index)
    rhat_modulation_index = np.array(rhat_modulation_index)
    n_units = np.sum(~np.isnan(robs_modulation_index) & ~np.isnan(rhat_modulation_index))

    # Create scatter plot comparing observed vs predicted modulation indices
    plt.figure(figsize=(8, 6))
    plt.plot(robs_modulation_index, rhat_modulation_index, 'C0.', alpha=0.6)
    plt.plot(plt.xlim(), plt.xlim(), 'k--', alpha=0.5, label='Unity line')
    plt.xlabel('Observed Modulation Index')
    plt.ylabel('Model Modulation Index')
    plt.title(f'Model: {list(results.keys())[i]}\nUnits: {n_units}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save chart
    import os
    os.makedirs('modulation_indices', exist_ok=True)
    plt.savefig(f'modulation_indices/modulation_comparison_{list(results.keys())[i]}.png', dpi=300, bbox_inches='tight')

    plt.show()

#%% Generate Comprehensive PDF Report
# Create detailed plots for each unit across all datasets
pdf_name = 'gratings_results.pdf'
n_units_total = np.sum([d['robs']['n_units'] for d in gratings_results.values()])

print(f"Generating PDF report with {n_units_total} units...")

with PdfPages(pdf_name) as pdf:
    with tqdm(total=n_units_total, desc='Plotting Gratings Results') as pbar:
        for session, r in gratings_results.items():
            for unit_idx in range(r['robs']['n_units']):
                # Generate comparison plot for this unit
                fig, axs = plot_gratings_comparison(r, unit_idx)

                # Add dataset information to title
                original_title = axs[0].title.get_text()
                axs[0].set_title(f'{session} - {original_title}')

                # Save to PDF and close figure to save memory
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                pbar.update(1)

print(f"✅ PDF report saved as '{pdf_name}'")



# %%
