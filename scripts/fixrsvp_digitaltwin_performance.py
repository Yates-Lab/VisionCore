
# this is to suppress errors in the attempts at compilation that happen for one of the loaded models because it crashed
import sys
sys.path.append('..')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device, get_session, get_complete_sessions
from models.data import prepare_data
from models.config_loader import load_dataset_configs

import matplotlib.pyplot as plt

import torch
from torchvision.utils import make_grid

import matplotlib as mpl


# embed TrueType fonts in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# (optional) pick a clean sans‐serif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

enable_autoreload()
device = get_free_device()

from mcfarland_sim import run_mcfarland_on_dataset, extract_metrics
from utils import get_model_and_dataset_configs


#%% Get model and data
model, dataset_configs = get_model_and_dataset_configs()
model = model.to(device)

#%% Load outputs and analyzers from local file
# to regenerate, run fixrsvp_lotc_model.py with run_analysis = True
import dill
with open('mcfarland_outputs.pkl', 'rb') as f:
    outputs = dill.load(f)
with open('mcfarland_analyzers.pkl', 'rb') as f:
    analyzers = dill.load(f)


#%% Extract relevant metrics for plotting
metrics = extract_metrics(outputs, min_total_spikes=1000, min_var=.1, eps_rho=1e-3)


#%% CNN performance.


from mcfarland_sim import bootstrap_mean_ci
ccs = []
ccmaxs = []
bpss = []
for j in range(len(outputs)):
    cc = outputs[j]['ccnorm']['ccnorm']
    ccmax = outputs[j]['ccnorm']['ccmax']
    ccmaxs.append(ccmax)
    bps = outputs[j]['bps_results']['gaborium']['bps'][outputs[j]['neuron_mask']]
    ccs.append(cc)
    bpss.append(bps)

cc = np.concatenate(ccs)
ccmax = np.concatenate(ccmaxs)
bps = np.concatenate(bpss)
bins = np.linspace(0, 1, 10)
plt.plot(cc, ccmax, '.')
for i in range(len(bins)-1):
    ix = (ccmax > bins[i]) & (ccmax <= bins[i+1])
    plt.plot(np.nanmean(cc[ix])*np.array([1, 1]), np.array([bins[i], bins[i+1]]), 'r-')
    # add text with the n and the mean cc value
    plt.text(np.nanmean(cc[ix])+.5, (bins[i] + bins[i+1])/2, f"n={np.sum(ix)}, cc={np.nanmean(cc[ix]):.2f}")

plt.xlabel('Performance of model (CC_Norm)')
plt.ylabel('Reliability of neuron (CC_Max)')
plt.title("The model is better on more reliable neurons")

ix = (bps > 0.2) & (ccmax > 0.85)
plt.figure()
plt.hist(cc[ix], bins=np.linspace(0, 1, 50))
mu = np.nanmean(cc[ix])
bootstrap_mean_ci(cc[ix], seed=0)
plt.axvline(mu, color='r', linestyle='--')
plt.title(f"CCNorm (mu={mu:.2f}, n={np.sum(ix)})")
