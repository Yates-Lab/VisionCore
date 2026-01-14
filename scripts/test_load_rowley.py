#%%

import sys

sys.path.append('./scripts')
import numpy as np

from DataYatesV1 import enable_autoreload, get_free_device
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

#%%

from models.config_loader import load_dataset_configs

dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_240_rowley.yaml"
# dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_240_all.yaml"
    
dataset_configs = load_dataset_configs(dataset_configs_path)



# %%

from models.data import prepare_data

train_data, val_data, dataset_config = prepare_data(dataset_configs[0])
# %%
