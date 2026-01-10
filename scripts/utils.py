
'''
This code defines a collection of shared helper functions that are used across
many of the analysis scripts. These utilities handle common tasks such as
loading model configurations and dataset configurations, ensuring a single,
consistent interface for accessing experimental and modeling parameters. 
Centralizing this functionality allows changes to configuration logic to propagate.
'''

#%% Imports
import sys
sys.path.append('..')
import numpy as np

from models.config_loader import load_dataset_configs
from eval.eval_stack_multidataset import load_model

def get_model_and_dataset_configs():
    dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_240_all.yaml"
    checkpoint_dir = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/multidataset_120_long/checkpoints"

    model_type = 'resnet_none_convgru'
    model, model_info = load_model(
        model_type=model_type,
        model_index=0, # none for best model
        checkpoint_path=None,
        checkpoint_dir=checkpoint_dir,
        device='cpu'
    )

    model.model.eval()
    model.model.convnet.use_checkpointing = False 
    dataset_configs = load_dataset_configs(dataset_configs_path)

    return model, dataset_configs

