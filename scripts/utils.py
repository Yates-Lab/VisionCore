
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

def get_model_and_dataset_configs(mode='standard'):

    if mode == 'standard':
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
    elif mode == 'frozencore':
        checkpoint_path = "/mnt/ssd/YatesMarmoV1/conv_model_fits/experiments/frozencore_readouts_120/checkpoints/frozencore_resnet_none_convgru_bs256_ds30_lr1e-3_wd1.0e-5_warmup5/epoch=46-val_bps_overall=0.5462.ckpt"
        dataset_configs_path = "/home/jake/repos/VisionCore/experiments/dataset_configs/multi_basic_120_long_rowley.yaml"
        from training.pl_modules import FrozenCoreModel
        import torch
        # Device to load on
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #%% Load the model
        print(f"Loading FrozenCoreModel from: {checkpoint_path}")

        model = FrozenCoreModel.load_from_checkpoint(
            checkpoint_path,
            map_location='cpu',
            strict=False
        )

        model.to(device)
        model.eval()
        
        dataset_configs = load_dataset_configs(dataset_configs_path)
    
    # loop over dataset configs, check if fixrsvp is present and if not, remove it
    for i in range(len(dataset_configs)-1, -1, -1):
        if 'fixrsvp' not in dataset_configs[i]['types']:
            dataset_configs[i]['types'] += ['fixrsvp']

    return model, dataset_configs

