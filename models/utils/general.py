import sys
import lightning as pl
import torch

class ValidateOnTrainStart(pl.Callback):
    """
    Callback to run validation at the start of training.
    """
    def __init__(self):
        pass

    def on_train_start(self, trainer, pl_module):
        return trainer.fit_loop.epoch_loop.val_loop.run()

def get_valid_dfs(dset, n_lags):
    """
    Generate a binary mask for valid data frames based on trial boundaries and DPI validity.
    
    This function creates a mask that identifies valid frames for analysis by:
    1. Identifying trial boundaries
    2. Excluding the first frame of each trial
    3. Ensuring DPI (eye tracking) data is valid
    4. Ensuring temporal continuity for the specified number of lags
    
    Parameters:
    -----------
    dset : DictDataset
        Dataset containing trial indices and DPI validity information
    n_lags : int
        Number of time lags to ensure continuity for
        
    Returns:
    --------
    dfs : torch.Tensor
        Binary mask tensor of shape [n_frames, 1] where 1 indicates valid frames
    """
    dpi_valid = dset['dpi_valid']
    new_trials = torch.diff(dset['trial_inds'], prepend=torch.tensor([-1])) != 0
    dfs = ~new_trials
    dfs &= (dpi_valid > 0)

    for _ in range(n_lags):
        dfs &= torch.roll(dfs, 1)
    
    dfs = dfs.float()
    dfs = dfs[:, None]
    return dfs

def get_optimizer(parameters, config:dict):
    """
    Create an optimizer based on configuration dictionary.
    
    Parameters:
    -----------
    parameters : iterable
        Model parameters to optimize
    config : dict
        Configuration dictionary with optimizer settings.
        Must contain 'optimizer' key specifying the optimizer type.
        Remaining keys are passed as keyword arguments to the optimizer.
        
    Returns:
    --------
    optimizer : torch.optim.Optimizer
        Configured optimizer instance
    """
    name = config.pop('optimizer')
    if name == 'SGD':
        optimizer = torch.optim.SGD(parameters, **config)
    elif name == 'Adam':
        optimizer = torch.optim.Adam(parameters, **config)
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, **config)
    elif name == 'SGDScheduleFree': 
        from schedulefree import SGDScheduleFree
        optimizer = SGDScheduleFree(parameters, **config)
    elif name == 'AdamWScheduleFree':
        from schedulefree import AdamWScheduleFree
        optimizer = AdamWScheduleFree(parameters, **config)
    else:
        raise ValueError(f"Optimizer {name} not recognized")
    return optimizer

