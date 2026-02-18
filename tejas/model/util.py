from DataYatesV1 import  get_complete_sessions
from models.config_loader import load_dataset_configs
from models.data import prepare_data
import os
import contextlib
from DataYatesV1 import calc_sta
import numpy as np

def get_dataset_from_config(subject, date, dataset_configs_path):
    """
    Build a single dataset containing only fixrsvp trials from train and val splits.

    Loads the dataset config for the given session, calls prepare_data to get train/val
    datasets, then restricts to fixrsvp indices from both splits and returns one
    dataset object plus the config.

    Args:
        subject (str): Subject identifier (e.g. 'Allen', 'Ellie')
        date (str): Session date string in YYYY-MM-DD format (e.g. '2022-03-04')
        dataset_configs_path (str): Path to YAML file listing dataset configs
        dataset_type (str): Type of dataset to extract indices for (default: 'fixrsvp')

    Returns:
        dataset (DictDataset): Shallow copy of train dataset with inds set to 
            dataset_type indices only. Shape of inds: (N, 2) where N is total 
            number of indices from both train and val splits
        dataset_config (dict): Config dict for this session containing keys like
            'cids' (list of cluster IDs), 'session' (str), etc.
    
    Raises:
        AssertionError: If session {subject}_{date} is not in complete sessions list
        ValueError: If config not found for the session or dataset cannot be prepared
    """
    assert f'{subject}_{date}' in [sess.name for sess in get_complete_sessions()], f"Session {subject}_{date} not found"

    # =========================================================================
    # Load config and locate this session
    # =========================================================================
    dataset_configs = load_dataset_configs(dataset_configs_path)
    try:
        dataset_idx = next(i for i, cfg in enumerate(dataset_configs) if cfg['session'] == f"{subject}_{date}")
    except Exception as e:
        raise ValueError(f"config not found for {subject}_{date}")
    # =========================================================================
    # Prepare train/val datasets (suppress prepare_data stdout/stderr)
    # =========================================================================
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        try:
            train_dset, val_dset, dataset_config = prepare_data(dataset_configs[dataset_idx], strict=False)
        except Exception as e:
            raise ValueError(f"dataset not found for {subject}_{date}")

    return train_dset, val_dset, dataset_config


def get_dataset_info(dataset_configs_path, subject, date, image_shape):
    # subject = 'Allen'
    # date = '2022-04-13'

    # dataset_configs_path = '/home/tejas/VisionCore/experiments/dataset_configs/multi_basic_240_gaborium_20lags.yaml'
    train_dset, val_dset, dataset_config = get_dataset_from_config(subject, date, dataset_configs_path)
    cids = dataset_config['cids']


    train_dset_loaded = train_dset[:]


    stim = train_dset_loaded['stim']
    robs = train_dset_loaded['robs']
    dfs = train_dset_loaded['dfs']

    assert image_shape[0] == image_shape[1] and (stim.shape[-1] - image_shape[0]) % 2 == 0, "Image shape must be square and resulting crop size must be even"

    crop_size = (stim.shape[-1] - image_shape[0]) // 2
    y_slice = slice(crop_size, -crop_size) if crop_size > 0 else slice(None)
    x_slice = slice(crop_size, -crop_size) if crop_size > 0 else slice(None)
    n_lags = 5
    # Calculate spike-triggered averages (STAs)
    stas = calc_sta(stim.detach().cpu().squeeze()[:, 0, y_slice, x_slice],
                    robs.cpu(),
                    range(n_lags),
                    dfs=dfs.cpu().squeeze(),
                    progress=True).cpu().squeeze().numpy()

    # # Calculate spike-triggered second moments (STEs)
    # # Uses squared stimulus values via stim_modifier
    stes = calc_sta(stim.detach().cpu().squeeze()[:, 0, y_slice, x_slice],
                    robs.cpu(),
                    range(n_lags),
                    dfs=dfs.cpu().squeeze(),
                    stim_modifier=lambda x: x**2,
                    progress=True).cpu().squeeze().numpy()

    # plot_stas(stas[:, :, None, :, :])
    # plt.show()
    # plot_stas(stes[:, :, None, :, :])
    # plt.show()
    peak_lags = np.array([stes[cc].std((1,2)).argmax() for cc in range(stes.shape[0])])

    return {'peak_lags': peak_lags, 
    'stas': stas, 
    'stes': stes, 
    'cids': cids,
    'train_dset': train_dset, 
    'val_dset': val_dset, 
    'dataset_config': dataset_config,
    'stim': stim,
    'robs': robs,
    'dfs': dfs,
    'crop_size': crop_size}
