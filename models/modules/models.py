"""
Model architectures for DataYatesV1.

This module contains the main model classes including single-dataset and multi-dataset variants.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
from .norm_act_pool import get_activation_layer
import torch._dynamo as dynamo

# Type aliases for clarity
ConfigDict = Dict[str, Any]

class ModularV1Model(nn.Module):
    """
    A modular V1 model architecture that allows easy swapping of components.

    The basic model architecture is as follows:
    stim (B, C_stim, T, H, W) ─► adapter ─► frontend ─► convcore ─► (B,C_conv,T,H,W) features
                                                            |
                                            behaviour  ─► MLP  (C_b)
                                                            |
                                                concat (or FiLM) along channel dim
                                                    (B,C_mod,T,H,W)
                                                            ▼
                                                        ConvGRU
                                                        (B,C_rec, T, H, W)
                                                            ▼
                                                        readout (factorised Gaussian or linear)
                                                        (B, n_units)
                                                            ▼
                                                        activation (nn.Softplus()) ─► spikes
                                            

    This model is built from modular components that can be mixed and matched:
    - Adapter: Converts stimulus to a common format (e.g., smooth and grid_sample)
    - Frontend: Processes stimulus with a "front end" (da, conv, etc.) - "da" is the dynamic adaptation model from Clark et al., 2013.
    - ConvNet: Feature extraction (DenseNet, CNN, ResNet, etc.)
    - Modulator: Behavioral modulation (MLP, Linear, etc.) - Optional, combines with convnet output before recurrent or readout.
    - Recurrent: Temporal processing (ConvLSTM, ConvGRU, etc.) - Optional
    - Readout: Output layer (DynamicGaussian, Linear, etc.)

    The model can be configured to skip certain components (like recurrent layers)
    by setting their type to 'none'.
    """
    def __init__(self, config: ConfigDict):
        """
        Initialize the model with the given configuration.

        Args:
            config: Dictionary containing model configuration
        """
        super().__init__()

        # Extract basic parameters
        self.height = config.get('height', None)
        self.width = config.get('width', None)
        self.sampling_rate = config.get('sampling_rate', 240)
        self.initial_input_channels = config.get('initial_input_channels', 1)

        # Set up activation function
        self.activation = get_activation_layer(config.get('output_activation', 'none'))
        print(f"Model activation: {self.activation.__class__.__name__}")

        # Set up baseline configuration
        baseline_config = config.get('baseline', {'enabled': False})
        self.baseline_enabled = baseline_config.get('enabled', False)
        self.baseline_activation_type = baseline_config.get('activation', 'relu')
        self.baseline_init_value = baseline_config.get('init_value', 0.001)

        # Build the model components
        self._build_model(config)

    def _build_model(self, config: ConfigDict, verbose=False):
        """Build all model components based on configuration."""
        # Import factory functions to avoid circular imports
        from ..factory import create_frontend, create_convnet, create_modulator, create_recurrent, create_readout

        # Track channel dimensions between components
        current_channels = self.initial_input_channels
        if verbose:
            print(f"Initial channels: {current_channels}")

        # get all configs
        adapter_config = config.get('adapter', {'type': 'none', 'params': {}})
        frontend_config = config.get('frontend', {'type': 'none', 'params': {}})
        convnet_config = config.get('convnet', {'type': 'densenet', 'params': {}})
        modulator_config = config.get('modulator', {'type': 'none', 'mode': 'concatenate', 'params': {}})
        recurrent_config = config.get('recurrent', {'type': 'none', 'params': {}})
        readout_config = config.get('readout', {'type': 'gaussian', 'params': {}})

        # Build adapter
        assert adapter_config['type'] in ['none', 'adapter'], 'Adapter must be "none" or "adapter"'
        adapter_type = adapter_config['type']
        adapter_params = adapter_config['params']
        self.adapter, current_channels = create_frontend(
            frontend_type=adapter_type,
            in_channels=current_channels,
            sampling_rate=self.sampling_rate,
            **adapter_params
        )

        # Build frontend
        frontend_type = frontend_config['type']
        frontend_params = frontend_config['params']
        self.frontend, current_channels = create_frontend(
            frontend_type=frontend_type,
            in_channels=current_channels,
            sampling_rate=self.sampling_rate,
            **frontend_params
        )

        # Build convnet
        if verbose:
            print(f"Current channels before convnet: {current_channels}")
        convnet_type = convnet_config['type']
        convnet_params = convnet_config['params']
        self.convnet, current_channels = create_convnet(
            convnet_type=convnet_type,
            in_channels=current_channels,
            **convnet_params
        )
        if verbose:
            print(f"Current channels after convnet: {current_channels}")

        # Build modulator (needed before recurrent to know dimensions)
        modulator_type = modulator_config['type']
        modulator_params = modulator_config['params']
        modulator_params['feature_dim'] = current_channels
        self.modulator, modulator_dim = create_modulator(
            modulator_type=modulator_type,
            **modulator_params
        )

        # Build recurrent
        recurrent_type = recurrent_config['type']
        recurrent_params = recurrent_config['params']

        # update input_dims depending on recurrent type
        if recurrent_type in ['lstm', 'gru']:
            assert readout_config['type'] == 'linear', 'LSTM/GRU only compatible with linear readout. There are no spatial dimensions'
            # error out with message
            raise NotImplementedError('LSTM/GRU used, flattening conv output. This is not supported yet.')

        # update for modulator
        if self.modulator is not None and modulator_dim > 0:
            # For concat modulators, add the modulator output channels
            # For FiLM modulators, channel count stays the same
            modulator_type = modulator_config.get('type', 'none')
            if modulator_type == 'concat':
                current_channels += modulator_dim
            else:
                pass

        # create recurrent
        if verbose:
            print(f"Current channels before recurrent: {current_channels}")
        self.recurrent, current_channels = create_recurrent(
            recurrent_type=recurrent_type,
            input_dim=current_channels,
            **recurrent_params
        )

        # Build readout
        if verbose:
            print(f"Current channels before readout: {current_channels}")
        readout_type = readout_config['type']
        readout_params = readout_config['params']
        self.readout = create_readout(
            readout_type=readout_type,
            in_channels=current_channels,
            **readout_params
        )

        # Set up baseline parameters if enabled
        if self.baseline_enabled:
            n_units = readout_params.get('n_units', 16)
            self.baseline = nn.Parameter(torch.full((n_units,), self.baseline_init_value))

            # Set up baseline activation function
            if self.baseline_activation_type == 'relu':
                self.baseline_activation = nn.ReLU()
            elif self.baseline_activation_type == 'softplus':
                self.baseline_activation = nn.Softplus()
            else:
                raise ValueError(f"Unknown baseline activation: {self.baseline_activation_type}")
        else:
            self.baseline = None
            self.baseline_activation = None

    def core_forward(self, x, behavior=None):

        # Process through frontend
        x = self.frontend(x)

        # Process through convnet
        feats = self.convnet(x)

        # Process through modulator
        if self.modulator is not None and behavior is not None:
            feats = self.modulator(feats, behavior)

        # Process through recurrent
        feats = self.recurrent(feats)
        return feats

    def forward(self, stimulus, behavior=None):
        """
        Forward pass through the model.

        Args:
            stimulus: Visual stimulus tensor with shape (N, C, T, H, W)
            behavior: Optional behavioral data with shape (N, n_vars)

        Returns:
            Tensor: Model predictions with shape (N, n_units)
        """
        # Process through adapter
        x = self.adapter(stimulus)

        # if x is not None:
        #     dynamo.mark_dynamic(x, 0)  # batch
        
        # if behavior is not None:
        #     dynamo.mark_dynamic(behavior, 0)

        x_core = self.core_forward(x, behavior)

        # Process through readout
        output = self.readout(x_core)

        # Apply activation function
        output = self.activation(output)

        # Add baseline if enabled
        if self.baseline_enabled:
            baseline_output = self.baseline_activation(self.baseline)
            output = output + baseline_output

        return output


class MultiDatasetV1Model(ModularV1Model):
    """
    Multi-dataset variant of ModularV1Model.
    
    This model supports training on multiple datasets simultaneously by having:
    - Multiple adapters (one per dataset)
    - Shared frontend 
    - Shared convnet, modulator, and recurrent components
    - Multiple readouts (one per dataset)
    
    During training, the model routes data through the appropriate frontend/readout
    based on the dataset_idx parameter.
    """
    
    def __init__(self, model_config: ConfigDict, dataset_configs: List[ConfigDict]):
        """
        Initialize the multi-dataset model.
        
        Args:
            model_config: Main model configuration (shared components)
            dataset_configs: List of dataset-specific configurations
        """
        # Don't call super().__init__() as we need custom initialization
        nn.Module.__init__(self)
        
        # Store configurations
        self.model_config = model_config
        self.dataset_configs = dataset_configs
        self.num_datasets = len(dataset_configs)
        
        # Extract basic parameters from model config
        self.height = model_config.get('height', None)
        self.width = model_config.get('width', None)
        self.sampling_rate = model_config.get('sampling_rate', 240)
        self.initial_input_channels = model_config.get('initial_input_channels', 1)
        
        # Set up activation function
        self.activation = get_activation_layer(model_config.get('output_activation', 'none'))

        # Set up baseline configuration
        baseline_config = model_config.get('baseline', {'enabled': False})
        self.baseline_enabled = baseline_config.get('enabled', False)
        self.baseline_activation_type = baseline_config.get('activation', 'relu')
        self.baseline_init_value = baseline_config.get('init_value', 0.001)

        # Build the model components
        self._build_multidataset_model()
    
    def _build_multidataset_model(self):
        """Build all model components for multidataset training."""
        # Import factory functions to avoid circular imports
        from ..factory import create_frontend, create_convnet, create_modulator, create_recurrent, create_readout

        # Build per-dataset adapters
        self.adapters = nn.ModuleList()
        frontend_output_channels = None  # Will be set from first frontend

        default_adapter_config = self.model_config.get('adapter', {'type': 'none', 'params': {}})

        for dataset_config in self.dataset_configs:
            adapter_config = dataset_config.get('adapter', default_adapter_config)
            adapter_params = adapter_config.get('params') or {}

            adapter, _ = create_frontend(
                frontend_type=adapter_config['type'],
                in_channels=self.initial_input_channels,
                sampling_rate=self.sampling_rate,
                **adapter_params
            )

            self.adapters.append(adapter)

        # build front end
        frontend_config = self.model_config.get('frontend', {'type': 'none', 'params': {}})
        frontend_type = frontend_config['type']
        frontend_params = frontend_config.get('params') or {}
        self.frontend, frontend_output_channels = create_frontend(
            frontend_type=frontend_type,
            in_channels=self.initial_input_channels,
            sampling_rate=self.sampling_rate,
            **frontend_params
        )

        # Build shared convnet
        convnet_config = self.model_config.get('convnet', {'type': 'densenet', 'params': {}})
        convnet_type = convnet_config['type']
        convnet_params = convnet_config.get('params') or {}
        self.convnet, convnet_output_channels = create_convnet(
            convnet_type=convnet_type,
            in_channels=frontend_output_channels,
            **convnet_params
        )
        print(f"Convnet output channels: {convnet_output_channels}")

        # Build shared modulator
        modulator_config = self.model_config.get('modulator', {'type': 'none', 'params': {}})
        modulator_type = modulator_config['type']
        modulator_params = modulator_config.get('params') or {}
        if modulator_params:
            modulator_params = modulator_params.copy()
        else:
            modulator_params = {}
        modulator_params['feature_dim'] = convnet_output_channels
        self.modulator, modulator_dim = create_modulator(
            modulator_type=modulator_type,
            **modulator_params
        )
        print(f"Modulator output channels: {modulator_dim}")
        
        # Calculate channels after modulation
        current_channels = convnet_output_channels
        if self.modulator is not None and modulator_dim > 0:
            if modulator_type == 'concat':
                current_channels += modulator_dim
            elif modulator_type in ['film', 'stn']:
                # FiLM and STN don't change channel count
                pass
            elif modulator_type in ['pc', 'convgru']:
                # PC and ConvGRU modulators output their own dimension
                # modulator_dim already contains the correct output dimension
                current_channels = modulator_dim
            else:
                raise ValueError(f"Unknown modulator type: {modulator_type}")
        
        print(f"Channels after modulation: {current_channels}")

        # Build shared recurrent
        recurrent_config = self.model_config.get('recurrent', {'type': 'none', 'params': {}})
        recurrent_type = recurrent_config['type']
        recurrent_params = recurrent_config.get('params') or {}
        self.recurrent, recurrent_output_channels = create_recurrent(
            recurrent_type=recurrent_type,
            input_dim=current_channels,
            **recurrent_params
        )
        
        # Build per-dataset readouts
        self.readouts = nn.ModuleList()

        # Get default readout config from model config
        model_readout_config = self.model_config.get('readout', {'type': 'gaussian', 'params': {}})

        for dataset_config in self.dataset_configs:
            # Use dataset-specific readout config if available, otherwise fall back to model config
            if 'readout' in dataset_config:
                readout_config = dataset_config['readout']
            else:
                readout_config = model_readout_config

            readout_type = readout_config['type']
            # Handle case where params is None (e.g., when YAML has only comments)
            readout_params = readout_config.get('params') or {}
            if readout_params:
                readout_params = readout_params.copy()
            else:
                readout_params = {}

            # Set n_units based on dataset cids
            cids = dataset_config.get('cids', [])
            readout_params['n_units'] = len(cids)

            readout = create_readout(
                readout_type=readout_type,
                in_channels=recurrent_output_channels,
                **readout_params
            )
            self.readouts.append(readout)

        # Set up per-dataset baseline parameters if enabled
        if self.baseline_enabled:
            self.baselines = nn.ParameterList()
            for dataset_config in self.dataset_configs:
                cids = dataset_config.get('cids', [])
                n_units = len(cids)
                baseline = nn.Parameter(torch.full((n_units,), self.baseline_init_value))
                self.baselines.append(baseline)

            # Set up baseline activation function
            if self.baseline_activation_type == 'relu':
                self.baseline_activation = nn.ReLU()
            elif self.baseline_activation_type == 'softplus':
                self.baseline_activation = nn.Softplus()
            else:
                raise ValueError(f"Unknown baseline activation: {self.baseline_activation_type}")
        else:
            self.baselines = None
            self.baseline_activation = None


    def core_forward(self, stimulus=None, behavior=None):

        # route through frontend
        feats = self.frontend(stimulus)

        # Process through shared convnet
        feats = self.convnet(feats)

        # Process through shared modulator
        if self.modulator is not None and behavior is not None:
            feats = self.modulator(feats, behavior)

        # Process through shared recurrent
        x_recurrent = self.recurrent(feats)

        return x_recurrent
    
    def forward(self, stimulus=None, dataset_idx: int = 0, behavior=None, history=None):
        """
        Forward pass through the model for a specific dataset.

        Args:
            stimulus: Visual stimulus tensor with shape (N, C, T, H, W) or None for modulator-only models
            dataset_idx: Index of the dataset (determines frontend/readout)
            behavior: Optional behavioral data with shape (N, n_vars)
            history: Optional spike history tensor (unused in base class, used in subclasses)

        Returns:
            Tensor: Model predictions with shape (N, n_units_for_dataset)
        """
        x = self.adapters[dataset_idx](stimulus)

        if x is None:
            # Modulator-only mode: create minimal features for modulator
            # Use 1 channel to match what convnet outputs during initialization
            B = behavior.shape[0]
            device = next(self.parameters()).device
            x = torch.ones(B, 1, 1, 1, 1, device=device, dtype=behavior.dtype)

        x = self.core_forward(x, behavior)

        # Route through appropriate readout
        output = self.readouts[dataset_idx](x)

        # Apply activation function
        output = self.activation(output)

        # Add baseline if enabled
        if self.baseline_enabled:
            baseline_output = self.baseline_activation(self.baselines[dataset_idx])
            output = output + baseline_output

        return output

class MultiDatasetV1ModelSpikeHistory(MultiDatasetV1Model):
    """
    Multi-dataset V1 model with spike history processing.

    This model extends MultiDatasetV1Model by adding per-dataset MLPs that process
    recent spike history and modulate the output. The history MLPs can use spectral
    normalization to prevent instability during simulation.
    """

    def __init__(self, model_config: ConfigDict, dataset_configs: List[ConfigDict]):
        super().__init__(model_config, dataset_configs)
        from models.modules.mlp import MLP

        # Configure MLP for processing spike history
        # This will be an MLP that takes in the num_lags x n_units input and pushes it
        # through an MLP with a bottleneck and then projects to n_units output to combine
        # with the readout output
        self.spike_history = nn.ModuleList()
        self.history_config = model_config.get('history_encoder', {
            'type': 'mlp',
            'params': {
                'num_lags': 5,
                'hidden_dims': [32, 10],
                'act_type': 'relu',
                'spectral_norm': False
            }
        })

        for dataset_idx in range(len(self.readouts)):
            n_units = self.readouts[dataset_idx].n_units

            # Create a copy of params to avoid mutation
            config = self.history_config['params'].copy()
            num_lags = config.pop('num_lags', 5)
            config['input_dim'] = num_lags * n_units
            config['output_dim'] = n_units

            self.spike_history.append(MLP(**config))

    def forward(self, stimulus=None, dataset_idx: int = 0, behavior=None, history=None):
        """
        Forward pass with spike history processing.

        Args:
            stimulus: Visual stimulus tensor with shape (N, C, T, H, W) or None
            dataset_idx: Index of the dataset (determines adapter/readout)
            behavior: Optional behavioral data with shape (N, n_vars)
            history: Spike history tensor with shape (N, num_lags, n_units) or (N, num_lags * n_units)

        Returns:
            Tensor: Model predictions with shape (N, n_units_for_dataset)
        """
        x = self.adapters[dataset_idx](stimulus)

        if x is None:
            # Modulator-only mode: create minimal features for modulator
            # Use 0 channels so concat modulator only returns behavior embedding
            B = behavior.shape[0]
            device = next(self.parameters()).device
            x = torch.ones(B, 0, 1, 1, 1, device=device, dtype=behavior.dtype)

        x = self.core_forward(x, behavior)

        # Route through appropriate readout
        output = self.readouts[dataset_idx](x)

        # Process history through MLP
        # Flatten history if needed: (B, num_lags, n_units) -> (B, num_lags * n_units)
        if history is not None:
            if history.dim() == 3:
                B, num_lags, n_units = history.shape
                history = history.reshape(B, num_lags * n_units)
            history_output = self.spike_history[dataset_idx](history)
            # Combine with readout output
            output = output + history_output

        # Apply activation function
        output = self.activation(output)

        # Add baseline if enabled
        if self.baseline_enabled:
            baseline_output = self.baseline_activation(self.baselines[dataset_idx])
            output = output + baseline_output

        return output