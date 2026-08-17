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


def require_behavior(modulator, behavior, where="core_forward"):
    """Refuse to run a behavior-conditioned model without a behavior tensor.

    A `concat` modulator widens the feature stack by `modulator_dim` channels,
    and the recurrent layer is built for that width. Skipping the modulator
    because `behavior` happens to be None therefore feeds the recurrent stack
    the wrong channel count: a crash if you are lucky, silently wrong features
    if you are not.

    Parameters
    ----------
    modulator : nn.Module or None
        The model's modulator. None (a `none`-modulator twin) always passes.
    behavior : torch.Tensor or None
        The behavior tensor for this forward pass.
    where : str
        Call site name, for the error message.

    Raises
    ------
    ValueError
        If the model has a modulator but no behavior was supplied.
    """
    if modulator is None or behavior is not None:
        return

    behavior_dim = getattr(modulator, "behavior_dim", None)
    dim_hint = (f"shape (batch, {behavior_dim})" if behavior_dim is not None
                else "the modulator's behavior_dim")
    raise ValueError(
        f"{where} received behavior=None on a model with a "
        f"{type(modulator).__name__}. This model is behavior-conditioned: its "
        f"configured behavior path cannot be skipped safely (feature-space "
        f"concatenation changes tensor width, while output residuals change "
        f"neural logits). Pass "
        f"a tensor of {dim_hint}. Note that passing zeros is a deliberate "
        f"ablation, not a neutral default -- for the current twin the behavior "
        f"input is entirely eye-movement derived (eye velocity through a "
        f"raised-cosine basis, plus raw eye position).")

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
            if modulator_type in ['concat', 'mlp_behavior']:
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
        require_behavior(self.modulator, behavior, where="ModularV1Model.core_forward")
        if self.modulator is not None:
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
        base_input_temporal_support = model_config.get(
            'base_input_temporal_support', None
        )
        if base_input_temporal_support is None:
            self.base_input_temporal_support = None
        else:
            base_input_temporal_support = int(base_input_temporal_support)
            if base_input_temporal_support <= 0:
                raise ValueError(
                    "base_input_temporal_support must be positive"
                )
            self.base_input_temporal_support = base_input_temporal_support
        base_input_crop = model_config.get('base_input_crop', None)
        if base_input_crop is None:
            self.base_input_crop = None
        elif isinstance(base_input_crop, int):
            if base_input_crop <= 0:
                raise ValueError("base_input_crop must be positive")
            self.base_input_crop = (base_input_crop, base_input_crop)
        else:
            if len(base_input_crop) != 2 or any(int(v) <= 0 for v in base_input_crop):
                raise ValueError(
                    "base_input_crop must be a positive integer or [height, width]"
                )
            self.base_input_crop = tuple(int(v) for v in base_input_crop)
        self.feature_behavior_source = model_config.get(
            'feature_behavior_source', 'behavior'
        )
        if self.feature_behavior_source not in {'behavior', 'output_behavior'}:
            raise ValueError(
                "feature_behavior_source must be 'behavior' or "
                f"'output_behavior', got {self.feature_behavior_source!r}"
            )
        
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
            if modulator_type in ['concat', 'mlp_behavior']:
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

        # Optional identity-initialized visual boosting branch.  The mature
        # model supplies the baseline prediction while a second, explicitly
        # configured feed-forward core learns only the residual visual signal.
        # Its per-neuron feature projections are initialized to zero, making a
        # compatible warm start exactly identical to the source checkpoint.
        # Keeping this path separate from the feature modulator also prevents
        # behavior channels from becoming an accidental visual shortcut.
        auxiliary_visual_config = self.model_config.get(
            'auxiliary_visual', {'type': 'none', 'params': {}}
        )
        auxiliary_visual_type = auxiliary_visual_config.get('type', 'none')
        if auxiliary_visual_type in (None, 'none'):
            self.auxiliary_convnet = None
            self.auxiliary_readouts = None
            self.auxiliary_visual_output_channels = None
        else:
            from .readout import (
                DynamicGaussianReadout,
                ResidualGaussianReadout,
            )

            if not all(
                isinstance(readout, DynamicGaussianReadout)
                for readout in self.readouts
            ):
                raise TypeError(
                    "auxiliary_visual requires Gaussian base readouts"
                )
            auxiliary_params = dict(
                auxiliary_visual_config.get('params') or {}
            )
            auxiliary_core_params = dict(
                auxiliary_params.pop('core', {}) or {}
            )
            auxiliary_readout_params = dict(
                auxiliary_params.pop('readout', {}) or {}
            )
            if auxiliary_params:
                raise ValueError(
                    "Unknown auxiliary_visual params: "
                    f"{sorted(auxiliary_params)}"
                )
            self.auxiliary_convnet, auxiliary_channels = create_convnet(
                convnet_type=auxiliary_visual_type,
                in_channels=frontend_output_channels,
                **auxiliary_core_params,
            )
            self.auxiliary_visual_output_channels = int(auxiliary_channels)
            auxiliary_readout_params.setdefault(
                'feature_mode', 'independent'
            )
            if auxiliary_readout_params['feature_mode'] != 'independent':
                raise ValueError(
                    "auxiliary_visual currently requires an independent "
                    "feature projection because its channel basis differs "
                    "from the mature core"
                )
            self.auxiliary_readouts = nn.ModuleList(
                ResidualGaussianReadout(
                    in_channels=auxiliary_channels,
                    n_units=len(config.get('cids', [])),
                    **auxiliary_readout_params,
                )
                for config in self.dataset_configs
            )

        # A second identity-initialized component may reuse the mature
        # auxiliary core without changing either visual feature bank.  This
        # is deliberately separate from ``auxiliary_readouts``: each neuron
        # gets two independently localized projections of the same smooth
        # auxiliary features, which can express center/surround or paired
        # subfields while keeping the stimulus Jacobian as a sum of localized
        # feed-forward terms.
        auxiliary_residual_config = self.model_config.get(
            'auxiliary_residual_readout', {'type': 'none', 'params': {}}
        )
        auxiliary_residual_type = auxiliary_residual_config.get(
            'type', 'none'
        )
        if auxiliary_residual_type in (None, 'none'):
            self.auxiliary_residual_readouts = None
        elif auxiliary_residual_type == 'residual_gaussian':
            from .readout import ResidualGaussianReadout

            if self.auxiliary_convnet is None:
                raise ValueError(
                    "auxiliary_residual_readout requires auxiliary_visual"
                )
            auxiliary_residual_params = dict(
                auxiliary_residual_config.get('params') or {}
            )
            input_scope = auxiliary_residual_params.pop(
                'input_scope', 'auxiliary_visual'
            )
            if input_scope != 'auxiliary_visual':
                raise ValueError(
                    "auxiliary_residual_readout input_scope must be "
                    "'auxiliary_visual'"
                )
            self.auxiliary_residual_readouts = nn.ModuleList(
                ResidualGaussianReadout(
                    in_channels=self.auxiliary_visual_output_channels,
                    n_units=len(config.get('cids', [])),
                    **auxiliary_residual_params,
                )
                for config in self.dataset_configs
            )
        else:
            raise ValueError(
                "Unknown auxiliary residual readout type: "
                f"{auxiliary_residual_type}"
            )

        # Optional identity-gated smooth residual core.  Unlike a new
        # component on one of the mature feature maps, this branch can learn
        # genuinely missing temporal/spatial features.  Its independent
        # Gaussian projections start at exactly zero, so random core weights
        # cannot change the parent prediction until the readout opens.
        residual_visual_config = self.model_config.get(
            'residual_visual', {'type': 'none', 'params': {}}
        )
        residual_visual_type = residual_visual_config.get('type', 'none')
        if residual_visual_type in (None, 'none'):
            self.residual_convnet = None
            self.residual_visual_readouts = None
        else:
            from .readout import ResidualGaussianReadout

            residual_visual_params = dict(
                residual_visual_config.get('params') or {}
            )
            residual_core_params = dict(
                residual_visual_params.pop('core', {}) or {}
            )
            residual_visual_readout_params = dict(
                residual_visual_params.pop('readout', {}) or {}
            )
            if residual_visual_params:
                raise ValueError(
                    "Unknown residual_visual params: "
                    f"{sorted(residual_visual_params)}"
                )
            self.residual_convnet, residual_visual_channels = create_convnet(
                convnet_type=residual_visual_type,
                in_channels=frontend_output_channels,
                **residual_core_params,
            )
            residual_visual_readout_params.setdefault(
                'feature_mode', 'independent'
            )
            if residual_visual_readout_params['feature_mode'] != 'independent':
                raise ValueError(
                    "residual_visual requires an independent feature "
                    "projection because its channel basis is newly learned"
                )
            self.residual_visual_readouts = nn.ModuleList(
                ResidualGaussianReadout(
                    in_channels=residual_visual_channels,
                    n_units=len(config.get('cids', [])),
                    **residual_visual_readout_params,
                )
                for config in self.dataset_configs
            )

        # Optional zero-initialized second Gaussian component.  It operates
        # on the visual portion of the already-smooth feature map and is
        # summed with the mature readout before behavior/output nonlinearities.
        # Configurations without this block retain the historical module tree
        # and checkpoint contract exactly.
        residual_readout_config = self.model_config.get(
            'residual_readout', {'type': 'none', 'params': {}}
        )
        residual_readout_type = residual_readout_config.get('type', 'none')
        if residual_readout_type in (None, 'none'):
            self.residual_readouts = None
            self.residual_readout_input_channels = None
        elif residual_readout_type == 'residual_gaussian':
            from .readout import (
                DynamicGaussianReadout,
                ResidualGaussianReadout,
            )

            if not all(
                isinstance(readout, DynamicGaussianReadout)
                for readout in self.readouts
            ):
                raise TypeError(
                    "residual_gaussian requires Gaussian base readouts"
                )
            residual_params = dict(
                residual_readout_config.get('params') or {}
            )
            input_scope = residual_params.pop('input_scope', 'visual')
            if input_scope != 'visual':
                raise ValueError(
                    "residual_readout input_scope currently must be 'visual'"
                )
            if recurrent_output_channels < convnet_output_channels:
                raise ValueError(
                    "The recurrent output does not retain all visual channels "
                    "required by residual_readout"
                )
            self.residual_readout_input_channels = convnet_output_channels
            self.residual_readouts = nn.ModuleList(
                ResidualGaussianReadout(
                    in_channels=convnet_output_channels,
                    n_units=len(config.get('cids', [])),
                    **residual_params,
                )
                for config in self.dataset_configs
            )
        else:
            raise ValueError(
                f"Unknown residual readout type: {residual_readout_type}"
            )

        # Optional neuron-specific behavior residual after the readout and
        # before the output nonlinearity.  This remains separate from the
        # legacy feature-space `modulator`, preserving old configurations and
        # checkpoints byte-for-byte when the block is absent.
        output_modulator_config = self.model_config.get(
            'output_modulator', {'type': 'none', 'params': {}}
        )
        output_modulator_type = output_modulator_config.get('type', 'none')
        self.output_behavior_mode = output_modulator_config.get(
            'input_mode', 'output_or_feature'
        )
        if self.output_behavior_mode not in {
            'output_or_feature',
            'concatenate_feature_and_output',
        }:
            raise ValueError(
                "output_modulator input_mode must be 'output_or_feature' or "
                "'concatenate_feature_and_output', got "
                f"{self.output_behavior_mode!r}"
            )
        if output_modulator_type in (None, 'none'):
            self.output_modulator = None
        elif output_modulator_type == 'mlp_behavior_residual':
            from .modulator import MultiDatasetBehaviorOutputModulator

            output_modulator_params = dict(
                output_modulator_config.get('params') or {}
            )
            self.output_modulator = MultiDatasetBehaviorOutputModulator(
                output_modulator_params,
                [len(config.get('cids', [])) for config in self.dataset_configs],
            )
        else:
            raise ValueError(
                f"Unknown output modulator type: {output_modulator_type}"
            )

        # Optional second output-only behavior residual initialized as an
        # exact identity.  This is deliberately separate from the spike-fit
        # output_modulator: it can receive a behavior-residual distillation
        # warm start without overwriting the independently learned student
        # behavior head.  Sequential gain/offset heads still only rescale a
        # neuron's stimulus Jacobian at fixed behavior; neither can create new
        # spatial or temporal sensitivity.
        distilled_config = self.model_config.get(
            'distilled_output_modulator', {'type': 'none', 'params': {}}
        )
        distilled_type = distilled_config.get('type', 'none')
        if distilled_type in (None, 'none'):
            self.distilled_output_modulator = None
        elif distilled_type == 'mlp_behavior_residual':
            from .modulator import MultiDatasetBehaviorOutputModulator

            distilled_params = dict(distilled_config.get('params') or {})
            self.distilled_output_modulator = MultiDatasetBehaviorOutputModulator(
                distilled_params,
                [len(config.get('cids', [])) for config in self.dataset_configs],
            )
        else:
            raise ValueError(
                "Unknown distilled output modulator type: "
                f"{distilled_type}"
            )

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


    def _crop_base_temporal_stimulus(self, stimulus):
        """Keep the newest frames on a mature core with shorter history."""
        if stimulus is None or self.base_input_temporal_support is None:
            return stimulus
        target_time = self.base_input_temporal_support
        available_time = stimulus.shape[-3]
        if available_time < target_time:
            raise ValueError(
                "base_input_temporal_support exceeds the supplied stimulus: "
                f"crop={target_time}, stimulus={available_time}"
            )
        # CombinedEmbeddedDataset preserves the configured lag order.  Every
        # native-rate model-selection config lists [0, 1, ...], so the newest
        # frame is at temporal index zero and the mature support is the leading
        # slice, not the tail.
        return stimulus[:, :, :target_time, ...]

    def _crop_base_spatial_stimulus(self, stimulus):
        """Center-crop only the mature path of a dual-aperture model."""
        if stimulus is None or self.base_input_crop is None:
            return stimulus
        target_h, target_w = self.base_input_crop
        height, width = stimulus.shape[-2:]
        if height < target_h or width < target_w:
            raise ValueError(
                "base_input_crop exceeds the supplied stimulus: "
                f"crop={self.base_input_crop}, stimulus={(height, width)}"
            )
        top = (height - target_h) // 2
        left = (width - target_w) // 2
        return stimulus[..., top:top + target_h, left:left + target_w]

    def _crop_base_stimulus(self, stimulus):
        """Apply the mature path's temporal and spatial support contracts."""
        stimulus = self._crop_base_temporal_stimulus(stimulus)
        return self._crop_base_spatial_stimulus(stimulus)

    def core_forward(self, stimulus=None, behavior=None):

        stimulus = self._crop_base_stimulus(stimulus)

        # route through frontend
        feats = self.frontend(stimulus)

        # Process through shared convnet
        feats = self.convnet(feats)

        # Process through shared modulator
        require_behavior(self.modulator, behavior,
                         where="MultiDatasetModel.core_forward")
        if self.modulator is not None:
            feats = self.modulator(feats, behavior)

        # Process through shared recurrent
        x_recurrent = self.recurrent(feats)

        return x_recurrent

    def core_forward_spatial_map(self, stimulus=None, behavior=None):
        """Run a translation-preserving core path when the core provides one.

        This is used by the Figure 4 spatial-information scorer. Historical
        cores keep their existing behavior; the Dekel core exposes a dedicated
        large-field path that retains its native deepest-stage lattice.
        """
        # A dual-aperture model normally evaluates its mature core on the
        # center crop.  Preserve exact ordinary-forward behavior when Figure 4
        # supplies precisely the auxiliary branch's training aperture.  Truly
        # larger counterfactual fields still use the historical convolutional
        # spatial-map path below.
        # A longer-history residual branch must never silently lengthen the
        # mature core's temporal receptive field, including in the Figure 4
        # translation-preserving path.
        stimulus = self._crop_base_temporal_stimulus(stimulus)
        auxiliary_input_size = getattr(
            getattr(self, "auxiliary_convnet", None),
            "input_size",
            None,
        )
        if (
            stimulus is not None
            and self.base_input_crop is not None
            and auxiliary_input_size is not None
            and tuple(stimulus.shape[-2:]) == tuple(auxiliary_input_size)
        ):
            stimulus = self._crop_base_spatial_stimulus(stimulus)
        feats = self.frontend(stimulus)
        spatial_forward = getattr(self.convnet, "forward_spatial_map", None)
        feats = spatial_forward(feats) if spatial_forward is not None else self.convnet(feats)

        require_behavior(
            self.modulator,
            behavior,
            where="MultiDatasetModel.core_forward_spatial_map",
        )
        if self.modulator is not None:
            feats = self.modulator(feats, behavior)
        return self.recurrent(feats)

    def auxiliary_visual_forward(self, stimulus, dataset_idx: int):
        """Return all logits drawn from the smooth auxiliary visual core."""
        if self.auxiliary_convnet is None:
            return None
        feats = self.frontend(stimulus)
        feats = self.auxiliary_convnet(feats)
        output = self.auxiliary_readouts[dataset_idx](
            feats,
            self.readouts[dataset_idx],
        )
        if self.auxiliary_residual_readouts is not None:
            output = output + self.auxiliary_residual_readouts[dataset_idx](
                feats,
                self.readouts[dataset_idx],
            )
        return output

    def auxiliary_visual_forward_spatial_map(self, stimulus):
        """Return translation-preserving auxiliary features for Figure 4."""
        if self.auxiliary_convnet is None:
            return None
        feats = self.frontend(stimulus)
        spatial_forward = getattr(
            self.auxiliary_convnet,
            "forward_spatial_map",
            None,
        )
        return (
            spatial_forward(feats)
            if spatial_forward is not None
            else self.auxiliary_convnet(feats)
        )

    def residual_visual_forward(self, stimulus, dataset_idx: int):
        """Return logits from the identity-gated smooth residual core."""
        if self.residual_convnet is None:
            return None
        feats = self.frontend(stimulus)
        feats = self.residual_convnet(feats)
        return self.residual_visual_readouts[dataset_idx](
            feats,
            self.readouts[dataset_idx],
        )

    def residual_visual_forward_spatial_map(self, stimulus):
        """Return translation-preserving residual-core features for Figure 4."""
        if self.residual_convnet is None:
            return None
        feats = self.frontend(stimulus)
        spatial_forward = getattr(
            self.residual_convnet,
            "forward_spatial_map",
            None,
        )
        return (
            spatial_forward(feats)
            if spatial_forward is not None
            else self.residual_convnet(feats)
        )
    
    def forward(
        self,
        stimulus=None,
        dataset_idx: int = 0,
        behavior=None,
        history=None,
        output_behavior=None,
    ):
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

        adapted_stimulus = x
        feature_behavior = self.resolve_feature_behavior(
            behavior, output_behavior
        )
        x = self.core_forward(x, feature_behavior)

        # Route through appropriate readout
        output = self.readouts[dataset_idx](x)
        if self.residual_readouts is not None:
            residual_features = x[
                :, :self.residual_readout_input_channels
            ]
            output = output + self.residual_readouts[dataset_idx](
                residual_features,
                self.readouts[dataset_idx],
            )
        auxiliary_output = self.auxiliary_visual_forward(
            adapted_stimulus,
            dataset_idx,
        )
        if auxiliary_output is not None:
            output = output + auxiliary_output
        residual_visual_output = self.residual_visual_forward(
            adapted_stimulus,
            dataset_idx,
        )
        if residual_visual_output is not None:
            output = output + residual_visual_output

        # Apply the neuron-specific residual to logits.  Zero-initialized
        # projections make this exactly equivalent to the old path until the
        # behavior head begins learning.
        # By default the output residual receives the historical model-wide
        # behavior tensor.  A separate tensor lets native-rate visual models
        # retain their established feature modulation while testing a
        # lower-rate behavior preprocessing contract at the neuron-specific
        # output head.  Existing callers and checkpoints are unchanged.
        residual_behavior = self.resolve_output_behavior(
            behavior, output_behavior
        )
        require_behavior(
            self.output_modulator,
            residual_behavior,
            where="MultiDatasetV1Model.forward(output_modulator)",
        )
        if self.output_modulator is not None:
            output = self.output_modulator(output, residual_behavior, dataset_idx)
        require_behavior(
            self.distilled_output_modulator,
            residual_behavior,
            where="MultiDatasetV1Model.forward(distilled_output_modulator)",
        )
        if self.distilled_output_modulator is not None:
            output = self.distilled_output_modulator(
                output, residual_behavior, dataset_idx
            )

        # Apply activation function
        output = self.activation(output)

        # Add baseline if enabled
        if self.baseline_enabled:
            baseline_output = self.baseline_activation(self.baselines[dataset_idx])
            output = output + baseline_output

        return output

    def resolve_output_behavior(self, behavior, output_behavior=None):
        """Select or combine behavior tensors for the output residual.

        The default preserves the original M20 contract.  The concatenation
        mode lets a residual use both the native-rate M16 covariates and an
        independently reconstructed lower-rate tensor without duplicating the
        former in host memory.
        """
        if self.output_behavior_mode == 'output_or_feature':
            return behavior if output_behavior is None else output_behavior
        if behavior is None or output_behavior is None:
            raise ValueError(
                "concatenate_feature_and_output requires both behavior and "
                "output_behavior tensors"
            )
        if behavior.shape[:-1] != output_behavior.shape[:-1]:
            raise ValueError(
                "behavior/output_behavior batch dimensions differ: "
                f"{tuple(behavior.shape)} vs {tuple(output_behavior.shape)}"
            )
        return torch.cat([behavior, output_behavior], dim=-1)

    def resolve_feature_behavior(self, behavior, output_behavior=None):
        """Route one declared behavior contract into the feature modulator."""
        if self.feature_behavior_source == 'behavior':
            return behavior
        if output_behavior is None:
            raise ValueError(
                "feature_behavior_source='output_behavior' requires an "
                "output_behavior tensor"
            )
        return output_behavior

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
