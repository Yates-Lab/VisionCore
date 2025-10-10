"""
Neural network modules for DataYatesV1.

This package contains modular components for building neural network models.
"""

from .frontend import DAModel, TemporalBasis, AffineAdapter, LearnableTemporalConv
from .convnet import VanillaCNN, ResNet, DenseNet, BaseConvNet
from .x3d import X3DNet, X3DUnit
from .recurrent import ConvLSTM, ConvGRU
from .modulator import ConcatModulator, FiLMModulator, SpatialTransformerModulator, ConvGRUModulator, PredictiveCodingModulator, MODULATORS
from .readout import DynamicGaussianReadout, DynamicGaussianReadoutEI, DynamicGaussianSN
from .common import SplitRelu, chomp

# For backward compatibility
from .conv_blocks import ConvBlock

# Model architectures
from .models import ModularV1Model, MultiDatasetV1Model

# Polar-V1 modules
from .polar_convnet import PolarConvNet, PyramidAdapter, QuadratureFilterBank2D, PolarDecompose
from .polar_modulator import PolarModulator, MinimalGazeEncoder, BehaviorEncoder
from .polar_recurrent import PolarRecurrent, PolarDynamics, TemporalSummarizer, init_kxy
from .polar_readout import PolarMultiLevelReadout, GaussianReadout
from .polar_jepa import PolarJEPA
