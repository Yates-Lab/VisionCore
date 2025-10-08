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
