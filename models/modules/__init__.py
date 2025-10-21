"""
Neural network modules for DataYatesV1.

This package contains modular components for building neural network models.
"""

from .frontend import DAModel, TemporalBasis, AffineAdapter, LearnableTemporalConv
from .convnet import VanillaCNN, ResNet, DenseNet, BaseConvNet
from .recurrent import ConvGRU
from .modulator import ConcatModulator, FiLMModulator, MODULATORS
from .readout import DynamicGaussianReadout, DynamicGaussianReadoutEI, DynamicGaussianSN
from .common import SplitRelu, chomp
from .conv_blocks import ConvBlock, ResBlock

# Model architectures
from .models import ModularV1Model, MultiDatasetV1Model

