# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This defines several neural networks
They take an image/ sequence as input and send a pyramid/ sequence of pyramids as output
"""
import numpy as np
import torch
import torch.nn as nn

from metavision_core_ml.core.modules import ConvLayer, PreActBlock
from metavision_core_ml.core.temporal_modules import time_to_batch, SequenceWise, ConvRNN


class Vanilla(nn.Module):
    """
    Baseline architecture getting 0.4 mAP on the HD Event-based Automotive Detection Dataset.

    It consists of Squeeze-Excite Blocks to stride 16 and then 5 levels of Convolutional-RNNs.

    Each level can then be fed to a special head for predicting bounding boxes for example.
    """

    def __init__(self, cin=1, base=16, cout=256):
        super(Vanilla, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 5

        self.conv1 = SequenceWise(nn.Sequential(
            ConvLayer(cin, self.base * 2, kernel_size=7, stride=2, padding=3, norm='BatchNorm2d'),
            PreActBlock(self.base * 2, self.base * 4, 2),
            PreActBlock(self.base * 4, self.base * 4, 1),
            PreActBlock(self.base * 4, self.base * 8, 1),
        ))

        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2))
        for i in range(self.levels - 1):
            self.conv2.append(ConvRNN(cout, cout, stride=2))

    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for conv in self.conv2:
            x = conv(x)
            y = time_to_batch(x)[0]
            outs.append(y)
        return outs

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()


class Vanilla_VGG(nn.Module):
    """
    Baseline architecture getting 0.3 mAP on the HD Event-based Automotive Detection Dataset.

    It consists of VGG blocks to stride 16 and then 5 levels of Convolutional-RNNs.

    Each level can then be fed to a special head for predicting bounding boxes for example.
    """

    def __init__(self, cin=1, base=16, cout=256):
        super(Vanilla_VGG, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 5

        self.conv1 = SequenceWise(nn.Sequential(
            ConvLayer(cin, base, kernel_size=3, stride=1, padding=1),
            ConvLayer(base, base * 2, kernel_size=3, stride=2, padding=1),
            ConvLayer(base * 2, base * 2, kernel_size=3, stride=1, padding=1),
            ConvLayer(base * 2, base * 2, kernel_size=3, stride=2, padding=1),
            ConvLayer(base * 2, base * 4, kernel_size=3, stride=1, padding=1),
            ConvLayer(base * 4, base * 8, kernel_size=3, stride=2, padding=1),
        ))

        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2))
        for i in range(self.levels - 1):
            self.conv2.append(ConvRNN(cout, cout, stride=2))

    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for conv in self.conv2:
            x = conv(x)
            y = time_to_batch(x)[0]
            outs.append(y)
        return outs

    def reset(self, mask=None):
        for name, module in self.conv2.named_modules():
            if hasattr(module, "reset"):
                module.reset(mask)

    @torch.jit.export
    def reset_all(self):
        for module in self.conv2:
            module.reset_all()


class Vanilla_VGGRU(Vanilla_VGG):
    """
    Baseline architecture with an alternate RNN Cell

    It consists of VGG blocks to stride 16 and then 5 levels of Convolutional-RNNs of the GRU type.

    Each level can then be fed to a special head for predicting bounding boxes for example.
    """

    def __init__(self, cin=1, base=16, cout=256):
        super(Vanilla_VGGRU, self).__init__()
        self.cin = cin
        self.cout = cout
        self.base = base
        self.levels = 5

        self.conv1 = SequenceWise(nn.Sequential(
            ConvLayer(cin, base, kernel_size=3, stride=1, padding=1),
            ConvLayer(base, base * 2, kernel_size=3, stride=2, padding=1),
            ConvLayer(base * 2, base * 2, kernel_size=3, stride=1, padding=1),
            ConvLayer(base * 2, base * 2, kernel_size=3, stride=2, padding=1),
            ConvLayer(base * 2, base * 4, kernel_size=3, stride=1, padding=1),
            ConvLayer(base * 4, base * 8, kernel_size=3, stride=2, padding=1),
        ))

        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvRNN(self.base * 8, cout, stride=2, cell="gru"))
        for i in range(self.levels - 1):
            self.conv2.append(ConvRNN(cout, cout, stride=2, cell="gru"))
