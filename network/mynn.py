"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
sys.path.append("..")
from config import cfg
import mindspore.nn as nn
from math import sqrt
import mindspore
from mindspore.common.initializer import HeNormal
from itertools import repeat
from mindspore.nn.cell import Cell
import numpy as np


def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = getattr(cfg.MODEL,'BNFUNC')
    normalizationLayer = layer(in_channels,  momentum= 0.9)
    #normalizationLayer = nn.BatchNorm2d(in_channels, momentum=0.9)
    return normalizationLayer

#初始化权重
def initialize_weights(*models):
   for model in models:
        for module in model.cells():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Dense):
                HeNormal(module.weight)




class WideBasic(nn.Cell):
    """
    WideBasic
    """
    def __init__(self,  stride=1):
        super(WideBasic, self).__init__()
        conv1 = nn.Conv2d(256 + 48, 256, kernel_size=3, pad_mode="same", has_bias=True, weight_init='HeNormal')
        conv2 = nn.Conv2d(256 + 48, 256, kernel_size=3, pad_mode="same", has_bias=True, weight_init='Normal')

        self.final_seg = nn.SequentialCell(
            nn.Conv2d(256 + 48, 256, kernel_size=3, pad_mode="same",has_bias=True, weight_init='HeNormal'),
            Norm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, pad_mode="same", has_bias=True),
            Norm2d(256),
            nn.ReLU(),
            )



    def construct(self, x):
        """
        basic construct
        """

        identity = x

        out = self.final_seg(identity)



        return out


# inputs = mindspore.Tensor(np.ones((2, 304, 24, 24)).astype("float32"))
# model = WideBasic()
# output = model(inputs)
#print(output)