import sys

sys.path.append('..')

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.STN import SpatialTransformer, Re_SpatialTransformer
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #  A sequential container:
        self.double_conv = nn.Sequential(
            # Applies a 3D convolution over an input signal:
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            # Applies Group Normalization over a mini-batch of inputs:
            nn.GroupNorm(out_channels//4, out_channels),
            # Applies the LeakyReLU function element-wise:
            nn.LeakyReLU(0.2),
            # Repeat:
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels//4, out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):  # x je npr. slika, ki jo obdelujemo
        return self.double_conv(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),  # Applies a 3D max pooling over an input signal
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
