"""
Convolutional blocks used in the Lightweight OpenPose PE model

EnhancePseEstimation/src/models
@author: Angel Villar-Corrales 
    Adapted from Daniil Osokin:
        https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
"""

from torch import nn


def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1,
         stride=1, relu=True, bias=True):
    """
    Convolutional layer, including Bath Normalization and ReLU activation function if specified
    in the parameters
    """

    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    """
    Two consecutive Convolutional layers, each of them followed by BN and ReLU nonlinearity.
    MobileNet Backbone is constructed using these blocks
    """

    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_dw_no_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    """
    Two consecutive Convolutional layers, each of them followed by a ELU nonlinearity.
    Used to construct the CPM block
    """

    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.ELU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    )

#
