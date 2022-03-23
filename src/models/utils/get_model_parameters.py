"""
Methods for accessing a subset of the model parameters

EnhancePoseEstimation/src/models
@author: Angel Villar-Corrales 
    Adapted from Daniil Osokin:
        https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
"""

from torch import nn


def get_parameters(model, predicate):
    """
    Method for accessing a subset of parameters from the model
    """

    for module in model.modules():
        for param_name, param in module.named_parameters():
            if predicate(module, param_name):
                yield param



def get_parameters_conv(model, name):
    """
    Obtaining the parameters from the convolutional layers

    Args:
    -----
    model: nn.Module
        instanciated neural network model
    name: string
        type of parameter to yield: weight, bias, ...

    Returns:
    --------
    params:
        parameters from the models
    """

    param_condition = lambda m, p: isinstance(m, nn.Conv2d) and m.groups == 1 and p == name
    params = get_parameters(model, param_condition)

    return params


def get_parameters_conv_depthwise(model, name):
    """
    Obtaining the parameters from the convolutional layers with same number of
    input and output channels

    Args:
    -----
    model: nn.Module
        instanciated neural network model
    name: string
        type of parameter to yield: weight, bias, ...

    Returns:
    --------
    params:
        parameters from the models
    """

    param_condition = lambda m, p: isinstance(m, nn.Conv2d) and m.groups == m.in_channels\
                                              and m.in_channels == m.out_channels and p == name
    params = get_parameters(model, param_condition)

    return params


def get_parameters_bn(model, name):
    """
    Obtaining the parameters from the batch normalization layers

    Args:
    -----
    model: nn.Module
        instanciated neural network model
    name: string
        type of parameter to yield: weight, bias, ...

    Returns:
    --------
    params:
        parameters from the models
    """

    param_condition = lambda m, p: isinstance(m, nn.BatchNorm2d) and p == name
    params = get_parameters(model, param_condition)

    return params
