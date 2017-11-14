import torch
import torch.nn as nn
import math


def EPE(input, target):
    """
    Endpoint-error
    :param input: 
    :param target: 
    :return: 
    """
    return (target - input).abs()


def TPE(input, target):
    """
    Three-pixel-error
    :param input: 
    :param target: 
    :return: 
    """
    pass
