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
    return abs(target - input)


def TPE(input, target):
    """
    Three-pixel-error
    :param input: 
    :param target: 
    :return: 
    """
    pass
