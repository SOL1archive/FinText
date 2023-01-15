import torch
import torch.nn as nn
import torch.functional as F

from transformers import ElectraTokenizer, ElectraModel

def CNN_Layer(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding
            ),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=kernel_size)
    )
    