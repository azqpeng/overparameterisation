import os
import numpy as np
import torch
from torch import nn

"""
This class creates a neural network with variable depth and width.
"""
class mlp(nn.Module):
    # create model
    def __init__(self, input_size=10, depth=3):
        super(mlp, self).__init__()
        self.flatten = nn.Flatten()

        # create linear blocks
        linear_blocks = [layer for i in range(depth-1) for layer in (nn.Linear(input_size, input_size), nn.ReLU())] 

        # stack
        self.stack = nn.Sequential(*linear_blocks, nn.Linear(input_size, 1), nn.ReLU())

    # feed
    def forward(self, input_data):
        input_data = self.flatten(input_data)
        logits = self.stack(input_data)
        return logits