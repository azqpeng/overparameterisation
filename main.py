import os
import numpy as np
import torch
from mlp import mlp
from torch import nn 


# seeeeeeeds
torch.manual_seed(1)

# go through different types of models, and train them.
for depth in range(1, 10):
    for width in range(1, 10):
        # initialization
        model = mlp(input_size=width, depth=depth)

        # training
        
