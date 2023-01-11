import os
import numpy as np
import torch
from torch import nn 
from mlp import mlp

# initialization
torch.manual_seed(0)
model = mlp()

# randomize weights (no_grad)
with torch.no_grad():
    model.stack[0].weight = torch.nn.Parameter(torch.rand(10, 10))
    model.stack[2].weight = torch.nn.Parameter(torch.rand(10, 10))
    model.stack[4].weight = torch.nn.Parameter(torch.rand(1, 10))

# Generate input on distribution from -1 to 1. 
size = 500

# dataset = np.empty((size, 1),)
# with torch.no_grad():
#     for i in range(size):
#         input = 2*(torch.rand(1, 10)) - 1
#         label = model(input)
#         np.append(dataset, (input, label))

input_data = 2*(torch.rand(size, 1, 10)) - 1
output_data = torch.zeros(size, 1, 1)
with torch.no_grad():
    for i in range(size):
        output_data[i] = model(input_data[i])


# # Split into training and test data
dataset = np.concatenate((input_data.numpy(), output_data.numpy()), axis=2)

training_data = dataset[:int(size*0.8)]
test_data = dataset[int(size*0.8):]

# Save data
np.save("training_data.npy", training_data)
np.save("test_data.npy", test_data)