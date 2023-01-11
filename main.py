import os
import numpy as np
import torch
import seaborn
from mlp import mlp
from torch import nn 
from torch.utils.data import DataLoader

training_data = np.load("training_data.npy")
test_data = np.load("test_data.npy")
print(training_data.shape)



train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
print(train_dataloader)
for batch, data in enumerate(train_dataloader):
        data = torch.split(data, 10, dim=2)
        print(data[0].shape)
        print(data[1].shape)
        break

# seeeeeeeds
torch.manual_seed(1)

"""
This function creates multilayer perceptrons for a range of depths and widths, and returns a 3D numpy array containing the 
errors of those compared to the base map. 
"""
def wdTuner(training_data, test_data, depth=10, width=10):
    for d in range(1, depth):
        for w in range(1, width):
            # initialization
            model = mlp(input_size=width, depth=depth)

            # train that model
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()
            return 0
                


"""
This function creates a heatmap using a 3D numpy array with labelled axis
"""
def heatMapper():
    return 0




"""
This function trains the model using the training data. 
"""
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, data in enumerate(dataloader):
        
        print(data.shape)
        print(data.type)
        X=1
        y=1

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
