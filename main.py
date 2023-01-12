import os
import numpy as np
import torch
import seaborn
from mlp import mlp
from torch import nn 
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

training_data = np.load("training_data.npy")
test_data = np.load("test_data.npy")

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# seeeeeeeds
torch.manual_seed(1)

"""
This function creates multilayer perceptrons for a range of depths and widths, and returns a 2D numpy array containing the 
errors of those compared to the base map. 
"""
def wdTuner(training_data, test_data, depth=10, width=10, epochs = 10, fileName = "errorArray.npy"):
    errorArray = np.empty((depth, width))
    for d in range(1, depth):
        for w in range(1, width):
            # initialization
            model = mlp(input_size=w, depth=d)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
            loss_fn = nn.MSELoss()
            epochs = 10
            # train that!
            for t in range(epochs):
                #print(f"Epoch {t+1}\n-------------------------------")
                train_loop(train_dataloader, model, loss_fn, optimizer)
                error = test_loop(test_dataloader, model, loss_fn)
            errorArray[d, w] = error
    # saves array as filef
    np.save(fileName, errorArray)
    return errorArray


"""
This function trains the model using the training data. 
"""
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, data in enumerate(train_dataloader):
        (X,y) = torch.split(data, 10, dim=2)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for every 100 batches...
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    #print("Training arc: Complete!")

"""
This function tests the model and returns the error.
"""
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_error = 0

    # computes total loss through average of batch losses
    with torch.no_grad():
        for data in dataloader:
            (X,y) = torch.split(data, 10, dim=2)
            pred = model(X)
            test_error += loss_fn(pred, y).item()
    test_error /= num_batches

    #print(f"Test Error: \n Accuracy: {(100*test_error):>0.1f}%, Avg loss: {test_error:>8f} \n")
    return test_error


# generation
m = wdTuner(training_data, test_data, 30, 100, 25)