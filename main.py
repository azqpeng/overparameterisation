import os
import numpy as np
import torch
from mlp import mlp
from torch import nn 
from torch.utils.data import DataLoader

training_data = np.load("training_data.npy");
test_data = np.load("test_data.npy");

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# seeeeeeeds
torch.manual_seed(1)

# go through different types of models, and train them.
for depth in range(1, 10):
    for width in range(1, 10):
        # initialization
        model = mlp(input_size=width, depth=depth)

        # training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        for epoch in range(10):
            for batch in train_dataloader:
                optimizer.zero_grad()
                output = model(batch)
                loss = loss_fn(output, batch)
                loss.backward()
                optimizer.step()
        
        # testing
        for batch in test_dataloader:
            output = model(batch)
            loss = loss_fn(output, batch)
        
        # print results of regression        