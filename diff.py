import sys, os, torch, copy
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

# function to learn from
def f(x, y): return 0.25*x*y + np.sin(2*x)

# make copy of modules
def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

# dataset for training
def get_points(n):
    x = 2*torch.rand(n) - 1
    y = 2*torch.rand(n) - 1
    z = f(x,y)
    return torch.stack((x,y,z)).T

class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self.dim = 3
        self.lin = nn.Linear(self.dim, self.dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.lin(x))

class Model(nn.Module):
    def __init__(self, n_layers):
        super(Model, self).__init__()
        self.dim = 3
        self.n_layers = n_layers

        self.layers = clones(Layer(), self.n_layers)
        self.layers.append(nn.Linear(self.dim, self.dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# single pass of the dataset
def single_pass(data, model, opt, batch_size=10):
    # indicies for shuffling dataset
    inds = torch.randperm(data.shape[0])
    for i in range(batch_size - 1):
        # get next batch
        ind = inds[i: i+batch_size]
        x = data[inds]
        
        # pass
        opt.zero_grad()
        z = model(x)
        loss = torch.sum((z - 1)**2)
        print(loss.item())

        loss.backward()
        opt.step()

if __name__ == '__main__':
    n = 100
    layers = 3
    epochs = 10
    assert n % epochs == 0

    # load data
    data = get_points(n)

    # build model + get optimizer
    model = Model(layers)
    opt = optim.Adam(model.parameters(), lr=1e-1)

    # main loop
    for e in range(epochs):
        single_pass(data, model, opt)
