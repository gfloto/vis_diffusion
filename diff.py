import sys, os, torch, copy
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

from utils import manifold

# dataset for training
def get_points(n):
    x = 3*torch.rand(n) - 1.5
    y = 3*torch.rand(n) - 1.5
    z = manifold(x,y)
    return torch.stack((x,y,z)).T

# get noise levels, assuming that sigma is beta
def get_ab(n):
    assert n == 100 # otherwise check the alpha goes from [0, 1]
    alpha = []
    beta = []

    # load forward process
    for i in range(n):
        b = 1e-4 + (i)*(0.1 - 1e-4)/(n-1)
        a = (1 - b) * alpha[-1] if i > 0 else (1 - b)
        alpha.append(a)
        beta.append(b)
    return torch.tensor(alpha), torch.tensor(beta)

class Layer(nn.Module):
    def __init__(self, dim, mid):
        super(Layer, self).__init__()
        self.lin = nn.Linear(dim, mid)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.lin(x))

class Model(nn.Module):
    def __init__(self, n_layers):
        super(Model, self).__init__()
        self.dim = 4 # to include the time
        self.mid = 10
        self.n_layers = n_layers

        self.layers = nn.ModuleList([Layer(self.dim, self.mid)])
        for _ in range(n_layers - 2):
            self.layers.append(Layer(self.mid, self.mid))
        self.layers.append(nn.Linear(self.mid, self.dim-1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# get q from x0 to xt
class Q:
    def __init__(self, alpha, beta, T, device):
        self.alpha = torch.tensor(alpha).to(device)
        self.beta = torch.tensor(beta).to(device)
        self.c = self.beta / (2*(1 - self.beta)*(1 - self.alpha)) # weight for loss
        self.T = 100

    # return sample and weight for loss
    def sample(self, x, eps, t):
        a = self.alpha[t]
        return torch.sqrt(a)*x + torch.sqrt(1 - a)*eps, self.c[t]

# single pass of the dataset
def single_pass(data, model, opt, q, device, batch_size=10):
    loss_track = []
    # NOTE: a single pass is over the data where t is random
    # indicies for shuffling dataset
    inds = torch.randperm(data.shape[0])
    for i in range((data.shape[0] // batch_size)):
        if i % q.T == 0: times = torch.randperm(q.T)

        # get next batch
        t = times[i%q.T]
        ind = inds[i: i+batch_size]
        x = data[inds]
        
        # get xt
        eps = torch.randn_like(x).to(device)
        (xt, c) = q.sample(x, eps, t)

        # stack t to make model conditioned on time step
        t_stack = t * torch.ones((xt.shape[0], 1)).to(device)
        xt = torch.hstack((xt, t_stack))
        
        # pass
        opt.zero_grad()
        z = model(xt)
        loss = torch.mean((eps - z)**2)

        loss.backward()
        opt.step()

        loss_track.append(loss.detach().item())
    return np.mean(loss_track)

if __name__ == '__main__':
    T = 100
    n = 10000
    layers = 8
    epochs = 100
    assert n % epochs == 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load data
    data = get_points(n)

    # build model + get optimizer
    model = Model(layers)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # to device
    model = model.to(device)
    data = data.to(device)

    # get beta and alpha for q network
    alpha, beta = get_ab(T)
    q = Q(alpha, beta, T, device)

    # main loop
    for e in range(epochs):
        loss = single_pass(data, model, opt, q, device)
        print(loss)

    torch.save(model.state_dict(), 'model.pt')

