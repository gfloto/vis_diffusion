import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

from utils import manifold, get_points

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
        self.kn = nn.Parameter(2*torch.randn(2,3), requires_grad=True)

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
        self.T = 100
        #self.c = self.beta / (2*(1 - self.beta)*(1 - self.alpha)) # weight for loss

    # return sample and weight for loss
    def sample(self, x, eps, t, kn, i, device):
        a = self.alpha[t]
        out = torch.sqrt(a)*x + kn[i]*torch.sqrt(1 - a) + (1 - a)*eps 
        return out 

# single pass of the dataset
def single_pass(data, model, opt, q, device, batch_size=10):
    loss_track = []
    (data1, data2) = data

    # NOTE: a single pass is over the data where t is random
    # indicies for shuffling dataset
    inds = torch.randperm(data1.shape[0])
    for i in range((data1.shape[0] // batch_size)):
        if i % q.T == 0: times = torch.randperm(q.T)

        # get next batch
        t = times[i%q.T]
        ind = inds[i: i+batch_size]
        x1 = data1[inds]
        x2 = data2[inds]
        
        # get xt
        eps1 = torch.randn_like(x1).to(device)
        eps2 = torch.randn_like(x2).to(device)
        xt1 = q.sample(x1, eps1, t, model.kn, 0, device)
        xt2 = q.sample(x2, eps2, t, model.kn, 1, device)

        # stack t to make model conditioned on time step
        t_stack1 = t * torch.ones((xt1.shape[0], 1)).to(device)
        t_stack2 = t * torch.ones((xt2.shape[0], 1)).to(device)
        xt1 = torch.hstack((xt1, t_stack1))
        xt2 = torch.hstack((xt2, t_stack2))
        
        # pass
        opt.zero_grad()
        z1 = model(xt1)
        z2 = model(xt2)
        #print(z1[0].detach().cpu(), xt1[0].detach().cpu())

        # loss
        loss1 = torch.mean((eps1 - z1)**2)
        loss2 = torch.mean((eps2 - z2)**2)

        leng = torch.linalg.norm(model.kn, axis=1)
        kn_loss = torch.mean((leng - 5)**2)
        diff_loss = (loss1 + loss2) / 2

        loss = diff_loss + kn_loss
        loss.backward()
        opt.step()

        loss_track.append(loss.detach().item())
    return np.mean(loss_track)

if __name__ == '__main__':
    T = 100
    n = 1000
    layers = 8
    epochs = 500
    assert n % epochs == 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load data
    (data1, data2) = get_points(n, k=1)

    # build model + get optimizer
    model = Model(layers)

    # to device
    model = model.to(device)
    data1 = data1.to(device)
    data2 = data2.to(device)
    data = (data1, data2)

    # get beta and alpha for q network
    alpha, beta = get_ab(T)
    q = Q(alpha, beta, T, device)

    # optimizer
    opt = optim.Adam(model.parameters(), lr=5e-4)

    # main loop
    kn_track = [model.kn.detach().cpu().numpy()]
    for e in range(epochs):
        loss = single_pass(data, model, opt, q, device)
        kn_track.append(model.kn.detach().cpu().numpy())
        print(loss)
        print(kn_track[-1])

    torch.save(model.state_dict(), 'model.pt')
    kn_track = np.array(kn_track)
    np.save('kn_track.npy', kn_track)
    print('done')

