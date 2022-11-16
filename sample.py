import sys, os, torch
import numpy as np
import matplotlib.pyplot as plt

from diff import Model, get_ab
from utils import manifold, make_manifold, gif_save

# for debugging the gamma implimentation
def omega(gamma, beta, t):
    if t == 0: return gamma.get(t)

    om = gamma.get(t)
    for s in range(t):
        w = gamma.get_w(s, t)
        g = gamma.get(s)
        print('w: ', w)
        print('g: ', g)
        om += np.sqrt(w) * g
    return om

# gamma function: in this case q(x_t|x_0) = f(x_0) + gamma*k_n
class Gamma:
    def __init__(self, alpha, beta):
        self.gbank = {0 : np.sqrt(1 - alpha[0])}
        self.alpha = alpha
        self.beta = beta
    
    # main pass, cached recursion
    def get(self, t):
        if t == 0: return self.gbank[0]
        
        g_out = torch.sqrt(1 - self.alpha[t])
        for s in range(t):
            w = self.get_w(s, t)
            
            # used cached gamma for speedup
            if s in self.gbank: 
                g = self.gbank[s]
            else: 
                g = self.get(s)
                self.gbank.update({s : g})
            
            # update output
            g_out -= torch.sqrt(w) * g 
        return g_out     
            
    # coeffs
    def get_w(self, s, t):
        a = 1 - self.beta
        q = a[s+1:t+1]
        return torch.prod(q)

# sample and save
def sample(n):
    T = 100
    layers = 8

    # load model
    model = Model(layers)
    model.load_state_dict(torch.load('model.pt'))
    kn = torch.load('kn.pt')

    # for sampling coeffs
    alpha, beta = get_ab(T)

    # for sampling gamma
    gamma = Gamma(alpha, beta)

    # get data
    x = torch.randn((n, 3))
    x[:n//2] += 2.7
    x[n//2:] -= 2.7
    store = [x.numpy()]
    for t in range(T-1, -1, -1):
        with torch.no_grad():
            # stack t to make model conditioned on time step
            t_stack = t * torch.ones((x.shape[0], 1))
            x_inp = torch.hstack((x, t_stack))

            # get output, calculate other things needed
            eps = model(x_inp)

            # noise for sampling
            z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

            # classic update
            f1 = 1 / np.sqrt(1 - beta[t])
            f2 = beta[t] / np.sqrt(1 - alpha[t])
            sig = np.sqrt( (1 - alpha[t-1]) / (1 - alpha[t]) * beta[t] )
            
            # fokker planck addition
            y = torch.ones_like(x)
            y[:n//2] *= 3
            y[n//2:] *= -3
            
            g = gamma.get(t)
            #print(f1, eps[0], g*y[0])

            # update, NOTE: this should be + !!! 
            x = f1*(x -  f2*eps) - g*y #+ sig*z
            print(x_inp[0].numpy(), x[0].numpy(), eps[0].numpy(), g*y[0].numpy(), f1, f2)
            store.append(x.detach().numpy())

    #score = np.abs(x[:,2] - manifold(x[:,1], x[:,2]))
    #inds = np.argsort(score)[:3]
    #print(score[inds])

    out = np.stack(store, axis=0)
    np.save('out.npy', out)

if __name__ == '__main__':
    n = 32
    sample(n)
    x = np.load('out.npy') 

    d = 3.5 # for plotting
    b = 3 # domain bound
    k = 20 # number of points in
    X, Y, Z = make_manifold(b, k)

    for a in range(x.shape[0]-1+20):
        i = min(a, x.shape[0]-1)
        # setup plots
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.view_init(elev=25 + 0.25*a, azim=135 + 0.45*a)
        ax1.set(xlim=(-d, d), ylim=(-d, d), zlim=(-d-2, d+2))
        ax1.set_box_aspect([1,1,1])

        k = max(0, i-10)
        ax1.plot_wireframe(X, Y, Z, color='k')
        ax1.scatter(x[i,:n//2,0], x[i,:n//2,1], x[i,:n//2,2], color='b')
        ax1.scatter(x[i,n//2:,0], x[i,n//2:,1], x[i,n//2:,2], color='g')
        for j in range(x.shape[1]):
            col = 'dodgerblue' if j < n//2 else 'limegreen'
            ax1.plot(x[k:i+1,j,0], x[k:i+1,j,1], x[k:i+1,j,2], color=col, linewidth=2)
            ax1.plot(x[k:i+1,j,0], x[k:i+1,j,1], x[k:i+1,j,2], color=col, linewidth=2)

        plt.savefig('imgs/img_{}.png'.format(a))
        plt.close()
    
    gif_save('imgs', 'diffusion')
        