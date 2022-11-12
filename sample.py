import sys, os, torch
import numpy as np
import matplotlib.pyplot as plt

from diff import Model, get_ab
from utils import manifold, make_manifold, gif_save

# sample and save
def sample(g):
    T = 100
    layers = 8

    # load model
    model = Model(layers)
    model.load_state_dict(torch.load('model.pt'))

    # for sampling coeffs
    alpha, beta = get_ab(T)

    # get data
    x = torch.randn((g, 3))
    x[:g//2, [0,1]] += 2
    x[g//2:, [0,1]] -= 2
    store = [x.numpy()]
    for t in range(T-1, -1, -1):
        with torch.no_grad():
            # stack t to make model conditioned on time step
            t_stack = t * torch.ones((x.shape[0], 1))
            x_inp = torch.hstack((x, t_stack))

            # get output, calculate other things needed
            eps = model(x_inp)

            z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            a = alpha[t]
            b = beta[t]

            # update
            f1 = 1 / np.sqrt(1-b)
            f2 = (1 - (1-b)) / np.sqrt(1-a)
            sig = np.sqrt( (1 - alpha[t-1]) / (1 - alpha[t]) * b )
            x = f1*(x -  f2*eps) + sig*z
            
            store.append(x.detach().numpy())

    #score = np.abs(x[:,2] - manifold(x[:,1], x[:,2]))
    #inds = np.argsort(score)[:3]
    #print(score[inds])

    out = np.stack(store, axis=0)
    np.save('out.npy', out)

if __name__ == '__main__':
    g = 50
    sample(g)
    x = np.load('out.npy') 

    d = 3 # for plotting
    b = 3 # domain bound
    k = 20 # number of points in
    X, Y, Z = make_manifold(b, k)

    for a in range(x.shape[0]-1+100):
        i = min(a, x.shape[0]-1)
        # setup plots
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.view_init(elev=25 + 0.25*a, azim=45 + 0.375*a)
        ax1.set(xlim=(-d, d), ylim=(-d, d), zlim=(-d-2, d+2))
        ax1.set_box_aspect([1,1,1])

        k = max(0, i-10)
        ax1.plot_wireframe(X, Y, Z, color='k')
        ax1.scatter(x[i,:g//2,0], x[i,:g//2,1], x[i,:g//2,2], color='b')
        ax1.scatter(x[i,g//2:,0], x[i,g//2:,1], x[i,g//2:,2], color='g')
        for j in range(x.shape[1]):
            col = 'dodgerblue' if j < g//2 else 'limegreen'
            ax1.plot(x[k:i+1,j,0], x[k:i+1,j,1], x[k:i+1,j,2], color=col, linewidth=2)
            ax1.plot(x[k:i+1,j,0], x[k:i+1,j,1], x[k:i+1,j,2], color=col, linewidth=2)

        plt.savefig('imgs/img_{}.png'.format(a))
        plt.close()
    
    gif_save('imgs', 'diffusion')
        