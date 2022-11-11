import sys, os, torch
import numpy as np
import matplotlib.pyplot as plt

from diff import Model, get_ab
from utils import manifold, make_manifold

# sample and save
def sample():
    T = 100
    layers = 8

    # load model
    model = Model(layers)
    model.load_state_dict(torch.load('model.pt'))

    # for sampling coeffs
    alpha, beta = get_ab(T)

    # get data
    x = torch.randn((15, 3))
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
            x = f1*(x -  f2*eps) + 0.75 * sig * z
            
            store.append(x.detach().numpy())

    #score = np.abs(x[:,2] - manifold(x[:,1], x[:,2]))
    #inds = np.argsort(score)[:3]
    #print(score[inds])

    out = np.stack(store, axis=0)
    np.save('out.npy', out)

if __name__ == '__main__':
    sample()
    x = np.load('out.npy') 

    b = 1 # domain bound
    k = 20 # number of points in
    X, Y, Z = make_manifold(b, k)

    for a in range(x.shape[0]-1+100):
        i = min(a, x.shape[0]-1)
        # setup plots
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.view_init(elev=50 - 0.25*a, azim=-125 + 0.375*a)
        ax1.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5))
        ax1.set_box_aspect([1,1,1])

        k = max(0, i-10)
        ax1.plot_wireframe(X, Y, Z, color='k')
        ax1.scatter(x[i,:,0], x[i,:,1], x[i,:,2], color='b')
        for j in range(x.shape[1]):
            ax1.plot(x[k:i+1,j,0], x[k:i+1,j,1], x[k:i+1,j,2], color='dodgerblue', linewidth=2)

        plt.savefig('imgs2/img_{}.png'.format(a))
        plt.close()