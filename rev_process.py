import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import gif_save, make_manifold

if __name__ == '__main__':
    n = 100 # number of iterations in plotting

    domain_bound = 1
    domain_points = 20
    X, Y, Z = make_manifold(domain_bound, domain_points)

    # choose starting point
    x_0 = np.array([-0.5, -0.5, Z[5,5]])
    x_track = np.empty((n, 3))
    x_track[0] = x_0
    a_track = [1]

    # load forward process
    for i in range(1, n):
        beta = 1e-4 + (i-1)*(0.1 - 1e-4)/(n-2)
        a = (1 - beta) * a_track[-1]
        a_track.append(a)
        x_track[i] = np.sqrt(1-beta) * x_track[i-1] + beta*np.random.randn(3)

    # flip around
    x_track = x_track[::-1]

    os.makedirs('imgs', exist_ok=True)
    os.makedirs('videos', exist_ok=True)

    # load image
    img = np.array(Image.open('assets/hands.jpg'))
    for i in range(n-1):
        # setup plots
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        ax1.view_init(elev=30 + -0.15*i, azim=-125 + 0.375*i)

        ax1.plot_wireframe(X, Y, Z, color='k')
        ax1.scatter(x_track[i,0], x_track[i,1], x_track[i,2], color='b')
        ax1.scatter(x_0[0], x_0[1], x_0[2], color='r')
        ax1.plot(x_track[:i+1,0], x_track[:i+1,1], x_track[:i+1,2], color='dodgerblue', linewidth=2)

        img_noise = 3*255 * np.linalg.norm(x_track[i] - x_0) * np.random.randn(img.shape[0], img.shape[1]) + img

        ax2.imshow(img_noise, cmap='gray')
        ax2.axis('off')

        plt.savefig('imgs/img_{}.png'.format(i))
        plt.close()

    # plot still frame
    for k in range(99, 125):
        i = 99
        # setup plots
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        ax1.view_init(elev=30 + -0.15*k, azim=-125 + 0.375*k)

        ax1.plot_wireframe(X, Y, Z, color='k')
        ax1.scatter(x_track[i,0], x_track[i,1], x_track[i,2], color='b')
        ax1.scatter(x_0[0], x_0[1], x_0[2], color='r')
        ax1.plot(x_track[:i+1,0], x_track[:i+1,1], x_track[:i+1,2], color='dodgerblue', linewidth=2)

        img_noise = 3*255 * np.linalg.norm(x_track[i] - x_0) * np.random.randn(img.shape[0], img.shape[1]) + img

        ax2.imshow(img_noise, cmap='gray')
        ax2.axis('off')

        plt.savefig('imgs/img_{}.png'.format(k))
        plt.close()

    gif_save('imgs', 'videos/rev_process.gif')
