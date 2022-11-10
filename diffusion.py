import sys
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

import os
import re
from PIL import Image

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# saves images into a gif
def gif_save(path, name):
    pull_path = os.path.join(path)
    ims = os.listdir(pull_path)   
    ims = sorted(ims, key=natural_key)

    # read files
    video = []
    for i, f in enumerate(ims):
        file = os.path.join(pull_path, f)
        frame = Image.open(file)
        video.append(frame)
        

    # save gif
    save_path = os.path.join(path, name+'.gif') 
    video[0].save(save_path, format='gif',
                   append_images=video[1:],
                   save_all=True,
                   duration=60, loop=0)   
    return

def f(x, y): return (-0.4*x*y + 0.225*x**2 + 0.35*y**3 + 0.25*np.sin(4*x + y)) - 0.4
def sig(x): return 1/(1 + np.exp(-x))

if __name__ == '__main__':
    # main here
    r = 1	
    n = 100
    x = np.linspace(-r, r, 20)
    y = np.linspace(-r, r, 20)

    # make manifold
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

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

    #plt.plot(a_track)
    #plt.show()
    #sys.exit()

    # load image
    img = np.array(Image.open('hands.jpg'))
    for i in range(n-1):
        # setup plots
        fig = plt.figure(figsize=(16,8))
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
        fig = plt.figure(figsize=(16,8))
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


    gif_save('imgs', 'diffusion')
