import sys, os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from utils import gif_save, natural_key, make_manifold, manifold, make_sphere

if __name__ == '__main__':
    gif_save('imgs', 'diffusion')
    sys.exit()
    #n = 100 # number of iterations in plotting

    b = 1 # domain bound
    k = 20 # number of points
    X, Y, Z = make_manifold(b, k)

    # choose starting points
    m = 25
    z = 2*np.random.rand(m, 3) - 1
    for i in range(m):
        z[i, 2] = manifold(z[i,0], z[i,1])

    # colors
    colors = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds', 'Brwnyl', 'Burg', 'Mint']
    c = np.random.randint(0, len(colors), size=(m))

    # plotting
    a_track = [1]
    for i in range(1, n):
        # important: setting the beta term
        beta = 1e-4 + (i-1)*(0.1 - 1e-4)/(n-2)
        a = (1 - beta) * a_track[-1]
        a_track.append(a)
        
        data = [go.Surface(x=X, y=Y, z=Z, colorscale='Greys')]

        #fig = plt.figure(figsize=(8,8))
        #ax = fig.add_subplot(111, projection='3d')
        #ax.view_init(elev=30 + -0.15*i, azim=-125 + 0.375*i)
        #ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5))
        #ax.set_box_aspect([1,1,1])

        #ax.plot_wireframe(X, Y, Z, color='k')
        #ax.scatter(z[:,0], z[:,1], z[:,2])

        # plot sphere
        for j in range(m):
            (xs, ys, zs) = make_sphere(np.sqrt(a_track[-1]) * z[j], np.sqrt(1 - a_track[-1]))
            data.append(go.Surface(x=xs, y=ys, z=zs, colorscale=colors[c[j]]))
            spot = np.random.randint(len(colors))
            #fig.plot_surface(xs, ys, zs)
        fig = go.Figure(data=data)
        fig.update_layout( autosize=False, width=800, height=800,)
        fig.write_image('imgs/img_{}.png'.format(i))

        #plt.savefig('imgs2/img_{}.png'.format(i))
        #plt.close()

    gif_save('imgs', 'diffusion')
    sys.exit()
##############################################################

    # load forward process
    for i in range(1, n):
        beta = 1e-4 + (i-1)*(0.1 - 1e-4)/(n-2)
        a = (1 - beta) * a_track[-1]
        a_track.append(a)
        x_track[i] = np.sqrt(1-beta) * x_track[i-1] + beta*np.random.randn(3)

    # flip around
    x_track = x_track[::-1]

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
