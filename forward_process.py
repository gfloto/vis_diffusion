import os
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from utils import gif_save, make_manifold, manifold, make_sphere

if __name__ == '__main__':
    num_points = 25 # number of points on manifold
    num_frames = 100 # number of frames to plot for forward process
    img_size = 400

    domain_size = 1 # plotting domain 
    domain_points = 20 # number of points to cross
    X, Y, Z = make_manifold(domain_size, domain_points)

    # make random points on manifold surface 
    z = 2 * np.random.rand(num_points, 3) - 1
    for i in range(num_points):
        z[i, 2] = manifold(z[i,0], z[i,1])

    # colors
    colors = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds', 'Brwnyl', 'Burg', 'Mint']
    c = np.random.randint(0, len(colors), size=(num_points))

    os.makedirs('imgs', exist_ok=True)
    os.makedirs('videos', exist_ok=True)

    # main plotting loop
    frame_count = 0
    alpha_track = [1]
    for i in tqdm(range(num_frames)):
        # diffusion schedule
        beta = 1e-4 + i * (0.1 - 1e-4) / (num_frames)
        alpha = (1 - beta) * alpha_track[-1]
        alpha_track.append(alpha)

        # plot manifold surface         
        data = [go.Surface(
            x=X,
            y=Y,
            z=Z, 
            colorscale='Greys',
            showscale=False,
        )]

        # plot sphere (represent std. dev.)
        for j in range(num_points):
            (xs, ys, zs) = make_sphere(np.sqrt(alpha_track[-1]) * z[j], np.sqrt(1 - alpha_track[-1]))
            data.append(
                go.Surface(
                    x=xs,
                    y=ys,
                    z=zs,
                    colorscale=colors[c[j]],
                    showscale=False,
                )
            )

        fig = go.Figure(data=data)

        # set camera angle
        fig.update_layout(
            scene=dict(
                camera=dict(
                    eye=dict(x=1.25, y=1.25, z=1.75)
                )
            )
        )

        fig.update_layout(
            autosize=False,
            width=img_size,
            height=img_size,
            margin=dict(l=0, r=0, b=0, t=0),
        )

        # setup plots
        fig = plt.figure(figsize=(16,8))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        ax1.view_init(elev=30 + -0.15*i, azim=-125 + 0.375*i)

        # save both images
        if i == 0:
            # plot a few static frames at the beginning of the gif...
            for _ in range(1):
                fig.write_image(f'imgs/img_{frame_count}.png')
                frame_count += 1
        else:
            fig.write_image(f'imgs/img_{frame_count}.png')
            frame_count += 1
        plt.close()

    gif_save('imgs', 'videos/forward_sphere.gif')
