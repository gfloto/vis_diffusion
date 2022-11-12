import sys, os, re, torch
import numpy as np
from PIL import Image

# dataset for training
def get_points_(n):
    x = 3*torch.rand(n) - 1.5
    y = 3*torch.rand(n) - 1.5
    z = manifold(x,y)
    return torch.stack((x,y,z)).T

# sample point according to mixture of gaussian
def get_points(n):
    # sample from each mixture w same prob
    x1 = torch.randn(n)
    x2 = torch.randn(n)
    y1 = torch.randn(n)
    y2 = torch.randn(n)
    
    # add offsets
    d = torch.ones_like(x1)

    # apply offset
    x1 += d
    x2 -= d
    y1 += d
    y2 -= d

    z1 = manifold(x1, y1)
    z2 = manifold(x2, y2)
    return (torch.stack((x1, y1, z1)).T, torch.stack((x2, y2, z2)).T)

# make sphere for plotting forward process
def make_sphere(spot, r):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    return (x+spot[0], y+spot[1], z+spot[2])

# make manifold
def make_manifold(r, n):
    x = np.linspace(-r, r, n)
    y = np.linspace(-r, r, n)

    # make manifold
    X, Y = np.meshgrid(x, y)
    Z = manifold(X, Y)
    return X, Y, Z

# example of latent manifold in 3d space
def manifold(x, y): 
    return (-0.2*x*y + 0.25*np.sin(2*x + y)) - 0.015*y**3 * x

# sorts keys according to alpha numeric
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
        #if i == 0 or i == len(ims) - 1:
        #    for _ in range(14):
                #video.append(frame)
        video.append(frame)
        

    # save gif
    save_path = os.path.join(path, name+'.gif') 
    video[0].save(save_path, format='gif',
                   append_images=video[1:],
                   save_all=True,
                   duration=60, loop=0)   
    return