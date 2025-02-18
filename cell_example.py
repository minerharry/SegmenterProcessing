import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import math
import skimage.io
from scipy.special import expit

fig,ax = plt.subplots()
center = (0,0)
radius = 1

stimcenter = (-0.7,0)
stimcrossradius = 0.4

k = math.exp(-(stimcrossradius**2)*20) #y-offset

def f(x,y):
    r = np.linalg.norm([x-stimcenter[0],y-stimcenter[1]],axis=0)
    res = stimcrossradius - r
    res[res > 0] = (stimcrossradius**2 - r**2)[res > 0]*7
    # return expit(res)*0.6-0.3
    # res[:] = -3
    # return res
    # res[res < 0] -= 0.5
    return expit(res)
    # return (np.exp(-np.linalg.norm([x-stimcenter[0],y-stimcenter[1]],axis=0)*20)-k)

def f2(x,y):
    return f(-x,y)
# print(f)

# im = f()

x = y = np.linspace(-1.5, 1.5, 1000)
X, Y = np.meshgrid(x, y)
zs = np.array((f(np.ravel(X), np.ravel(Y)) + 0.7*f2(np.ravel(X),np.ravel(Y)))*0.8)
Z = zs.reshape(X.shape)

axim = skimage.io.imshow(Z,extent=[-1.5,1.5,-1.5,1.5],cmap="Blues")
cb = axim.colorbar
if cb:
    cb.remove()

print(axim)

cell = Ellipse(center,radius*2,radius*1.5,transform=ax.transData,ec='black',facecolor='none')
axim.set_clip_path(cell)

ax.add_patch(cell)

ax.axis('off')
plt.show()