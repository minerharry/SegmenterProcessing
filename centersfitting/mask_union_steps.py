
import itertools
from tkinter.filedialog import askopenfilename
from skimage.io import imshow
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.animation import FuncAnimation
import tqdm

def centroid(m:np.ndarray,id=None):
    if id:
        y_center, x_center = np.argwhere(m==1).mean(0)
    else:
        y_center, x_center = np.argwhere(m).mean(0)
    return np.array([y_center,x_center])

def triple_step(m1:np.ndarray,m2:np.ndarray,m3:np.ndarray)->np.ndarray:
    """looks at the difference in (the center of mass of m1 and m2) and (the center of mass of m2 and m3)"""
    return centroid(m3&m2) - centroid(m2&m1)

if __name__ == "__main__":
    maskspath = askopenfilename()
    file = tifffile.TiffFile(maskspath)
    
    # f = plt.figure()
    # gs = f.add_gridspec(2,6)
    # im0 = f.add_subplot(gs[0, 0:2])
    # im1 = f.add_subplot(gs[0, 2:4])
    # im2 = f.add_subplot(gs[0, 4:6])
    # comb1 = f.add_subplot(gs[1, 0:3])
    # comb2 = f.add_subplot(gs[1, 3:6])

    # axs = [im0,im1,im2,comb1,comb2]

    # bbox = None

    imIter = iter(file.series[0])
    images = [next(imIter).asarray() for _ in range(3)]

    data = [centroid(images[0])]

    gim = images[0]

    for nim in tqdm.tqdm(imIter,total=len(file.series[0])-3):
        step = triple_step(*images)
        if np.any(np.isnan(step)):
            step = np.array([0,0])
        else:
            gim = images[0]
        data.append(data[-1]+step)
        _,*images = *images,nim.asarray()

    print(data)

    plt.imshow(gim)
    plt.plot(*(np.array(data).T[::-1]))
    plt.show()



    # def animate(k):
    #     global bbox,images
    #     _,*images = *images,next(imIter).asarray()

    #     cy,cx = np.where(images[0])
    #     try:
    #         bbox = [[min(cy)-40,max(cy)+40],[min(cx)-40,max(cx)+40]]
    #     except:
    #         pass

    #     im0.imshow(images[0])
    #     im1.imshow(images[1])
    #     im2.imshow(images[2])
    #     comb1.imshow(images[0]&images[1])
    #     comb2.imshow(images[1]&images[2])

    #     for ax in axs:
    #         ax.set_ylim(bbox[0])
    #         ax.set_xlim(bbox[1])
    #     return ax
        

    # anim = FuncAnimation(f,animate,tqdm.trange(100))
    # anim.save("anim.mp4")
    # plt.show()    