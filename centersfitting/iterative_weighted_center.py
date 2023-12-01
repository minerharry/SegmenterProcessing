# from cmath import exp
# from time import sleep
from typing import Any, Callable
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from skimage.io import imshow,imsave,imread
from tifffile import TiffFile

def decay(d2:float|np.ndarray,sd:float):
    return np.exp(-d2/100)

def dist2(p1,p2):
    return np.sum(np.square(p1-p2),axis=2)


def iterative_weighted_centers(mask:np.ndarray,iters=4,__center=None,decay_func:Any=decay):
    if iters == 0:
        if __center is None:
            raise ValueError("Iterations must be an integer greater than zero")
        else:
            return [__center]

    posy,posx = np.where(mask)

    weight:np.ndarray
    if __center is None:
        weight = np.ones(mask.shape)/np.sum(mask)
        __center = np.argwhere(mask).mean(0)
    else:
        temp = (np.fromfunction(lambda y,x:dist2(np.transpose((y,x)),__center),mask.shape)).transpose()
        # print(temp)
        d2s = np.zeros(mask.shape)
        d2s[posy,posx] = temp[posy,posx]
        # plt.imshow(temp)
        # plt.show()
        # print(d2s)

        decayed = decay(d2s,1)
        # print(decayed)
        # plt.imshow(decayed)
        # plt.show()
        weight = np.zeros(mask.shape)
        weight[posy,posx] = decayed[posy,posx]
        # print(np.sum(weight))
        weight /= np.sum(weight)
        # print(weight)


    masses = weight[posy,posx].reshape(-1,1);
    CM = np.sum(np.array([posy,posx]).transpose()*masses,axis=0) #get array of vector positions, multiply by masses at that position
    # print(CM)
    return [__center] + iterative_weighted_centers(mask,iters = iters-1, __center = CM,decay_func=decay_func)

def iterative_weighted_center(mask:np.ndarray,iters:int=4,decay_func:Callable[[float|np.ndarray,float],float|np.ndarray]=decay):
    return iterative_weighted_centers(mask,iters=iters,decay_func=decay_func)[-1];


if __name__ == "__main__":
    maskspath = Path(r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\temp\tracks_masks")
    for p in [maskspath/"2023.4.2 OptoTiam Exp 53_movie3_track2.TIF",maskspath/"2023.4.2 OptoTiam Exp 53_movie3_track3.TIF",maskspath/"2023.4.2 OptoTiam Exp 53_movie3_track5.TIF"]:
        filename = maskspath/p
        print(filename)
        file = TiffFile(filename)
        im = file.series[0][0].asarray();
        CM = iterative_weighted_center(im)
        # print(CM)
        # CM = np
        print([int(CM[0]),int(CM[1])])
        im[int(CM[0]),int(CM[1])] = 10
        plt.imshow(im)
        plt.show()