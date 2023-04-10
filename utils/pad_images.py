import os
from pathlib import Path
from cv2 import detail_HomographyBasedEstimator
import numpy as np
from tqdm import tqdm
import filegetter as filegetter
from skimage.io import imread,imsave



def padding(array:np.ndarray, xx:int, yy:int):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=[(a, aa), (b, bb)], mode='edge');


if __name__ == "__main__":
    dir = Path(filegetter.askdirectory());

    size = (1024,1344);


    for n in tqdm(os.listdir(dir)):
        if (not n.endswith(("tif","TIF"))):
            continue;
        im = imread(dir/n);
        # print(im.shape);
        im = padding(im,*size);
        # print(im.shape);
        imsave(dir/n,im,check_contrast=False);
        
