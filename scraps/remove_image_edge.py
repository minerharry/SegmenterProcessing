import numpy as np;
from skimage.io import imread,imsave;
from pathlib import Path;
import os;
from tqdm import tqdm;

in_folder = "";
in_folder = Path(in_folder);

names = os.listdir(in_folder);
for name in tqdm(names):
    im = imread(in_folder/name);
    im[:,0] = 0;
    im[0,:] = 0;
    im[-1,:] = 0;
    im[:,-1] = 0;