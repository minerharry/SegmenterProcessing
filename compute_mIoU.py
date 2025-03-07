from pathlib import Path

from tqdm import tqdm
from utils.filegetter import adir
from imageio.v3 import imread
import numpy as np

d1 = Path(adir(key="mIoU dir1"))
d2 = Path(adir(key="mIoU dir2"))


IoUs = []
for f in tqdm(d1.glob("*.tif")):
    im1 = imread(d1/f.name)
    im2 = imread(d2/f.name)

    im1 = ~np.isclose(im1,0)
    im2 = ~np.isclose(im2,0)

    iou = np.sum(im1 * im2) / np.sum(im1 | im2)
    IoUs.append(iou)

mean = np.mean(IoUs)
print("mean IoU:",mean)
