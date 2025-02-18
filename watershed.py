from pathlib import Path
from matplotlib.image import AxesImage
from skimage.segmentation import watershed
from skimage.io import imread,imshow,imsave
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.measure import label
from scipy.ndimage import sobel

folder = Path(r"C:\Users\Harrison Truscott\OneDrive - University of North Carolina at Chapel Hill\Bear Lab\optotaxis calibration\data\segmentation_iteration_testing\continuous_out_testing\example")

image = imread(folder/"example_img.TIF")
segmented = imread(folder/"example_fastai_preds.TIF")
raw = imread(folder/"example_raw_difference.TIF")

##strategy 1: use the distance to the background to detect separate cells using the B/W mask
## from https://www.mathworks.com/company/newsletters/articles/the-watershed-transform-strategies-for-image-segmentation.html
## and https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
from scipy import ndimage as ndi
distance = ndi.distance_transform_edt(segmented)
# mask = np.zeros(distance.shape, dtype=bool)

import numpy as np
ax:AxesImage = imshow(raw)
coords = peak_local_max(raw, footprint=np.ones((10, 10)), labels=label(image>0),threshold_abs=0)
print(len(coords))
print(coords)
# ax.axes.scatter(coords[:,1],coords[:,0],marker='.')
plt.figure("image")
plt.imshow(image)
plt.figure("segmented distance")
plt.imshow(-distance)
plt.figure("product with product peaks")
im = watershed(-distance,mask=segmented,connectivity=1)
i2 = distance * raw
print(i2.__array__())

imsave("dum.TIF",i2)

peaks = peak_local_max(i2,footprint=np.ones((7,7)),labels=label(i2>0),threshold_abs=0)
plt.imshow(i2)

plt.scatter(peaks[:,1],peaks[:,0],marker='.',color='red')
plt.figure("sobel")
# s = sobel(i2)
# plt.imshow(sobel)

plt.figure("distance only watershed")
plt.imshow(im)
im2 = watershed(-i2,mask=segmented,connectivity=1)
plt.figure("product watershed")
plt.imshow(im2)
plt.figure("raw")
plt.imshow(raw)
# plt.scatter(peaks[:,1],peaks[:,0],marker='.',color='red')
plt.figure("segmented with product peaks")
plt.imshow(segmented)
plt.scatter(peaks[:,1],peaks[:,0],marker='.',color='red')
# imshow(raw)
plt.show()