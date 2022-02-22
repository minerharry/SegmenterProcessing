import numpy as np
from skimage.io import imread,imsave,imshow
inmask = 'working_masks/random_s2_t125.TIF'
outmask = 'working_masks/random_s2_t125.TIF';
image = imread(inmask);
print(image.max());
image = np.invert(image);
imsave(outmask,image)