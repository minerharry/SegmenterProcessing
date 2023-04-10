import numpy as np
from skimage.io import imread,imsave,imshow
from utils.filegetter import askopenfilename,asksaveasfilename
inmask = askopenfilename();
# outmask = asksaveasfilename();
image = imread(inmask);
print(image.max());
bitData = image
print(bitData);
print(np.max(bitData));
print(np.min(bitData));
vals,counts = np.unique(bitData, return_counts=True)
index = np.argmax(counts)
print(vals[index]);

# image = np.invert(image);
# imsave(outmask,image)