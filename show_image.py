from utils.filegetter import afn
import matplotlib.pyplot as plt
from skimage.io import imread

fn = afn(title="image")
im = imread(fn)
plt.imshow(im)
plt.show()