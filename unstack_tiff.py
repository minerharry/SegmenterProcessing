from filegetter import askopenfilename
from skimage.io import imread
from pympler.asizeof import asizeof
from tifffile import TiffFile

tiff = askopenfilename()

print(tiff)
start = asizeof(locals())
file = TiffFile(tiff);

import code
code.interact(local=locals())