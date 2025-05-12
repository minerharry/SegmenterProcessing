import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

##CLASS TO MAKE DIVERGENT COLORMAPS WORK!
class MidpointNormalize(colors.Normalize):
    """Custom normalization subclass to center diverging colormaps at a specified anchor point value (default:0) 
    such that a data value of the specified midpoint always maps to the center of the underlying colormap (colormap position 0.5).
    Usage:

    ```
    image = imread("image.png")
    norm = MidpointNormalize(vmin=np.min(image), vmax=np.max(image), midpoint=0) #or don't specify vmin, vmax to have them set automatically when used on a set of data
    cmap = 'RdBu_r' 
    plt.imshow(image, cmap=cmap, norm=norm)
    ```
    """
    def __init__(self, vmin:float|None=None, vmax:float|None=None, midpoint:float=0, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))