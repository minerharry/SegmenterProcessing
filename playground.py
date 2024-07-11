from utils.filegetter import afn, afns, adir
from skimage.io import imread, imsave, imshow
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.measure import label
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

def read_mask(path):
    m = imread(path)
    m = binary_fill_holes(m)
    return m

def read_image(path):
    im = imread(path)
    im = rescale_intensity(im)
    return im

if __name__ == "__main__":
    from IPython import embed; embed()