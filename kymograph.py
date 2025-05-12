import math
from typing import Iterable
import matplotlib
import matplotlib.axes
import numpy as np
from PIL import Image
from scipy import ndimage


def shape[T,N](values:Iterable[T],axes:Iterable[int],default:N=None,length:int|None=None)->list[T|N]:
    res:list[T|N] = [default]*(length if length is not None else max(axes)+1)
    for v,a in zip(values,axes):
        res[a] = v
    return res

#used from https://github.com/ome/training-scripts/blob/master/practical/python/server/Kymograph.py. Used with modification (to just use np arrays) under GNU public license
#assuming channel has already been selected - since we can just assume they use a numpy view to select the channel
def get_line_data(stack:np.ndarray, x1:float, y1:float, x2:float, y2:float, line_w:float=2, channel_order:str="TYX",sum_flatten=True):#the_z=0, the_c=0, the_t=0):
    """
    Grab pixel data covering the specified line, and rotates it horizontally. Works on a stack of images (presumed default) or a single image (exclude "T" from channel_order)

    Uses current rendering settings and returns 8-bit data.
    Rotates it so that x1,y1 is to the left,
    Returning a numpy 2d array. Used by Kymograph.py script.
    Uses PIL to handle rotating and interpolating the data. Converts to numpy
    to PIL and back (may change dtype.)

    @param pixels:          PixelsWrapper object
    @param x1, y1, x2, y2:  Coordinates of line
    @param line_w:          Width of the line we want
    @param channel_order:   Channel order string of stack (Default: "TYX")
    @param sum_flatten:     Optional: whether to sum the input values and return a 2d [1d if not a time series] array with channel order ([Time],Distance). 
            If false, returns 3d [2d if not time series] array with channel order ([Time],Distance,Offset)
    """

    if stack.ndim != len(channel_order):
        if stack.ndim == 2: #assume flat array
            channel_order = "YX"
        else:
            raise ValueError(f"Channel order string \"{channel_order}\" must have same length as # of dimensions of array, {stack.ndim}")
    print(channel_order)

    xdim = channel_order.find("X")
    ydim = channel_order.find("Y")
    tdim = channel_order.find("T")
    if tdim == -1:
        tdim = None

    size_x = stack.shape[xdim]
    size_y = stack.shape[ydim]

    line_dx = x2 - x1
    line_dy = y2 - y1

    rads = math.atan2(line_dy, line_dx)

    # How much extra Height do we need, top and bottom?
    extra_h = abs(math.cos(rads) * line_w)
    bottom = math.ceil(max(y1, y2) + extra_h/2)
    top = math.floor(min(y1, y2) - extra_h/2)

    # How much extra width do we need, left and right?
    extra_w = abs(math.cos(rads) * line_w)
    left = math.floor(min(x1, x2) - extra_w)
    right = math.ceil(max(x1, x2) + extra_w)

    # What's the larger area we need? - Are we outside the image?
    pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    if left < 0:
        pad_left = abs(left)
        left = 0
    x = left+pad_left
    if top < 0:
        pad_top = abs(top)
        top = 0
    y = top+pad_top
    if right > size_x:
        pad_right = right-size_x
        right = size_x
    x2 = right+pad_left
    if bottom > size_y:
        pad_bottom = bottom-size_y
        bottom = size_y
    y2 = bottom+pad_top

    pad_sizes = shape([(pad_left,pad_right),(pad_top,pad_bottom)],(xdim,ydim),(0,0),length=stack.ndim) #0,0 + explicit length makes sure all nonspecified axes stay the same. Works in the cast of unspecified time dimension
    padstack = np.pad(stack,pad_sizes)
    print(padstack.shape)

    crop = shape((slice(x,x2),slice(y,y2)),(xdim,ydim),slice(None,None),length=stack.ndim) #only slice x and y, leave all other dimensions (r.e.: time) untouched
    substack = padstack[tuple(crop)] #perform crop
    print(crop)
    print(substack.shape)

    rotated = ndimage.rotate(substack,math.degrees(rads),axes=(xdim,ydim))
    print(rotated.shape)


    # finally we need to crop to the length of the line. We know the rotation was about the center of the image, so dividing by two gets us what we need
    length = int(math.sqrt(math.pow(line_dx, 2) + math.pow(line_dy, 2)))
    rot_w, rot_h = rotated.shape[xdim],rotated.shape[ydim]
    crop_x = (rot_w - length)//2
    crop_x2 = crop_x + length
    crop_y = (rot_h - line_w)//2
    crop_y2 = crop_y + line_w
    rotcrop = shape([slice(crop_x,crop_x2),slice(crop_y,crop_y2)],(xdim,ydim),slice(None,None),length=stack.ndim);
    cropped = rotated[tuple(rotcrop)]
    print(cropped.shape)

    cropped = np.moveaxis(cropped,(tdim,xdim,ydim),(0,1,2)) if tdim else np.moveaxis(cropped,(xdim,ydim),(0,1))#make time axis first, then distance axis, then thickness. if sum_flatten, we'll sum over this thickness

    if sum_flatten:
        flattened = np.sum(cropped,axis=-1) #last axis is always thickness
        return flattened
    else:
        return cropped


def make_kymograph(stack:np.ndarray,start_xy:tuple[float,float],end_xy:tuple[float,float],width:float,channel_order="TYX"):
    return get_line_data(stack,*start_xy,*end_xy,line_w=width,channel_order=channel_order)

