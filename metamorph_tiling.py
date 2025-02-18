import functools
import math
import itertools
import os
from pathlib import Path
import re
from typing import Any, Collection, Iterable, Literal, Protocol, Sequence, Sized

import matplotlib
from matplotlib.artist import Artist
from matplotlib.axes import Axes
import matplotlib.patches
from matplotlib.text import Text
import numpy as np
import scipy
import scipy.linalg

header = """"Stage Memory List", Version 6.0
0, 0, 0, 0, 0, 0, 0, "UserSteps", "UserSteps"
0
{n_stages}
"""

xOffset = {"4x":21500,"10x":8593,"20x":4332}
yOffset = {"4x":16300,"10x":6640,"20x":3320}

xOverlap = 0.06
yOverlap = 0.06

Tiling = Collection[tuple[str,tuple[int,int]]] #[(name, (x,y)),]

default_centers = ((0.35,0.65),(0.35,0.65))

class Tiler(Protocol):
    def __call__(self,
        xLeft:int,xRight:int,yTop:int,yBottom:int,
        xOffset:dict[str,int]=xOffset,
        yOffset:dict[str,int]=yOffset,
        xOverlap:float=xOverlap,
        yOverlap:float=yOverlap,
        magnification:str="4x")->Tiling: ...
    
def monkey_tiler(axis:Literal[0,1,"x","y"],L:int, R:int)->Tiler:

    def monkey_in_the_middle_tiling(xLeft:int,xRight:int,yTop:int,yBottom:int,
            xOffset:dict[str,int]=xOffset,
            yOffset:dict[str,int]=yOffset,
            xOverlap:float=xOverlap,
            yOverlap:float=yOverlap,
            magnification:str="4x")->Tiling:
        return monkey_in_the_middle(axis,L,R,xLeft,xRight,yTop,yBottom,xOffset,yOffset,xOverlap,yOverlap,magnification);

    return monkey_in_the_middle_tiling;

def monkey_in_the_middle(axis:Literal[0,1,"x","y"],
                         L:int, R:int, #"Left" (low), "Right" (high)
                         xLeft:int,xRight:int,yTop:int,yBottom:int,
                         xOffset:dict[str,int]=xOffset,
                         yOffset:dict[str,int]=yOffset,
                         xOverlap:float=xOverlap,
                         yOverlap:float=yOverlap,magnification:str="4x")->Tiling:
    ###OK so wtf is this
    ## Issue: in the far-red objective, the *center* (horizontally) of the images is just totally fucked. However, this is still fixable in software - we only need
    ## one good look at the cells. We could massively increase the overlap, but that would be super inefficient. Instead, we can make pairs of tiles that together
    ## cover the space of one larger tile and whose sides match the broken bit in the middle, like this:
    ##
    ##  -----------------------------------------
    ##  |             |           |             |
    ##  |<-----L----->|<----W---->|<-----R----->|
    ##  |             |           |             |
    ##  -----------------------------------------
    ##               -----------------------------------------
    ##               |             |           |             |
    ##               |<-----L----->|<----W---->|<-----R----->|
    ##               |             |           |             |
    ##               -----------------------------------------
    ##
    ## Note that this requires that both L and R to be greater than W; W = {...}Offset - L - R
    ##
    ## To go for best coverage of W, we don't want to just align the left/right side of W with L/R; instead, we should center the alignment of W with whichever side (L/R) is shorter

    axes = ("x","y")
    if isinstance(axis,str):
        axis:int = axes.index(axis)
    offsets = [xOffset,yOffset]
    W = offsets[axis][magnification] - L - R

    if (W > L or W > R):
        raise ValueError(f"Unusable region too wide for monkey-in-the-middle; bad middle region of size {W} must be smaller than the good side regions {(L,R)}")
    
    ##measured from the low (left) side of the image
    W_center = L + W//2
    shift:int
    if L > R: #center W on R
        shift = L + W + R//2 - W_center #center of R minus center of W
    else:
        shift = W_center - L//2 #center of L minus center of W

    ## now, we're going to masquerade as larger tiles to get_tiling, then split every resulting tile in twain
    tile_size = L + W + R + shift
    
    big_offsets = [xOffset.copy(),yOffset.copy()]
    big_offsets[axis][magnification] = tile_size

    big_tiles = get_tiling(xLeft,xRight,yTop,yBottom,big_offsets[0],big_offsets[1],xOverlap,yOverlap,magnification)

    #now, split the tiles into the real tiles
    tiles:list[tuple[str,tuple[int,int]]] = []
    for (name,pos) in big_tiles:
        pos = list(pos)
        pos[axis] -= shift//2
        t1 = (f"{name}_L",(pos[0],pos[1]))
        pos[axis] += shift
        t2 = (f"{name}_R",(pos[0],pos[1]))
        tiles += [t1,t2]

    return tiles



def get_tiling(
        xLeft:int,xRight:int,yTop:int,yBottom:int,
        xOffset:dict[str,int]=xOffset,
        yOffset:dict[str,int]=yOffset,
        xOverlap:float=xOverlap,
        yOverlap:float=yOverlap,
        magnification:str="4x")->Tiling:
    boundss = [xLeft,xRight],[yTop,yBottom]
    offsets = xOffset[magnification],yOffset[magnification]
    overlaps = xOverlap,yOverlap ##overlap is on both sides of a tile; for a tile in the middle of a row with x offset 0.2, the center 0.6 will be unique to that tile
    tile_coords = [[],[]]
    axes = ["x","y"]
    for axis in [0,1]:
        bounds,offset,overlap = boundss[axis],offsets[axis],overlaps[axis]
        #assert bounds[1] >= bounds[0]
        tile_edgewidth = int(offset*overlap)
        width = bounds[1] - bounds[0]
        if width == 0: #identical bounds given, hardcoded single tile
            n_tiles = 1
            tile_offset = 0
            eff_width = 0
            print("warning: interpreting 0-width region as singletile")
        else:
            tile_offset = offset - tile_edgewidth
            eff_width = width - tile_edgewidth
            if eff_width < 0:
                tile_offset *= -1
            n_tiles = max(math.ceil(eff_width/tile_offset),0)+1
        print(f"{axes[axis]} tile_offset:",tile_offset,"eff_width:",eff_width,"n_tiles:",n_tiles)
        tile_coords[axis] = [int((i-n_tiles/2+1/2)*tile_offset + (bounds[0]+bounds[1])/2) for i in range(n_tiles)]
    
    tiles = []
    #print(list(itertools.product(enumerate(tile_coords[0]),tile_coords[1]))))
    for (xn,x),(yn,y) in itertools.product(enumerate(tile_coords[0]),enumerate(tile_coords[1])):
        tiles.append((f"s{yn+1}_{xn+1}",(x,y)))
    
    return tiles

def zify_tiles(tiles:Tiling,*points:tuple[float,float,float],method:Literal['linear','quadratic']='linear')->Iterable[int]:

    #solve ax + by + c = z via [(x,y,1),...] @ (a,b,c) = [z,...] = A @ x = Z
    print(points)
    ptarray = np.array(points)
    print(ptarray.shape)
    # from IPython import embed; embed()
    if method == 'linear':
        A = np.concatenate([ptarray[:,:2],np.ones((ptarray.shape[0],1))],axis=-1)
        Z = ptarray[:,2]

        (a,b,c),residuals,rank,sv = scipy.linalg.lstsq(A,Z)
        a2,b2 = 0,0
    elif method == 'quadratic':
        A = np.concatenate([ptarray[:,:2]**2,ptarray[:,:2],np.ones((ptarray.shape[0],1))],axis=-1)
        Z = ptarray[:,2]

        (a2,b2,a,b,c),residuals,rank,sv = scipy.linalg.lstsq(A,Z)
    else:
        raise ValueError(method)

    #get z of each tile via z = ax + by + c 
    for (name,(x,y)) in tiles:
        yield a*x + b*y + c + a2*x**2 + b2*y**2

file_format = "\"{name}\", {x}, {y}, {z}, 0, 0, FALSE, -9999, TRUE, TRUE, 0, -1, \"\"\n"

def write_tiling(filename:os.PathLike|str,tiles:Tiling,z:float|Iterable[float]=0):
    from typing import Iterable
    if not isinstance(z,Iterable):
        z = itertools.cycle([z])
    with open(filename,"w") as f:
        f.write(header.format(n_stages=len(tiles)))
        for (name,(x,y)),z in zip(tiles,z):
            f.write(file_format.format(name=name,x=x,y=y,z=z))

def read_tiling(filename:os.PathLike|str)->tuple[Tiling,list[float]]:
    with open(filename,"r") as f:
        lines = f.readlines()
        n_stages = int(lines[3])
        tiles = []
        zs = []
        for line in lines[4:]:
            name,x,y,z,*_ = line.split(", ");
            name = name.strip("\"")
            x = int(x)
            y = int(y)
            z = float(z)
            tiles.append((name,(x,y)))
            zs.append(z)
        return tiles,zs
        
def plot_tiling(mag:str,
                tiles:Tiling,
                z:float|Iterable[float]=0,
                ax:Axes|None=None,
                color:Any=None,
                do_text:bool=True,
                im_center:tuple[float,float]=(0.5,0.5)
                )->tuple[list[matplotlib.patches.Patch],list[Text]]:


    tile_size = np.array((xOffset[mag],yOffset[mag]))
    rects = []
    texts = []
    for (name,pos) in tiles:
        corner = pos - tile_size*im_center
        rect = (matplotlib.patches.Rectangle(corner,tile_size[0],tile_size[1],fill=False,color=color));
        if do_text:
            text = Text(corner[0],corner[1],name)
        else:
            text = None
        if ax:
            ax.add_patch(rect)
            if text: ax.add_artist(text)
    
        rects.append(rect)
        texts.append(text)

    return rects,texts

def plot_z(tiles:Tiling,
           z:float|Iterable[float],
           zpoints:Iterable[tuple[tuple[int,int]|str,float]],
           ax:Axes):
    if not isinstance(z,Iterable):
        z = itertools.cycle([z])
    tile_zpoints = [(x,y,z) for ((name,(x,y)),z) in zip(tiles,z)]
    # make_zpoint(tiles,)
    manual_zpoints = [make_zpoint(tiles,idx,z) for idx,z in zpoints]

    if ax.name != "3d":
        raise ValueError("Axes must be 3d!")
    
    tilearr = np.array(tile_zpoints).T
    ax.scatter(tilearr[0],tilearr[1],tilearr[2])

    manarr = np.array(manual_zpoints).T
    ax.scatter(manarr[0],manarr[1],manarr[2])


def make_zpoint(tiling:Tiling,idx:tuple[int,int]|str,z:float):
    point = dict(tiling)[f"s{idx[0]}_{idx[1]}" if isinstance(idx,tuple) else idx]
    return (point[0],point[1],z)

def apply_zpoints(tiling:Tiling,points:Iterable[tuple[tuple[int,int]|str,float]],method:Literal['linear','quadratic']='linear'):
    z_tiles = [make_zpoint(tiling,idx,zpoint) for idx,zpoint in points]
    return zify_tiles(tiling,*z_tiles,method=method)

def mag_adjust_bounds(orig_bounds:tuple[tuple[int,int],tuple[int,int]],
                      dest_mag:str,
                      orig_mag:str,
                      center_ranges:tuple[tuple[float,float],tuple[float,float]]=default_centers
                      )->tuple[tuple[int,int],tuple[int,int]]:
    x_bounds,y_bounds = orig_bounds
    print(f"pre transform xbounds: {x_bounds} ybounds: {y_bounds}")
    
    def sign(num:int):
        if num > 0:
            return 1
        if num < 0:
            return -1
        return 0
        
    width_off = xOffset[orig_mag] - xOffset[dest_mag]
    height_off = yOffset[orig_mag] - yOffset[dest_mag]
    xsign = sign(width_off)
    ysign = sign(height_off)

    coeffs = (
            (max(xsign*center_ranges[0][0],xsign*center_ranges[0][1]),max(xsign*(1-center_ranges[0][0]),xsign*(1-center_ranges[0][1]))),
            (max(ysign*center_ranges[1][0],ysign*center_ranges[1][1]),max(ysign*(1-center_ranges[1][0]),ysign*(1-center_ranges[1][1])))
            )
    
    print(f"transform coefficients: {coeffs}")
        
    x_bounds = (
        int(x_bounds[0] - coeffs[0][0]*width_off),
        int(x_bounds[1] + coeffs[0][1]*width_off)
    )
    y_bounds = (
        int(y_bounds[0] - coeffs[1][0]*height_off),
        int(y_bounds[1] + coeffs[1][1]*height_off)
    )
    
    print(f"post transform xbounds: {x_bounds} ybounds: {y_bounds}")
    return x_bounds,y_bounds

def main(output:str|os.PathLike|None,
         mag:str,
         x_bounds:tuple[int,int],
         y_bounds:tuple[int,int],
         z:int|Iterable[tuple[tuple[int,int],float]], #stagepos,z
         orig_mag:str|None=None,
         center_ranges:tuple[tuple[float,float],tuple[float,float]] = default_centers,
         get_tiling:Tiler = get_tiling): #xlow,high; ylow,high


    ## SIZE TRANSFORMATION
    # if you want the same region tiled in two magnifications, using the settings of one doesn't work for the other because the higher magnification
    # will fit more images in the edges
    # One solution is to simply use the higher magnification to tile the lower, so the relevant region is still captured. however, it would be nice
    # if the total area imaged was comparable between the two so less cropping is needed
    # to do this, we can add half* the width of the source mag images, minus half* the width of the destination mag images, to the edges of the x bounds,
    # and similarly to the edges of the y bounds. The shift is based on precisely half of the image when the objectives are precisely aligned (center of
    # both images is the same); however, this is rarely the case. Therefore, there needs to be some margin for error left in the half based on where the center
    # could actually be.
    # for example, if the center was at 0.25*w,0.25*h, the offsets should be (0.25*source - 0.25*dest,0.75*source - 0.75*dest).
    # we can add in room for error by making the left/right coefficients not add up to 1; if the dest image is smaller than the source, we should err
    # on the side of a larger coefficient so more is added to the tiling area, and if the dest image is larger than the source, we should err on the side
    # of a smaller coefficient so less is removed. (using the higher mag tiling directly on a lower mag is the latter case with both coefficients 0!).
    # Therefore, if we think our center is (0.5,0.5) +/- (0.2,0.2), our coefficients should be (0.3,0.3) for high->low mag, and (0.7,0.7) for low->high mag
    
    if orig_mag is not None:
        x_bounds,y_bounds = mag_adjust_bounds((x_bounds,y_bounds),mag,orig_mag,center_ranges);
    
    topleft = (x_bounds[0],y_bounds[0]) #x,y
    bottomright = (x_bounds[1],y_bounds[1]) #x,y
    
    #zpoint_1 = None
    print(f"tiling from {topleft} to {bottomright} at z position {z}")
    
    
    tiles = get_tiling(topleft[0],bottomright[0],topleft[1],bottomright[1],magnification=mag)

    
    if isinstance(z,Iterable):
        tile_z = apply_zpoints(tiles,z)
    else:
        tile_z = z

    if output:
        write_tiling(output,tiles,z=tile_z)
        print(f"File {Path(output).name} created successfully")

    return tiles,tile_z,(x_bounds,y_bounds)

def get_corners(xs:tuple[float,float],ys:tuple[float,float]):
    return [xs[0],xs[0],xs[1],xs[1],xs[0]],[ys[0],ys[1],ys[1],ys[0],ys[0]]

if __name__ == "__main__":
    orig_mag = None
    

    #note: 
    # y_bounds = (-172465,-168726)
    #x_center = -223529
    #x_radius = 2000
    #x_radius = 36450 #good radius for fixing, center ~1/3-1/2
    #x_bounds = (x_center - x_radius, x_center + x_radius) \
    

    # mag="4x"
    # z=4560
    

    #yOverlap = 0.4 # for that reeeeally annoying streak in the top of the cy5 image
    
    # mag = "20x"
    # z = 4505
    # yOverlap = 0.15

    mag = "4x"
    z = 4800
    
    ##Manual bounds:
    if True:
        x_bounds = (-340963,-280633);
        y_bounds = (-328001,-340568);
    
        output = fr"C:\Olympus\app\mmproc\DATA\auto_tiling_{mag}.STG"
    


    ##Gradient Snap
    if False:
        x_bounds = (-281795,-261000)
        y_bounds= (-19097,-227235)
        
        x_mid = -292624 #shift window based on midpoints
        y_anchors = (-151722,-332550)
        y_mid = (y_anchors[1] + y_anchors[0])//2

        x_bounds = (x_mid - (x_bounds[1]-x_bounds[0])//2, x_mid + (x_bounds[1] - x_bounds[0])//2)
        y_bounds = (y_mid + (y_bounds[1]-y_bounds[0])//2, y_mid - (y_bounds[1] - y_bounds[0])//2)
        output = fr"C:\Olympus\app\mmproc\DATA\gradient_tiling_{mag}.STG"






    zpoints = None
    # zpoints = [
    #     ((4,1),4502),
    #     ((3,4),1000),
    #     ((3,4),1000)
    # ]

    
    # output = fr"C:\Olympus\app\mmproc\DATA\auto_tiling_{mag}.STG"
    
    
    
    
    #if uncommented, will do size transformation
    # orig_mag = "4x"

    
    tiler:Tiler = get_tiling

    do_monkey = False
    if do_monkey:
        width = int(xOffset[mag]/(2.8))
        tiler = monkey_tiler('x',width,width);
    


    tiling,z_tile,(x_bounds2,y_bounds2) = main(output,mag,x_bounds,y_bounds,zpoints or z,orig_mag=orig_mag,get_tiling=tiler)


    mpl_test = False
    if mpl_test:
        import matplotlib.pyplot as plt
        ax:Axes
        fig,ax = plt.subplots()

        plot_tiling(mag,tiling,z=z,ax=ax,color='blue',do_text=False,im_center=(0.5,0.5))

        # tiling2,ztile2,bounds = main(output,'4x',x_bounds,y_bounds,zpoints or z,orig_mag=orig_mag)
        # plot_tiling('4x',tiling2,z=z,ax=ax,color='red',do_text=False)

        corners = get_corners(x_bounds,y_bounds)
        ax.plot((corners)[0],(corners)[1],color='black')

        # corners2 = get_corners(x_bounds2,y_bounds2)
        # ax.plot((corners2)[0],(corners2)[1],color='purple')

        # corners3 = get_corners(*bounds)
        # ax.plot(*corners3,color='green')
        
        ax.autoscale_view()

        
        # for rect in rects:
        #     ax.add_patch(rect);

        
        from IPython import embed; embed()
    
    