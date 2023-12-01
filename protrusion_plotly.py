import os
from pathlib import Path
import re
from telnetlib import IP
from typing import Any, Callable, Sequence
# from matplotlib import gridspec
# from matplotlib.animation import FuncAnimation
# from matplotlib.figure import Figure, SubFigure
# import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import sparse
from tqdm import tqdm
from libraries.centers import get_appmedoid
from libraries.movie_reading import ImageMapMovie, Movie, MovieSequence
# from utils.paging import paging_bar,Index
from utils.filegetter import afn
from tifffile import TiffFile,TiffPage,TiffFrame
import IPython

from utils.parse_moviefolder import get_movie



# file = Path(afn())

Image = go.Image|go.Heatmap

def imshow(image:np.ndarray,**kwargs)->Image:
    im = px.imshow(image,**kwargs).data[0]
    im.xaxis = None
    im.yaxis = None
    return im


rett = tuple[tuple[Image,go.Scatter,go.Scatter]|None,Image,Image,tuple[Image,go.Scatter]|Image|None,Image,Image]|None;
def plot_protrusions(tfile,msequence:MovieSequence|None=None,figure:go.Figure|None=None,do_vis:bool=False):
    tiff = TiffFile(tfile)

    nangles = 80
    angles = np.linspace(-np.pi,np.pi,nangles)
    anghistlist = []
    angdistlist = []
    anghistdiff = []
    angdistdiff = []


    
    def loop(inp)->rett:
        frame:int
        t:TiffPage|TiffFrame
        # vis:bool
        frame,t = inp

        # result:rett

        im = t.asarray()
        if not np.any(im):
            anghistlist.append(np.nan*np.ones(nangles-1))
            angdistlist.append(np.nan*np.ones(nangles-1))
            if do_vis:
                res:list[Any] = [None,]*6
                if msequence:
                    image = msequence[frame+1];
                    res[3] = imshow(image)
                res[1] = imshow(np.array(angdistlist));
                res[2] = imshow(np.array(anghistlist))
                res[4] = imshow(np.array(angdistdiff))
                res[5] = imshow(np.array(anghistdiff))
                return tuple(res)
            else:
                return None

        center = np.array(get_appmedoid(mask=im)) #center [y,x]
        pixels = np.argwhere(im)
        ypixels,xpixels = (pixels.T) #array of coordinates [y,x]
        pangles:np.ndarray = np.arctan2(ypixels-center[0],xpixels-center[1])

        pbinlocs = np.digitize(pangles,angles) #returns index of rightmost edge of bins for each pixel
        radii = np.linalg.norm(pixels-center,axis=1);
        rgroups:list[np.ndarray] = [radii[pbinlocs == l] for l in range(1,nangles)]
        maxradii = np.array([rs.max(initial=0) for rs in rgroups])
        angdistlist.append(maxradii)
        angdistdiff.append(angdistlist[-1] - angdistlist[-2] if len(angdistlist) > 1 else np.zeros(nangles-1))
        
        angle_hist = np.histogram(pangles,bins=angles)
        anghistlist.append(angle_hist[0])
        anghistdiff.append(anghistlist[-1] - anghistlist[-2] if len(anghistlist) > 1 else np.zeros(nangles-1))

        centerangles = np.convolve(angles,[0.5,0.5],mode='valid') #centers of bins
        bpoints = (np.sin(centerangles)*maxradii+center[0],np.cos(centerangles)*maxradii+center[1])

        if not do_vis:
            return None

        imax:tuple[Image,go.Scatter,go.Scatter]
        a1:Image
        a2:Image
        imax2:tuple[Image,go.Scatter]|None = None
        d1:Image
        d2:Image
        

        linedata = go.Scatter(x=bpoints[1][maxradii != 0],y=bpoints[0][maxradii != 0],marker={"line":{"color":"blue"}},mode='lines')
        imax = imshow(im),linedata,go.Scatter(x=[center[1]],y=[center[0]])
        
        if msequence:
            image = msequence[frame+1];
            imax2 = (imshow(image),linedata,)

        # bb = (np.min(ypixels)-5,np.max(ypixels)+5),(np.min(xpixels)-5,np.max(xpixels)+5)
        # imax.set_xlim(bb[1])
        # imax.set_ylim(bb[0])
        # imax2.set_xlim(bb[1])
        # imax2.set_ylim(bb[0])
        

        # a1.set_title("maxradius")
        a1 = imshow(np.array(angdistlist))

        # a2.set_title("numpixels")
        a2 = imshow(np.array(anghistlist))

        dar1 = np.array(angdistdiff,dtype=float)
        dar2 = np.array(anghistdiff,dtype=float)
        rmin,rmax = np.array([-1,1])*max(abs(dar1.min()),abs(dar1.max()),abs(dar2.min()),abs(dar2.max()))
        # d1.set_title("deriv_maxradius")
        d1 = imshow(dar1,color_continuous_scale=px.colors.diverging.RdBu)
        # d2.set_title("deriv_numpixels")
        d2 = imshow(dar2,color_continuous_scale=px.colors.diverging.RdBu)

        return imax,a1,a2,imax2,d1,d2

    nframes = len(tiff.series[0])
    start = int(tiff.shaped_metadata[0]["startframe"])
    # print(start)
    frames = ((k[0]+start,k[1]) for k in enumerate(tqdm(tiff.series[0])))

    shapedata = [loop(frame) for frame in frames]
    framedata:list[go.Frame] = []
    prev:rett = None
    for f in shapedata:
        framedata.append(fr := go.Frame())
        data = []
        if f is None:
            continue
        nprev = list(f)
        if f[0] is None:
            if prev is None or prev[0] is None:
                #huh
                pass
            else:
                data.extend(prev[0])
                nprev[0] = prev[0]
        else:
            data.extend(f[0])
        
        data.append(f[1])
        data.append(f[2])
        
        if f[3] is None:
            if prev is None or prev[3] is None:
                #huh
                pass
            else:
                data.extend(prev[3])
                nprev[3] = prev[3]

        data.append(f[4])
        data.append(f[5])

        prev = tuple(nprev) #type:ignore
        fr.data = data



    if figure is None:
        figure = make_subplots(2,3,subplot_titles=["mask","maxradius","numpixels","image","maxradius deriv","numpixels deriv"])

    figure.add_traces(framedata[0].data,[0,0,0,0,1,1,1],[0,1,2,0,1,2])

    return figure    
    # del anim
        

if __name__ == "__main__":
    tfolder = Path(r"C:\Users\Harrison Truscott\Downloads\53_tracks_masks")
    files = os.listdir(tfolder);

    movie_path = r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\images\2023.4.2 OptoTiam Exp 53"
    
    mreader:Movie[int] = get_movie(movie_path)
    do_vis = True

    def plot_file(file:str):
        print("plotting track:",tfolder/file)
        movie = mreader[int(re.match(r".*movie(\d+)_.*",file).group(1))]
        f = plot_protrusions(tfolder/file,movie,do_vis=do_vis)
        f.show()
    def plot_findex(index:int,event):
        return plot_file(files[index])
    
    IPython.embed()
    


        




