import os
from pathlib import Path
import re
# from telnetlib import IP
# from typing import Any, Callable, Sequence
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure, SubFigure
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from tqdm import tqdm
from libraries.centers import get_appmedoid
from libraries.movie_reading import ImageMapMovie, Movie, MovieSequence
from utils.paging import paging_bar,LegacyIndex
from utils.filegetter import afn
from tifffile import TiffFile,TiffPage,TiffFrame
import IPython

from libraries.parse_moviefolder import get_movie



# file = Path(afn())




def plot_protrusions(tfile,msequence:MovieSequence|None=None,figure:Figure|None=None,do_vis:bool=False):
    tiff = TiffFile(tfile)

    nangles = 80
    angles = np.linspace(-np.pi,np.pi,nangles)
    anghistlist = []
    angdistlist = []
    anghistdiff = []
    angdistdiff = []
    f = figure or plt.figure()
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=f)

    imax = f.add_subplot(spec[0, 0])
    a1 = f.add_subplot(spec[0,1])
    a2 = f.add_subplot(spec[0,2])
    imax2 = f.add_subplot(spec[1, 0])
    d1 = f.add_subplot(spec[1,1])
    d2 = f.add_subplot(spec[1,2])

    def loop(inp):
        frame:int
        t:TiffPage|TiffFrame
        vis:bool
        frame,t,vis = inp

        im = t.asarray()
        if not np.any(im):
            anghistlist.append(np.nan*np.ones(nangles-1))
            angdistlist.append(np.nan*np.ones(nangles-1))
            if do_vis:
                imax.imshow(im)
                if msequence:
                    image = msequence[frame+1];
                    imax2.imshow(image)

                a1.set_title("maxradius")
                a1.imshow(np.array(angdistlist))
                a2.set_title("numpixels")
                a2.imshow(np.array(anghistlist))
            return

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

        def vis2():
            # print("visualizing")
            imax.cla()
            a1.cla()
            a2.cla()
            imax2.cla()
            d1.cla()
            d2.cla()
            imax.plot(bpoints[1][maxradii != 0],bpoints[0][maxradii != 0],color='blue')
            imax.imshow(im)
            imax.scatter(center[1],center[0])
            
            if msequence:
                image = msequence[frame+1];
                imax2.imshow(image)
                imax2.plot(bpoints[1][maxradii != 0],bpoints[0][maxradii != 0],color='blue')
                imax.scatter(center[1],center[0])

            bb = (np.min(ypixels)-5,np.max(ypixels)+5),(np.min(xpixels)-5,np.max(xpixels)+5)
            imax.set_xlim(bb[1])
            imax.set_ylim(bb[0])
            imax2.set_xlim(bb[1])
            imax2.set_ylim(bb[0])
            

            a1.set_title("maxradius")
            a1.imshow(np.array(angdistlist))
            a2.set_title("numpixels")
            a2.imshow(np.array(anghistlist))

            dar1 = np.array(angdistdiff,dtype=float)
            dar2 = np.array(anghistdiff,dtype=float)
            rmin,rmax = np.array([-1,1])*max(abs(dar1.min()),abs(dar1.max()),abs(dar2.min()),abs(dar2.max()))
            d1.set_title("deriv_maxradius")
            d1.imshow(dar1,cmap="bwr",vmin=rmin,vmax=rmax)
            d2.set_title("deriv_numpixels")
            d2.imshow(dar2,cmap="bwr",vmin=rmin,vmax=rmax)


            plt.draw()
            # plt.pause(0.1)
        if do_vis or vis:
            vis2()
    nframes = len(tiff.series[0])
    start = int(tiff.shaped_metadata[0]["startframe"])
    print(start)
    frames = ((k[0]+start,k[1],k[0] >= nframes-1) for k in tqdm(enumerate(tiff.series[0])))
    print(type(frames))
    print(hasattr(frames,"__len__"))
    # print(frames)
    # print(len(frames))
    # frames = [(k[0]+start,k[1],False) for k in frames]
    # frames[-1] = (frames[-1][0],frames[-1][1],True)
    anim = FuncAnimation(figure,loop,frames,repeat=False,save_count=nframes)
    plt.show()
    # del anim
        

if __name__ == "__main__":
    tfolder = Path(r"C:\Users\Harrison Truscott\Downloads\53_tracks_masks")
    files = os.listdir(tfolder);

    movie_path = r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\images\2023.4.2 OptoTiam Exp 53"
    
    mreader:Movie[int] = get_movie(movie_path)
    do_vis = True

    figure:SubFigure
    def plot_file(file:str):
        for ax in figure.axes:
            ax.remove()
        print("plotting track:",tfolder/file)
        movie = mreader[int(re.match(r".*movie(\d+)_.*",file).group(1))]
        plot_protrusions(tfolder/file,movie,figure,do_vis=do_vis)
    def plot_findex(index:int,event):
        return plot_file(files[index])
    
    figure = paging_bar(*LegacyIndex(plot_findex,loop=True,max=len(files)-1).callbacks,init_axes=False)
    plot_findex(0,None)
    # plt.ion()
    IPython.embed()


        




