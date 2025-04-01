from itertools import chain
import itertools
from typing import Any, Callable, Iterable, Sequence, TypeVar
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np

from utils.parse_tracks import FijiManualTrack, QCTracksArray

from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path

# def plot_track_arrays(arrays:QCTracksArray,)
T = TypeVar("T")
def z(o:Iterable[T]|T)->Iterable[T]:
    if not isinstance(o,Iterable) or isinstance(o,str):
        return itertools.cycle([o])
    return o


#for gradient plotting
def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def plot_tracks(tracks:Iterable[Sequence[tuple[int,int]]|np.ndarray],ax:plt.Axes|None=None,
                start_ball=True,
                title:str|None=None,
                labels:Iterable[str]|None=None,colors:str|Iterable[Any]|None=None,
                gradient:bool=False, #IF TRUE: interprets "color" as a colormap instead
                unit:str|None="microns",scale:ScaleBar|None|Callable[[],ScaleBar]=lambda: ScaleBar(1,units="um", location="upper right", label="Cell diameter", fixed_value=30, frameon=False),
                legend:bool|str="lower left",
                markersize=4,
                ):
    if not ax:
        ax = plt.subplot()
    assert isinstance(ax,plt.Axes)
    if colors is None:
        colors = (f"C{i}" for i in itertools.count()) #infinite generator for the color cycle, make all components of the track have the same color

    for m,label,color in zip(tracks,z(labels),z(colors)):
        print(label)
        m = np.array(m)
        assert m.shape[1] == 2
        if not gradient:
            ax.plot(m[:,0],m[:,1],color=color,label=label,marker='.',markersize=markersize)
            ax.plot(*m[0],color=color,marker='o')
            if start_ball: ax.arrow(*m[-2],*((m[-1]-m[-2])),color=color,shape='full', lw=0, length_includes_head=True, head_starts_at_zero=True, head_width=0.9)
        else:
            #color values
            Z = np.linspace(0.0, 1.0, m.shape[0])
            segments = make_segments(m[:,0],m[:,1])
            
            lc = LineCollection(segments,array=Z,cmap=color,label=label)
            ax.scatter(m[:,0],m[:,1],cmap=color,c=Z,marker='.',s=markersize)
            ax.add_collection(lc)

            cmap = mpl.colormaps[color] if not isinstance(color,Colormap) else color
            startc,endc = cmap([0,1])
            ax.plot(*m[0],c=startc,marker='o')
            if start_ball: ax.arrow(*m[-2],*((m[-1]-m[-2])),color=endc,shape='full', lw=0, length_includes_head=True, head_starts_at_zero=True, head_width=0.9)

    ax.set_xlabel("X position" + (f" ({unit})" if unit else ""),fontsize=34)
    ax.set_ylabel("Y position" + (f" ({unit})" if unit else ""),fontsize=34)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.invert_yaxis()
    ax.set_aspect(1.0)

    if scale:
        if callable(scale):
            scale = scale()
        ax.add_artist(scale)

    if legend:
        if isinstance(legend,str):
            ax.legend(loc=legend)
        else:
            ax.legend()

    if title:
        ax.set_title(title,fontsize=50)

    return ax


if __name__ == "__main__":
    # seg_analysis_folder = Path.home()/r"Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis"
    seg_analysis_folder = Path.home()/r"OneDrive - University of North Carolina at Chapel Hill\Bear Lab\optotaxis calibration\data\Segmentation Analysis"

    manual_raw = seg_analysis_folder/"2023.4.2 OptoTiam Exp 53 $manual/scaled_qc_tracks_raw.pkl"
    manual_smoothed = seg_analysis_folder/"2023.4.2 OptoTiam Exp 53 $manual/scaled_qc_tracks_smoothed.pkl"
    # manual_source = r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.4.2 OptoTiam Exp 53 $manual\manual tracks\down3 in pixels per frame.csv"
    auto_raw = seg_analysis_folder/"2023.4.2 OptoTiam Exp 53/scaled_qc_tracks_raw.pkl"
    auto_smoothed = seg_analysis_folder/"2023.4.2 OptoTiam Exp 53/scaled_qc_tracks_smoothed.pkl"

    # microns_per_pixel = 1.625


    movie = 7
    man_track = 5
    auto_track = 7

    # movie = 8
    # man_track = 3
    # auto_track = 14


    for i in ["Raw","Smoothed","Gaussed"]:

        fig = plt.figure();


        if i == "Raw":
            # print(QCTracksArray(manual_raw)[movie].keys())
            # print(QCTracksArray(manual_smoothed)[movie].keys())
            # print(QCTracksArray(auto_raw)[movie].keys())
            # print(QCTracksArray(auto_smoothed)[movie].keys())
            m = np.array(QCTracksArray(manual_raw)[movie][man_track])
            a = np.array(QCTracksArray(auto_raw)[movie][auto_track])
        elif i == "Smoothed":
            m = np.array(QCTracksArray(manual_smoothed)[movie][man_track])
            a = np.array(QCTracksArray(auto_smoothed)[movie][auto_track])
        else:
            from scipy.ndimage import gaussian_filter1d
            def do_gauss(pos:np.ndarray,width:float):
                pos = pos.copy()
                for i in [0,1]:
                    pos[:,i] = gaussian_filter1d(pos[:,i],width)
                return pos
            m = do_gauss(np.array(QCTracksArray(manual_raw)[movie][man_track]),1.4)
            a = do_gauss(np.array(QCTracksArray(auto_raw)[movie][auto_track]),1.4)
            

        ax = plt.subplot();

        plot_tracks([m,a],ax=ax,labels=["manual","automatic"],colors=["black","red"],title=i + " Tracks: Manual vs Automatic",markersize=0.5)


    plt.show()