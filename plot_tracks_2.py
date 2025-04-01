from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from plot_tracks import plot_tracks
from utils.parse_tracks import QCTracksArray
import matplotlib as mpl


if __name__ == "__main__":
    seg_analysis_folder = Path.home()/r"OneDrive - University of North Carolina at Chapel Hill\Bear Lab\optotaxis calibration\data\Segmentation Analysis"
    auto_raw = seg_analysis_folder/"2023.4.2 OptoTiam Exp 53/scaled_qc_tracks_raw.pkl"

    movie = 6

    tracks = QCTracksArray(auto_raw)[movie]

    labels = []
    trackarrays = []
    for trackid,tr in tracks.items():
        labels.append(f"Track {trackid}")
        trackarrays.append(tr)

    print(len(trackarrays)) 
    print(len(labels))

    ax = plot_tracks(trackarrays,labels=None,title="Movie 6 Tracks",colors=mpl.colormaps["RdBu"],gradient=True,legend=False);
    
    from imageio.v3 import imread
    im = imread(r"C:\Users\miner\OneDrive - University of North Carolina at Chapel Hill\Bear Lab\Honors Thesis\Figures\Figure 2 Assets\trackmasks_combined.TIF")
    # ax.invert_yaxis()
    ax.imshow(im,extent=(0,im.shape[1]*1.625,im.shape[0]*1.625,0),cmap="Greys_r")
    

    plt.show()