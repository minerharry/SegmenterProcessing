import matplotlib.pyplot as plt
import numpy as np

from utils.parse_tracks import FijiManualTrack, QCTracksArray

from matplotlib_scalebar.scalebar import ScaleBar


manual_raw = r"c:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.4.2 OptoTiam Exp 53 $manual\scaled_qc_tracks_raw.pkl"
manual_smoothed = r"c:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.4.2 OptoTiam Exp 53 $manual\scaled_qc_tracks_smoothed.pkl"
# manual_source = r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.4.2 OptoTiam Exp 53 $manual\manual tracks\down3 in pixels per frame.csv"
auto_raw = r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.4.2 OptoTiam Exp 53\scaled_qc_tracks_raw.pkl"
auto_smoothed = r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.4.2 OptoTiam Exp 53\scaled_qc_tracks_smoothed.pkl"

# microns_per_pixel = 1.625

#TODO: improve labelling, add micron scale bar and cell diameter marker

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
        


    plt.plot(m[:,0],m[:,1],color="black",label="manual",marker='.')
    plt.plot(*m[0],color="black",marker='o')
    plt.arrow(*m[-2],*((m[-1]-m[-2])),color="black",shape='full', lw=0, length_includes_head=True, head_starts_at_zero=True, head_width=0.9)

    plt.plot(a[:,0],a[:,1],color="red",label="auto",marker='.')
    plt.plot(*a[0],color="red",marker='o')
    plt.arrow(*a[-13],*((a[-1]-a[-13])),color="red",shape='full', lw=0, length_includes_head=True, head_starts_at_zero=True, head_width=0.9)


    plt.xlabel("X position (microns)")
    plt.ylabel("Y position (microns)")


    # scale = ScaleBar(1,units = "um", location  = "center right", rotation="vertical");
    # plt.gca().add_artist(scale)
    cellSize = ScaleBar(1,units="um", location="upper right", label="Cell diameter", fixed_value=30, frameon=False)
    plt.gca().add_artist(cellSize)

    plt.title(i + " Tracks: Manual vs Automatic")
    plt.legend(loc="lower left")


plt.show()