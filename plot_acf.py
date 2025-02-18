import itertools
import numpy as np
from utils.acf import autocorrelate
from utils.filegetter import afn
from utils.parse_tracks import QCTracks, QCTracksArray, QCTracksDict
import matplotlib.pyplot as plt

paths = [
    ("auto 50 smoothed",r"c:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.3.24 OptoTiam Exp 50\qc_tracks_smoothed.pkl"),
    ("manual 50 raw",r"c:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.3.24 OptoTiam Exp 50 $manual\qc_tracks_raw.pkl"),
    ("auto 53 smoothed",r"c:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.4.2 OptoTiam Exp 53\qc_tracks_smoothed.pkl"),
    ("manual 53 raw",r"c:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.4.2 OptoTiam Exp 53 $manual\qc_tracks_raw.pkl")
         ]
for n,p in paths:
    tracks = QCTracksArray(p)
    # array = QCTracksArray()


    pos_acfs = {}
    vel_acfs = {}
    for movie in tracks:
        for tid,track in tracks[movie].items():
            t = (movie,tid)
            pos_acfs[t] = (autocorrelate(track))
            vel = np.diff(track)
            vel_acfs[t] = (autocorrelate(vel))

    def avgNestedLists(nested_vals):
        """
        Averages a 2-D array and returns a 1-D array of all of the columns
        averaged together, regardless of their dimensions.
        """
        output = []
        maximum = 0
        for lst in nested_vals:
            if len(lst) > maximum:
                maximum = len(lst)
        for index in range(maximum): # Go through each index of longest list
            temp = []
            for lst in nested_vals: # Go through each list
                if index < len(lst): # If not an index error
                    temp.append(lst[index])
            output.append(np.nanmean(temp))
        return output

    ## ragged list mean; see https://stackoverflow.com/a/38619350/13682828
    avg_pos_acf = avgNestedLists(pos_acfs.values())
    avg_vel_acf = avgNestedLists(vel_acfs.values())

    # plt.figure("Position ACFs (all)")
    # for t,acf in pos_acfs.items():
    #     plt.plot(acf,label=t)

    # plt.figure("Average Position ACF")
    # plt.plot(avg_pos_acf)

    f = plt.figure()
    plt.title(f"{n} Velocity ACFs (all)")
    for t,acf in vel_acfs.items():
        plt.plot(acf,label=t)
    f.savefig(f"out/acfs/{n}_all.png")
    

    f = plt.figure()
    plt.title(f"{n} Average Velocity ACF")
    plt.plot(avg_vel_acf)
    f.savefig(f"out/acfs/{n}_avg.png")

    