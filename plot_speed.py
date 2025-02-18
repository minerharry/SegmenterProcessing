import itertools
import numpy as np
from utils.acf import autocorrelate
from utils.filegetter import afn,afns
from utils.parse_tracks import QCTracks, QCTracksArray, QCTracksDict
import matplotlib.pyplot as plt
from pathlib import Path

paths = [
    ("auto 50 smoothed",r"c:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.3.24 OptoTiam Exp 50\qc_tracks_smoothed.pkl"),
    ("manual 50 raw",r"c:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.3.24 OptoTiam Exp 50 $manual\qc_tracks_raw.pkl"),
    ("auto 53 smoothed",r"c:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.4.2 OptoTiam Exp 53\qc_tracks_smoothed.pkl"),
    ("manual 53 raw",r"c:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.4.2 OptoTiam Exp 53 $manual\qc_tracks_raw.pkl")
         ]
trackss = [(n,QCTracksArray(t)) for n,t in paths]

speeds = {}
for name,tracks in trackss:
    plt.figure(name)
    plt.title(f"{name} speeds")
    for movie in tracks:
        for tid,track in tracks[movie].items():
            t = (movie,tid)
            # print(t)
            speeds[t] = np.linalg.norm(np.diff(track),axis=1)
            plt.plot(speeds[t])

    plt.savefig(f"out/speeds/{name}.png")

plt.show()