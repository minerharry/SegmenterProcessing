


import csv
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np


def graph_angles_hist(angles:list[float],ax:Axes,n_bins=30):
    print(angles)
    angles = np.deg2rad(angles);
    angles %= np.pi*2

    angle_bins = np.linspace(0,np.pi,n_bins+1,endpoint=True);

    hist,bins = np.histogram(angles,angle_bins)
    print(hist)

    return ax.bar(bins[:-1],hist,align='edge',width=np.diff(bins)[0]);


if __name__ == "__main__":
    # angles_folder = r"C:\Users\Harrison Truscott\Downloads\vasp_red_s1\FAAS_afXZaP\adhesion_props\FAAI\per_image_angles"
    # idx = 16

    # file = Path(angles_folder)/f"{idx}.csv"
    #"FAAS_hOOUXP"

    file = r"C:\Users\Harrison Truscott\Downloads\vasp_red_s1\FAAS_afXZaP\adhesion_props\FAAI\per_image_FAAI.csv"

    with open(file,'r') as f:
        reader = csv.reader(f)
        # angles = [float(l[0]) for l in reader]
        faai = [float(l) for l in next(reader)]

    # fig,ax = plt.subplots(1,1,subplot_kw={'projection':'polar'})
    # ax.set_title(f"S1 Frame {idx} FA Angles")

    fig,ax = plt.subplots(1,1)
    ax.set_title(f"S1 FAAI")
    ax.set_ylim(0,90)
    ax.set_ylabel("FAAI (degrees)")
    ax.set_xlabel("frame")
    ax.plot(np.arange(len(faai))+1,faai)

    # graph_angles_hist(angles,ax);

    from IPython import embed; embed()

