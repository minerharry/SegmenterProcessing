from collections import namedtuple
from pathlib import Path
import pickle
from typing import Any, Callable, DefaultDict, Protocol
from matplotlib.axes import Axes
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import numpy as np
from tqdm import tqdm
from libraries.analysis import analyze_experiment_tracks, scale_tracks
from libraries.parsend import StageDict, group_stage_basenames
from libraries.smoothing import mavg
from utils.parse_tracks import QCTracks, QCTracksArray
from utils.associations import AssociateTracks
import matplotlib.pyplot as plt
import IPython

centertype = "approximate-medoid"
centerx,centery = centertype + 'x',centertype + 'y'

def do_mavg(pos:pd.DataFrame,width:int,power:int):
    pos = pos.copy()
    for i in [centerx,centery]:
        pos[i] = mavg(pos[i],width,power=power)
    return pos

def do_gauss(pos:pd.DataFrame,width:int):
    pos = pos.copy()
    for i in [centerx,centery]:
        pos[i] = gaussian_filter1d(pos[i],width)
    return pos

#dum but lazy
class Smoother(Protocol):
    def __call__(self, pos:pd.DataFrame,*args: Any) -> pd.DataFrame: ...

funcs:dict[str,Smoother] = {'mavg':do_mavg,'gauss':do_gauss}

if __name__ == "__main__":
    do_state = False
    if not do_state:
        pre_keys = list(globals().keys()).copy()
        exp = "2023.4.2 OptoTiam Exp 53"
        expm = exp + " $manual"

        seg_folder = Path(r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis")
        sources = [QCTracks(seg_folder/e/"scaled_qc_tracks_raw.pkl") for e in [exp,expm]]
        names = ["automatic","manual"]

        nd_loc = Path(r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\images")/exp/"p.nd"
        stagedict = StageDict(nd_loc)
        groups = group_stage_basenames(stagedict)

        smoothings:list[None|tuple[str,tuple]] = [None]
        for width in range(2,4):
            for power in range(1,5):
                smoothings.append(("mavg",(width,power)))
        for width in np.linspace(1,2,10)[1:]:
            smoothings.append(("gauss",(width,)))

        criteria = ("FMI",0),("FMI",1),"Persistence",("trackVelocity",0),("trackVelocity",1),("trackVelocity",2),"trackLength"
        #x fmi, y fmi, persistence, x velocity, y velocity, speed

        do_paired = True

        if do_paired:
            associations = {m:AssociateTracks(sources[0][m],sources[1][m]) for m in tqdm(set(sources[0].keys()).union(sources[1].keys()),leave=False,desc="associating") if m in sources[0] and m in sources[1]}
        else:
            associations = None

        comparisons:dict[str|tuple[str,int],dict[None|tuple[str,tuple],dict[str,tuple[float,float]]]] = DefaultDict(lambda: DefaultDict(dict)) #indexed by comparison criterium, smoothing type, group

        for smoothing in tqdm(smoothings,desc="smoothing"):
            tracks:list[dict[int,dict[int,pd.DataFrame]]] = []
            for source in sources:
                if smoothing is None:
                    tracks.append(source)
                else:
                    tracks.append(stracks := {})
                    for movie,mtracks in source.items():
                        stracks[movie] = strack = {}
                        for tid,frame in mtracks.items():
                            strack[tid] = funcs[smoothing[0]](frame,*smoothing[1]);
            for group,members in tqdm(groups.items(),leave=False,desc="group"):
                movies = [m[1] for m in members if all(m[1] in t for t in tracks)]
                analyses = [analyze_experiment_tracks({m:track[m] for m in movies if m in track},'approximate-medoid',do_progressbar=False) for track in tracks]
                for c in tqdm(criteria,leave=False,desc="criteria"):
                    name,index = c if isinstance(c,tuple) else (c,None)
                    # print({k:v for k,v in analyses[0].items() if k not in ["scaled_tracks"]})
                    data:list[dict[int,dict[int,float|tuple[float]]]] = [analy[name] for analy in analyses]
                    flat_data:list[list[float]]
                    if not do_paired:
                        flat_data= [
                            sum([
                                [k[index] if isinstance(k,tuple) else k for k in datdict[m].values()]
                                for m in datdict
                            ],[])
                            for datdict in data
                        ]
                    else:
                        assert associations is not None
                        pair_data:list[tuple[float,float]] = []
                        for m in movies:
                            assocs = associations[m]
                            pair_data += [tuple([k[index] if isinstance(k,tuple) else k for k in (data[0][m][a],data[1][m][b])]) for a,b in assocs]
                        flat_data = [[p[0] for p in pair_data],[p[1] for p in pair_data]]

                    ##now, finally, we can compare
                    if not do_paired:
                        res = (np.subtract(np.mean(flat_data[1]),np.mean(flat_data[0])),
                            stats.ttest_ind(a=flat_data[0],b=flat_data[1],equal_var=True)[1])
                    else:
                        res = (np.mean(np.subtract(flat_data[1],flat_data[0])),
                            stats.ttest_rel(a=flat_data[0],b=flat_data[1])[1])
                        
                    comparisons[c][smoothing][group] = res

        criteria_names = {k:str(k) for k in criteria}
        criteria_names.update({("FMI",0):"fmi.x",("FMI",1):"fmi.y","Persistence":"D/T",("trackVelocity",0):"velocity.x",("trackVelocity",1):"velocity.y",("trackVelocity",2):"speed"})
        

        ##get nice viewing of smoothing results
        quick_smooth:dict[str,dict[Any,dict[str,float]]] = {} 
        for c in criteria:
            name = criteria_names[c]
            # smooths = comparisons[c];
            smooths = {"1-1" if k is None else f"{k[1][0]}-{k[1][1]}":{j:round(l[1],8) for j,l in v.items()} for k,v in comparisons[c].items() if k is None or k[0] == "mavg"}
            quick_smooth[name] = smooths

        ##get nice viewing of gaussian results
        quick_gauss:dict[str,dict[Any,dict[str,float]]] = {} 
        for c in criteria:
            name = criteria_names[c]
            # smooths = comparisons[c];
            gausss = {1.0 if k is None else round(k[1][0],5):{j:round(l[1],8) for j,l in v.items()} for k,v in comparisons[c].items() if k is None or k[0] == "gauss"}
            quick_gauss[name] = gausss

        stat_smooth:dict[str,dict[Any,dict[str,float]]] = {} 
        for c in criteria:
            name = criteria_names[c]
            # smooths = comparisons[c];
            smooths = {"1-1" if k is None else f"{k[1][0]}-{k[1][1]}":{j:round(l[0],8) for j,l in v.items()} for k,v in comparisons[c].items() if k is None or k[0] == "mavg"}
            stat_smooth[name] = smooths

        stat_gauss:dict[str,dict[Any,dict[str,float]]] = {} 
        for c in criteria:
            name = criteria_names[c]
            # smooths = comparisons[c];
            gausss = {1.0 if k is None else k[1][0]:{j:round(l[0],8) for j,l in v.items()} for k,v in comparisons[c].items() if k is None or k[0] == "gauss"}
            stat_gauss[name] = gausss

        # vars = ["quick_smooth","quick_gauss","groups","comparisons"]
        print(pre_keys)
        print(globals()["quick_smooth"])
        d = {k:v for k,v in globals().items() if k not in pre_keys}
        print(locals()["quick_smooth"])
        d.update({k:v for k,v in locals().items() if k not in pre_keys})
        try:
            print(d["quick_smooth"])
            pd.to_pickle(d,"state.pkl")
        except Exception as e:
            print(e)
    else:
        state = pd.read_pickle("state.pkl")
        print(state.keys())
        print(state["quick_smooth"])
        globals().update(state);

    # plt.ion()
    def plot_quicks():
        for name,quick in [("average",quick_smooth),("gauss",quick_gauss),("average statistic",stat_smooth),("gauss statistic",stat_gauss)]:
            quick:dict[str,dict[Any,dict[str,float]]]
            f = plt.figure(name + (" paired" if do_paired else " unpaired"))
            # f.title(name)
            plots:list[Axes] = np.concatenate(f.subplots(2,4))
            for ax,cname in zip(plots,quick.keys()):
                ax.set_title(cname)
                if isinstance(next(iter(quick[cname].keys())),(float,int)): #numeric, can be used as x axis
                    # print("numeric key",quick[cname].keys())
                    nums = quick[cname].keys()
                else: #nonnumeric, use as labels
                    # print("nonnumeric key",quick[cname].keys())
                    nums = range(len(quick[cname]))
                    ax.set_xticks(nums)
                    ax.set_xticklabels(quick[cname].keys())
                for group in groups.keys():
                    ax.plot(nums,[n[group] for n in quick[cname].values()],label=group)
                if "statistic" in name:
                    ax.plot(nums,[0]*len(nums),linestyle='--',color='grey',label='0')
                    ax.autoscale()
                else:
                    ax.plot(nums,[0.05]*len(nums),linestyle='--',color='grey',label="p=0.05")
                    ax.set_ylim(0,1)
                    ax.set_ylabel("p-value")
                # ax.set_xlabel("smoothing criteria")
            ax.legend()
            f.canvas.draw_idle()
            # f.draw_idle()
    # plt.show()

    plot_quicks()

    # print(comparisons)
    IPython.embed()