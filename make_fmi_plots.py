from collections import UserDict
from contextlib import nullcontext
from csv import DictReader, DictWriter
import csv
import itertools
from logging import raiseExceptions
from operator import itemgetter
import os
from pathlib import Path
import re
from typing import Any, DefaultDict, Iterable, Literal, Sequence, TextIO, TypeVar

from matplotlib import pyplot as plt
from ordered_set import OrderedSet
from utils.filegetter import afns,afn
from utils.parse_tracks import MergedTrackAnalysis, TrackAnalysis
from libraries.parsend import StageDict, group_stage_basenames
from utils.statplots import plot_CI
from utils.zipdict import zip_dict

suffix_regex = ' \\${0}'
def strsuffix(exp:str,suffixes=None):
    if isinstance(suffixes,str):
        suffixes = [suffixes]
    elif not isinstance(suffixes,Iterable):
        suffixes = ["\\S*"]
    for suffix in suffixes:
        r = suffix_regex.format(suffix)
        exp = re.sub(r,'',exp)
    return exp

K = TypeVar("K")
class IdentityDefault(dict[K,K]):
    def __missing__(self, key):
        return key

def StringableTrackAnalysis(file):
    tracks:dict[str,dict[str,dict[str,Any]]] = DefaultDict(dict)
    with (open(file,'r') if not isinstance(file,TextIO) else nullcontext(file)) as f:
        reader = DictReader(f);
        for row in reader:
            if row["trackid"] != "average":
                tracks[row["movie"]][row["trackid"]] = row;
    return dict(tracks)

groupspec = None|dict[str,Iterable[tuple[str,int|tuple[int,int]]]]
def make_fmi_plots(filenames:Sequence[str|tuple[str,...]|TrackAnalysis],axes:list[Literal["x","y"]]="y",selections:list[list[tuple[int|tuple[int,int],int]]|None]|list[tuple[int|tuple[int,int],int]]|None=None,auto_groups:bool|Iterable[bool]=True,names:None|Iterable[str|None]=None,grouplist:None|groupspec|Iterable[groupspec]=None):
    """Create FMI plots from tracking data analysis. Creates one figure per filename. Can accept multiple filenames, which will produce a MergedTrackAnalysis. Also accepts an existing TrackAnalysis object; make sure to specify names if so.
    
    filenames:Sequence[str|tuple[str]|TrackAnalysis], sequence of analysis inputs. Each entry must be an analysis object, a path to a TrackAnalysis .csv file, or
    a tuple of TrackAnalysis .csv files.

    axes:list[Literal["x","y"]], which FMI axis to use the analysis of (default y for vertical images); can be "x", "y", or both. Creates separate figures for each filename for each axis

    grouplist:groupspec|Iterable[groupspec] how to combine individual stages from each movie into larger groups of shared analysis. each groupspec object is a dict of {groupname:Iterable[stage]}, where each stage is specified
    as a tuple of [stagename,stagekey]. stage key is either an integer stage number or a valid key for the TrackAnalysis object; if multiple filenames are passed, the MergedTrackAnalysis object accepts
    tuples of (file#,stage#). grouplist can either be passed as a single dict to apply the same grouping to each track analysis, or as an iterable of dicts to have different groups per analysis

    selections: which tracks to include from each stage. For each track analysis, the selection spec is a list of (stage,track#) tuples. If the analysis is a MergedTrackAnalysis, stage can either be an integer or a tuple (file#,stage#). 
    Selection spec can also be passed as a single list to apply the same selection to each analysis, or as a list of lists to apply different selections to each one.

    auto_groups:bool whether to use contextual .nd files/folder structure to automatically assign groups based on metamorph stage names."""


    if len(filenames) == 0:
        return
    if isinstance(auto_groups,bool):
        auto_groups = itertools.cycle([auto_groups])
    if selections is None:
        selections = itertools.cycle([None])
    else:
        assert len(selections) > 0 
        if isinstance(selections[0],tuple):
            selections = itertools.cycle([selections])
    if names is None:
        names = itertools.cycle([None])
    if grouplist is None:
        grouplist = itertools.cycle([None])
    elif isinstance(grouplist,dict):
        grouplist = itertools.cycle([grouplist])
    groups:None|dict[str,Iterable[tuple[str,int|tuple[int,int]]]]
    for t in axes:
        for n,selection,auto_group,ni,groups in zip(filenames,selections,auto_groups,names,grouplist):

            name = ni or ""
            if isinstance(n,str):
                loc = Path(n)
                name = loc.name 
                if "$manual" in str(loc):
                    name = "manual " + name
                else:
                    name = "automatic " + name
                if selection:
                    name = "selected " + name
            elif isinstance(n,tuple):
                loc = tuple((Path(p) for p in n))
                assert ni is not None
            else:
                loc = None
                assert ni is not None

            out = f"output/analysis/figures/{name}"
            plt.figure(name,figsize=(5.5,4.8))
            plt.title(name + " " + t)

            if loc is None:
                pass
            elif isinstance(loc,tuple):
                data = MergedTrackAnalysis(loc)
            else:
                data = TrackAnalysis(loc);
            # print("data:",data)

            
            
            exclude:list[int] = [];
            excludeName:list[str] = [];
            flipGroups:list[str] = ["up"]


            if auto_group and groups is None:
                assert loc is not None
                print("auto_grouping")
                if not isinstance(loc,tuple):
                    loc = (loc,)
                multi = len(loc) > 1
                groups = {}
                for l in loc:
                    exp = l.parent.name
                    exp = strsuffix(exp)
                    images = l.parent.parent.parent/"images"/exp #to gcp_transfer, then images twice
                    nds = [x.path for x in os.scandir(images) if x.name.endswith(".nd")]
                    print("found nd file:",nds[0]);
                    maps = StageDict(nds[0]);
                    grps = group_stage_basenames(maps)
                    if multi:
                        grps = {f"{exp}: nam":val for nam,val in grps.items()}
                    groups.update(grps)
            else:
                auto_group = False
                    

            movies = list(data.keys());
            if groups is None:
                groups = {f"Movie{i}":[(f"Movie{i}",i)] for i in movies};


            orientation = "horizontal"

            poss = range(1,len(groups)+1)


            order = OrderedSet(["down","downshallow","control","upshallow","up"] if auto_group else groups.keys())
            print(order)
            order.intersection_update(groups.keys())
            print(order)

            fmidict = {}

            pos_selection_exists:dict[int,bool] = {}
            # print(groups)
            for (groupName,stages),pos in zip([(k,groups[k]) for k in order],poss):
                # print(groupName,pos)
                
                fmidict[groupName] = fmis = []
                fullfmis = []
                color = []

                # print(stages)
                for name,num in stages:
                    if name in excludeName or num in exclude:
                        print("continuing")
                        continue;
                    factor = -1 if groupName in flipGroups else 1
                    if num not in data:
                        continue
                    for tid,dat in data[num].items():
                        color.append("red" if selection and (num,tid) not in selection else "black")
                        fullfmis.append(float(dat[f"FMI.{t}"])*factor)
                        if selection and (num,tid) in selection:
                            fmis.append(float(dat[f"FMI.{t}"])*factor)
                # print(groupName)
                if len(fullfmis) == 0:
                    print(f"No fmi data for group {groupName}, skipping...")
                    continue
                # print(fullfmis)

                # print(stages)

                print(f"plotting group {groupName} with {len(fmis)} samples")
                plot_CI(pos,fullfmis,orientation="horizontal",value_marker='.',values_color=color,plot_values=True,mean_color="black",interval_color="red",plot_significance=True)
                if len(fmis) != 0:
                    plot_CI(pos+0.5,fmis,orientation="horizontal",value_marker='.',plot_values=True,plot_mean=True,mean_color="purple",interval_color="black",plot_significance=True)
                pos_selection_exists[pos] = (len(fmis) != 0)


            displayNames = IdentityDefault(); #IdentityDefault({"control":"No Light","down":"Steep","up":"Shallow"})

            dnames = [displayNames[n] for n in order]
            print(poss,dnames)
            dnames = [n for p,n in zip(poss,dnames) if p in pos_selection_exists]
            poss = [p for p in poss if p in pos_selection_exists] 
            print(poss,dnames)
            dnames += ["selected " + d for p,d in zip(poss,dnames) if pos_selection_exists[p]]
            poss += [p + 0.5 for p in poss if pos_selection_exists[p]]
            print(poss,dnames)
            if orientation == "vertical":
                plt.ylabel("FMI")
                plt.ylim(-0.5,0.5)
                plt.xlabel("Gradient Position")
                plt.xticks(poss,dnames)
                plt.plot([0,poss[-1]+1],[0,0],linestyle='--',color="black")
            else:
                plt.xlabel("FMI")
                plt.xlim(-0.5,0.5)
                plt.ylabel("Gradient Position")
                plt.yticks(poss,dnames)
                plt.plot([0,0],[0,poss[-1]+1],linestyle='--',color="black")
            # dox = "y" in input("has dox?\n")
            # manual = "Manual" if "$manual" in str(anal_location) else "Automatic"
            # dox = "1 ug per mL Dox" if "53" in str(anal_location) else "No Dox";

            # # plt.title(f"OptoTiEXITam1 {'1 ug/mL' if dox else 'No'} Dox {'Manual' if manual else 'Automatic'} Tracking");
            # smoothing = "Raw" if "raw" in str(anal_location) else "Smoothed"
            # name = f"{dox} {smoothing} {manual} Tracks"

            plt.savefig(f"{out}.png")

    # plt.show()

if __name__ == "__main__":
    #automatic selections:
    # auto = [(1, 6), (1, 22), (1, 2), (1, 21), (1, 0), (2, 2), (2, 6), (2, 49), (2, 61), (3, 5), (3, 2), (3, 32), (3, 4), (3, 3), (3, 18), (3, 43), (3, 7), (4, 1), (4, 2), (4, 10), (4, 5), (4, 17), (4, 24), (4, 60), (4, 4), (4, 8), (4, 6), (4, 7), (5, 163), (5, 1), (5, 31), (5, 15), (5, 69), (5, 57), (5, 257), (6, 5), (6, 2), (6, 23), (6, 3), (6, 1), (6, 11), (6, 8), (6, 80), (6, 20), (6, 52), (7, 8), (7, 2), (7, 7), (7, 22), (7, 4), (7, 1), (7, 13), (7, 102), (7, 71), (8, 14), (9, 11), (9, 6), (9, 28), (9, 5), (9, 25), (9, 7), (10, 7), (10, 3), (10, 25), (10, 33), (10, 30)]

    # #manual selections:
    # man = [(1, 2), (1, 5), (1, 1), (1, 11), (1, 9), (2, 4), (2, 1), (2, 2), (2, 7), (3, 12), (3, 8), (3, 11), (3, 2), (3, 1), (3, 15), (3, 4), (3, 15), (4, 1), (4, 13), (4, 2), (4, 9), (4, 4), (4, 14), (4, 15), (4, 12), (4, 7), (4, 11), (4, 8), (5, 12), (5, 6), (5, 5), (5, 2), (5, 9), (5, 3), (5, 3), (6, 2), (6, 9), (6, 11), (6, 8), (6, 1), (6, 14), (6, 15), (6, 6), (6, 6), (6, 16), (7, 6), (7, 10), (7, 5), (7, 3), (7, 1), (7, 2), (7, 9), (7, 1), (7, 4), (8, 3), (9, 12), (9, 10), (9, 4), (9, 9), (9, 3), (9, 7), (10, 4), (10, 10), (10, 3), (10, 8), (10, 1)]

    names = [afn(title="Select Track Analysis (csv) files",filetypes=[("Track Analysis CSV Files","*.csv")])]
    
    # selection:list[list[tuple[int,int]]|None] = [man if "$manual" in str(n) else auto for n in names]

    # groups:groupspec = {"JimUp (same as HUp)":[("up1 (replicate)",4)],"MitchUp":[("up3",3)],"HUp":[("up1",1)],"MarkUp":[("up2",2)],"MitchDown":[("control3",11)],"HDown":[("control1",9)],"MarkDown":[("control2",10)]}
    make_fmi_plots(names,selections=None,grouplist=None)

    plt.show()
    
    



