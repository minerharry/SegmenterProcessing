from collections import UserDict
from contextlib import nullcontext
from csv import DictReader, DictWriter
import csv
from dataclasses import InitVar, asdict, dataclass, field
import itertools
from logging import raiseExceptions
import math
from operator import itemgetter
import os
from pathlib import Path
import re
from typing import Any, DefaultDict, Iterable, Literal, NamedTuple, Sequence, Sized, TextIO, TypeVar

from matplotlib import pyplot as plt
from utils.filegetter import afns,afn
from utils.identity_default import IdentityDefault
from utils.parse_tracks import MergedTrackAnalysis, TrackAnalysis
from libraries.parsend import StageDict, group_stage_basenames, try_fetch_nd
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

def StringableTrackAnalysis(file):
    tracks:dict[str,dict[str,dict[str,Any]]] = DefaultDict(dict)
    with (open(file,'r') if not isinstance(file,TextIO) else nullcontext(file)) as f:
        reader = DictReader(f);
        for row in reader:
            if row["trackid"] != "average":
                tracks[row["movie"]][row["trackid"]] = row;
    return dict(tracks)


def sizematch(i1:Iterable,i2:Sized):
    return not isinstance(i1,Sized) or len(i1) == len(i2)
groupspec = None|dict[str,Iterable[tuple[str,int|tuple[int,int]]]]
def make_fmi_plots(
        filenames:Sequence[str|tuple[str,...]|TrackAnalysis]=None,
        axes:list[Literal["x","y"]]|Literal["x","y"]="y",
        selections:list[list[tuple[int|tuple[int,int],int]]|None]|list[tuple[int|tuple[int,int],int]]|None=None,
        auto_groups:bool|Iterable[bool]=True,
        names:None|Iterable[str|None]=None,
        grouplist:None|groupspec|Iterable[groupspec]=None,
        exclude_stages:list[int|tuple[int,int]|str]|list[list[int|tuple[int,int]|str]]|None=None,
        figfolder:str|Path = Path(f"output/analysis/figures/"),
        records:bool=True,
        rose:bool=True,):
    """Create FMI plots from tracking data analysis. Creates one figure per filename. Can accept multiple filenames, which will produce a MergedTrackAnalysis. Also accepts an existing TrackAnalysis object; make sure to specify names if so.
    
    filenames:Sequence[str|tuple[str]|TrackAnalysis], sequence of analysis inputs. Each entry must be an analysis object, a path to a TrackAnalysis .csv file, or
    a tuple of TrackAnalysis .csv files.

    axes:list[Literal["x","y"]], which FMI axis to use the analysis of (default y for vertical images); can be "x", "y", or both. Creates separate figures for each filename for each axis

    grouplist:groupspec|Iterable[groupspec] how to combine individual stages from each movie into larger groups of shared analysis. each groupspec object is a dict of {groupname:Iterable[stage]}, where each stage is specified
    as a tuple of [stagename,stagekey]. stage key is either an integer stage number or a valid key for the TrackAnalysis object; if multiple filenames are passed, the MergedTrackAnalysis object accepts
    tuples of (file#,stage#). grouplist can either be passed as a single dict to apply the same grouping to each track analysis, or as an iterable of dicts to have different groups per analysis

    selections: which tracks to include from each stage. For each track analysis, the selection spec is a list of (stage,track#) tuples. If the analysis is a MergedTrackAnalysis, stage can either be an integer or a tuple (file#,stage#). 
    Selection spec can also be passed as a single list to apply the same selection to each analysis, or as a list of lists to apply different selections to each one.

    auto_groups:bool whether to use contextual .nd files/folder structure to automatically assign groups based on metamorph stage names.
    
    rose:bool whether to also make rose plots. Will make a folder of individual rose plots per figure of fmis, with labeled sub-images
    """
    if filenames is None:
        raise ValueError("Must supply track analyses!")

    if len(filenames) == 0:
        return
    if isinstance(auto_groups,bool):
        auto_groups = itertools.cycle([auto_groups])
    selects:Iterable[list[tuple[int|tuple[int,int],int]]|None] = itertools.cycle([[]])
    if selections is None:
        selects = itertools.cycle([None])
    else:
        if isinstance(selections[0],tuple):
            selects = itertools.cycle([selections]) #type:ignore
    assert sizematch(selects,filenames)
    
    exclude:Iterable[list[int|str|tuple[int,int]]] = itertools.cycle([[]]);
    if exclude_stages is not None:
        if not isinstance(exclude_stages[0],list):
            exclude = itertools.cycle(exclude_stages)
    assert sizematch(exclude,filenames)
    if names is None:
        names = itertools.cycle([None])
    assert sizematch(names,filenames)
    if grouplist is None:
        grouplist = itertools.cycle([None])
    elif isinstance(grouplist,dict):
        grouplist:Iterable[groupspec] = itertools.cycle([grouplist])
    assert sizematch(grouplist,filenames)
    groups:None|dict[str,Iterable[tuple[str,int|tuple[int,int]]]]
    excludes:list[int|str|tuple[int,int]]
    for t in axes:
        for n,selection,auto_group,ni,groups,excludes in zip(filenames,selects,auto_groups,names,grouplist,exclude):

            name = ni or ""
            if isinstance(n,str):
                loc = Path(n)
                name = loc.name 
                if "$manual" in str(loc):
                    name = "manual " + name
                else:
                    name = "automatic " + name
            elif isinstance(n,tuple):
                loc = tuple((Path(p) for p in n))
                assert ni is not None
            else:
                loc = None
                assert ni is not None

            assert name != ""
            if selection:
                name = "selected " + name

            figfolder = Path(figfolder)
            fmi_out = figfolder/"fmi"/f"{name}"
            rose_out = figfolder/"rose"/f"{name}"
            fmifig = plt.figure(name,figsize=(5.5,4.8))
            fmiax = fmifig.subplots(1,1)

            if loc is None:
                assert isinstance(n,TrackAnalysis)
                data = n
                pass
            elif isinstance(loc,tuple):
                data = MergedTrackAnalysis(loc)
            else:
                data = TrackAnalysis(loc);
            # print("data:",data)

            
            
            
            flipGroups:list[str] = ["down"] 


            if auto_group and groups is None:
                assert loc is not None
                print("auto_grouping")
                if not isinstance(loc,tuple):
                    loc = (loc,)
                multi = len(loc) > 1
                groups = DefaultDict(list)
                for i,l in enumerate(loc):
                    exp = l.parent.name
                    exp = strsuffix(exp)
                    images = l.parent.parent.parent/"images"/exp #to gcp_transfer, then images twice
                    nds = []
                    if images.exists():
                        nds = [x.path for x in os.scandir(images) if x.name.endswith(".nd")]
                        print("found nd file:",nds[0]);
                    else:
                        try:
                            nd = try_fetch_nd(exp,as_file=True)
                            print("remote .nd found: ",nd)
                            nds = [nd]
                        except FileNotFoundError:
                            auto_group = False
                            print("auto grouping aborted")

                    if nds:
                        maps = StageDict(nds[0]);
                        grps = {k:[(v1,(i,v2)) for (v1,v2) in v] for k,v in group_stage_basenames(maps).items()}
                        print("groupies:",grps)
                        
                        ##FORCE MERGE
                        if False:
                            grp2 = {}
                            if "control" in grps:
                                grp2 = {"control":grps["control"]}
                                del grps["control"]
                            merge = sum(grps.values(),[])
                            grp2["illuminated"] = merge
                            grps = grp2
                        
                        for k,v in grps.items():
                            groups[k] += v
            else:
                auto_group = False
                    
            print(f"groups: {groups}")

            movies = list(data.keys());
            if groups is None:
                groups = {f"Movie{i}":[(f"Movie{i}",i)] for i in movies};


            orientation = "vertical"

            poss = range(1,len(groups)+1)


            order = [o for o in ["downshallow","control","up","down","illuminated"] if o in groups.keys()] if auto_group else groups.keys()
            
            removed = [k for k in groups.keys() if k not in order]
            if len(removed) > 0:
                print("Autogroup Removed the following Stages:",removed)

            fmidict:dict[str,tuple[list[float],list[float]]] = {}
            rosedict = {}

            rosefigs = {}

            displayNames = IdentityDefault({"control":"No Light","down":"Down Steep","up":"Up Shallow","upshallow":"Up Shallow","illuminated":"Light"})# IdentityDefault();
            displayNames = IdentityDefault({"control":"No Light","down":"Steep","up":"Shallow","upshallow":"Up Shallow","illuminated":"Light"})# IdentityDefault();

            pos_selection_exists:dict[int,bool] = {}
            # print(groups)
            for (groupName,stages),pos in zip([(k,groups[k]) for k in order],poss):
                # print(groupName,pos)
                select_fmis = []
                fullfmis = []
                fmidict[groupName] = (fullfmis,select_fmis)
                color = []

                select_roses = []
                full_roses:list[tuple[float,float]] = []
                rosedict[groupName] = (full_roses,select_roses)

                # print(stages)
                for nam,num in stages:
                    if nam in excludes or num in excludes:
                        # print("continuing")
                        continue;
                    factor = 1 if groupName in flipGroups else -1
                    # factor = -1
                    if num not in data:
                        continue
                    for tid,dat in data[num].items():
                        color.append("red" if selection and (num,tid) not in selection else "black")
                           
                        fullfmis.append(float(dat[f"FMI.{t}"])*factor)
                        if selection and (num,tid) in selection:
                            select_fmis.append(float(dat[f"FMI.{t}"])*factor)
                        if rose:
                            # print(dat)
                            # print((float(dat[f"Displacement.x (microns)"]),float(dat[f"Displacement.y (microns)"])))
                            full_roses.append((float(dat[f"Displacement.x (microns)"]),float(dat[f"Displacement.y (microns)"])))
                            if selection and (num,tid) in selection:
                                select_roses.append((float(dat[f"Displacement.x (microns)"]),(float(dat[f"Displacement.y (microns)"]))))
                            
                # print(groupName)
                if len(fullfmis) == 0:
                    print(f"No fmi data for group {groupName}, skipping...")
                    continue
                # print(fullfmis)

                # print(stages)
                print(f"plotting group {groupName} with {len(select_fmis)} samples")
                plot_CI(pos,fullfmis,orientation=orientation,value_marker='.',values_color=color,plot_values=False,mean_color="black",interval_color="black",plot_significance=True,ax=fmiax)
                if len(select_fmis) != 0:
                    plot_CI(pos+0.5,select_fmis,orientation=orientation,value_marker='.',plot_values=False,plot_mean=True,mean_color="purple",interval_color="black",plot_significance=True,ax=fmiax)
                pos_selection_exists[pos] = (len(select_fmis) != 0)

                if groupName == "up4 y":
                    from IPython import embed; embed()
                if rose:
                    rosefigs[groupName] = rosefig = plt.figure()
                    from utils.circular_hist import circular_hist
                    ax = rosefig.subplots(1,1,subplot_kw={"projection":"polar"})
                    angles = list(map(lambda p: math.atan2(-p[1],p[0]),full_roses));
                    # print(angles)
                    circular_hist(ax,angles)
                    dispname = displayNames[groupName]
                    # if groupName in flipGroups:
                    #     dispname += "*"
                    ax.set_title(dispname)
                    if len(select_roses) > 0:
                        rosefigs["selected " + groupName] = rosefig = plt.figure()
                        ax = rosefig.subplots(1,1,subplot_kw={"projection":"polar"})
                        angles = list(map(lambda p: math.atan2(-p[1],p[0]),select_roses));
                        # print(angles)
                        circular_hist(ax,angles)
                        ax.set_title("selected " + dispname)
                    # plt.show()





            dnames = [displayNames[n] + ("*" if n in flipGroups and False else "") for n in order]
            # print(poss,dnames)
            dnames = [n for p,n in zip(poss,dnames) if p in pos_selection_exists]
            poss = [p for p in poss if p in pos_selection_exists] 
            # print(poss,dnames)
            dnames += ["selected " + d for p,d in zip(poss,dnames) if pos_selection_exists[p]]
            poss += [p + 0.5 for p in poss if pos_selection_exists[p]]
            # print(poss,dnames)
            
            plt.title(name + " " + t)
            if orientation == "vertical":
                fmiax.set_ylabel("FMI")
                fmiax.set_ylim(-0.1,0.25)
                # fmiax.set_xlabel("Gradient Position")
                fmiax.set_xticks(poss,labels=dnames)
                fmiax.plot([0.5,poss[-1]+0.5],[0,0],linestyle='--',color="black")
            else:
                fmiax.set_xlabel("FMI")
                # fmiax.set_xlim(-0.5,0.5)
                # fmiax.set_ylabel("Gradient Position")
                fmiax.set_yticks(poss,labels=dnames)
                fmiax.plot([0,0],[0.5,poss[-1]+0.5],linestyle='--',color="black")
            # dox = "y" in input("has dox?\n")
            # manual = "Manual" if "$manual" in str(anal_location) else "Automatic"
            # dox = "1 ug per mL Dox" if "53" in str(anal_location) else "No Dox";

            # # plt.title(f"OptoTiEXITam1 {'1 ug/mL' if dox else 'No'} Dox {'Manual' if manual else 'Automatic'} Tracking");
            # smoothing = "Raw" if "raw" in str(anal_location) else "Smoothed"
            # name = f"{dox} {smoothing} {manual} Tracks"
            # plt.show()

            fmi_out.parent.mkdir(parents=True,exist_ok=True)
            # fmifig.tight_layout(pad=1)
            fmifig.savefig(f"{fmi_out}.png",dpi=fmifig.dpi)
            
            if records:
                if selection:
                    raise NotImplemented() #simply doesn't do it
                datapath = Path(fmi_out)/"data.csv"
                datapath.parent.mkdir(parents=True,exist_ok=True)
                import csv
                with open(datapath,"w") as f:
                    writer = csv.writer(f)
                    groupnames = fmidict.keys()
                    writer.writerow(groupnames)
                    for fmis in itertools.zip_longest(*[fmidict[g][0] for g in groupnames]):
                        writer.writerow(fmis)
                               
                
                recordpath = Path(fmi_out)/"record.json"
                @dataclass
                class StageInfo:
                    filename:str
                    stagenum:int
                    stagename:str
                    cellcount:int
                group_sources = {}

                for name,stages in groups.items():
                    group_sources[name] = infos = []
                    for stagename,key in stages:
                        if key in excludes or stagename in excludes:
                            continue
                        if key not in data:
                            continue
                        num = len(data[key])
                        file:Path
                        stagenum:int
                        if isinstance(key,tuple):
                            assert isinstance(loc,tuple)
                            file = loc[key[0]]
                            stagenum = key[1]
                        else:
                            assert isinstance(loc,Path),(key)
                            file = loc
                            stagenum = key
                        info = StageInfo(str(file),stagenum,stagename,num);
                        infos.append(info)
                
                
                @dataclass
                class FMIRecord: #params, sources, N
                    # groups:InitVar[dict[str, Iterable[tuple[str, int | tuple[int, int]]]]]
                    group_sources:dict[str,list[StageInfo]] #list of (filename,stage#,stagename)

                    total_counts:dict[str,int] = field(init=False)

                    def __post_init__(self):
                        self.total_counts = {group:self.total_count(group) for group in self.group_sources}

                    def total_count(self,group:str):
                        return sum([k.cellcount for k in self.group_sources[group]])


                record = FMIRecord(group_sources)

                import json
                with open(recordpath,"w") as f:
                    json.dump(asdict(record),f,indent=1)


            if rose:
                rose_out.mkdir(parents=True,exist_ok=True)
                for n,r in rosefigs.items():
                    r.savefig(rose_out/f"{n}.png")

            

    # plt.show()

if __name__ == "__main__":
    #automatic selections:
    # auto = [(1, 6), (1, 22), (1, 2), (1, 21), (1, 0), (2, 2), (2, 6), (2, 49), (2, 61), (3, 5), (3, 2), (3, 32), (3, 4), (3, 3), (3, 18), (3, 43), (3, 7), (4, 1), (4, 2), (4, 10), (4, 5), (4, 17), (4, 24), (4, 60), (4, 4), (4, 8), (4, 6), (4, 7), (5, 163), (5, 1), (5, 31), (5, 15), (5, 69), (5, 57), (5, 257), (6, 5), (6, 2), (6, 23), (6, 3), (6, 1), (6, 11), (6, 8), (6, 80), (6, 20), (6, 52), (7, 8), (7, 2), (7, 7), (7, 22), (7, 4), (7, 1), (7, 13), (7, 102), (7, 71), (8, 14), (9, 11), (9, 6), (9, 28), (9, 5), (9, 25), (9, 7), (10, 7), (10, 3), (10, 25), (10, 33), (10, 30)]

    # #manual selections:
    # man = [(1, 2), (1, 5), (1, 1), (1, 11), (1, 9), (2, 4), (2, 1), (2, 2), (2, 7), (3, 12), (3, 8), (3, 11), (3, 2), (3, 1), (3, 15), (3, 4), (3, 15), (4, 1), (4, 13), (4, 2), (4, 9), (4, 4), (4, 14), (4, 15), (4, 12), (4, 7), (4, 11), (4, 8), (5, 12), (5, 6), (5, 5), (5, 2), (5, 9), (5, 3), (5, 3), (6, 2), (6, 9), (6, 11), (6, 8), (6, 1), (6, 14), (6, 15), (6, 6), (6, 6), (6, 16), (7, 6), (7, 10), (7, 5), (7, 3), (7, 1), (7, 2), (7, 9), (7, 1), (7, 4), (8, 3), (9, 12), (9, 10), (9, 4), (9, 9), (9, 3), (9, 7), (10, 4), (10, 10), (10, 3), (10, 8), (10, 1)]

    names = [afn(key="Track Analysis",title="Select Track Analysis (csv) files",filetypes=[("Track Analysis CSV Files","*.csv")])]
    
    # selection:list[list[tuple[int,int]]|None] = [man if "$manual" in str(n) else auto for n in names]

    # groups:groupspec = {"JimUp (same as HUp)":[("up1 (replicate)",4)],"MitchUp":[("up3",3)],"HUp":[("up1",1)],"MarkUp":[("up2",2)],"MitchDown":[("control3",11)],"HDown":[("control1",9)],"MarkDown":[("control2",10)]}
    make_fmi_plots(names,auto_groups=True,grouplist=None)

    plt.show()