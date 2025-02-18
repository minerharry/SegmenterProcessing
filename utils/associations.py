from __future__ import annotations
from ast import literal_eval
import math
import os
from pathlib import Path
import re
from typing import Literal, overload
import numpy as np
import pandas as pd
from tqdm import tqdm
from libraries.centroidtracker import CentroidTracker

from utils import bidict
from utils.parse_tracks import QCTracks
import copy
from joblib import Memory

associations_cachedir = "associations/cache"
associations_cache = Memory(associations_cachedir, verbose=0)


#TODO: BROKEN! Associating cells that are way too far apart. This is either or both of the following reasons:
#1: constraints are simply too wide and need to be tightened
#2: apparent closeness at one point in the track allows the whole track to be associated. There needs to be significantly more filtering for the final step
#   -- this is more complicated than it first seems. It might be good to require that every timepoint in the track be allowed by the distance metric
#           That can be more easily achieved by storing every cell that matches the filtering in order of closeness in addition to the best fit
#       Alternatively, talk to max about a more mathematical version of truly 3d association, instead of just this 2d
@associations_cache.cache
def AssociateTracks(track1:dict[int,pd.DataFrame]|dict[int,dict[int,tuple[int,int]]]|dict[int,list[tuple[int,int]]],track2:dict[int,pd.DataFrame]|dict[int,dict[int,tuple[int,int]]],
        centertype:str="approximate-medoid",default_radius:float=30,
        bar:bool|dict=False):

    if len(track1) == 0 or len(track2) == 0:
        return []

    class Cell:
        def __init__(self,pos:tuple[float,float],area:float|None,tid:int):
            self.pos = pos;
            self.area = area
            self.tid = tid
            # self.name = name or str(id(self))

        @property
        def radius(self):
            return math.sqrt(self.area) if self.area else default_radius

        @staticmethod
        def cell_distance(t0:Cell,t1:Cell)->float:
            return math.dist(t0.pos,t1.pos)
        
        @staticmethod
        def cells_filter(t0:Cell,t1:Cell,disappeared_time:int,dist:float)->bool:
            radius = max(t0.radius,t1.radius)
            # if t0.tid == 40 and t1.tid == 4:
            #     print(t0,t1,radius,dist)
            return dist < 2*radius
        
        def __str__(self) -> str:
            return f"Cell with tid {self.tid}, area {self.area}, radius {self.radius}, and position {self.pos}"
    
    def from_frame(frame:pd.Series,tid:int): 
        assert isinstance(frame,pd.Series)
        return Cell((frame[centertype+'x'],frame[centertype+'y']),frame['area'],tid)

    frames = set()
    for ts in [track1,track2]:
        for f in ts.values():
            frames.update(f['frame'] if isinstance(f,pd.DataFrame) else f.keys())

    o1 = set()
    o2 = set()
    links = []
    if bar:
        kw = {"desc":"Associating Tracks"};
        try:
            kw.update(bar)
        except:
            pass
        frames = tqdm(frames,**kw)
    for f in frames:
        tracker = CentroidTracker[Cell](Cell.cell_distance,Cell.cells_filter,0,0)
        
        f1 = tracker.update([from_frame(track[track['frame']==f].iloc[0],tid) if isinstance(track,pd.DataFrame) else Cell(track[f],None,tid) for tid,track in track1.items() if f in (track['frame'].values if isinstance(track,pd.DataFrame) else track)])

        #tracker updates the dict directly, so make sure it's a saved version
        f1 = copy.deepcopy(f1)

        f2 = tracker.update([from_frame(track[track['frame']==f].iloc[0],tid) if isinstance(track,pd.DataFrame) else Cell(track[f],None,tid) for tid,track in track2.items() if f in (track['frame'].values if isinstance(track,pd.DataFrame) else track)])
        
        #all objects that are in both mean cells that are in both snapshots
        ls = ([(f1[i].tid,f2[i].tid) for i in set(f1.keys()).intersection(f2.keys())])

        o1.update(g[0] for g in ls)
        o2.update(g[1] for g in ls)
        links.extend(ls)

    if len(o1) == 0 or len(o2) == 0:
        ##no objects successfully linked
        return []

    o1 = list(o1)
    o2 = list(o2)


    arr = np.zeros([len(o1),len(o2)])
    idx_map:list[bidict[int,int]] = [bidict((o,i) for i,o in enumerate(oi)) for oi in [o1,o2]] #bidirectional index map, by default object->index
    for link in links:
        index = tuple([m[obj] for obj,m in zip(link,idx_map)])
        arr[index] += 1
    
    a1:np.ndarray = arr.max(axis=1)
    a1 = a1.argsort()[::-1] #we want biggest first
    a2:np.ndarray = arr.argmax(axis=1)[a1]

    associations = [(idx_map[0].inverse[e1][0],idx_map[1].inverse[e2][0]) for e1,e2 in zip(a1,a2)]
    return associations


@associations_cache.cache(ignore=["savepath"])
def get_full_associations(
            d1:QCTracks,
            d2:QCTracks,
            savepath:str|None=None,
            names:tuple[str,str]=("Track1","Track2"),
            originpaths:None|tuple[str,str]=None,
            )\
                ->tuple[
                    dict[int,list[tuple[int,int]]],
                    tuple[list[tuple[int,int]],list[tuple[int,int]]],
                    tuple[list[tuple[int,int]],list[tuple[int,int]]]]:
    """associates movies and gets some more in depth statistics, optionally saving to a human-readable format
    Returns: tuple of:
     Associations: dict[int,list[tuple[int,int]]] - movie-indexed association output
     Inclusions: tuple[list[tuple[int,int]],list[tuple[int,int]]] - lists of (movie,trackid) for tracks1 and tracks2 respectively
      Lists the tracks that are associated with tracks from the other movie. that is, all elements in these lists were successfully associated
     Remainders: tuple[list[tuple[int,int]],list[tuple[int,int]]] - lists of (movie,trackid) for tracks1 and tracks2 respectively
      Lists the tracks that ARE NOT associated with tracks from the other movie. Only tracks that are in one movie but not the other.
     """

    s1 = [] #(movie,track) paired from group 1
    s2 = [] #(movie,track) paired from group 2

    r1 = [] #(movie,track) remainder (unpaired) from group 1
    r2 = [] #(movie,track) remainder (unpaired) from group 2

    assocs:dict[int,list[tuple[int,int]]] = {}

    for movie in set(d1.keys()).intersection(d2.keys()):
        t1 = d1[movie]
        t2 = d2[movie]
        # print(t1.keys())
        # print(t2.keys())
        associations = AssociateTracks(t1,t2,bar={"leave":False});
        assocs[movie] = associations
        print(movie,associations)
        s1.extend([(movie,a[0]) for a in associations])
        s2.extend([(movie,a[1]) for a in associations])

        r1.extend([(movie,m) for m in set(t1.keys()).difference([a[0] for a in associations])])
        r2.extend([(movie,m) for m in set(t2.keys()).difference([a[1] for a in associations])])

    if savepath:
        save_association(savepath,assocs,(s1,s2),(r2,r2),names=names,read_paths=originpaths)
    
    return assocs,(s1,s2),(r1,r2)

def save_association(path:str|Path,
                    associations:dict[int,list[tuple[int,int]]],
                    inclusions:tuple[list[tuple[int,int]],list[tuple[int,int]]],
                    remainders:tuple[list[tuple[int,int]],list[tuple[int,int]]],
                    names:tuple[str,str]=("Track1","Track2"),
                    read_paths:None|tuple[str,str]=None):
    print("saving association...")
    s1,s2 = inclusions
    r1,r2 = remainders
    with open(path,'w') as f:
        indent = "  "
        if read_paths:
            f.write(f"Original Paths:\n")
            f.write(f"{indent}{names[0]}: {read_paths[0]}\n")
            f.write(f"{indent}{names[1]}: {read_paths[1]}\n")
        f.write(f"Associations: (Movie: [{names},])\n")
        f.write("{\n")
        for k,assoc in associations.items():
            f.write(f"{indent}{k}: {assoc},\n")
        f.write("}\n\n")

        f.write("Inclusions: [(Movie,Trackn),]\n")
        f.write(f" {names[0]}:\n")
        f.write(f"  {s1}\n")
        f.write(f" {names[1]}:\n")
        f.write(f"  {s2}\n\n")

        f.write("Remainders: [(Movie,Trackn),]\n")
        f.write(f" {names[0]}:\n")
        f.write(f"  {r1}\n")
        f.write(f" {names[1]}:\n")
        f.write(f"  {r2}")    


# @overload
# def try_read_association(path:str,recalculate:Literal[False]=False)\
#     ->None|tuple[
#         dict[int,list[tuple[int,int]]],
#         tuple[list[tuple[int,int]],list[tuple[int,int]]],
#         tuple[list[tuple[int,int]],list[tuple[int,int]]]]: ...

# @overload
# def try_read_association(path:str,recalculate:Literal[True]=True)\
#     ->tuple[
#         dict[int,list[tuple[int,int]]],
#         tuple[list[tuple[int,int]],list[tuple[int,int]]],
#         tuple[list[tuple[int,int]],list[tuple[int,int]]]]: ...


# def try_read_association(path:str)->tuple[
#         dict[int,list[tuple[int,int]]],
#         tuple[list[tuple[int,int]],list[tuple[int,int]]],
#         tuple[list[tuple[int,int]],list[tuple[int,int]]]]:
#     if not os.path.exists(path) or not path.endswith(".txt"):
#         raise FileNotFoundError()
#     with open(path,"r") as f:
#         lines = f.readlines()
#     total = "".join(lines)
    
#     assoc_regex = re.compile(r"Associations: \(.*\)*(\{.*\})",flags=re.S)
#     assocs:dict[int,list[tuple[int,int]]] = literal_eval(re.findall(assoc_regex,total)[0])

#     included_regex = re.compile(r"Inclusions:\n.*:\n\s*(\[[\(\)\d, ]*\])\n.*:\n\s*(\[[\(\)\d, ]*\])")
#     finds = re.findall(included_regex,total)[0]
#     inclusions:tuple[list[tuple[int,int]],list[tuple[int,int]]] = (literal_eval(finds[0]),literal_eval(finds[1]))

#     remainder_regex = re.compile(r"Remainders:\n.*:\n\s*(\[[\(\)\d, ]*\])\n.*:\n\s*(\[[\(\)\d, ]*\])")
#     finds = re.findall(remainder_regex,total)[0]
#     remainders:tuple[list[tuple[int,int]],list[tuple[int,int]]] = (literal_eval(finds[0]),literal_eval(finds[1]))

#     return assocs,inclusions,remainders


        
