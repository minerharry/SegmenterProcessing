"""Utils to parse manual tracking files"""
from __future__ import annotations
import builtins

from contextlib import nullcontext
from csv import DictReader
import math
from os import PathLike
import pickle
from typing import DefaultDict, Hashable, Iterable, Sequence, TextIO, Any
import numpy as np
import copy
from tqdm import tqdm
import pandas as pd

from libraries.centroidtracker import CentroidTracker
from utils import bidict

class FijiManualTrack(dict[int,dict[int,tuple[int,int]]]):
    def __new__(cls: type[FijiManualTrack],file:TextIO|PathLike|str) -> FijiManualTrack:
        tracks:dict[int,dict[int,tuple[int,int]]] = DefaultDict(dict)
        with (open(file,'r') if not isinstance(file,TextIO) else nullcontext(file)) as f:
            reader = DictReader(f);
            for row in reader:
                tracks[int(row['Track n°'])][int(row['Slice n°'])] = ((int(row['X']),int(row['Y'])));
        obj = super().__new__(cls)
        obj.update(tracks)
        return obj
    
    def __init__(self,*args,**kwargs):
        """Read tracks from fiji manual track *.csv files; returns a trackid indexed dictionary of frame:(x,y)"""
        pass

class TrackAnalysis(dict[int, dict[int, dict[str, Any]]]):
    """dict[int, dict[int, dict[str, Any]]]
    a movie and trackid indexed dictionary of analysis parameters"""
    def __new__(cls,file:TextIO|PathLike|str|dict[int, dict[int, dict[str, Any]]]):
        tracks:dict[int,dict[int,dict[str,Any]]] = DefaultDict(dict)
        if isinstance(file,dict):
            tracks.update(file)
        else:
            with (open(file,'r') if not isinstance(file,TextIO) else nullcontext(file)) as f:
                reader = DictReader(f);
                for row in reader:
                    if row["trackid"] != "average":
                        tracks[int(row["movie"])][int(row["trackid"])] = row;
        obj = super().__new__(cls)
        obj.update(tracks)
        return obj
    
    def __init__(self,*args,**kwargs):
        """Read tracks from track analysis *.csv files; returns a movie and trackid indexed dictionary of analysis parameters"""
        pass

class MergedTrackAnalysis(TrackAnalysis):
    """dict[int, dict[int, dict[str, Any]]]
    a movie and trackid indexed dictionary of analysis parameters"""
    def __new__(cls,files:Iterable[TextIO|PathLike|str]):
        sub_analysis = [TrackAnalysis(f) for f in files]
        merged:dict[int,dict[int,dict[str,Any]]] = {}
        total = 0
        for an in sub_analysis:
            shifted = {k+total:v for k,v in an.items()}
            merged.update(shifted)
            total += len(an)
        obj = super().__new__(cls,merged)
        setattr(obj,"sub_analyses", sub_analysis)
        return obj
    
    def __init__(self,*args,**kwargs):
        """Read tracks from track analysis *.csv files; returns a movie and trackid indexed dictionary of analysis parameters"""
        self.sub_analyses:list[TrackAnalysis];
        pass

    def __getitem__(self, key: int|tuple[int,int]) -> dict[int, dict[str, Any]]:
        if isinstance(key,tuple):
            return self.sub_analyses[key[0]][key[1]]
        else:
            return self[key]
        
    def __contains__(self, key: int|tuple[int,int]) -> builtins.bool:
        if super().__contains__(key):
            return True
        elif isinstance(key,tuple):
            if key[1] in self.sub_analyses[key[0]]:
                return True
        return False


class QCTracks(dict[int,dict[int,pd.DataFrame]]):
    """dict[int,dict[int,pd.DataFrame]]
    Movie- and Trackid- indexed nested dict of dataframes"""
    def __new__(cls: type[QCTracks], file:TextIO|PathLike|str) -> QCTracks:
        with (open(file,'rb') if not isinstance(file,TextIO) else nullcontext(file)) as f:
            try:
                trackDict = pickle.load(f);
            except:
                trackDict = pd.read_pickle(f);
        out = super().__new__(cls)
        out.update(trackDict)
        return out
    
    def __init__(self,*args,**kwargs):
        """Read tracks from *_qc_tracks.pkl; returns a movie and trackid indexed nested dictionary of dataframes"""
        pass


class QCTracksArray(dict[int, dict[int, list[tuple[int, int]]]]):
    """dict[int, dict[int, list[tuple[int, int]]]]
    Movie- and Trackid- indexed nested dict of lists of (x,y) points"""
    def __new__(cls: type[QCTracksArray], file:TextIO|PathLike|str|QCTracks,centertype:str="approximate-medoid") -> QCTracksArray:
        tracks:dict[int,dict[int,list[tuple[int,int]]]] = DefaultDict(dict)
        cx, cy = centertype + 'x',centertype + 'y'
        if isinstance(file,dict):
            assert all(all([isinstance(f,dict)] + [isinstance(fd,pd.DataFrame) for fd in f.values()]) for f in file.values())
            qctracks = file
        else:
            qctracks = QCTracks(file);
        for movie in qctracks:
            for tid,track in qctracks[movie].items():
                x,y = (track[cx]),track[cy]
                tracks[movie][tid] = list(zip(x,y))
        obj = super().__new__(cls)
        obj.update(tracks)
        return obj
    
    def __init__(self,*args,**kwargs):
        """Read tracks from *_qc_tracks.pkl; returns a movie and trackid indexed nested dictionary of arrays of points"""
        pass

class QCTracksDict(dict[int, dict[int, dict[int, tuple[int, int]]]]):
    """dict[int, dict[int, dict[int, tuple[int, int]]]]
    Movie- and Trackid- indexed nested dict of track dictionaries frame:(x,y)"""
    def __new__(cls: type[QCTracksDict], file:TextIO|PathLike|str|QCTracks,centertype:str="approximate-medoid") -> QCTracksDict:
        tracks:dict[int,dict[int,dict[int,tuple[int,int]]]] = DefaultDict(lambda: DefaultDict(dict))
        cx, cy = centertype + 'x',centertype + 'y'
        if isinstance(file,dict):
            assert all(all([isinstance(f,dict)] + [isinstance(fd,pd.DataFrame) for fd in f.values()]) for f in file.values())
            qctracks = file
        else:
            qctracks = QCTracks(file);
        for movie in qctracks:
            for tid,track in qctracks[movie].items():
                for frame,x,y in zip(track['frame'],track[cx],track[cy]):
                    tracks[movie][tid][int(frame)] = (int(x),int(y))
        obj = super().__new__(cls)
        obj.update({k:dict(v) for k,v in tracks.items()})
        return obj
    
    def __init__(self,*args,**kwargs):
        """Read tracks from *_qc_tracks.pkl; returns a movie and trackid indexed nested dictionary of frame-indexed (x,y) points"""
        pass

class QCTracksFrameDict(dict[int, dict[int, dict[int, pd.Series]]]):
    """dict[int, dict[int, dict[int, pd.Series]]]
    Movie- and Trackid- indexed nested dict of track dictionaries frame:Series"""
    def __new__(cls: type[QCTracksFrameDict], file:TextIO|PathLike|str|QCTracks,centertype:str="approximate-medoid") -> QCTracksFrameDict:
        tracks:dict[int,dict[int,dict[int,pd.Series]]] = DefaultDict(lambda: DefaultDict(dict))
        cx, cy = centertype + 'x',centertype + 'y'
        if isinstance(file,dict):
            assert all(all([isinstance(f,dict)] + [isinstance(fd,pd.DataFrame) for fd in f.values()]) for f in file.values())
            qctracks = file
        else:
            qctracks = QCTracks(file);
        for movie in qctracks:
            for tid,track in qctracks[movie].items():
                for frame,(l,t) in zip(track['frame'],track.iterrows()):
                    tracks[movie][tid][int(frame)] = t
        obj = super().__new__(cls)
        obj.update({k:dict(v) for k,v in tracks.items()})
        return obj
    
    def __init__(self,*args,**kwargs):
        """Read tracks from *_qc_tracks.pkl; returns a movie and trackid indexed nested dictionary of frame-indexed (x,y) points"""
        pass