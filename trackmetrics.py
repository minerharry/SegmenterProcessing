"""High level metrics of tracks that go beyond FMI, including spatial relationships to other tracks in the movie"""
from math import dist
from typing import cast
from utils.parse_tracks import QCTracks,QCTracksDict, QCTracksFrameDict

def minimum_distance(tracks:QCTracksDict|QCTracks)->dict[int,dict[int,float]]:
    if isinstance(tracks,QCTracks):
        tracks = QCTracksDict(tracks)
    res:dict[int,dict[int,float]] = {}
    for movie in tracks:
        distances:dict[int,float] 
        res[movie] = distances = {}
        for tid,track in tracks[movie].items():
            mindist = float('inf')
            for frame in track:
                # mindist = float('inf')
                for otid,otrack in tracks[movie].items():
                    if tid == otid:
                        continue
                    if frame in otrack:
                        # print(track[frame],otrack[frame])
                        mindist = min(mindist,dist(track[frame],otrack[frame]))
            distances[tid] = mindist
    return res

def average_mindistance(tracks:QCTracksDict|QCTracks)->dict[int,dict[int,float]]:
    if isinstance(tracks,QCTracks):
        tracks = QCTracksDict(tracks)
    res:dict[int,dict[int,float]] = {}
    for movie in tracks:
        distances:dict[int,float] 
        res[movie] = distances = {}
        for tid,track in tracks[movie].items():
            ds = []
            for frame in track:
                mindist = float('inf')
                for otid,otrack in tracks[movie].items():
                    if tid == otid:
                        continue
                    if frame in otrack:
                        # print(track[frame],otrack[frame])
                        mindist = min(mindist,dist(track[frame],otrack[frame]))
                if mindist != float('inf'):
                    ds.append(mindist)
            distances[tid] = sum(ds)/len(ds)
    return res

def average_area(tracks:QCTracksFrameDict|QCTracks)->dict[int,dict[int,float]]:
    if isinstance(tracks,QCTracks):
        tracks = QCTracksFrameDict(tracks)    
    res:dict[int,dict[int,float]] = {}
    for movie in tracks:
        distances:dict[int,float] 
        res[movie] = distances = {}
        for tid,track in tracks[movie].items():
            ds = []
            for frame in track.values():
                ds.append(frame['area'])
            distances[tid] = sum(ds)/len(ds)
    return res