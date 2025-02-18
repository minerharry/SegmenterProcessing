from typing import DefaultDict, Dict, List, Tuple
import numpy as np
import matplotlib
import sys
import copy

from pandas import DataFrame

def apply_qc(sample_in:Dict[int,Dict[int,DataFrame]],
        minTrackLength,
        keep:Dict[int,List[int]],
        trim:Dict[Tuple[int,int],Tuple[int,int]],
        removemov:List[int],
        exclude:List[Tuple[int,int]]
        )->Tuple[Dict[int,Dict[int,int]],Dict[int,Dict[int,DataFrame]]]:
    
    sample=copy.deepcopy(sample_in)
    #a sample containes movies, each movie contains tracks
    
    #initialize the status of all tracks in the sample as 1 (active)
    sampTrStatus:Dict[int,Dict[int,int]]={}
    for imov in sample:
        sampTrStatus[imov] = {}
        for itr in sample[imov]:
            sampTrStatus[imov][itr] = 1
    
    
    #CONDITIONS ON TRACKS
    for imov in sample:
        for itr in sample[imov]:        
            #if track length less than min length deactivate track
            if len(sample[imov][itr]) < minTrackLength:
                sampTrStatus[imov][itr]=0
    

    #exclude tracks after visual inspection
    #exclude=[[2,3],[2,4]]
    #exclude=[[1,7]]
                
    for mov,track in exclude:
        sampTrStatus[mov][track]=0
    

    #only keep certain tracks
    #input: dict of elements of the form {movie:[track1,track2,...]}

    for mov,tracks in keep.items():
        #for all the tracks in movie i[0]-1 turn off all the tracks
        for itracks in sampTrStatus[mov]:
            sampTrStatus[mov][itracks]=0
        #turn on the desired tracks
        for itracks in tracks:
            sampTrStatus[mov][itracks]=1
            

    #remove movies
    for mov in removemov:
        for itr in sampTrStatus[mov]: 
            sampTrStatus[mov][itr]=0;        
            
            
    #trim tracks
    #input: dict (trim) of elements with the form {(movie, track):(begginning frame, end frame)]
    #trim={(7,1):(1,53)}
    trims:Dict[int,Dict[int,Tuple[int,int]]] = {};
    for (mov,track),(start,end) in trim.items():    
        if track == -1:
            trims[mov] = DefaultDict(lambda: (start,end));
        else:
            trims[mov][track] = (start,end);

    for mov,t in trims.items():
        for (track,(start,end)) in t.items():
            #print(i)
            #get 'frame' column
            framec=sample[mov][track]['frame']
            #print (framec)
            #get index of desired first frame
            firstframe = framec.iloc[0]
            lastframe = framec.iloc[-1]
            if start < firstframe or start > lastframe:
                raise Exception(f'in movie {mov} track {track} beggining of trimming {start} is out of range {(firstframe,lastframe)}');
            if end < firstframe or end > lastframe:
                raise Exception(f'in movie {mov} track {end} end of trimming {end} is out of range {(firstframe,lastframe)}');

            if start >= end :
                raise Exception(f'in movie {mov} track {track} end of trimming {end} is smaller or equal than beggining of trimming {start}')
                
            ifirstframe = framec[framec==start].index[0]
            #get index of desired last frame
            ilastframe = framec[framec==end].index[0]

            

            #trim track
            sample[mov][track]=sample[mov][track].loc[ifirstframe:ilastframe+1]    
    
    return sampTrStatus, sample
