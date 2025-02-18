## The goal is to look at good tracks and to check whether they always have mask overlap between them.
## We will start with the automatic tracks and count how many frames the masks don't overlap and mark them. 
## Then, I can go through with view_dots_mpl.py and manually confirm
import os
import pickle
from typing import Any, DefaultDict, NamedTuple
import joblib
import numpy as np
from tqdm import tqdm
from libraries.analysis import analyze_experiment_tracks
from libraries.overlap import get_labeled_overlap
from libraries.movie_reading import ImageMapSequence,Movie
from utils.filegetter import afn,skip_cached_popups
from pathlib import Path
from utils.parse_tracks import QCTracks
from utils.plotly_utils import CallbackIndexer, Subplotter, Toolbar,figureSpec
# from utils.trackmasks import get_trackmasks, read_trackmask
from utils.extractmasks import extract_labeled_cellmasks
from libraries.parse_moviefolder import get_movie
from IPython import embed

# while True:
with skip_cached_popups():
    autracks = QCTracks(autrackloc := Path(afn(title="automatic",key="autotrack")))

experiment = autrackloc.parent.name
lmasks = extract_labeled_cellmasks(experiment=experiment)

labeledmasks = get_movie(lmasks)

autanalysis = analyze_experiment_tracks(autracks,'approximate-medoid',do_progressbar=False)


def nested_getitem(d:dict|Any,t:tuple):
    if len(t) == 0:
        return d
    return nested_getitem(d[t[0]],t[1:])


Datum = NamedTuple("Datum",intersections=np.ndarray,currlabel=int,nextlabel=int)

cache = joblib.Memory("caches/overlaps")

@cache.cache
def get_maps(autracks:QCTracks,labeledmasks:Movie,depth:int=4):
    if os.path.exists("abasdasd.pkl"):
        #for depth=4 only
        with open("abasdasd.pkl","rb") as f:
            return pickle.load(f)
    
        

    missingmaps:list[tuple[int,int,int,int]] = []
    extramaps:list[tuple[int,int,int,int]] = []
    multimaps:list[tuple[int,int,int,int]] = []

    notabledata:dict[tuple[int,int,int,int],Datum] = {} #movie, trackid, frame, offset

    for m in tqdm(autracks): #autracks
        frames = labeledmasks[m]
        for t,track in tqdm(autracks[m].items(),leave=False):
            for f in tqdm(track['frame'],leave=False):
                data = track[track["frame"]==f]
                # print(data["frame"])
                for i in range(1,depth+1): #looking for how far into the future this test will work for
                    try:
                        if f+i not in track["frame"]:
                            #only look at frames and their next frame
                            continue
                        currlabel = data['label'].iloc[0]
                        if currlabel == -1:
                            #don't consider frames without valid mask
                            continue
                        currmask = currlabel == frames[f]
                        
                        try:
                            intersections = get_labeled_overlap(currmask,frames[f+i])
                        except IndexError:
                            ##this shouldn't happen
                            #reached the end of the movie
                            raise Exception("Movie end reached but track continues???")

                        nextlabel = track[track["frame"]==f+i]['label'].iloc[0]

                        multi = len(intersections) > 1
                        missing = nextlabel != -1 and nextlabel not in intersections
                        extra = set(intersections).difference(set([nextlabel]))

                        if multi: multimaps.append((m,t,f,i)); #print("multi!")
                        if missing: missingmaps.append((m,t,f,i)); #print("missing!")
                        if extra: extramaps.append((m,t,f,i)); #print("extra!")

                        notabledata[m,t,f,i] = Datum(intersections=intersections,currlabel=currlabel,nextlabel=nextlabel)
                    except Exception as e:
                        embed()
    # res = {(k1,k2,k3,k4):v for k1,v1 in notabledata.items() for k2,v2 in v1.items() for k3,v3 in v2.items() for k4,v in v3.items()}
    return notabledata,{"missing":missingmaps,"extra":extramaps,"multi":multimaps}

autracks = {1:autracks[1]}
notabledata,namedmaps = get_maps(autracks,labeledmasks,depth=100)

globals().update(**namedmaps)
embed()

            
##Results: Experiment 53 - smallest non-overlap is 24 frames :) definitely 5-10 is not an issue in the slightest