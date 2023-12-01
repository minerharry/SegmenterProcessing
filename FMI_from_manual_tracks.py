from csv import DictReader
import math
from typing import DefaultDict
from utils.filegetter import askopenfilename
from utils.parse_tracks import FijiManualTrack
infile = askopenfilename();

tracks:dict[int,dict[int,tuple[int,int]]] = FijiManualTrack(askopenfilename());



FMI = dict[int,tuple[float,float]]();
disp = dict[int,tuple[float,float]]();
dist = dict[int,float]();

for track,framedict in tracks.items():
    fs = sorted(framedict.keys());
    frames = [framedict[f] for f in fs]
    
    accDist = 0;
    prev = start = frames[0];
    for frame in frames[1:]:
        d = math.sqrt((prev[0]-frame[0])**2+(prev[1]-frame[1])**2);
        accDist += d;
        prev = frame;
    end = frames[-1];
    
    dist[track] = accDist;
    disp[track] = (end[0]-start[0],end[1]-start[1]);
    FMI[track] = (disp[track][0]/accDist,disp[track][1]/accDist);

print(FMI);
print(disp);
print(dist);
    
    
    