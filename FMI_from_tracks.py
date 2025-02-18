from csv import DictReader
import math
from typing import DefaultDict
infile = "C:\\Users\\Harrison Truscott\\Downloads\\mov4_manual\\manualtrack_results.csv"

tracks = DefaultDict[int,list[tuple[int,int]]](lambda:[]);

with open(infile,'r',encoding='utf-8-sig') as f:
    reader = DictReader(f);
    for row in reader:
        tracks[int(row['track#'])].append((int(row['x']),int(row['y'])));

FMI = dict[int,tuple[float,float]]();
disp = dict[int,tuple[float,float]]();
dist = dict[int,float]();

for track,frames in tracks.items():
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
    
    
    