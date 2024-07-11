from csv import DictWriter
from itertools import zip_longest
import os
from pathlib import Path
from typing import Iterable
from libraries.parsend import StageDict, group_stage_basenames
from utils.filegetter import afns
from utils.parse_tracks import TrackAnalysis
from utils.zipdict import zip_dict

# anal_location = Path(afn())
for anal_location in [Path(f) for f in afns()]:
    print("analyzing experiment data:",anal_location)
    data = TrackAnalysis(anal_location);

    groups:None|dict[str,list[tuple[str,int]]] = None
    exclude:list[int] = [];
    excludeName:list[str] = [];

    exp = anal_location.parent.name
    details = anal_location.name.split("track_analysis")[1].removesuffix(".csv")
    # exp = exp.strip(" $manual")
    images = anal_location.parent.parent.parent/"images"/exp.strip(" $manual") #to gcp_transfer, then images twice
    nds = [x.path for x in os.scandir(images) if x.name.endswith(".nd")]
    print("found nd file:",nds[0]);
    maps = StageDict(nds[0]);
    groups = group_stage_basenames(maps)

    invert = ["up"]

    goodnames = {"control":"No Light","down":"Steep","up":"Shallow"}

    out_folder = Path(f"output/analysis/{exp}")
    os.makedirs(out_folder,exist_ok=True)

    vels = {}
    speeds = {}
    fmis = {}
    for gname,gs in groups.items():
        vel = vels[goodnames[gname]] = []
        speed = speeds[goodnames[gname]] = []
        fmi = fmis[goodnames[gname]] = []
        for mname,m in gs:
            for tid,tdata in data[m].items():
                v = float(tdata["Velocity.y (microns/min)"])
                if gname in invert:
                    v = -v
                vel.append(v)

                s = float(tdata["Speed (microns/min)"])
                speed.append(s)

                f = float(tdata["FMI.y"])
                if gname in invert:
                    f = -f
                fmi.append(f)

    with open(out_folder/f"velocities{details}.csv","w",newline="") as f:
        d = DictWriter(f,fieldnames=vels.keys());
        d.writeheader()
        d.writerows(zip_dict(vels))


    with open(out_folder/f"speeds{details}.csv","w",newline="") as f:
        d = DictWriter(f,fieldnames=speeds.keys());
        d.writeheader()
        d.writerows(zip_dict(speeds))

    with open(out_folder/f"FMIs{details}.csv",'w',newline='') as f:
        d = DictWriter(f,fieldnames=fmis.keys())
        d.writeheader()
        d.writerows(zip_dict(fmis));


