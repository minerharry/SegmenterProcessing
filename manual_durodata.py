from typing import DefaultDict
from libraries.analysis import analyze_experiment_tracks, save_tracks_analysis_csv
from utils.filegetter import afn,afns
import pandas as pd
from IPython import embed
from pathlib import Path

tracksqc_paths = afns()
for tracksqc_path in tracksqc_paths:
    name = input(f"descriptive shortname for file {Path(tracksqc_path).name}: ");

    ltracks:list[pd.DataFrame] = pd.read_pickle(tracksqc_path);

    region_tracks = DefaultDict(list)
    for t in ltracks:
        t['time'] = range(len(t))
        region_tracks[t['gel-region'].iloc[0]].append(t)

    tracks = {k:{int(m["movie"].iloc[0]):m for m in region_tracks[k]} for k in region_tracks}
    # print(tracks);
    analysis = analyze_experiment_tracks(tracks,'approximate-medoid')
    # embed()

    save_tracks_analysis_csv("output/durodata/" + name + ".csv", analysis, "um", "min")