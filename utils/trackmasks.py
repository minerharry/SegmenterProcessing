
import re
from typing import DefaultDict
import zipfile
import os
from pathlib import Path
import joblib
from tifffile import TiffFile
from tqdm import tqdm
from utils import SafeDict, cleardir
from utils.extractmasks import ensure_masks, extract_masks

class arrayholder:
    def __init__(self,arr):
        self.arr = arr
    def asarray(self):
        return self.arr

def read_trackmask(file:str|Path|os.PathLike):
    mask = TiffFile(file)
    startframe = int(mask.shaped_metadata[0]["startframe"])
    nframes = len(mask.series[0])
    frames = range(startframe,startframe+nframes)
    if len(mask.series[0][0].shape) > 2:
        stack = mask.series[0].shape[0]
        frames = range(startframe,startframe+nframes*stack)
        # from IPython import embed; embed()
        res = [arrayholder(a) for a in mask.series[0][0].asarray()]
        return frames,res
    return frames,mask.series[0]

def get_trackmasks(
            file:str|Path|os.PathLike|None=None,
            experiment:str|None=None,
            zip_name:str|Path|os.PathLike="tracks_masks.zip",
            analysis_folder:str|Path|os.PathLike=r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis",
            trackmasks_outfolder:str|Path|os.PathLike="trackmasks",
            force_reextract:bool=False,
            pattern:str="{experiment}_movie{movie}_track{track}.TIF",
            ):
    # if file is not None:
    #     experiment = Path(file).parent.name
    #     dest = Path(trackmasks_outfolder)/experiment
    #     source = Path(file)
    # elif experiment is not None:
    #     dest = Path(trackmasks_outfolder)/experiment;
    #     source = Path(analysis_folder)/experiment/zip_name
    # else:
    #     raise ValueError("At least one of file (absolute path) or experiment (to choose folder in analysis_folder) must be provided and must not be None");
    # assert ensure_masks(source,dest,force_reextract=force_reextract)
    dest = extract_masks(file,experiment,zip_name,analysis_folder,trackmasks_outfolder,force_reextract)
    experiment = dest.name
    regex = re.compile(pattern.format_map(SafeDict(experiment=re.escape(experiment),movie=r"(\d+)",track=r"(\d+)")));
    def keyify(name:str):
        m = re.match(regex,name);
        if m:
            return (int(m.group(1)),int(m.group(2)))
        else:
            return None
    resdict = DefaultDict[int,dict[int,Path]](dict)
    for file in os.listdir(dest):
        if (keys := keyify(file)) is not None:
            resdict[keys[0]][keys[1]] = dest/file
    
    return dest,dict(resdict)