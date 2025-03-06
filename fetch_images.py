import functools
import random
import os
import shutil
import re
from pathlib import Path,PurePosixPath
from threading import local
from tqdm import tqdm
from bidict import bidict
from libraries.filenames import numeric_basename
from typing import Iterable, Union
import gsutilwrap
# from fastprogress.fastprogress import master_bar,progress_bar

def fdouble(exp):
    return Path("D:/Harrison/")/exp/exp

def gpath(path,exp=None):
    if not exp: exp = Path(path).name
    return PurePosixPath("gs://optotaxisbucket/movies")/exp/exp
    
def singlegpath(path,exp=None):
    if not exp: exp = Path(path).name
    return PurePosixPath("gs://optotaxisbucket/movies")/exp

localMap = bidict({
    "random" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.1.19 Random Test", #low cell count
    "random2" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.7 Random Migration", #some bubbles, but good overall
    "migration1" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.1.20 Migration Test 1", #blurry
    "migration2" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.9 Migration Test 2", #no migration
    "migration4" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.16 Migration Test 4", #some contamination
    "migration5" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.23 Migration Test 5",
    "migration6" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.3.1 Migration Test 6", #low cell count, some small contamination
    "migration7" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.3.3 Migration Test 7",
    "migration8" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.3.8 Migration Test 8",
    "itsn1" : "C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/2022.12.20 ITSNAIOopto",
    "itsn2" : "C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/2023.01.02 ITSNAIOopto2/Original",
    "migration41" : "C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/2023.1.26 OptoITSN Test 41",
    "migration42" : "F:/Lab Data/2023.1.31 OptoITSN Test 42/2023.1.31 OptoITSN Test 42",
    "migration43" : fdouble("2023.2.1 OptoITSN Test 43"),
    "migration44" : fdouble("2023.2.3 OptoITSN Test 44"),
    "migration45" : fdouble("2023.2.5 OptoITSN Test 45"),
    "migration46" : fdouble("2023.2.7 OptoITSN Test 46"),
    "migration47" : fdouble("2023.2.9 OptoITSN Test 47"),
    "migration50" : fdouble("2023.3.24 OptoTiam Exp 50"),
    "migration51" : fdouble("2023.3.29 OptoTiam Exp 51"),
    "migration53" : fdouble("2023.4.2 OptoTiam Exp 53"),
    "migration54" : fdouble("2023.4.3 OptoTiam Exp 54"),
    "migration55" : fdouble("2023.4.5 OptoTiam Exp 55"),
    "migration56" : fdouble("2023.4.7 OptoTiam Exp 56"),
})

gcp_map = bidict({
    name:(gpath(path),singlegpath(path)) for name,path in localMap.items()
})

gcp_map["itsn2"] = (PurePosixPath("gs://optotaxisbucket/movies/2023.01.02 ITSNAIOopto2/Original"),)

sourceMap = {k:(localMap[k],*gcp_map[k]) if k in gcp_map else localMap[k] for k in localMap}


def is_gcp_path(path:Union[str,Path]):
    if not isinstance(path,Path):
        path = Path(path);
    return path.parts[0].lower() == "gs:";

def gs_str(p:Union[str,Path]):
    p = Path(p);
    out = ""
    if is_gcp_path(p):
        p = Path(*p.parts[1:])
        out = "gs://"
    out += p.as_posix();
    return out

def gs_ls(dir:Path):
    r =  gsutilwrap.ls(gs_str(dir/"*"),dont_recurse=True)
    return r

class SourceNotFound(Exception):
    pass

@functools.cache
def try_get_images(s:Path):
    if is_gcp_path(s):
        try:
            return [Path(g).name for g in gs_ls(s)]
        except FileNotFoundError:
            raise SourceNotFound
    else:
        try:
            return os.listdir(s)
        except FileNotFoundError:
            raise SourceNotFound

def get_images(sources:Iterable[Path|str]):
    for s in sources:
        try:
            return try_get_images(Path(s))
        except SourceNotFound:
            continue
    raise SourceNotFound(f"Unable to match any sources {sources}")

def try_copy_image(source:Path,imagename:str,dest:Path):
    if is_gcp_path(source):
        try:
            return gsutilwrap.copy(gs_str(source/imagename),gs_str(dest))
        except FileNotFoundError:
            raise SourceNotFound
    else:
        try:
            return shutil.copy(source/imagename,dest)
        except FileNotFoundError:
            raise SourceNotFound

sourceMap = {
    "random" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.1.19 Random Test", #low cell count
    "random2" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.7 Random Migration", #some bubbles, but good overall
    "migration1" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.1.20 Migration Test 1", #blurry
    "migration2" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.9 Migration Test 2", #no migration
    "migration4" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.16 Migration Test 4", #some contamination
    "migration5" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.23 Migration Test 5",
    "migration6" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.3.1 Migration Test 6", #low cell count, some small contamination
    "migration7" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.3.3 Migration Test 7",
    "migration8" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.3.8 Migration Test 8",
}

sources = ["random","random2","migration1","migration4","migration5","migration6","migration7","migration8"];

sources = ["migration45","migration46","migration47","migration50","migration51","migration53","migration54","migration55","migration56"];
sources = ["migration47","migration50","migration53","migration55","migration51"]
processFolder = Path("C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/optotaxis calibration/data/segmentation_iteration_testing/");

iteration = 3;
round = 7
outFolder = processFolder/f"iter{iteration}/round{round}/images";
outFolder = processFolder/f"iter{iteration}/round{round}/test_images";
# outFolder = processFolder/"evaluation_images_22-4-6";


num_select = 2;
basename_regex = numeric_basename 
if __name__ == "__main__":
    if not isinstance(sources,list):
        sources = [sources];
    if not os.path.exists(outFolder):
        os.makedirs(outFolder);
    for source in tqdm(sources):
        s = sourceMap[source]
        if isinstance(s,str):
            s = (s,)

        images = get_images(s)
        ims = random.sample(set(filter(lambda x: x.endswith(("tif","tiff","TIF","TIFF")),images)),num_select);
        

        for im in tqdm(ims,leave=False):
            try_copy_image(Path(s[0]),im,outFolder/re.sub(basename_regex,source,im))
