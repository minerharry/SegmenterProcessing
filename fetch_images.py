import functools
import random
import os
import shutil
import re
from pathlib import Path,PurePosixPath
from threading import local
from tqdm import tqdm
from bidict import bidict
from libraries.filenames import filename_format,filename_regex_anybasename,alphanumeric_match,tiff_ext,parse_filename,numeric_basename
from typing import Iterable, Union
import gsutilwrap
from utils import IdentityDefault
# from fastprogress.fastprogress import master_bar,progress_bar

def fdouble(exp):
    return Path("D:/Harrison/")/exp/exp

def grange(path:str,letters=["G","I"]): #google drive virtual drives
    return tuple(path.format_map(IdentityDefault(G=let)) for let in letters)

def gpath(path,exp=None):
    if not exp: exp = Path(path).name
    return PurePosixPath("gs://optotaxisbucket/movies")/exp/exp
    
def singlegpath(path,exp=None):
    if not exp: exp = Path(path).name
    return PurePosixPath("gs://optotaxisbucket/movies")/exp

keydict = bidict({
    "random" : "2022.1.19 Random Test",
    "random2" : "2022.2.7 Random Migration",
    "migration1" :"2022.1.20 Migration Test 1",
    "migration2" :"2022.2.9 Migration Test 2", 
    "migration4" :"2022.2.16 Migration Test 4",
    "migration5" :"2022.2.23 Migration Test 5",
    "migration6" :"2022.3.1 Migration Test 6", 
    "migration7" :"2022.3.3 Migration Test 7",
    "migration8" :"2022.3.8 Migration Test 8",
    "itsn1" : "2022.12.20 ITSNAIOopto",
    "itsn1_better" : "2022.12.20 ITSNAIOopto Better Segmentation",
    "itsn2" : "2023.01.02 ITSNAIOopto2",
    "migration41" : "2023.1.26 OptoITSN Test 41",
    "migration42" : "2023.1.31 OptoITSN Test 42",
    "migration43" : "2023.2.1 OptoITSN Test 43",
    "migration44" : "2023.2.3 OptoITSN Test 44",
    "migration45" : "2023.2.5 OptoITSN Test 45",
    "migration46" : "2023.2.7 OptoITSN Test 46",
    "migration47" : "2023.2.9 OptoITSN Test 47",
    "migration50" : "2023.3.24 OptoTiam Exp 50",
    "migration51" : "2023.3.29 OptoTiam Exp 51",
    "migration53" : "2023.4.2 OptoTiam Exp 53",
    "migration54" : "2023.4.3 OptoTiam Exp 54",
    "migration55" : "2023.4.5 OptoTiam Exp 55",
    "migration56" : "2023.4.7 OptoTiam Exp 56",
    "migration61" : "2023.5.17 OptoTiam Exp 61",
    "migration63" : "2023.5.25 OptoTiam Exp 63",
    "migration64" : "2023.6.1 OptoTiam Exp 64",
    "migration65" : "2023.6.2 OptoTiam Exp 65",
    "migration70" : "2023.10.10 OptoTiam Exp 70",
    "peg3" : "2024.6.27 OptoPLC FN+Peg Test 3",
    "peg4" : "2024.6.27 OptoPLC FN+Peg Test 4",
})

fauto = ()

localMap = ({
    "random" : grange("{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.1.19 Random Test"), #low cell count
    "random2" : grange("{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.7 Random Migration"), #some bubbles, but good overall
    "migration1" : grange("{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.1.20 Migration Test 1"), #blurry
    "migration2" : grange("{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.9 Migration Test 2"), #no migration
    "migration4" : grange("{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.16 Migration Test 4"), #some contamination
    "migration5" : grange("{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.23 Migration Test 5"),
    "migration6" : grange("{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.3.1 Migration Test 6"), #low cell count, some small contamination
    "migration7" : grange("{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.3.3 Migration Test 7"),
    "migration8" : grange("{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.3.8 Migration Test 8"),
    "itsn1" : Path.home()/"OneDrive - University of North Carolina at Chapel Hill/Bear Lab/2022.12.20 ITSNAIOopto",
    "itsn2" : Path.home()/"OneDrive - University of North Carolina at Chapel Hill/Bear Lab/2023.01.02 ITSNAIOopto2/Original",
    "migration41" : Path.home()/"OneDrive - University of North Carolina at Chapel Hill/Bear Lab/2023.1.26 OptoITSN Test 41",
    "migration42" : Path("F:/Lab Data/2023.1.31 OptoITSN Test 42/2023.1.31 OptoITSN Test 42"),
    "migration43" : fauto,
    "migration44" : fauto,
    "migration45" : fauto,
    "migration46" : fauto,
    "migration47" : fauto,
    "migration50" : fauto,
    "migration51" : fauto,
    "migration53" : fauto,
    "migration54" : fauto,
    "migration55" : fauto,
    "migration56" : fauto,
    "migration61" : fauto,
    "migration63" : fauto,
    "migration64" : fauto,
    "migration65" : fauto,
    "migration70" : fauto,
    "peg3" : fauto,
    "peg4" : fauto,
})

for c,v in localMap.items():
    if v is fauto:
        localMap[c] = fdouble(keydict[c])

gcp_map = bidict({
    name:tuple(sum([(gpath(p),singlegpath(p)) for p in (path if isinstance(path,tuple) else (path,))],tuple())) for name,path in localMap.items()
})

gcp_map["itsn2"] = (PurePosixPath("gs://optotaxisbucket/movies/2023.01.02 ITSNAIOopto2/Original"),)

sourceMap:dict[str,tuple[str|Path,...]] = {k:(*(Q if isinstance((Q := localMap[k]),tuple) else (Q,)),*gcp_map[k]) if k in gcp_map else (Q if isinstance((Q := localMap[k]),tuple) else (Q,)) for k in localMap}


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

def try_copy_image(sources:Path|str|tuple[Path|str,...],imagename:str,dest:Path):
    if not isinstance(sources,tuple):
        sources = (sources,)
    for source in sources:
        source = Path(source)
        if is_gcp_path(source):
            try:
                return gsutilwrap.copy(gs_str(source/imagename),gs_str(dest))
            except FileNotFoundError:
                continue
        else:
            try:
                return shutil.copy(source/imagename,dest)
            except FileNotFoundError:
                continue
    raise SourceNotFound

def get_adjacent_images(im:str,width:int,regex:str|re.Pattern=filename_regex_anybasename,include_center:bool=True):
    if width <= 0:
        raise ValueError(width)
    
    m = parse_filename(im,regex)[0]
    if not m:
        raise Exception("Improper regex / image name format")
    base,movie,frame,ext = m
    frame = int(frame)
    num_lower = int((width-1)/2)
    num_upper = width - num_lower - 1

    offsets = list(range(-num_lower,num_upper+1))
    assert 0 in offsets
    if not include_center:
        offsets.remove(0)

    names = [filename_format.format(base,movie,off,ext) for off in offsets]

    return names;


# sourceMap = {
#     "random" : "{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.1.19 Random Test", #low cell count
#     "random2" : "{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.7 Random Migration", #some bubbles, but good overall
#     "migration1" : "{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.1.20 Migration Test 1", #blurry
#     "migration2" : "{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.9 Migration Test 2", #no migration
#     "migration4" : "{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.16 Migration Test 4", #some contamination
#     "migration5" : "{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.23 Migration Test 5",
#     "migration6" : "{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.3.1 Migration Test 6", #low cell count, some small contamination
#     "migration7" : "{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.3.3 Migration Test 7",
#     "migration8" : "{G}:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.3.8 Migration Test 8",
# }

# sources = ["random","random2","migration1","migration4","migration5","migration6","migration7","migration8"];
# 
# sources = ["migration45","migration46","migration47","migration50","migration51","migration53","migration54","migration55","migration56"];
sources = ["migration47","migration50","migration53","migration55","migration51","migration64","migration70","peg3","peg4"]
processFolder = Path.home()/("OneDrive - University of North Carolina at Chapel Hill/Bear Lab/optotaxis calibration/data/segmentation_iteration_testing/");

iteration = 5;
round = 2
# outFolder = processFolder/f"iter{iteration}/round{round}/images";
outFolder = processFolder/f"iter{iteration}/round{round}/new_images";
# outFolder = processFolder/"evaluation_images_22-4-6";


num_select = 1;
basename_regex = numeric_basename 
width = 4
exclude_edge = True

if __name__ == "__main__":
    if not isinstance(sources,list):
        sources = [sources];
    if not os.path.exists(outFolder):
        os.makedirs(outFolder);
    for source in tqdm(sources):
        s = sourceMap[source]

        images = get_images(s)
        ims = random.sample(set(filter(lambda x: x.endswith(("tif","tiff","TIF","TIFF")),images)),num_select);
        
        if width != 1:
            wide_ims = sum([get_adjacent_images(im,width) for im in ims],start=[])
            ims = wide_ims

        for im in tqdm(ims,leave=False):
            try_copy_image(s,im,outFolder/re.sub(basename_regex,source,im))
