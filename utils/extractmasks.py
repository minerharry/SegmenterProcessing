
import functools
import re
from typing import DefaultDict
import zipfile
import os
from pathlib import Path
import joblib
from tifffile import TiffFile
from tqdm import tqdm
from utils import SafeDict, cleardir



def extract_masks(
            file:str|Path|os.PathLike|None,
            experiment:str|None,
            zip_name:str|Path|os.PathLike,
            analysis_folder:str|Path|os.PathLike,
            masks_outfolder:str|Path|os.PathLike,
            force_reextract:bool,
            ):
    """
    Extract masks from input location to some other folder. Tailored for extracting labeledmasks/trackmasks from segmentation analysis.
    If the zipfile is given, will extract from {file} to {masks_outfolder}/{experiment}
    If zipfile is not given, will extract from {analysis_folder}/{experiment}/{zip_name} to {masks_outfolder}/{experiment}
    If extraction has previously been completed, will not extract again.
    """
    if file is not None:
        experiment = Path(file).parent.name
        dest = Path(masks_outfolder)/experiment
        source = Path(file)
    elif experiment is not None:
        dest = Path(masks_outfolder)/experiment;
        source = Path(analysis_folder)/experiment/zip_name
    else:
        raise ValueError("At least one of file (absolute path) or experiment (to choose folder in analysis_folder) must be provided and must not be None");
    assert ensure_masks(source,dest,force_reextract=force_reextract)
    assert dest.name == experiment, str((dest.name, experiment))
    return dest

@functools.wraps(extract_masks)
def extract_labeled_cellmasks(
            file:str|Path|os.PathLike|None=None,
            experiment:str|None=None,
            zip_name:str|Path|os.PathLike="labeledmasks.zip",
            analysis_folder:str|Path|os.PathLike=r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis",
            labeledmasks_outfolder:str|Path|os.PathLike="labeledmasks",
            force_reextract:bool=False,
            ):
    return extract_masks(file,experiment,zip_name,analysis_folder,labeledmasks_outfolder,force_reextract)

@functools.wraps(extract_masks)
def extract_labeled_nucmasks(
            file:str|Path|os.PathLike|None=None,
            experiment:str|None=None,
            zip_name:str|Path|os.PathLike="labelednucs.zip",
            analysis_folder:str|Path|os.PathLike=r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis",
            labelednucs_outfolder:str|Path|os.PathLike="labelednucs",
            force_reextract:bool=False,
            ):
    return extract_masks(file,experiment,zip_name,analysis_folder,labelednucs_outfolder,force_reextract)

def ensure_masks(source:Path,dest:Path,force_reextract:bool=False):
    if force_reextract or not dest.exists() or not (dest/"extract_complete.flag").exists() or os.path.getmtime(source) > os.path.getctime(dest/"extract_complete.flag"):
        print("reextracting masks...")
        try:
            os.remove(dest/"extract_complete.flag")
        except FileNotFoundError:
            pass
        if source.suffix != ".zip":
            raise ValueError("Must provide zipfile")
        elif not source.exists():
            raise ValueError(f"Path to zipfile {source} does not exist")
        return unzip_masks(source,dest)
    else:
        return True

def unzip_masks(source:Path,dest:Path):
    if not dest.exists():
        os.makedirs(dest,exist_ok=True)
    cleardir(dest)
    with zipfile.ZipFile(source,mode='r') as file:
        file.extractall(dest,tqdm([m for m in file.filelist],desc="Extracting"))
    with open(dest/"extract_complete.flag","w") as f:
        pass
    return True
