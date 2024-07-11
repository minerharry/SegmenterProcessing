from pathlib import Path
from shutil import copy
import shutil

from tqdm import tqdm
from utils.filegetter import adir
masksdir = adir(key="masks") #"C:/Users/Harrison Truscott/Downloads/itsn_2_masks/Cell"
masksdir = Path(masksdir);
outmasksdir = Path(adir(key="outmasks"))

copy_im = False
if copy_im:
    imsdir = adir(key="images") #"C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/2023.01.02 ITSNAIOopto2/Original"
    imsdir = Path(imsdir);
    outimsdir = Path(adir(key="outimages"))

s = 4
for t in tqdm(range(1,30,4)):
    # t = 47;
    copyname = f"p_s{s}_t{t}.TIF"
    outname = f"0607_s{s}_t{t}.TIF";
    try:
        copy(masksdir/copyname,outmasksdir/outname);
    except shutil.SameFileError:
        pass
    if copy_im:
        try:
            copy(imsdir/copyname,outimsdir/outname);
        except shutil.SameFileError:
            pass