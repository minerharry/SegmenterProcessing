from pathlib import Path
from shutil import copy

masksdir = "C:/Users/Harrison Truscott/Downloads/itsn_2_masks/Cell"
imsdir = "C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/2023.01.02 ITSNAIOopto2/Original"
masksdir = Path(masksdir);
imsdir = Path(imsdir);
outiter = 1
outround = 10;

s = 7
t = 47;
copyname = f"p_s{s}_t{t}.TIF"
outname = f"itsn2_s{s}_t{t}.TIF";

basefolder = "C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/optotaxis calibration/data/segmentation_iteration_testing"
roundfolder = Path(basefolder)/f"iter{outiter}"/f"round{outround}";
mout = roundfolder/"input";
imout = roundfolder/"images";

copy(masksdir/copyname,mout/outname);
copy(imsdir/copyname,imout/outname);