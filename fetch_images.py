import random
import os
import shutil
import re
from pathlib import Path
from tqdm import tqdm
# from fastprogress.fastprogress import master_bar,progress_bar

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
    "itsn1" : "C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/2022.12.20 ITSNAIOopto",
    "itsn2" : "C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/2023.01.02 ITSNAIOopto2/Original",
    "migration41" : "C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/2023.1.26 OptoITSN Test 41",
    "migration42" : "F:/Lab Data/2023.1.31 OptoITSN Test 42/2023.1.31 OptoITSN Test 42",
    "migration43" : "F:/Lab Data/2023.2.1 OptoITSN Test 43/2023.2.1 OptoITSN Test 43",
    "migration44" : "F:/Lab Data/2023.2.3 OptoITSN Test 44/2023.2.3 OptoITSN Test 44",

}

sources = ["itsn1","itsn2","migration41","migration42","migration43","migration44"];

processFolder = Path("C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/optotaxis calibration/data/segmentation_iteration_testing/");

iteration = 1;
round = 11;
outFolder = processFolder/f"iter{iteration}/round{round}/images";
# outFolder = processFolder/"evaluation_images_22-4-6";


num_select = 5;
prefix_replace = "p[0-9]*";

if __name__ == "__main__":
    if not isinstance(sources,list):
        sources = [sources];
    if not os.path.exists(outFolder):
        os.makedirs(outFolder);
    for source in tqdm(sources):
        selectFolder = Path(sourceMap[source])
        # print(selectFolder);
        ims = random.sample(list(filter(lambda x: x.endswith(("tif","tiff","TIF","TIFF")),os.listdir(selectFolder))),num_select);
        for im in tqdm(ims,leave=False):
            shutil.copy(selectFolder/im,outFolder/re.sub(prefix_replace,source,im));