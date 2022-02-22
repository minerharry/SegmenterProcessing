import random
import os
import shutil
import re
from pathlib import Path

sourceMap = {
    "random" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.1.19 Random Test",
    "migration1" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.1.20 Migration Test 1",
    "random2" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.7 Random Migration",
    "migration4" : "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.16 Migration Test 4",
}

sources = ["random","random2","migration1","migration4"]

processFolder = Path("C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/optotaxis calibration/data/segmentation_iteration_testing/");
iteration = 3;
round = 2;
outFolder = processFolder/f"iter{iteration}/round{round}/images";

num_select = 3;
prefix_replace = "p[0-9]*";

if __name__ == "__main__":
    if not isinstance(sources,list):
        sources = [sources];
    if not os.path.exists(outFolder):
        os.makedirs(outFolder);
    for source in sources:
        selectFolder = Path(sourceMap[source])
        ims = random.sample(list(filter(lambda x: x.endswith(("tif","tiff","TIF","TIFF")),os.listdir(selectFolder))),num_select);
        for im in ims:
            shutil.copy(selectFolder/im,outFolder/re.sub(prefix_replace,source,im));