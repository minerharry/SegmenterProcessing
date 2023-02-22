import re
import os

filename_regex = re.compile('p[0-9]*_s([0-9]+)_t([0-9]+).*//.(TIF|TIFF|tif|tiff)');

input_folder = "G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.2.23 Migration Test 5"

files = os.listdir(input_folder);

movies_shape:dict[int,list[int]] = {};

for f in files:
    mat = filename_regex.match(f);
    if not mat:
        continue;
    stage,frame = mat.groups();
    if stage not in movies_shape:
        movies_shape[stage] = [];
    movies_shape[stage].append(stage);

for m in movies_shape:
    l = movies_shape[m].sorted();
    for i in range(min(l),max(l)+1):
        if i not in l:
            print(f"ERROR: p*_s{m}_t{i}.tif is missing");