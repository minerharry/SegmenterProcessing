import functools
from pathlib import Path
import shutil
from tqdm import tqdm
from utils.filegetter import adir
from fetch_images import sourceMap, get_images, try_copy_image
from libraries.filenames import filename_format,filename_regex_anybasename,alphanumeric_match,tiff_ext
import os
import re
import gsutilwrap

images_in = Path(adir(title="input images",key="adjacency_in"))
images_out = Path(adir(title="output dir",key="adjacency_out"))

regex = filename_regex_anybasename

num_stacked_images = 3

include_center_image = True


for im in tqdm(os.listdir(images_in)):
    m = re.match(regex,im);
    if not m:
        raise Exception("Improper regex / image name format")
    base,movie,frame,ext = m.groups()
    frame = int(frame)
    num_lower = int((num_stacked_images-1)/2)
    num_upper = num_stacked_images - num_lower - 1

    offsets = list(range(-num_lower,num_upper+1))
    assert 0 in offsets
    offsets.remove(0)

    s = sourceMap[base]
    if include_center_image:
        dest = images_out/im
        if not os.path.exists(dest):
            shutil.copy(images_in/im,dest)

    for o in offsets:
        match_regex = filename_format.format(alphanumeric_match,re.escape(movie),str(o+frame),tiff_ext)
        # if os.path.exists(source_dir):
        #     source = source_dir
        #     matches = [re.match(match_regex,f) for f in os.listdir(source_dir)]
        # elif isinstance(s,tuple):
        #     source = gs_str(gs_dir)
        #     matches = [re.match(match_regex,Path(f).name) for f in gs_ls(gs_dir)]
        # else:
        #     raise Exception(f"path {source_dir} does not exist")

        matches = [re.match(match_regex,f) for f in get_images(s)]
        
        matches = [m.string for m in matches if m is not None]

        if len(matches) == 0:
            raise Exception(f"unable to find image of format {match_regex} in source folder(s) {s}")
        elif len(matches) > 1:
            raise Exception(f"multiple matches for image of format {match_regex} found in source folder(s) {s}")
        
        name = matches[0]
        orig_components = re.match(regex,name).groups()
        dest = images_out/filename_format.format(base,*orig_components[1:])
        if os.path.exists(dest):
            continue
        else:
            try_copy_image(s,name,dest)
    

        