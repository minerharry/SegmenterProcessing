from operator import itemgetter

import win32com.client
from utils.filegetter import askdirectory
from pathlib import Path
import os
import re
from PIL import Image
from PIL.TiffTags import TAGS
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    # imdir = Path(askdirectory());
    imdir = r"F:\Lab Data\2023.10.3 OptoTiam Exp 68\2023.10.3 OptoTiam Exp 68"
    imdir = Path(imdir);
    # movie = input("movie number? ");

    for movie in tqdm(range(1,17)):

        files = os.listdir(imdir);
        try:
            p_file = next(filter(lambda x: x.endswith("nd"), files));
            basename = os.path.splitext(p_file)[0];
        except StopIteration:
            basename = "p";

        regex = re.compile(f"{basename}_s{movie}_t([0-9]+)\\.[tif|tiff|TIF|TIFF]");

        movie_files = [];
        for f in files:
            m = regex.match(f);
            if m:
                movie_files.append((m.group(1),f));

        movie_files.sort(key=itemgetter(0));


        zs = [];
        for _,f in tqdm(movie_files,leave=False):
            with Image.open(imdir/f) as img:
                meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}
            metaxml = meta_dict["ImageDescription"][0];
            root = ET.fromstring(metaxml);
            children = root.findall("PlaneInfo/prop[@id='z-position']");
            e = children[0]
            z = float(e.attrib['value']);
            zs.append(z);
        
        plt.figure(f"stage {movie}")
        plt.plot(zs);
    plt.show();