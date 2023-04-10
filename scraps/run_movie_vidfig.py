import os

from matplotlib.axes import Axes
from utils.filegetter import askdirectory
from videofig import videofig as vidfig
from skimage.io import imread
import re
from pathlib import Path
from tqdm import tqdm

regex = re.compile(".*s7_.*t([0-9]+)\\..*");

direct = askdirectory();

extras = 0;
def getframe(s:str):
    global extras
    m = re.match(regex,s);
    if m is None:
        print(s);
        extras += 1;
        return float('inf');
    else:
        return int(m.group(1));

imnames = sorted(tqdm(filter(None, (re.match(regex,s) for s in os.listdir(direct)))),key=lambda x: int(x.group(1)));
# print(imnames);
imgs = [imread(Path(direct)/p.group(0)) for p in tqdm(imnames)]

def re_func(frame:int,f:Axes):
    print(frame);
    f.imshow(imgs[frame]);

vidfig(len(imgs)-extras,re_func);