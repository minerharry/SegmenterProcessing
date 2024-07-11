from pathlib import Path
import re
from tqdm import tqdm
from utils.filegetter import askopenfilename,adir
from skimage.io import imsave
from tifffile import TiffFile
from utils.safedict import SafeDict
tiff = askopenfilename()
print(tiff)
file = TiffFile(tiff);

#keys: frame, idx, movie, custom([arg])
namepattern = "frame{frame:02d}.tif"

outfolder = Path(adir())


def getmovie():
    return input("movie #?");

def custom(msg:str=""):
    return input(msg);

movie = None if "{movie}" not in namepattern else getmovie()

for num,image in enumerate(tqdm(file.series[0])):
    arr = image.asarray();
    fname = namepattern.format_map(SafeDict({"frame":num+1,"idx":num,"movie":movie}))
    for m in re.finditer(r"\{custom:([^\}]*)\}",fname):
        fname = fname[:m.start()] + str(custom(m.group(1))) + fname[m.end():];
    imsave(outfolder/fname,arr,check_contrast=False)
    