from pathlib import Path
import re
from typing import Callable

from tqdm import tqdm
from utils import inquire
from utils.filegetter import adir

from tifffile import TiffFile
from utils.bftools import bfconvert, formatlist, get_omexml_metadata, showinf, supported_formats

def convert_files(
        indir:str|Path,
        outdir:str|Path,
        ext:str,
        file_wildcard:str|None=None,
        file_regex:str|re.Pattern|None=None,
        recurse:bool=True,
        extract_metadata:bool=True,
        **kwargs):
    """Uses bfconvert to recursively copy and all files from indir to outdir"""
    indir = Path(indir);
    outdir = Path(outdir);

    print(indir,outdir)

    ext = ext.lstrip("."); #ensure ext does not start with dot

    if not ext in supported_formats("write"):
        raise ValueError("Bioformats cannot write images of format " + ext);

    if file_regex is None:
        filter_func:Callable[[Path],bool] = lambda p: True
    else:
        filter_func:Callable[[Path],bool] = lambda p: re.compile(file_regex).match(p.name) is not None

    if file_wildcard is None:
        file_wildcard = "*"

    glob = indir.rglob if recurse else indir.glob;
    for f in tqdm(list(filter(filter_func,glob(file_wildcard)))):
        f = f.relative_to(indir);
        # bfconvert(str(indir/f),str(outdir/f.with_suffix("." + ext)),overwrite=True,**kwargs);
        if extract_metadata:
            orig_meta = get_omexml_metadata(str(indir/f));
            with open(str(outdir/f.with_suffix(".xml")),"w") as file:
                file.write(orig_meta);
    


if __name__ == "__main__":
    sourcedir = adir(key="indir",mangle_key=True);
    destdir = adir(key="outdir",mangle_key=True);
    
    # ext = inquire.inquire("Output file format?",supported_formats("write"),default="tif");
    convert_files(sourcedir,destdir,"tif",file_wildcard="*.vsi",extract_metadata=True);



