import re
from utils.filegetter import adir
from libraries.parse_moviefolder import filename_regex_alphanumeric,filename_format
import os
from pathlib import Path
from tqdm import tqdm

dir = adir()
dir = Path(dir)

new_basename = "0607"
for fname in tqdm(list(dir.rglob("*"))):
    name = fname.name
    m = re.match(filename_regex_alphanumeric,name)
    # print(m)
    if m:
        basename,*args = m.groups()
        newname = filename_format.format(new_basename,*args)
        # iname = dir/fname
        # outname = dir/fname.parent/name
        # print(iname,outname)
        os.rename(dir/fname,dir/fname.parent/newname)
