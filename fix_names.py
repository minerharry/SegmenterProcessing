import os
from pathlib import Path
import re
from utils.filegetter import askdirectory


in_regex = re.compile(r'p[0-9]*_s([0-9]+)_([0-9]+).*\.(TIF|TIFF|tif|tiff)');
out_pattern = "p_s{0}_t{1}.{2}"


while True:
    dir = Path(askdirectory())
    for p in dir.rglob("*"):
        if (m := re.match(in_regex,p.name)):
            # print(p)
            # print(m)
            # print(m.groups())
            # print(out_pattern.format(*m.groups()))
            # print(p.parent/out_pattern.format(*m.groups()))
            # exit()
            os.rename(p,p.parent/out_pattern.format(*m.groups()))