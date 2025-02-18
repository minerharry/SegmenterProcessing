from utils.filegetter import adir
import os
import shutil
from pathlib import Path
ref = Path(adir(key="ref"))
i = Path(adir(key="in"))
out = Path(adir(key="out"))
for l in os.listdir(ref):
    shutil.copy(i/l,out/l)