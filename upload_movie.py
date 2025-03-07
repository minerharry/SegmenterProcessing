from pathlib import Path
from gsutilwrap import rsync
from utils.filegetter import adir

if __name__:

    dir = Path(adir(key="movie upload"))

    folder = dir/"Phase"
    assert folder.exists()

    exp = dir.name


    
    dest = f"gs://optotaxisbucket/movies/{exp}/{exp}"

    rsync(folder,dest,multithreaded=True,recursive=False)

    