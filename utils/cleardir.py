import os
import shutil
import stat


def on_rm_error( func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod( path, stat.S_IWRITE )
    # os.unlink( path )

def cleardir(dir): #clears all files in dir without deleting dir
  for f in os.scandir(dir):
    # f = os.path.join(dir,f)
    if os.path.isdir(f): shutil.rmtree(f,onerror=on_rm_error); #just in case
    else: os.remove(f);