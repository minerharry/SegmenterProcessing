import re
import fastai
from contextlib import contextmanager
from fastai.learner import Learner,load_learner as ll
from fastcore.meta import delegates
import pathlib
import __main__

@contextmanager
def fix_posixpath():
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.Path
    yield
    pathlib.PosixPath = temp

def load_learner(fname,**kwargs)->Learner:
    namedict = {}
    with fix_posixpath():
        try:
            while True:
                try:
                    res = ll(fname,**kwargs)
                    break
                except AttributeError as e:
                    m = re.search(re.compile("Can't get attribute '(.*?)' on "),e.args[0])
                    if m is not None and (name := m.group(1)).isidentifier():
                        namedict[name] = getattr(__main__,name,None)
                        setattr(__main__,name,None)
                        res = ll(fname,**kwargs)
                        continue
                    else:
                        raise
        finally:
            for n,v in namedict.items():
                if v is not None:
                    setattr(__main__,n,v)
                else:
                    delattr(__main__,n)
    return res
