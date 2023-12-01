import dill
from utils.filegetter import afn
from fastai.learner import load_learner
import torch
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
fn = afn()
def mask_from_image(): pass
l = torch.load(fn,map_location=torch.device("cpu"))
from IPython import embed; embed()
