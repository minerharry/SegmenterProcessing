from typing import Literal
from fastai.layers import Module,SequentialEx
import torch
import torch.nn.functional as F
from fastai.learner import Learner
## REFERENCE: From fastai
# class MergeLayer(Module):
#     "Merge a shortcut with the result of the module by adding them or concatenating them if `dense=True`."
#     def __init__(self, dense:bool=False): self.dense=dense
#     def forward(self, x): return torch.cat([x,x.orig], dim=1) if self.dense else (x+x.orig)
# class ResizeToOrig(Module):
#     "Merge a shortcut with the result of the module by adding them or concatenating them if `dense=True`."
#     def __init__(self, mode='nearest'): self.mode = mode
#     def forward(self, x):
#         if x.orig.shape[-2:] != x.shape[-2:]:
#             x = F.interpolate(x, x.orig.shape[-2:], mode=self.mode)
#         return x
    
class ResizeMergeLayer(Module):
    "Merge a shortcut with the result of the module, either resizing the original (`resize_to='res'`) or the result (`resize_to=orig`) to match . concatenates them if `dense=True`."
    def __init__(self,mode:str='nearest',dense:bool=False,resize_to:Literal['res','orig']='res'):
        self.mode = mode
        self.dense = dense
        self.resize_to = resize_to
    def forward(self,x):
        orig = x.orig
        if orig.shape[-2:] != x.shape[-2:]:
            if self.resize_to == "res":
                orig = F.interpolate(orig,x.shape[-2],mode=self.mode)
            else:
                x = F.interpolate(x,orig.shape[-2],mode=self.mode)
        return torch.cat([x,orig],dim=1) if self.dense else (x+orig)

def halve_unet(learner:Learner):
    model = SequentialEx(*learner.layers[:8], ResizeMergeLayer(dense=True), *learner.layers[11:])
    learner.model = model.float()
    return learner

if __name__ == "__main__":
    from fastai.learner import load_learner
    import pathlib
    t = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    mask_from_image = None
    learner = load_learner(r"C:\Users\Harrison Truscott\Downloads\models_matt_matt_cell_unet_round4_nostack.pkl")
    pathlib.PosixPath = t

    learner = halve_unet(learner)
    import numpy as np
    t = torch.tensor(np.ndarray((4, 3, 100, 100))).float()
    # t = t.double()
    p = learner.model(t)
    from IPython import embed; embed()