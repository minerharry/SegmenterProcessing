import numpy as np
from skimage.exposure import rescale_intensity

def rescale(im:np.ndarray,min_quantile:float=0.1,max_quantile:float=0.97,out_range:str|tuple[float,float]|np.dtype|None='dtype'):
    min,max = np.quantile(im.ravel(),[min_quantile,max_quantile])
    return rescale_intensity(im,(min,max),out_range=out_range)
