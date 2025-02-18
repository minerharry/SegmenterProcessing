from typing import Iterable
import cv2
import joblib
import numpy as np

def get_mask_outlines(mask:np.ndarray)->list[np.ndarray]:
    """returns a list of point-(x,y) indexed numpy arrays containing outlines. Each outline is in pixel space."""
    if len(np.unique(mask)) == 1:
        return []
    # print(mask.shape)
    contours:list[np.ndarray]
    try:    
        contours,_ = cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # raise Exception()
    except:
        from IPython import embed; embed()
        contours = []

    return [c.squeeze(axis=1) for c in contours]

def get_labeled_mask_outlines(mask:np.ndarray,ignore:int|Iterable[int]=0)->list[tuple[int,list[np.ndarray]]]:
    """returns a list of tuples (label,outlines) for each nonzero label in the array. outlines are in the same format as get_mask_outlines."""
    if isinstance(ignore,int):
        ignore = [ignore]
    
    labeled_contours:list[tuple[int,list[np.ndarray]]] = []
    for label in np.unique(mask):
        if label in ignore:
            continue
        m = mask == label
        contours = get_mask_outlines(m)
        labeled_contours.append((label,contours))
        
    return labeled_contours