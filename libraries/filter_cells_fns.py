from typing import Tuple, Union
import numpy as np


def remove_multiple_nuclei_cells(labeledcellsmask:np.ndarray,labelednucsmask:np.ndarray)->np.ndarray:
    out = np.copy(labeledcellsmask)
    labels=set(out.ravel())
    for i in labels:
        if i!=0: #ignore background           
            #get obect i coordinates 
            icoord=np.where(out==i)
            #get values of nuclear mask for those coordinates
            nucvalues=labelednucsmask[icoord]
            #get labels list (get individual elements)
            nuclabels=set(nucvalues)
            #if there is more than one nuclei (2 objects counting backround)
            if len(nuclabels) > 2: 
                #remove cell object:
                out[out==i]=0
    return out
                
        
def remove_large_objects(labeledmask:np.ndarray,maxarea:float)->np.ndarray:
    out = np.copy(labeledmask)
    component_sizes = np.bincount(labeledmask.ravel())
    too_large = component_sizes > maxarea
    too_large_mask = too_large[labeledmask]
    out[too_large_mask] = 0
    return out

def remove_touching_edge(labeledmask:np.ndarray,margins:Union[int,Tuple[int,int],Tuple[int,int,int,int]]=1)->np.ndarray:
    '''remove regions that touch the edge of the image.

    margins param can be:
    - int - will remove regions that have pixels within distance margins from each edge; will not remove if margins=0
    - tuple[int,int] - filter regions within margins[0] of the first axis, within margins[1] of the second axis
    - tuple[int,int,int,int] - filter regions by: [-first axis, +first axis, -second axis, +second axis]
    '''
    if isinstance(margins,tuple) and len(margins) == 2:
        margins = (margins[0],margins[0],margins[1],margins[1]);
    elif isinstance(margins,int):
        margins = (margins,margins,margins,margins);
    

    out = np.copy(labeledmask)

    #scan each edge of the image
    first=out.shape[0]
    second=out.shape[1]

    def rm_idx(a,b):
        if out[a,b] != 0:
            out[out==out[a,b]] = 0;
    
    #i = first axis, j = second axis

    #negative first axis
    for i in range(margins[0]): 
        for j in range(second):
            rm_idx(i,j);            
    
    #positive first axis
    for i in range(first-margins[1],first):
        for j in range(second):
            rm_idx(i,j);

        
    #negative second axis
    for i in range(first):
        for j in range(margins[2]):
            rm_idx(i,j);
    
    
    #positive second axis
    for i in range(first):
        for j in range(second-margins[3],second):
            rm_idx(i,j);
    
    return out