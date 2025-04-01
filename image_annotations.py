from datetime import timedelta
import functools
import itertools
from typing import Callable, Concatenate, Iterable, overload
import cv2
import numpy as np
textfont = cv2.FONT_HERSHEY_SIMPLEX


def applicable[**P,**P2,R:Callable[[np.ndarray,int],np.ndarray]](f:Callable[Concatenate[tuple[int,...],P],R]):
    @overload
    def apply(image:np.ndarray,*args:P.args,**kwargs:P.kwargs)->np.ndarray: ...
    @overload
    def apply(image:Iterable[np.ndarray],*args:P.args,**kwargs:P.kwargs)->"map[np.ndarray]": ...
    def apply(image:np.ndarray|Iterable[np.ndarray],*args:P.args,**kwargs:P.kwargs):
        single = False
        if isinstance(image,np.ndarray):
            single = True
            image = [image]
        
        it = iter(image)
        im = next(it)
        it = itertools.chain([im],it)

        func = f(im.shape,*args,**kwargs)
        if single:
            return func(next(it),0)
        else:
            return map(func,it,itertools.count())
        
    return apply
        
@applicable
def draw_scalebar(shape:tuple[int,...],pixelscale:float,scaleLength:float=250,scaleUnit:str="um",textsize:float=1,scaleWidth:int=5,color:tuple[int,int,int]=(255,255,255),thickness:int=2):
    length = scaleLength/pixelscale
    text = f"{scaleLength} {scaleUnit}"

    (label_width, label_height), baseline = cv2.getTextSize(text, textfont, textsize, 1)
    # print(label_height,baseline)
    text_height = label_height + baseline

    scale_top = 5 + text_height + 3
    rect = [(shape[1]-5,scale_top),(int(shape[1]-5-length),scale_top+scaleWidth)]

    textcorner = (int(shape[1]-5-length/2-label_width/2),5+label_height) #bottom left corner of text

    def _apply(im,frame:int):
        im = cv2.putText(im.copy(),text,textcorner,textfont,textsize,color,thickness)
        im = cv2.rectangle(im,*rect,color,-1) #does filled rectangle
        return im

    return _apply


@applicable
def draw_timestamp(shape:tuple[int,...],delta:timedelta|None=None,textsize:float=1,smallformat:str="{minute}:{second:02.2f}",color:tuple[int,int,int]=(255,255,255),thickness:int=2):
    def _apply(im,frame:int):
        if delta:
            real = (delta*frame).total_seconds()
            hour,minute,second = int(real//3600),int((real%3600)//60),real%60
            time = smallformat % (minute,second) if smallformat and not hour else f"{hour}h {minute}m"
        else:
            time = str(frame)

        return cv2.putText(im.copy(),time,(0,im.shape[0]),textfont,textsize,color,thickness)    
    
    return _apply


