import builtins
from typing import Any, Callable, Generic, Literal, Sequence, TypeVar, overload
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import IPython


T = TypeVar("T")
R = TypeVar("R")
class Index(Generic[T,R]):
    """Turns next/previous callbacks into a single integer index callback"""
    def __init__(self,range:int|tuple[int,int]|tuple[int,int,int]|None,
            index_callback:Callable[[int],R]|Callable[[T],R],
            start_index:int=0,
            custom_range: Sequence[T]|None = None,
            loop=True) -> None:
        """
        Range: inputs to python's range function. If None will use custom_range
               Index Callback: function that takes an int / sequence element and returns the proper update value(s) for the output(s)
        Start Index: index in the range to start with. This is not the numeric value it starts at (unless the range starts from zero).
        Loop: whether to cycle through values when paging or to remain at the end. If loop is false and the index is at the final value,
            the index callback will not be updated if the user tries to advance
        Custom Range: Sequence of values to index over instead of an integer range. Item value can be whatever, will be passed to the index callback
        """
        r = builtins.range(range) if isinstance(range,int) else builtins.range(*range) if range is not None else custom_range
        if r is None:
            raise ValueError("At least one of range or custom_range must be not None")
        self.range = r
        self.ind = start_index
        self.callback = index_callback
        self.loop = loop

    @property
    def current(self)->T:
        return self.range[self.ind]

    def next(self,*args):
        self.ind += 1
        return self._callback(args)
    
    def prev(self,*args):
        self.ind -= 1
        return self._callback(args);

    def _callback(self,args:tuple)->R:
        if self.loop:
            self.ind = self.ind % len(self.range)
        else:
            self.ind = max(0,min(self.ind,len(self.range)-1))
        try:
            return self.callback(self.range[self.ind],callback_args=args)
        except:
            return self.callback(self.range[self.ind])
        
class LegacyIndex:
    def __init__(self,active_func:Callable[[int,MouseEvent],None],loop:bool=False,min:int=0,max:int=10,init_ind:int=0) -> None:
        self.ind = init_ind
        self.func = active_func
        self.loop = loop
        self.max = max
        self.min = min
        assert max >= min, "Max must be greater than min"
        if self.ind not in range(self.min,self.max+1):
            self.ind = self.min
        self.callbacks = (self.prev,self.next)

    def next(self, event:MouseEvent):
        self.ind += 1
        if self.loop:
            self.ind = (self.ind-self.min)%self.max + self.min
        else:
            self.ind = max(self.min,min(self.ind,self.max))
        self.func(self.ind,event)

    def prev(self, event:MouseEvent):
        self.ind -= 1
        if self.loop:
            self.ind = (self.ind-self.min)%self.max + self.min
        else:
            self.ind = max(self.min,min(self.ind,self.max))
        self.func(self.ind,event)

@overload
def paging_bar(prev_callback:Callable[[MouseEvent],None],next_callback:Callable[[MouseEvent],None],figure:Figure|None=None,return_buttons:Literal[False]=False,prevtext="Previous",nexttext="Next",init_axes:bool=True)->SubFigure: ...

@overload
def paging_bar(prev_callback:Callable[[MouseEvent],None],next_callback:Callable[[MouseEvent],None],figure:Figure|None=None,return_buttons:Literal[True]=True,prevtext="Previous",nexttext="Next",init_axes:bool=True)->tuple[SubFigure,tuple[Button,Button]]: ...

def paging_bar(prev_callback:Callable[[MouseEvent],None],next_callback:Callable[[MouseEvent],None],figure:Figure|None=None,return_buttons:bool=False,prevtext="Previous",nexttext="Next",init_axes:bool=True):
    if not figure:
        figure = plt.figure()
    disp_fig,button_fig = figure.subfigures(2,height_ratios=[0.9,0.1])
    disp_fig:SubFigure
    axprev = button_fig.add_axes([0.7, 0.2, 0.1, 0.8])
    axnext = button_fig.add_axes([0.81, 0.2, 0.1, 0.8])
    buttons = []
    bnext = Button(axnext, nexttext)
    bnext.on_clicked(next_callback)
    bprev = Button(axprev, prevtext)
    bprev.on_clicked(prev_callback)
    buttons.append(bnext)
    buttons.append(bprev)
    if init_axes:
        disp_fig.add_axes([0.1,0.1,0.8,0.8])
    disp_fig._buttons = buttons
    if return_buttons:
        return disp_fig,(bprev,bnext)
    else:
        return disp_fig


# T = TypeVar("T")
@overload
def slider_bar(callback:Callable[[float],None],
    label:str,
    valmin:float,
    valmax:float,
    /,
    valinit:float=0.5,
    valstep:list[float]|float|None=None,
    orientation:Literal["horizontal","vertical"]="horizontal",
    figure:Figure|None=None,return_slider:Literal[False]=False,init_axes:bool=True,
    **slider_kwargs)->SubFigure: ...

@overload
def slider_bar(callback:Callable[[float],None],
    label:str,
    valmin:float,
    valmax:float,
    /,
    valinit:float=0.5,
    valstep:list[float]|float|None=None,
    orientation:Literal["horizontal","vertical"]="horizontal",
    figure:Figure|None=None,return_slider:Literal[True]=True,init_axes:bool=True,
    **slider_kwargs)->tuple[SubFigure,Slider]: ...

def slider_bar(callback:Callable[[float],None],label:str,valmin:float,valmax:float,/,figure:Figure|None=None,return_slider:bool=False,init_axes:bool=True,
    orientation:Literal["horizontal","vertical"]="horizontal", **slider_kwargs):
    if not figure:
        figure = plt.figure()    
    disp_fig:SubFigure
    slider_fig:SubFigure
    if orientation == "horizontal":
        disp_fig,slider_fig = figure.subfigures(2,1,height_ratios=[0.9,0.1])
    else:
        slider_fig,disp_fig = figure.subfigures(1,2,width_ratios=[0.1,0.9])

    axslider = slider_fig.add_axes([0.25,0.3,0.6,0.6] if orientation == "horizontal" else [0.3,0.25,0.6,0.6])
    slider = Slider(axslider,label,valmin,valmax,orientation=orientation,**slider_kwargs)
    slider.on_changed(callback)
    if init_axes:
        disp_fig.add_axes([0.1,0.1,0.8,0.8])
    disp_fig._slider = slider
    if return_slider:
        return disp_fig,slider
    else:
        return disp_fig



# @overload
# def indexed_pager(index_callback:Callable[[int,MouseEvent],None],figure:Figure|None=None,return_buttons:Literal[False]=False,prevtext="Previous",nexttext="Next")->SubFigure: ...
# @overload
# def indexed_pager(index_callback:Callable[[int,MouseEvent],None],figure:Figure|None=None,return_buttons:Literal[True]=True,prevtext="Previous",nexttext="Next")->tuple[SubFigure,tuple[Button,Button]]: ...
# def indexed_pager(index_callback:Callable[[int,MouseEvent],None],figure:Figure|None=None,return_buttons:bool=False,prevtext="Previous",nexttext="Next"):
#     return paging_bar(Index(index_callback))


if __name__ == "__main__":
    # f = plt.figure()
    fig:SubFigure
    # button_fig:SubFigure

    s:Line2D|None = None
    p=np.linspace(0,10,200)

    def plot_sin(freq:float):
        print("plotting sin",freq)
        global s
        if s is None:
            s = plt.plot(p,np.sin(p*freq))[0]
        else:
            s.set_ydata(np.sin(p*freq))
        fig.canvas.draw_idle()

    # callback = LegacyIndex(plot_sin,loop=True)
    fig = slider_bar(plot_sin,"Frequency",0,10)
    
    
    # bnext = Button(axnext, 'Next')
    # bnext.on_clicked(callback.next)
    # bprev = Button(axprev, 'Previous')
    # bprev.on_clicked(callback.prev)
    plot_sin(2)
    # print(disp_fig)

    IPython.embed()