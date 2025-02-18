from abc import ABC
import builtins
from collections import OrderedDict
from copy import copy, deepcopy
from itertools import zip_longest
import itertools
import math
from typing import Any, Callable, DefaultDict, Generic, Hashable, Iterable, ParamSpec, Sequence, TypeVar
import dash
from dash import Dash, dcc, State, html
from dash_extensions.enrich import Output, Input
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
from plotly.subplots import make_subplots
from plotly.basedatatypes import BaseTraceType
from utils.animated_slider import PlaybackSliderAIO

from utils.paging import Index

#from https://stackoverflow.com/a/57503963/13682828
def factor_int(n:int):
    a = math.floor(math.sqrt(n))
    b = math.ceil(n/float(a))
    return a, b



figureSpec = list[tuple[go.Figure|BaseTraceType|None,...]|dict[Hashable,go.Figure|BaseTraceType|None]|None]

O = TypeVar("O")
class OutputCallback(Generic[O]):
    def __init__(self,callback:Callable[...,O],output_signature:Output) -> None:
        self.callback = callback
        self.output = output_signature

class Subplotter: #generates/updates figures from figureSpec info
    def __init__(self,spec:figureSpec|None=None,transition:go.layout.Transition|None=go.layout.Transition(duration=400)):
        self.transition = transition;
        self.update(None)
        self._figure = go.Figure()
        self.update(spec);
    
    def setTransition(self,transition:go.layout.Transition|None=go.layout.Transition(duration=400)):
        self.transition = transition
        self.layout.update(transition=self.transition)
    
    @property
    def next_trace(self):
        self._next_trace += 1
        return self._next_trace - 1
    
    @next_trace.setter
    def next_trace(self,idx:int):
        self._next_trace = idx

    def _update_trace(self,trace:BaseTraceType|None,subplot:int,key:Hashable):
        if subplot >= len(self.subplots):
            raise ValueError(f"subplot index {subplot} out of range for figure with {len(self.subplots)} subplots; unable to update/add trace")
        plot = self.subplots[subplot]
        if plot is None:
            plot = self.subplots[subplot] = {}
        if key in plot:
            if trace is None: #hiding trace
                plot[key].update(visible=False)
            else:
                plot[key].update(trace,visible=True,xaxis=self.axes[subplot][0],yaxis=self.axes[subplot][1])
        elif trace is not None:
            plot[key] = BaseTraceType.update(trace)
            self.tracedict[(subplot,key)] = self.next_trace
            plot[key].update(xaxis=self.axes[subplot][0],yaxis=self.axes[subplot][1])

    @property
    def layoutaxes(self):
        for (a1,a2) in self.axes:
            yield (a1[0] + "axis" + a1[1:],a2[0] + "axis" + a2[1:])

    def update(self,spec:figureSpec|None,clear_missing=False,subplots_shape:tuple[int,int]|None=None,
        subtitles:Sequence[str|None]|None=None,
        ranges:Sequence[tuple[tuple[float,float],tuple[float,float]]|None]|None=None,
        axis_specs:tuple[go.XAxis|None,go.YAxis|None]|list[tuple[go.XAxis|None,go.YAxis|None]|None]|None=None,
        layout:dict|go.Layout|None=None,
        subplot_kwargs:dict[str,Any]|None=None):
        """Supplying an argument of "None" will clear the figure and all previous traces.
        This is equivalent to passing clear_missing=True and passing an empty spec.\n
        To manually hide traces in clear_missing=False mode, pass None as either a trace or subplot specifier.
        giving subtitles=None will leave them as they are, but subtitles=[] will clear all subtitles. 
        if provided subtitles are shorter than the number of subplots, subplots will be titled until the list runs out and
        later ones will be blank. empty strings act as spacers and will clear previous titles; None elements leave existing
        titles for that subplot as they are.
        """

        self.updated = True
        if spec is None:
            self.subplots:list[dict[Hashable,BaseTraceType]|None] = []
            self.tracedict:dict[tuple[int,Hashable],int] = {}
            self.axes:list[tuple[str,str]] = []
            self.layout = go.Layout(transition=self.transition)
            self.subtitles:list[str]|None = None
            self.next_trace = 0;
            return

        #dictify specs
        dspec:list[dict[Hashable,go.Figure|BaseTraceType|None]] = []
        for plot in spec:
            if plot is None:
                dspec.append(DefaultDict(lambda: None))
            elif not isinstance(plot,dict):
                dspec.append({i:t for i,t in enumerate(plot)})
            else:
                dspec.append(plot)

        
        #update subplots
        if len(dspec) > len(self.subplots) or subplots_shape:
            print("recalculating subplots")
            nplots = len(dspec)
            subshape = subplots_shape;
            if subshape is not None:
                if subshape[0] > 0 or subshape[1] > 0:
                    if subshape[0] < 1:
                        subshape = (math.ceil(nplots/subshape[1]),subshape[1])
                    else:
                        subshape = (subshape[0],math.ceil(nplots/subshape[0]))
                else:
                    subshape = factor_int(nplots)
            else:
                subshape = factor_int(nplots)
            
            ##subplot titles
            if subtitles is not None and subtitles != self.subtitles:
                self.subtitles = [ssub if sub is None else sub for sub,ssub,_ in zip_longest(subtitles,self.subtitles or [],dspec,fillvalue='')]
            elif self.subtitles is None or len(self.subtitles) != len(self.subplots):
                self.subtitles = [sub for sub,_ in zip_longest(self.subtitles or [],dspec,fillvalue='')];

            subkwargs = dict(subplot_titles=self.subtitles)
            if subplot_kwargs:
                subkwargs.update(subplot_kwargs)
            sublayout = make_subplots(*subshape,**subkwargs).layout.update(template=None)
            self.layout.update(sublayout)
            self.axes = [("x","y")] + [(f"x{i}",f"y{i}") for i in range(2,nplots+1)]
            self.subplots.extend([None]*(nplots-len(self.subplots)))
            assert len(self.subplots) == len(dspec)
        elif subtitles:
            self.subtitles = [ssub if sub is None else sub for sub,ssub,_ in zip_longest(subtitles,self.subtitles or [],dspec,fillvalue='')]
            for ann,title in zip(self.layout.annotations,self.subtitles):
                ann.text = title

        if axis_specs is not None:
            laxes = list(self.layoutaxes)
            if not isinstance(axis_specs,list):
                axis_specs = [axis_specs]*len(laxes) #type:ignore
            assert axis_specs is not None
            for specs,axes in zip_longest(axis_specs,laxes,fillvalue=None):
                if not specs:
                    continue
                if not axes:
                    break
                (xaxis,yaxis) = axes
                (xspec,yspec) = specs
                if xspec:
                    self.layout[xaxis].update(xspec)
                if yspec:
                    self.layout[yaxis].update(yspec)

        if layout is not None:
            self.layout.update(layout)
        

        if ranges is not None:
            if len(ranges) > len(self.axes):
                print("Warning: more ranges provided than subplots, ignoring extra entries")
            for i in range(len(self.axes)):
                s = str(i) if i else ""
                self.layout[f"xaxis{s}"].update(range=ranges[i][0] if ranges[i] is not None else ranges[i])
                self.layout[f"yaxis{s}"].update(range=ranges[i][1] if ranges[i] is not None else ranges[i])

        used:set[tuple[int,Hashable]] = set()
        # print(dspec)
        #unpack spec
        for idx,plot in enumerate(dspec):
            for key,part in plot.items():
                if isinstance(part,go.Figure):
                    #only taking ONE trace from figures! assumed passed in as px.line or similar. Other traces get ignored.
                    d = part.data
                    if len(d) > 1:
                        print(f"WARNING: more than one trace detected in figure passed to Subplotter.update. Only the first trace will be plotted.\nFigure in subplot {idx}, trace key {key}. Figure {part} has too many traces. Trace kept: {d[0]}.")
                    self._update_trace(d[0],idx,key);
                else:
                    self._update_trace(part,idx,key)
                used.add((idx,key))
        
        if clear_missing:
            for idx,plot in enumerate(self.subplots):
                if plot is None:
                    continue
                for key in plot:
                    if (idx,key) not in used:
                        self._update_trace(None,idx,key)
                        
    def get_traces(self):
        ntraces = sum([len(l or []) for l in self.subplots])
        traces = [None]*ntraces
        for idx,plot in enumerate(self.subplots):
            if plot is None:
                continue;
            for key,trace in plot.items():
                traces[self.tracedict[(idx,key)]] = trace
        return traces

    @property
    def figure(self):
        if self.updated:
            new_traces = []
            for i,(fig_trace,trace) in enumerate(zip_longest(self._figure.data,self.get_traces(),fillvalue=None)):
                assert trace is not None
                if fig_trace is not None:
                    if fig_trace is not trace:
                        self._figure.data[i].update(trace)
                else:
                    new_traces.append(trace)
            self._figure.add_traces(new_traces)
            
            self._figure.update_layout(self.layout)
            
            # ## fix x and y axis ranges
            # full = self._figure.full_figure_for_development()
            # for i in range(len(self.axes)):
            #     s = str(i) if i else ""
            #     self._figure.layout[f"xaxis{s}"].update(range=full.layout[f"xaxis{s}"].range)
            #     self._figure.layout[f"yaxis{s}"].update(range=full.layout[f"yaxis{s}"].range)

        self.updated = False
        return self._figure;

class Toolbar:
    def __init__(self,
            content:dash.development.base_component.Component,
            paging:bool=False,
                paging_labels:tuple[str,str]=("previous","next"),
            slider:bool=False,
                slider_range:range=range(10),
            aio_id=None
                ):
        if not aio_id:
            aio_id = f"toolbar{id(self)}"
        self.bar = html.Div([],id=aio_id)
        self.html = html.Div([content,self.bar]);
        self.paging = paging
        self.doslider = slider
        if paging:
            self.next = html.Button(paging_labels[1],id=f"toolbar{id(self)}_next",accessKey="right");
            self.prev = html.Button(paging_labels[0],id=f"toolbar{id(self)}_prev",accessKey="left");
            self.bar.children.append(self.prev)
            self.bar.children.append(self.next)
        if slider:
            self.slider = PlaybackSliderAIO(aio_id=f"toolbar{id(self)}_slider",slider_props={"min":slider_range.start,"max":slider_range.stop-1,"step":slider_range.step,"value":slider_range.start,"marks":None,"tooltip":{"always_visible":True}})
            self.bar.children.append(self.slider)

    def __getstate__(self):
        print("state got")
        return super().__getstate__()


         
    def slider_signature(self,drag=True,state=False):
        if not self.doslider:
            raise Exception(f"no slider signatures for non-paging toolbar {self}")
        else:
            type = State if state else Input
            return type(self.slider.ids.slider(f"toolbar{id(self)}_slider"),"drag_value" if drag else "value");

    def slider_range_outsignatures(self,step=False):
        if not self.doslider:
            raise Exception(f"no slider signatures for non-paging toolbar {self}")
        else:        
            props = ["min","max"]
            if step:
                props.append("step")
            return [Output(self.slider.ids.slider(f"toolbar{id(self)}_slider"),m) for m in props];

    def slider_outsignature(self,drag=True):
        if not self.doslider:
            raise Exception(f"no slider signatures for non-paging toolbar {self}")
        else:
            return Output(self.slider.ids.slider(f"toolbar{id(self)}_slider"),"drag_value" if drag else "value");
        
    def paging_signatures(self,state=False):
        if not self.paging:
            raise Exception(f"no paging signatures for non-paging toolbar {self}")
        else:
            type = State if state else Input
            return (type(self.prev.id,"n_clicks"),type(self.next.id,"n_clicks"))


R = TypeVar("R")
def CallbackIndexer(app:Dash,
                    output_signature:Output|Iterable[Output],
                    input_signature:tuple[Input|Iterable[Input],Input|Iterable[Input]],
                    indexer:Index[Any,R],
                    # callback:Callable[[I],R]
                    ):
        outputs = [output_signature] if not isinstance(output_signature,Iterable) else output_signature
        inputs = [[inp] if not isinstance(inp,Iterable) else inp for inp in input_signature]
        @app.callback(
            *outputs,
            *inputs[0])
        def p_callback(*args)->R:
            return indexer.prev(*args)

        @app.callback(
            *outputs,
            *inputs[1])
        def n_callback(*args)->R:
            return indexer.next(*args)

            


def opacify_image(image:go.Figure,opacity:float,coloraxis:str="coloraxis"):
    assert 0 <= opacity <= 1
    image.layout[coloraxis].colorscale = opacify_colorscale(image.layout[coloraxis].colorscale,opacity)

def opacify_colorscale(colorscale:go.layout.Colorscale,opacity:float):
    assert 0 <= opacity <= 1
    scale = list(colorscale);
    for i in range(len(scale)):
        c:str = scale[i][1]
        if "#" in c:
            # color = tuple(int(c.removeprefix("#")[i:i+2], 16) for i in (0, 2, 4))
            color = pc.hex_to_rgb(c)
        else:
            # color = tuple(map(int, c.removeprefix("rgb(").removesuffix(")").split(', ')))
            color = pc.unlabel_rgb(c)
        scale[i] = (scale[i][0],f"rgba{color + (opacity,)}")
    return tuple(scale)

    




##better version found in paging.py: Index
# T = TypeVar("T")
# R = TypeVar("R")
# class CallbackIndexer(Generic[T,R]):
#     """Turns next/previous callbacks into a single integer index callback"""
#     def __init__(self,range:int|tuple[int,int]|tuple[int,int,int]|None,
#             input_signatures:tuple[Input,Input],
#             output_signature:Output|Iterable[Output],
#             index_callback:Callable[[int],R]|Callable[[T],R],
#             start_index:int=0,
#             custom_range: Sequence[T]|None = None,
#             loop=True) -> None:
#         """
#         Range: inputs to python's range function. If None will use custom_range
#         Input Signatures: (previous callback input, next callback input). Input value **will be ignored by the pager**, 
#             but all arguments are accessible through the callback's callback_args keyword arguments [optional].
#         Output Signature(s): where to send the return value of index_callback
#         Index Callback: function that takes an int / sequence element and returns the proper update value(s) for the output(s)
#         Start Index: index in the range to start with. This is not the numeric value it starts at (unless the range starts from zero).
#         Loop: whether to cycle through values when paging or to remain at the end. If loop is false and the index is at the final value,
#             the index callback will not be updated if the user tries to advance
#         Custom Range: Sequence of values to index over instead of an integer range. Item value can be whatever, will be passed to the index callback
#         """
#         self.range = builtins.range(range) if isinstance(range,int) else builtins.range(*range) if range is not None else custom_range
#         if self.range is None:
#             raise ValueError("At least one of range or custom_range must be not None")
#         self.index = start_index
#         self.inputs = input_signatures
#         self.outputs = [output_signature] if isinstance(output_signature,Output) else output_signature
#         self.callback = index_callback



    

