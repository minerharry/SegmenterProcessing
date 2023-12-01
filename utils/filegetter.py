import tkinter as tk
from tkinter import TclError, filedialog
from types import NoneType
from typing import Callable, DefaultDict, ParamSpec, Sequence, TypeVar, IO, Type, overload
import shelve
import os
import contextlib as ctx
root = tk.Tk()

cache_location = "utils/filegetter.cache"
def prep_cache(cache:shelve.Shelf):
    cache["keyed"] = dict()
    cache["located"] = DefaultDict(dict)

_skip_cached_popups = False
'''Set to true to skip popups for files/locations that exist in the cache'''

@ctx.contextmanager
def skip_cached_popups(b:bool=True):
    global _skip_cached_popups
    old_cache = _skip_cached_popups
    try:
        _skip_cached_popups = b
        yield b
    finally:
        _skip_cached_popups = old_cache


class NoSelectionError(Exception):
    pass

T = TypeVar("T")
P = ParamSpec("P",)
R = TypeVar("R",IO,str,Sequence[IO],Sequence[str])
# result = 
def cachewrap(f:Callable[P,R|None],blanktype:Type[T])->Callable[P,R|T]:

    @overload
    def loc_cache(*args,allow_blank=False,**kwargs)->R: ...
    @overload
    def loc_cache(*args,allow_blank=True,**kwargs)->R|T: ...
    def loc_cache(*args,initialdir=None,initialfile=None,key=None,allow_blank=False,skip_popup=False,**kwargs)->R|T:
        import sys
        call_location = os.path.abspath(sys.argv[0])
        func_name = f.__name__
        if initialdir and initialfile:
            if os.path.dirname(initialfile) != initialdir:
                raise ValueError("Both initialfile and initialdir specified, but initialfile is not a file in initialdir. Either ensure initialfile is a file in initialdir, or (recommended) only supply one of these arguments.")
        loc = initialfile if initialfile else initialdir
        if not loc:
            with shelve.open(cache_location) as cache:
                if "keyed" not in cache or "located" not in cache:
                    prep_cache(cache)
                # print(cache["keyed"],cache["located"])
                if key and key in cache["keyed"]:
                    loc = cache["keyed"][key]
                elif call_location in cache["located"]:
                    if func_name in cache["located"][call_location]:
                        loc = cache["located"][call_location][func_name]
                    elif None in cache["located"][call_location]:
                        loc = cache["located"][call_location][None]
                    else:
                        loc = None
                else:
                    loc = None

        if loc is not None and (skip_popup or _skip_cached_popups):
            return loc #skip the popup and return the cached value
        
        if "title" not in kwargs and key:
            kwargs["title"] = key
        if loc is None:
            res = f(*args,**kwargs);
        elif os.path.isdir(loc):
            res = f(*args,initialdir=loc,**kwargs);
        else:
            try:
                res = f(*args,initialfile=loc,initialdir=os.path.dirname(loc),**kwargs);
            except TclError: #directory function
                # this should only happen the same key is being used for functions of 
                # different types or if it's the first call from a new location
                res = f(*args,initialdir=os.path.dirname(loc),**kwargs)

        newloc:str
        if res is None:
            if allow_blank:
                return blanktype()
            else:
                raise NoSelectionError
        elif isinstance(res,str):
            if len(res) == 0:
                ##dialog canceled, nothing returned
                if allow_blank:
                    return blanktype()
                else:
                    raise NoSelectionError
            newloc = res
        elif isinstance(res,IO):
            #As far as I can tell, IO objects returned for reading files have the filename as IO.name. unsure if consistent.
            #seems to be a standard for tkinter's filedialog return types?
            newloc = res.name
        elif isinstance(res,Sequence): #multiple files, save the directory instead of one file
            if len(res) == 0:
                raise AssertionError
            if isinstance(res[0],str):
                newloc = os.path.dirname(res[0])
            elif isinstance(res[0],IO):
                newloc = os.path.dirname(res[0].name)
            else:
                raise AssertionError
        else:
            raise AssertionError
        
        with shelve.open(cache_location,writeback=True) as cache:
            if key:
                cache["keyed"][key] = newloc
            else:
                cache["located"][call_location][None] = newloc
                cache["located"][call_location][func_name] = newloc
            # print(cache["keyed"],cache["located"])

        if res or allow_blank:
            print(res)
            return res
        else:
            raise NoSelectionError();
    return loc_cache


        




askdirectory = cachewrap(filedialog.askdirectory,NoneType);
askopenfilehandle = cachewrap(filedialog.askopenfile,NoneType);
askopenfilename = cachewrap(filedialog.askopenfilename,NoneType);
askopenfilenames = cachewrap(filedialog.askopenfilenames,list);
askopenfiles = cachewrap(filedialog.askopenfiles,list);
asksaveasfilehandle = cachewrap(filedialog.asksaveasfile,NoneType);
asksaveasfilename = cachewrap(filedialog.asksaveasfilename,NoneType);
adir = askdirectory
afn = askopenfilename
afns = askopenfilenames
aofn = askopenfilename
aofns = askopenfilenames
asfn = asksaveasfilename
afh = askopenfilehandle
aofh = askopenfilehandle
fn = askopenfilename