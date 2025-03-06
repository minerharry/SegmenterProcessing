from abc import ABC, abstractclassmethod
from ast import literal_eval as leval
from dataclasses import dataclass
import functools
from itertools import groupby
import itertools
from multiprocessing import Value
from operator import attrgetter, itemgetter
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Collection, Container, Generator, Iterable, Mapping, Self, Sequence, Sized, TypeVar, overload

import bidict
import imageio
import numpy as np
import tifffile
from tqdm import tqdm
from stack_tiffs import Ensized, Sizeable, readiter, write_series, write_stack
from parsend import NDData, parseND

T = TypeVar("T")

def all_equal(f:Iterable[T],key:Callable[[T],Any]|None=None):
    g = groupby(f,key=key)
    return next(g,True) and not next(g,False)

class Singleton(Sizeable[T]): #this is dumb
    def __init__(self,val:T):
        self.val = val

    def __len__(self) -> int:
        return 1
    
    def __iter__(self):
        return iter([self.val])

@dataclass
class FilePart:
    fullpath:Path
    filename:str
    basename:str
    wavenum:int|None=None
    stagenum:int|None=None
    timepoint:int|None=None

    @staticmethod
    def from_name(p:Path,wavenames:Mapping[str,int]):
        parts = str(p.stem).split("_")
        res = FilePart(p,p.name,parts[0])
        for part in parts[1:]:
            if part[0] == "s":
                res.stagenum = int(part[1:])
            elif part[0] == "t":
                res.timepoint = int(part[1:])
            elif part[0] == "w":
                res.wavenum = wavenames[part[1:]]
            else:
                ValueError("Unrecognized metaseires filename part: " + part)
        return res
    
    def in_range(self,basename:str,timerange:Container[int|None],stagerange:Container[int|None],waverange:Container[int|None]):
        return self.basename == basename \
           and (self.wavenum is None or self.wavenum in waverange) \
           and (self.stagenum is None or self.stagenum in stagerange) \
           and (self.timepoint is None or self.timepoint in timerange)


# @functools.wraps(groupby)
def sorted_groups(i,key=None,sortkey=None,groupkey=None):
    return groupby(sorted(i,key=sortkey or key),key=groupkey or key);

class ND_Dimension(ABC):
    of_keyword = "of"
    
    @classmethod
    def missing(cls,num:int,name:str|None)->str:
        """First component of a missing message. Full context is "{dimN.Missing} {dimN-1.at} {dimN-2.at} {dimN-2.at} ..."""
        return f"Missing {cls.desc(num,name)}"
    
    @classmethod
    def of(cls,num:int,name:str|None)->str: 
        """Secondary modifier to description. Full context is "{dimN.Missing} {dimN-1.of} {dimN-2.of} {dimN-2.of} ..."""
        return f"{cls.of_keyword} {cls.desc(num,name)}"

    @classmethod
    def desc(cls,num:int,name:str|None)->str: 
        """Describe this object, without punctuation. E.g. wavelength 2: Phase [num = 2, name = Phase]"""
        ...

    @classmethod
    def name(cls,plural:bool=False)->str:
        """Name of this dimension, e.g. timepoint (plural: timepoints)"""
        ...

    @classmethod
    def __hash__(cls):
        return hash(id(cls))
        
class TimeDimension(ND_Dimension):
    @classmethod
    def desc(cls, num: int, name: str | None) -> str:
        assert name is None
        return f"time point #{num}"
    
    @classmethod
    def name(cls, plural:bool=False):
        return "timepoints" if plural else "timepoint"

class StageDimension(ND_Dimension):
    of_keyword = "at"
    @classmethod
    def desc(cls, num: int, name: str | None) -> str:
        return f"stage position #{num}: {name}"
    
    @classmethod
    def name(cls, plural:bool=False):
        return "stage positions" if plural else "stage position"
    
class WaveDimension(ND_Dimension):
    @classmethod
    def desc(cls, num: int, name: str | None) -> str:
        return f"wavelength #{num}: {name}"
    
    @classmethod
    def name(cls, plural:bool=False) -> str:
        return "wavelengths" if plural else "wavelengths"

class NDimensionalError(Exception):
        def __init__(self,*points:tuple[type[ND_Dimension],int,str|None]):
            self.dimensions = points

        def add_dimension(self,*points:tuple[type[ND_Dimension],int,str|None]):
            self.dimensions += points

        def __str__(self):
            return " ".join(["Error with " + self.dimensions[0][0].desc(self.dimensions[0][1],self.dimensions[0][2])] + [dim.of(num,name) for (dim,num,name) in self.dimensions[1:]])

class PointMissingError(NDimensionalError): #TODO: add body with more information, with a __str__ method that synthesizes it properly. Also move outside of the function!
    def __str__(self):
        return " ".join([self.dimensions[0][0].missing(self.dimensions[0][1],self.dimensions[0][2])] + [dim.of(num,name) for (dim,num,name) in self.dimensions[1:]])

class DuplicatePointError(NDimensionalError):
    def __str__(self):
        res = "Too many snapshots with the same parameters! "
        if len(self.dimensions) == 0:
            return res + "Single image specified, multiple copies found!"
        else:
            res += "Multiple copies of "
            res += " ".join(dim.of(num,name) for (dim,num,name) in self.dimensions[1:])
            res += " detected in folder"
            return res


Y = TypeVar("Y",covariant=True)
S = TypeVar("S",contravariant=True)
R = TypeVar("R",covariant=True)
class SizedGenerator(Generator[Y,S,R]):
    def __init__(self,gen:Generator[Y,S,R],length:int):
        self.gen = gen
        if isinstance(gen,SizedGenerator):
            raise ValueError()
        self.length = length

    def __len__(self):
        return self.length
    
    def send(self, value: S) -> Y:
        return self.gen.send(value)
    
    def close(self) -> None:
        return self.gen.close()
    
    def throw(self,*args,**kwargs):
        return self.gen.throw(*args,**kwargs)

    # def __getattr__(self, name: str) -> Any:
    #     return self.gen.__getattribute__(name)


def iter_nd(images:Iterable[str|Path|os.PathLike[str]],
            order:Iterable[type[ND_Dimension]] = (StageDimension,TimeDimension,WaveDimension),
            nd:str|os.PathLike[str]|Path|dict[str,str]|NDData|None=None,
            basename:str|None=None,
            timenums:Iterable[int|None]|int|None=None,
            stagenums:Iterable[int|None]|int|None=None,
            wavenums:Iterable[int|None]|int|None=None,
            wavenames:None|Iterable[str]|Mapping[int,str]=None,
            wavetimes:None|Mapping[int|None,Container[int]]=None):
    nd_base:str|None = None
    if nd is not None:
        if not isinstance(nd,Mapping):
            try:
                nd = Path(nd)
                nd_base = nd.stem
                nd = parseND(nd)
            except FileNotFoundError:
                raise
            except:
                raise ValueError(f"Can't understand nd input type: {type(nd)}")
        assert isinstance(nd,Mapping)  

    if not basename:
        if not nd:
            raise ValueError("At least one of nd or basename must be provided")

        #try to get basename from ND
        if nd_base:
            basename = nd_base
        elif "basename" in nd:
            basename = nd.get("basename")
    assert basename is not None

    
    if isinstance(timenums,int):
        ntime = timenums
        timenums = range(1,ntime+1)
    if not timenums:
        if not nd:
            timenums = [None]
            raise ValueError("At least one of nd or timenums must be provided. To specify to use no timepoints, please pass timenums=[None] instead of timepoints=None")
        else:
            ntime:int|None = leval(nd.get("NTimePoints","None")) if nd["DoTimelapse"] == "TRUE" else None
            timenums = range(1,ntime+1) if ntime else [None]
    else:
        timenums = list(timenums)
    
    if isinstance(stagenums,int):
        nstage = stagenums
        stagenums = range(1,nstage+1)
    if not stagenums:
        if not nd:
            stagenums = [None]
            raise ValueError("At least one of nd or stagenums must be provided. To specify to use no stage positions, please pass stagenums=[None] instead of stagenums=None")
        else:
            nstage:int|None = leval(nd.get("NStagePositions","None")) if nd["DoStage"] == "TRUE" else None
            stagenums = range(1,nstage+1) if nstage else [None]
    else:
        stagenums = list(stagenums)
    # assert stagenums is not None

    if isinstance(wavenums,int):
        nwave = wavenums
        wavenums = range(1,wavenums+1)
    if not wavenums:
        if not nd:
            wavenums = [None]
            raise ValueError("At least one of nd or wavenums must be provided. To specify to use no wavelengths, please pass wavenums=[None] instead of wavenums=None")
        else:
            nwave:int|None = leval(nd.get("NWavelengths","None")) if nd["DoWave"] == "TRUE" else None
            wavenums = range(1,nwave+1) if nwave else [None]
    else:
        wavenums = list(wavenums)
    # assert wavenums is not None
    
    #sort out wavename/num bidict:
    if any(map(lambda x: x is not None,wavenums)): #any non-None wavelengths specified? Need names
        if wavenames is None:
            raise ValueError("Must provide wavelength names!")
        
        if not isinstance(wavenames,Mapping):
            wavenames = dict([(k,v) for k,v in zip(wavenums,wavenames) if k is not None])
        else:
            assert all([k in wavenames for k in wavenums if k is not None]) #make sure noone passed a dict[str,int] instead of dict[int,str]
            wavenames:Mapping[int,str] = wavenames
        wavenames = bidict.bidict(wavenames)
    else:
        wavenames = bidict.bidict()

    if not wavetimes:
        wavetimes = {} #wavenum,timepoints
        if isinstance(nd,NDData):
            for points in nd.getEntry("WavePointsCollected",default=[]):
                wave,*points = map(int,map(str.strip,points.split(",")))
                wavetimes[wave] = points
    if wavetimes:
        raise NotImplementedError("wavetimes")
        wave_offtimes = {num:{time for time in timenums if time not in wavetimes[num]} for num in wavenums}

    
    fileparts:list[FilePart] = [FilePart.from_name(Path(p),wavenames.inverse) for p in images]
    group = filter(lambda x: FilePart.in_range(x,basename,timenums,stagenums,wavenums),fileparts);

    

    #this is a little silly, but the issue is that a function with yield will always wait to execute until the first next() call.
    #this means that in this case, where we either want to return OR iterate deeper, it will wait and then return - still as an iterable
    def loop(group:Iterable[FilePart],dims:Sequence[tuple[type[ND_Dimension],str,Collection[tuple[int|None,str|None]]]]
             ) -> Path | \
                SizedGenerator[
                    tuple[
                        tuple[int | None, str | None],
                        Path | SizedGenerator[
                            tuple[
                                tuple[int | None, str | None],
                                Path
                            ], 
                            None, 
                            None
                        ]
                    ],
                    Any,
                    None]:
        if len(dims) == 0:
            #final dimension reached, check for uniqueness and return path
            group = list(group)
            if len(group) > 1:
                raise DuplicatePointError()
            elif len(group) == 0: #shouldn't happen but whatever
                raise PointMissingError()
            
            frame = next(iter(group))
            return frame.fullpath
        return SizedGenerator(loopgen(group,dims),len(dims[0][2]))

    def loopgen(group:Iterable[FilePart],dims:Sequence[tuple[type[ND_Dimension],str,Collection[tuple[int|None,str|None]]]]):
        (dimension,attribute,dimpoints),*dims = dims

        grouped = sorted_groups(group,key=attrgetter(attribute),sortkey=lambda x: (getattr(x,attribute) is None,getattr(x,attribute)))
        dimdict = {s:list(k) for s,k in grouped}
        for (num,name) in dimpoints:
            if num not in dimdict:
                if num is not None:
                    raise PointMissingError((dimension,num,name))
                else:
                    raise AssertionError(f"No {dimension.name(True)} found")
            
            try:
                yield (num,name),loop(dimdict[num],dims)
            except NDimensionalError as e:
                if num is not None: #don't include blank dimensions
                    e.add_dimension((dimension,num,name))
                raise


        
    
    
    dimension_dict:dict[type[ND_Dimension],Collection[tuple[int|None,str|None]]] = {
        TimeDimension: list(zip(timenums,itertools.cycle([None]))),
        StageDimension: [(num,(nd.get(f'Stage{num}') if nd else f"s{num}") if num is not None else "null") for num in stagenums],
        WaveDimension: [(num,wavenames[num] if num is not None else "null") for num in wavenums]
    }

    dimension_attributes:dict[type[ND_Dimension],str] = {
        TimeDimension: "timepoint",
        StageDimension: "stagenum",
        WaveDimension: "wavenum",
    }

    dimensions = [(dim,dimension_attributes[dim],dimension_dict[dim]) for dim in order]

    result = loop(group,dimensions)

    return result

    

def stack_nd(nd_loc:str|Path,output_folder:str|Path|os.PathLike[str]="stacks",
             source_exts:Collection[str]=(".tif",".tiff"),
             images_folder:str|Path|os.PathLike[str]="",
             copy_nd:bool=False):
    ## write nd file + folder into a single tiff file (WARNING: Potentially very large!)
    ## supports multistage, multitime, and multiwavelength.
    ## Different stages e.g. (_s{pos}_) are written as different series
    ## Different time points (_t{num}_) are written as sequential images in one series
    ## Different wavelengths (_w{num}{name}_) are written as different channels of one image

    NDData = parseND(nd_loc)
    assert len(source_exts) > 0
    ims_folder = Path(nd_loc).parent/images_folder
    filenames = (ims_folder).glob(f"*[{']['.join(source_exts)}]")

    stage_movies:(list|Singleton)[(list|Singleton)[(list|Singleton)[str|None]]] = []
    stagenums = []
    for (stagenum,stagename),times in iter_nd(filenames,nd=NDData):
        stagenums.append(stagenum)
        movie_frames:(list|Singleton)[(list|Singleton)[str|None]] = []
        for (timepoint,_),waves in times:
            timepoint_frames:(list|Singleton)[str|None] = []
            for (wavenum,wavename),frame in waves:
                print(stagename,timepoint,wavename,frame)
                if frame is None: #empty wavelength, but allowed by nd_data wavetimes
                    if wavenum is None:
                        #This will probably never happen because this currently breaks if there are no acquisitions at one timepoint
                        timepoint_frames = Singleton(None)
                    else:
                        assert isinstance(timepoint_frames,list)
                        timepoint_frames.append(None)
                else:
                    if wavenum is None: #single wavelength, grayscale
                        timepoint_frames = Singleton(str(ims_folder/frame))
                    else:
                        assert isinstance(timepoint_frames,list)
                        timepoint_frames.append(str(ims_folder/frame))

            if timepoint is None:
                movie_frames = Singleton(timepoint_frames)
            else:
                assert isinstance(movie_frames,list)
                movie_frames.append(timepoint_frames)

        if stagenum is None:
            stage_movies = Singleton(movie_frames)
        else:
            assert isinstance(stage_movies,list)
            stage_movies.append(movie_frames)
    
    

    #movie frames collected, time to write
    out = Path(nd_loc).parent/output_folder
    out.mkdir(exist_ok=True)
    if isinstance(stage_movies,Singleton):
        names = ["stack.tiff"]
    else:
        names = [f"Stage{i}.tiff" for i in stagenums]

    im_shape:tuple[int,...]|None = None
    im_dtype:np.dtype[Any]|None = None


    flatten = False
    def read_im(im:(list[str]|Singleton[str]),imread=imageio.v3.imread):
        def _imread(k:str|None):
            nonlocal im_shape,im_dtype
            if k is None:
                assert im_shape is not None and im_dtype is not None
                return np.zeros(im_shape,dtype=im_dtype)
            else:
                res = imread(k)
                if (im_shape is None):
                    im_shape = res.shape
                    im_dtype = res.dtype
                return res

        return _imread(im.val) if (isinstance(im,Singleton) and flatten) else np.stack([_imread(i) for i in im])

    def multi_readiter(iter:Sizeable[list[str|None]|Singleton[str|None]]):
        gen = (read_im(it) for it in tqdm(iter,desc="writing series",leave=False))
        return Ensized(gen,len(iter))
        

    for name,movie in zip(tqdm(names,desc="writing stage positions"),stage_movies):

        outfile = out/name
        if isinstance(movie,Singleton): #not time-series
            tifffile.imwrite(outfile,read_im(movie.val))
        else:
            write_series(outfile,multi_readiter(movie),photometric='minisblack',writerKwargs={"imagej":True})

    if copy_nd:
        shutil.copy(nd_loc,out/Path(nd_loc).name)

    print("Movie saved successfully")

def stack_and_copy(source:Path|os.PathLike[str]|str,dest:Path|os.PathLike[str]|str,source_im_exts:Collection[str]=(".tif",".tiff")):
    dest, source  = Path(dest), Path(source)
    dest.mkdir(exist_ok=True)
    nds = Path(source).glob("*.nd")
    any_nd = False 
    for nd in nds:
        print("Copying nd movie:",nd)
        stack_nd(nd,dest,source_exts=source_im_exts)
        any_nd = True
    
    nd_copy_excl = [*source_im_exts,".nd"]
    for path in Path(source).glob("*"):
        if path.is_file():
            #if there are any nd files, don't copy any other files with the source_im_exts extensions. 
            #kinda dumb method but assuming all relevant files should be copied by stack_nd
            if (not any_nd or path.suffix.lower() not in nd_copy_excl):
                #copy file to dest
                shutil.copy(path,dest/(path.relative_to(source)))
        else:
            #recursively copy subdirs
            stack_and_copy(path,dest/(path.relative_to(source)),source_im_exts=source_im_exts)


    


    
if __name__ == "__main__":
    # nd = r"F:\Lab Data\opto\2024.7.2 OptoPLC S345F FN Migration + Labeled FB Test\Multiwave\p.nd"
    # stack_nd(nd)
    
    # src = Path(r"F:\Lab Data\opto")
    # dst = Path(r"C:\Users\Harrison Truscott\OneDrive - University of North Carolina at Chapel Hill\Bear Lab\Other Data\opto")

    src = r"C:\Users\bearlab\Documents\Data_temp\Harrison\2024.7.31 OptoPLC S345F Protrusion Test\Phase\p.nd"
    dst = "../stacks"
    # stack_nd(src,dst)

    # src2 = src/"2024.6.25 OptoPLC FN+Peg Test 2"/"Phase"/"p.nd"
    # dst2 = dst/"2024.6.25 OptoPLC FN+Peg Test 2"/"Phase"

    stack_and_copy(r"C:\Users\bearlab\Documents\Data_temp\Harrison\2024.7.31 OptoPLC S345F Protrusion Test",r"C:\Users\bearlab\Documents\Data_temp\Harrison\2024.7.31 OptoPLC S345F Protrusion Test\stacks")
    # stack_nd(src2,output_folder=dst2)