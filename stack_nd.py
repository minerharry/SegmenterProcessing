from ast import literal_eval as leval
from dataclasses import dataclass
import functools
from itertools import groupby
from multiprocessing import Value
from operator import attrgetter, itemgetter
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Collection, Container, Iterable, Mapping, Self, Sized, overload

import bidict
import imageio
import numpy as np
import tifffile
from tqdm import tqdm
from stack_tiffs import Ensized, Sizeable, readiter, write_series, write_stack
from parsend import parseND

def all_equal[T](f:Iterable[T],key:Callable[[T],Any]|None=None):
    g = groupby(f,key=key)
    return next(g,True) and not next(g,False)

class Singleton[T](Sizeable[T]): #this is dumb
    def __init__(self,val:T):
        self.val = val

    def __len__(self) -> int:
        return 1
    
    def __iter__(self):
        return iter([self.val])

@dataclass
class FilePart:
    filename:str
    basename:str
    wavenum:int|None=None
    stagenum:int|None=None
    timepoint:int|None=None

    @staticmethod
    def from_name(p:Path,wavenames:Mapping[str,int]):
        parts = str(p.stem).split("_")
        res = FilePart(p.name,parts[0])
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
    
    def in_range(self,basename:str,timerange:Container[int],stagerange:Container[int],waverange:Container[int]):
        return self.basename == basename \
           and (self.wavenum is None or self.wavenum in waverange) \
           and (self.stagenum is None or self.stagenum in stagerange) \
           and (self.timepoint is None or self.timepoint in timerange)

@functools.wraps(groupby)
def sorted_groups(i,key=None):
    return groupby(sorted(i,key=key),key=key);


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

    basename = Path(nd_loc).stem
    ntime:int|None = leval(NDData.get("NTimePoints","None")) if NDData["DoTimelapse"] == "TRUE" else None
    nstage:int|None = leval(NDData.get("NStagePositions","None")) if NDData["DoStage"] == "TRUE" else None
    nwave:int|None = leval(NDData.get("NWavelengths","None")) if NDData["DoWave"] == "TRUE" else None

    timenums = range(1,ntime+1) if ntime else [None]
    stagenums = range(1,nstage+1) if nstage else [None]
    wavenums = range(1,nwave+1) if nwave else [None]
    wavenames = bidict.bidict({f"{num}{NDData[f'WaveName{num}']}":num for num in wavenums if num is not None})

    assert len(source_exts) > 0
    ims_folder = Path(nd_loc).parent/images_folder
    filenames = (ims_folder).glob(f"*[{']['.join(source_exts)}]")

    fileparts:list[FilePart] = [FilePart.from_name(p,wavenames) for p in filenames]
        
    group = filter(lambda x: FilePart.in_range(x,basename,timenums,stagenums,wavenums),fileparts);
    # from IPython import embed; embed()
    # group = sorted(group,key=lambda x:x.stagenum)

    waveTimes:dict[int,list[int]] = {}#wavenum,timepoints
    for points in NDData.getEntry("WavePointsCollected",default=[]):
        wave,*points = map(int,map(str.strip,points.split(",")))
        waveTimes[wave] = points
        

    stage_movies:(list|Singleton)[(list|Singleton)[(list|Singleton)[str|None]]] = []

    stagegroups:groupby[int|None,FilePart] = sorted_groups(group,key=attrgetter("stagenum"))
    
    stagedict = {s:list(k) for s,k in stagegroups} #groupby iterators are not stored in parallel so need to collect

    for stage in stagenums:
        if stage not in stagedict:
            if stage is not None:
                raise ValueError(f"Missing stage position #{stage}: {NDData[f'Stage{stage}']}")
            else:
                raise AssertionError("No stages found")
        
        timegroups:groupby[int|None,FilePart] = sorted_groups(stagedict[stage],key=attrgetter("timepoint"))

        movie_frames:(list|Singleton)[(list|Singleton)[str|None]] = []

        timedict = {t:list(k) for t,k in timegroups}
        # from IPython import embed; embed()
        for time in timenums:
            if time not in timedict:
                raise ValueError(f"Mising time point #{time}" + f" of stage #{stage}: {NDData[f'Stage{stage}']}" if stage else "");

            timepoint_frames:(list|Singleton)[str|None] = []

            wavegroups:groupby[int|None,FilePart] = sorted_groups(timedict[time],key=attrgetter("wavenum"))
            wavedict = {w:list(k) for w,k in wavegroups}
            for wave in wavenums:
                if wave in waveTimes:
                    if time not in waveTimes[wave]:
                        #wavelength is skipped this time point! register a blank frame
                        if wave is None: #single wavelength, grayscale. 
                            #This will probably never happen because this currently breaks if there are no acquisitions at one timepoint
                            timepoint_frames = Singleton(None)
                        else:
                            assert isinstance(timepoint_frames,list)
                            timepoint_frames.append(None)
                        continue
                    #otherwise, throw the following error

                if wave not in wavedict:
                    if wave is not None:
                        raise ValueError(f"Missing wavelength #{wave}: {wavenames.inv[wave]}"
                                         + f" at time point #{time}" if time is not None else ""
                                         + f" of stage #{stage}: {NDData[f'Stage{stage}']}" if stage is not None else "")
                    else:
                        raise AssertionError("No wavelengths found")
                
                wave_frames = list(wavedict[wave])

                if len(wave_frames) > 1:
                    raise ValueError("Too many snapshots with the same parameters! Multiple copies of" 
                            + f" wavelength #{wave}" + f": {wavenames.inv[wave]}" if wave is not None else ""
                            + f" at time point #{time}" if time is not None else ""
                            + f" of stage #{stage}: {NDData[f'Stage{stage}']}" if stage is not None else ""
                            + "detected in folder.")

                if wave is None: #single wavelength, grayscale
                    timepoint_frames = Singleton(str(ims_folder/wave_frames[0].filename))
                else:
                    assert isinstance(timepoint_frames,list)
                    timepoint_frames.append(str(ims_folder/wave_frames[0].filename))

            if time is None:
                movie_frames = Singleton(timepoint_frames)
            else:
                assert isinstance(movie_frames,list)
                movie_frames.append(timepoint_frames)

        if stage is None:
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
    
    src = Path(r"F:\Lab Data\opto")
    dst = Path(r"C:\Users\Harrison Truscott\OneDrive - University of North Carolina at Chapel Hill\Bear Lab\Other Data\opto")

    # src2 = src/"2024.6.25 OptoPLC FN+Peg Test 2"/"Phase"/"p.nd"
    # dst2 = dst/"2024.6.25 OptoPLC FN+Peg Test 2"/"Phase"

    stack_and_copy(src,dst)
    # stack_nd(src2,output_folder=dst2)