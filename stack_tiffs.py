import glob
from typing import Any, Collection, Iterable, Iterator, Protocol, Sized, Tuple
import numpy as np
import tifffile
import fnmatch
import re
import os
# from fastprogress import progress_ba
from tqdm import tqdm

class Sizeable[T](Sized,Iterable[T],Protocol):
    ...

class Ensized[T](Sizeable[T]):
    def __init__(self,iterator:Iterator[T],length:int):
        self.iterator = iterator
        self.length = length
    def __iter__(self):
        return self.iterator
    def __len__(self):
        return self.length


def peek_iterator(iterator: Iterator[Any]) -> Tuple[Any, Iterator[Any]]:
    """Return first item of iterator and iterator.

    >>> first, it = peek_iterator(iter((0, 1, 2)))
    >>> first
    0
    >>> list(it)
    [0, 1, 2]

    """
    first = next(iterator)

    def newiter(first=first, iterator=iterator):
        yield first
        yield from iterator

    return first, newiter()

def readiter(files:Sizeable[str|os.PathLike[str]]):
    iter = (tifffile.imread(file) for file in tqdm(files,desc="reading images",leave=False))
    return Ensized(iter,len(files))
        

# from https://stackoverflow.com/a/47270916/13682828
def stack_files(series,output,progress=True,**kwargs):
    with tifffile.TiffWriter(output) as stack:
        for s in tqdm(series) if progress else series:
            stack.write(
                s, 
                photometric='minisblack',
                **kwargs
            )

def write_stack(output,series:Iterable[Sizeable[np.ndarray]],progress=True,writerKwargs={},**kwargs):
    with tifffile.TiffWriter(output,**writerKwargs) as stack:
        for it in tqdm(series) if progress else series:
            length = len(it)
            images = iter(it)
            if length > 0:
                im,images = peek_iterator(images);
                series_shape = (length,*im.shape);
                series_dtype = im.dtype;
                stack.write(images,shape=series_shape,dtype=series_dtype,contiguous=False,**kwargs);
            else:
                stack.write(None,contiguous=False);

def write_series(output,images:Sizeable[np.ndarray],writerKwargs={},**kwargs):
    return write_stack(output,[images],writerKwargs=writerKwargs,progress=False,**kwargs);


if __name__ == "__main__":
    from utils.filegetter import askdirectory, asksaveasfilename
    serieses = [1];
    files = {}
    parent_input = r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\images\2023.4.2 OptoTiam Exp 53"
    exclude = [];
    basename = "p";
    for series in serieses:
        if series in exclude:
            files[series] = [];
        input_files = glob.glob(f"{parent_input}\\{basename}_s{series}_*.TIF");
        input_files.sort(key=lambda s: int(re.match(f".*{basename}_s{series}_t([0-9]*)\\.TIF",s).group(1)));
        files[series] = input_files;


    output_file = "C:/Users/Harrison Truscott/Downloads/53_mov1.tif"
    if output_file == '':
        print("no output file selected >:(");
        exit();
    write_stack([readiter(files[s]) for s in serieses],output_file);
    