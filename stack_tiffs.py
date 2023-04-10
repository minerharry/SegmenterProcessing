import glob
from typing import Any, Iterable, Iterator, Tuple
import numpy as np
import tifffile
import fnmatch
import re
import os
# from fastprogress import progress_ba
from tqdm import tqdm

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

def readiter(files):
    for file in tqdm(files,desc="reading images",leave=False):
        yield tifffile.imread(file);

# from https://stackoverflow.com/a/47270916/13682828
def stack_files(series,output,**kwargs):
    with tifffile.TiffWriter(output) as stack:
        for s in tqdm(series):
            stack.write(
                s, 
                photometric='minisblack',
                **kwargs
            )

def write_series(series:Iterable[Tuple[int,Iterator[np.ndarray]]],output,writerKwargs={},**kwargs):
    with tifffile.TiffWriter(output,**writerKwargs,imagej=True) as stack:
        for length,images in tqdm(series):
            if length > 0:
                im,images = peek_iterator(images);
                series_shape = (length,*im.shape);
                series_dtype = im.dtype;
                stack.write(images,shape=series_shape,dtype=series_dtype,contiguous=False,**kwargs);
            else:
                stack.write(None,contiguous=False);



if __name__ == "__main__":
    from filegetter import askdirectory, asksaveasfilename
    serieses = range(1,9);
    files = {}
    parent_input = r"C:\Users\Harrison Truscott\Downloads\itsn_1_masks\Cell"
    exclude = [];
    basename = "p";
    for series in serieses:
        if series in exclude:
            files[series] = [];
        input_files = glob.glob(f"{parent_input}\\{basename}_s{series}_*.TIF");
        input_files.sort(key=lambda s: int(re.match(f".*{basename}_s{series}_t([0-9]*)\\.TIF",s).group(1)));
        files[series] = input_files;


    output_file = "C:/Users/Harrison Truscott/Downloads/cellmasks.tif"
    if output_file == '':
        print("no output file selected >:(");
        exit();
    write_series([(len(files[s]),readiter(files[s])) for s in serieses],output_file);
    