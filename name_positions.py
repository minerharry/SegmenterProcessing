header = """# Define the number of dimensions we are working on
dim = 2

# Define the image coordinates
"""
rowformat = "{name}; ; ({x}, {y})\n"

image_offset = {"4x":(21500,-16300),"10x":(8593,-6640),"20x":(4332,-3320)} ##width,height of image in microscope units

image_size = (1344,1024) ##width, height of image in pixels

from ast import literal_eval
import itertools
import os
import shutil
from typing import Iterable, NamedTuple, NewType, TypeAlias

from tqdm import tqdm
from parsend import parseND
from pathlib import Path
from tifffile import TiffFile

def parseSTG(file:str|Path):
    file = Path(file)
    with open(file,"r") as f:
        vers_info = f.readline().rstrip("\n")
        head_data = [f.readline().rstrip("\n") for _ in range(3)]
        stage_data:dict[str,tuple[str,...]] = {}
        for l in f.readlines():
            if l.strip() == "":
                continue
            # print(l)
            data = map(str.strip,l.rstrip("\n").split(","))
            t = literal_eval(next(data))
            stage_data[t] = tuple(data)
    return vers_info,head_data,stage_data


def create_position_list(nd_file:str,pos_file:str,out_file:str,overwrite:bool=False,mag:str="4x"):
    """Takes in an nd and stg file and creates a TileConfiguration file for imageJ stitching using positions from the stg file"""
    if not nd_file.endswith(".nd"):
        raise Exception("nd_file must be an .nd file")
    if not pos_file.endswith(".STG"):
        raise Exception("pos_file must be an .STG file of stage positions")
    
    NDData = parseND(nd_file)
    _,_,STGData = parseSTG(pos_file)
    posdata:dict[str,tuple[int,int]] = {}

    npos = int(NDData["NStagePositions"])
    print(npos)
    for i in range(1,npos+1):
        posname = NDData[f"Stage{i}"]
        fname = f"p_s{i}.TIF"
        d = STGData[posname]
        pos = map(int,d[:2]) ##x,y; these are in **metamorph units**
        pixel_pos = [p/m*w for p,m,w in zip(pos,image_offset[mag],image_size)]

        posdata[fname] = tuple(pixel_pos)

    out_file = Path(out_file)
    if overwrite or not os.path.exists(out_file):
        os.makedirs(out_file.parent,exist_ok=True)
        with open(out_file,"w") as f:
            f.write(header)
            for fname,(x,y) in posdata.items():
                f.write(rowformat.format(name=fname,x=x,y=y))
    else:
        raise FileExistsError(out_file)

def create_position_list_nd(nd_file:str,out_file:str,overwrite:bool=False,image_folder:str|Path|None=None):
    """Takes in an nd file and creates a TileConfiguration file for imageJ stitching using positions encoded in the metadata"""
    nd_file = Path(nd_file)
    if nd_file.suffix != (".nd"):
        raise Exception("nd_file must be an .nd file")

    NDData = parseND(nd_file)
    if image_folder is None:
        image_folder = nd_file.parent;
    image_folder = Path(image_folder)
    posdata:dict[str,tuple[int,int]] = {}

    npos = int(NDData["NStagePositions"])
    print(npos)
    for i in range(1,npos+1):
        posname = NDData[f"Stage{i}"]
        fname = f"p_s{i}.TIF"

        #get position from file
        tiff = TiffFile(image_folder/fname);
        meta = tiff.metaseries_metadata;
        plane = meta['PlaneInfo'];
        pos = int(plane['stage-position-x']),int(plane['stage-position-y']); ##x,y; these are in **metamorph units**

        pixel_pos = [p/m*w for p,m,w in zip(pos,image_offset[mag],image_size)]
        print(pos,pixel_pos);
        posdata[fname] = tuple(pixel_pos)

    out_file = Path(out_file)
    if overwrite or not os.path.exists(out_file):
        os.makedirs(out_file.parent,exist_ok=True)
        with open(out_file,"w") as f:
            f.write(header)
            for fname,(x,y) in posdata.items():
                f.write(rowformat.format(name=fname,x=x,y=y))
    else:
        raise FileExistsError(out_file)
    
def prep_images_for_stitching(
        images_folder:str|Path,
        dest_folder:str|Path,
        mag:str="4x",
        nd_loc:str|Path="p.nd",
        stg_loc:str|Path|None=None,
        images:Iterable[int|str]|None=None):
    images_folder = Path(images_folder);
    
    dest_folder = images_folder/Path(dest_folder) #if dest_folder, nd_loc, or stg_loc are absolute, a/b -> b
    
    nd_loc = images_folder/Path(nd_loc)
    if nd_loc.suffix != (".nd"):
        raise Exception("nd_loc must be a .nd file")
    
    stg_loc = stg_loc if stg_loc is None else images_folder/Path(stg_loc)
    if stg_loc and stg_loc.suffix.lower() != (".stg"):
        raise Exception("stg_loc must be a .stg file")
    
    NDData = parseND(nd_loc)
    if stg_loc:
        _,_,STGData = parseSTG(stg_loc)
    else:
        STGData = None

    
    ## IMAGE DATA

    PosName = NewType("PosName",str)
    PosData = NamedTuple("PosData",(('pos',tuple[int,int]),('filename',Path)))
    posnames:list[PosName] = []
    imdata:dict[PosName,PosData] = {}

    npos = int(NDData["NStagePositions"])
    if images is None:
        images = range(1,npos+1)
    print(npos)
    for i in range(1,npos+1):
        posname = PosName(NDData[f"Stage{i}"])
        posnames.append(posname)

        fname = images_folder/f"p_s{i}.TIF"

        if STGData:
            pos = map(int,STGData[posname][:2]) ##x,y; these are in **metamorph units**
        else:
            try:
                #get position from file
                tiff = TiffFile(fname);
            except FileNotFoundError:
                continue
            meta = tiff.metaseries_metadata;
            assert meta is not None
            plane = meta['PlaneInfo'];
            pos = int(plane['stage-position-x']),int(plane['stage-position-y']); ##x,y; these are in **metamorph units**
        

        pixel_pos = [p/m*w for p,m,w in zip(pos,image_offset[mag],image_size)]

        imdata[posname] = PosData((pixel_pos[0],pixel_pos[1]),fname)

    
    ## Make Output folder
    
    outfile = dest_folder/"TileConfiguration.txt"
    if (not dest_folder.exists()):
        dest_folder.mkdir()

    with open(outfile,"w") as f:
        f.write(header)

    for im in tqdm(images):
        if isinstance(im,str):
            imname = PosName(im)
            if (imname not in posnames):
                raise ValueError(f"Image {im} not found")
        else:
            if im-1 not in range(len(posnames)):
                raise ValueError(f"Image index {im} out of range")
            imname = posnames[im-1]
        
        data = imdata[imname]

        shutil.copy(data.filename,dest_folder/data.filename.name)

        with open(outfile,"a") as f:
            f.write(rowformat.format(name=data.filename.name,x=data.pos[0],y=data.pos[1]))
        
def split_backtiling(
        images_folder:str|Path,
        dest_folder:str|Path,
        mag:str="4x",
        nd_loc:str|Path="p.nd",
        stg_loc:str|Path|None=None):
    """
    Takes an input folder of image files with positions named via backtiling convention, `backtile{orig_tile}_[...]`.
    For each original backtiling, creates a subfolder in the destination folder (named `orig_tile`) containing those backtiled images,
    each with its own TileConfiguration.txt
    """
    images_folder = Path(images_folder)

    dest_folder = images_folder/Path(dest_folder) #if dest_folder, nd_loc, or stg_loc are absolute, a/b -> b

    nd_loc = images_folder/Path(nd_loc)
    if nd_loc.suffix != (".nd"):
        raise Exception("nd_loc must be a .nd file")
    
    stg_loc = stg_loc if stg_loc is None else images_folder/Path(stg_loc)
    if stg_loc and stg_loc.suffix.lower() != (".stg"):
        raise Exception("stg_loc must be a .stg file")
    
    def get_orig(x:str):
        return x.split("backtile{")[1].split('}')[0]
    
    NDData = parseND(nd_loc)
    positions = [NDData[f"Stage{i}"] for i in range(1,1+int(NDData["NStagePositions"]))]
    assert len(positions) == len(set(positions)), "Duplicate Stage position names!" #shouldn't be possible from metamorph but /shrug

    positions.sort(key=get_orig)
    if not (dest_folder.exists()):
        dest_folder.mkdir()

    for name,images in itertools.groupby(positions, get_orig):

        sub = dest_folder/name
        sub.mkdir(exist_ok=True)
        
        images = list(images)

        #fill the subfolder with the selected images and TileConfiguration.txt
        prep_images_for_stitching(images_folder,sub,mag=mag,nd_loc=nd_loc,stg_loc=stg_loc,images=images) 

    



    


    

if __name__ == "__main__":

    src = r"C:\Users\James Bear\Documents\2024.6.14 OptoPLC FN+Peg Test\Fluorescent"
    out = "backtiles"
    split_backtiling(src,out,mag="10x")

    from IPython import embed; embed()
    # import typer
    # typer.run(create_position_list_nd)

        

        


