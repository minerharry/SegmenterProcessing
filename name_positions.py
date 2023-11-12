header = """# Define the number of dimensions we are working on
dim = 2

# Define the image coordinates
"""
rowformat = "{name}; ; ({x}, {y})\n"

image_offset = [21500,-16300] ##width,height of image in microscope units

image_size = [1344,1024] ##width, height of image in pixels

from ast import literal_eval
import os
from parsend import parseND
from pathlib import Path

def parseSTG(file:str):
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


def create_position_list(nd_file:str,pos_file:str,out_file:str,overwrite:bool=False):
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
        pixel_pos = [p/m*w for p,m,w in zip(pos,image_offset,image_size)]

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

if __name__ == "__main__":
    import typer
    typer.run(create_position_list)

        

        


