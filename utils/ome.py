from pathlib import Path
from typing import Iterable
from bs4 import BeautifulSoup
import numpy as np
from tifffile import TiffFile

def parse_ome_metadata(file:TiffFile|str|Path=None,xmlstring:str=None):
    if file:
        if not isinstance(file,TiffFile):
            file = TiffFile(file)
        xmlstring = file.ome_metadata
    if xmlstring:
        return BeautifulSoup(xmlstring,features="lxml-xml")
    raise Exception()


def pixel_to_absolute_coordinates(pos:tuple[float,float]|Iterable[tuple[float,float]]=(0,0),meta:BeautifulSoup|str=None,file:TiffFile|str|Path=None):
    """Convert pixel coordinates to coordinates in image space using ome metadata. 
    If no pos is specified, returns coordinates of top left of image.
    Returns tuple ((x,y),(xunit,yunit)); xunit,yunit are strings"""
    if isinstance(meta,str):
        meta = parse_ome_metadata(xmlstring=meta)
    if meta is None:
        if file:
            meta = parse_ome_metadata(file=file)
        else:
            raise Exception
        
    image = meta.find_all("OME:Image")[0] #TODO: Multi-image support??
    pixels = image.find("OME:Pixels")
    pixel_scale = [float(pixels["PhysicalSizeX"]),float(pixels["PhysicalSizeY"])]
    pixel_units = [pixels["PhysicalSizeXUnit"],pixels["PhysicalSizeYUnit"]]

    plane = pixels.find("OME:Plane") #TODO: Multi-plane support??
    
    image_pos = [float(plane["PositionX"]),float(plane["PositionY"])]
    image_units = [plane["PositionXUnit"],plane["PositionYUnit"]]
    assert image_units == pixel_units, "Plane and Pixel units don't match!"

    result_pos = np.add(image_pos,np.multiply(pixel_scale,pos)) #result = image_pos + pixel_scale*pos. Should broadcast naturally?
    return result_pos,image_units




    