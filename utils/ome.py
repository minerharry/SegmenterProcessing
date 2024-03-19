import os
from pathlib import Path
from typing import Iterable
from bs4 import BeautifulSoup
import numpy as np
from tifffile import TiffFile

from utils.bftools import get_omexml_metadata

def parse_ome_metadata(file:TiffFile|str|Path|None=None,xml:BeautifulSoup|Path|str|None=None):
    # print(file)
    if isinstance(xml,BeautifulSoup):
        return xml;
    if xml is None and file is not None:
        if isinstance(file,TiffFile):
            xml = file.ome_metadata
            if (xml is None):
                raise Exception(f"File {file} has no OME-xml metadata!")
            xml.replace("OME:","") #hack to make tifffile ome_metadata compatible with bftools ome_metadata
        else:
            xml = get_omexml_metadata(str(file));
            # print(xmlstring)
    if xml:
        if Path(xml).exists():
            with open(xml,'r') as f:
                return BeautifulSoup(f,features="lxml-xml")
        if isinstance(xml,str):
            return BeautifulSoup(xml,features="lxml-xml")
    raise Exception(f"Could not parse xml data for file {file} and xml {xml}");


def pixel_to_absolute_coordinates(pos:tuple[float,float]|Iterable[tuple[float,float]]=(0,0),meta:BeautifulSoup|Path|str|None=None,file:TiffFile|str|Path|None=None):
    """Convert pixel coordinates to coordinates in image space using ome metadata. 
    If no pos is specified, returns coordinates of top left of image.
    Returns tuple ((x,y),(xunit,yunit)); xunit,yunit are strings"""
    meta = parse_ome_metadata(file=file,xml=meta);
        
    image = meta.find_all("Image")[0] #TODO: Multi-image support??
    pixels = image.find("Pixels")
    pixel_scale = [float(pixels["PhysicalSizeX"]),float(pixels["PhysicalSizeY"])]
    pixel_units = [pixels["PhysicalSizeXUnit"],pixels["PhysicalSizeYUnit"]]

    plane = pixels.find("Plane") #TODO: Multi-plane support??
    
    image_pos = [float(plane["PositionX"]),float(plane["PositionY"])]
    image_units = [plane["PositionXUnit"],plane["PositionYUnit"]]
    assert image_units == pixel_units, "Plane and Pixel units don't match!"

    result_pos = np.add(image_pos,np.multiply(pixel_scale,pos)) #result = image_pos + pixel_scale*pos. Should broadcast naturally?
    return result_pos,image_units

def get_pixel_scale(meta:BeautifulSoup|Path|str|None=None,file:TiffFile|str|Path|None=None):
    meta = parse_ome_metadata(file=file,xml=meta);
        
    image = meta.find_all("Image")[0] #TODO: Multi-image support??
    pixels = image.find("Pixels")
    pixel_scale = (float(pixels["PhysicalSizeX"]),float(pixels["PhysicalSizeY"]))
    pixel_units = (pixels["PhysicalSizeXUnit"],pixels["PhysicalSizeYUnit"])
    return pixel_scale,pixel_units

def absolute_coordinate_to_pixels(pos:tuple[float,float]|Iterable[tuple[float,float]]=(0,0),meta:BeautifulSoup|Path|str|None=None,file:TiffFile|str|Path|None=None):
    """Convert pixel coordinates to coordinates in image space using ome metadata. 
    If no pos is specified, returns pixel value of (0,0) in absolute coordinates
    ASSUMES INPUT IS IN CORRECT UNITS!!!!
    Returns length-2 array [x,y] in pixelspace, or list of arrays [[x1,y1],[x2,y2],...] if input is list of points"""
    meta = parse_ome_metadata(file=file,xml=meta);
        
    pos = np.array(pos); #works with single point or list of points thanks to numpy broadcasting
        
    image = meta.find_all("Image")[0] #TODO: Multi-image support??
    pixels = image.find("Pixels")
    pixel_scale = np.array([float(pixels["PhysicalSizeX"]),float(pixels["PhysicalSizeY"])])
    pixel_units = [pixels["PhysicalSizeXUnit"],pixels["PhysicalSizeYUnit"]]

    plane = pixels.find("Plane") #TODO: Multi-plane support??
    
    image_pos = np.array([float(plane["PositionX"]),float(plane["PositionY"])])
    image_units = [plane["PositionXUnit"],plane["PositionYUnit"]]
    assert image_units == pixel_units, "Plane and Pixel units don't match!"

    result_pos = np.divide(np.subtract(pos, image_pos),pixel_scale) #result = (pos - image_pos) / pixel_scale. Should broadcast naturally?
    return result_pos

    