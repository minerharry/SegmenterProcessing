from abc import ABC, abstractmethod
from typing import Any, Iterable
import numpy as np

class Metadata(ABC):

    @abstractmethod
    def __getitem__(self,key:str)->Any: ...

    @property
    def PhysicalSize(self):
        return (self.PhysicalSizeX,self.PhysicalSizeY);

    @property
    @abstractmethod
    def PhysicalSizeY(self)->float: ...

    @property
    @abstractmethod
    def PhysicalSizeX(self)->float: ...


    @property
    def PhysicalSizeUnits(self):
        return (self.PhysicalSizeXUnit,self.PhysicalSizeYUnit);

    @property
    @abstractmethod
    def PhysicalSizeXUnit(self)->float: ...

    @property
    @abstractmethod
    def PhysicalSizeYUnit(self)->float: ...


    @property
    def Position(self):
        return (self.PositionX,self.PositionY);

    @property
    @abstractmethod
    def PositionX(self)->float: ...

    @property
    @abstractmethod
    def PositionY(self)->float: ...


    @property
    def PositionUnits(self):
        return (self.PositionXUnit,self.PositionYUnit);

    @property
    @abstractmethod
    def PositionXUnit(self)->float: ...

    @property
    @abstractmethod
    def PositionYUnit(self)->float: ...


def pixel_to_absolute_coordinates(meta:Metadata, pos:tuple[float,float]|Iterable[tuple[float,float]]=(0,0)):
    """Convert pixel coordinates to coordinates in image space using ome metadata. 
    If no pos is specified, returns coordinates of top left of image.
    Returns tuple ((x,y),(xunit,yunit)); xunit,yunit are strings"""

    pos = np.array(pos);

    pixel_scale = meta.PhysicalSize;
    pixel_units = meta.PhysicalSizeUnits

    image_pos = meta.Position
    image_units = meta.PositionUnits
    assert image_units == pixel_units, "Plane and Pixel units don't match!"

    result_pos = np.add(image_pos,np.multiply(pixel_scale,pos)) #result = image_pos + pixel_scale*pos. Should broadcast naturally?
    return result_pos,image_units

def get_pixel_scale(meta:Metadata):
    return meta.PhysicalSize,meta.PhysicalSizeUnits;

def absolute_coordinate_to_pixels(meta:Metadata, pos:tuple[float,float]|Iterable[tuple[float,float]]=(0,0)):
    """Convert pixel coordinates to coordinates in image space using ome metadata. 
    If no pos is specified, returns pixel value of (0,0) in absolute coordinates
    ASSUMES INPUT IS IN CORRECT UNITS!!!!
    Returns length-2 array [x,y] in pixelspace, or list of arrays [[x1,y1],[x2,y2],...] if input is list of points"""        
    
    pos = np.array(pos); #works with single point or list of points thanks to numpy broadcasting

    pixel_scale = meta.PhysicalSize
    pixel_units = meta.PhysicalSizeUnits

    image_pos = meta.Position
    image_units = meta.PositionUnits
    assert image_units == pixel_units, "Plane and Pixel units don't match!"

    result_pos = np.divide(np.subtract(pos, image_pos),pixel_scale) #result = (pos - image_pos) / pixel_scale. Should broadcast naturally?
    return result_pos

    