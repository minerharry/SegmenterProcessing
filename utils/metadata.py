from abc import ABC, abstractmethod
from os import PathLike
from typing import IO, Any, Iterable
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



from tifffile import TiffFile
class MetamorphMetadata(Metadata):
    def __init__(self,file:TiffFile|PathLike|IO[bytes]):
        self.file = file
        self.parse_metaseries_metadata()

    size:tuple[float,float];
    sizeunits: tuple[str,str];

    position:tuple[float,float];
    positionunits:tuple[str,str];

    def __getitem__(self, key: str) -> Any:
        raise AttributeError();

    @property
    def PhysicalSizeX(self):
        return self.size[0];
    @property
    def PhysicalSizeY(self):
        return self.size[1];

    @property
    def PhysicalSizeXUnit(self):
        return self.sizeunits[0];
    @property
    def PhysicalSizeYUnit(self):
        return self.sizeunits[1];

    @property
    def PositionX(self):
        return self.position[0];
    @property
    def PositionY(self):
        return self.position[1];

    @property
    def PositionXUnit(self):
        return self.positionunits[0];

    @property
    def PositionYUnit(self):
        return self.positionunits[1];

    def parse_metaseries_metadata(self):
        if not isinstance(self.file,TiffFile):
            self.file = TiffFile(self.file)

        meta = self.file.metaseries_metadata
        assert meta is not None
        assert meta['ApplicationName'] == 'MetaMorph'

        plane = meta['PlaneInfo']
        if plane['spatial-calibration-state'] == True:
            self.size = (plane['spatial-calibration-x'], plane['spatial-calibration-y'])
            self.sizeunits = (plane['spatial-calibration-units'],)*2
        else:
            self.size = (1,1)
            self.sizeunits = ('pixel','pixel')

        if self.sizeunits[0] != 'pixel':
            raise Exception(f"No calibration registered for {self.sizeunits[0]}->stage units conversion") 
        else:
            image_offset = {"4x":(21500,-16300),"10x":(8593,-6640),"20x":(4332,-3320)} ##width,height of image in microscope units (X,Y)
            image_size = (1344,1024) ##width, height of image in pixels

            mags = [k for k in image_offset if k in plane['_MagSetting_']]
            if len(mags) != 1:
                raise Exception(f"Cannot parse magnification for mag setting {plane['_MagSetting_']}")
            else:
                mag = mags[0]
            self.size = image_offset[mag][0]/image_size[0], image_offset[mag][1]/image_size[1]
            self.sizeunits = ("stage_units","stage_units")
        
        self.position = (plane['stage-position-x'],plane['stage-position-y']);
        self.positionunits = ('stage_units','stage_units')






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

    