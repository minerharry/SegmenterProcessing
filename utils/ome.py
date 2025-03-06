from abc import abstractmethod
from pathlib import Path
from typing import Any
from bs4 import BeautifulSoup
from tifffile import TiffFile
from utils.bftools import get_omexml_metadata
from utils.metadata import Metadata

class OMEMetadata(Metadata):
    def __init__(self,file:TiffFile|str|Path|None=None,xml:BeautifulSoup|Path|str|None=None) -> None:
        self.file = file;
        self.xml = xml;
        self.parse_omexml_metadata();

    def __getitem__(self, key: str) -> Any:
        raise AttributeError();

    size:tuple[float,float];
    sizeunits: tuple[str,str];

    position:tuple[float,float];
    positionunits:tuple[str,str];

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

    def _get_omexml_metadata(self)->BeautifulSoup:
        return parse_ome_metadata(file=self.file,xml=self.xml);

    def parse_omexml_metadata(self):
        xml = self._get_omexml_metadata()
        image = xml.find_all("Image")[0] #TODO: Multi-image support??
        pixels = image.find("Pixels")
        self.size = (float(pixels["PhysicalSizeX"]),float(pixels["PhysicalSizeY"]))
        self.sizeunits = (pixels["PhysicalSizeXUnit"],pixels["PhysicalSizeYUnit"])

        plane = pixels.find("Plane") #TODO: Multi-plane support??
        
        self.position = (float(plane["PositionX"]),float(plane["PositionY"]))
        self.positionunits = (plane["PositionXUnit"],plane["PositionYUnit"])
        assert self.sizeunits == self.positionunits, "Plane and Pixel units don't match! Unit conversions not supported"




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
    elif xml:
        if Path(xml).exists():
            with open(xml,'r') as f:
                return BeautifulSoup(f,features="lxml-xml")
        if isinstance(xml,str):
            return BeautifulSoup(xml,features="lxml-xml")
    raise Exception(f"Could not parse xml data for file {file} and xml {xml}");
