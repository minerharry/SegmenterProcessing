from abc import ABC
from dataclasses import dataclass
import itertools
from typing import Collection
import bioformats
from bioformats import ImageReader
import javabridge
import numpy as np
from utils.filegetter import afn, skip_cached_popups
from javabridge.wrappers import JWrapper
from javabridge.jutil import to_string,iterate_collection
import cv2 as cv

from utils.ome import OMEMetadata
javabridge.start_vm(class_path=bioformats.JARS)

## ROI Format from this snippet of the extractZeissROIs code
# roiTypes={'Rectangle','Circle','Ellipse','Polygon','Bezier'};
# roiDefinitionStrings.Rectangle{1}='Global Layer|Rectangle|Geometry|Top';
# roiDefinitionStrings.Rectangle{2}='Global Layer|Rectangle|Geometry|Left';
# roiDefinitionStrings.Rectangle{3}='Global Layer|Rectangle|Geometry|Width';
# roiDefinitionStrings.Rectangle{4}='Global Layer|Rectangle|Geometry|Height';
# roiDefinitionStrings.Circle{1}='Global Layer|Circle|Geometry|CenterX';
# roiDefinitionStrings.Circle{2}='Global Layer|Circle|Geometry|CenterY';
# roiDefinitionStrings.Circle{3}='Global Layer|Circle|Geometry|Radius';
# roiDefinitionStrings.Ellipse{1}='Global Layer|Ellipse|Geometry|CenterX';
# roiDefinitionStrings.Ellipse{2}='Global Layer|Ellipse|Geometry|CenterY';
# roiDefinitionStrings.Ellipse{3}='Global Layer|Ellipse|Geometry|RadiusX'; 
# roiDefinitionStrings.Ellipse{4}='Global Layer|Ellipse|Geometry|RadiusY'; 
# roiDefinitionStrings.Ellipse{5}='Global Layer|Ellipse|Rotation';
# roiDefinitionStrings.Polygon{1}='Global Layer|ClosedPolyline|Geometry|Points';
# roiDefinitionStrings.Bezier{1}='Global Layer|ClosedBezier|Geometry|Points';

##MODIFICATION: I don't know why (some weird parsing thing?) but it doesn't like "Global Layer". Maybe it's because it's in global metadata not series metadata, idk.
##but the upshot is that instead of global layer, here it shows up as just layer. I'm just checking for substring now, so it should be compatible with both.


class ROI(ABC):
    _meta_keys:list[str|tuple[str,str]] = [] #list of elements used to construct each ROI. Note that each one will be of the form A|B|C|D|E #{i} for each ROI i, so these are a prefix
    @classmethod
    def meta_keys(cls):
        return cls._meta_keys
    
    @classmethod
    def required_keys(cls):
        return [k for k in cls.meta_keys() if isinstance(k,str)]
    
    @classmethod
    def optional_keys(cls):
        return [k for k in cls.meta_keys() if isinstance(k,tuple)]
    
    @classmethod
    def create(cls,*args:str,**kwargs:str):
        return cls(*map(float,args),**{k:float(v) for k,v in kwargs.items()})

@dataclass
class RectROI(ROI):
    Top:float
    Left:float
    Width:float
    Height:float
    Rotation:float=0

    _meta_keys = ['Layer|Rectangle|Geometry|Top',
                  'Layer|Rectangle|Geometry|Left',
                  'Layer|Rectangle|Geometry|Width',
                  'Layer|Rectangle|Geometry|Height',
                  ("Rotation",'Layer|Rectangle|Rotation')]

@dataclass
class CircleROI(ROI):
    CenterX:float
    CenterY:float
    Radius:float

    _meta_keys = ['Layer|Circle|Geometry|CenterX',
                  'Layer|Circle|Geometry|CenterY',
                  'Layer|Circle|Geometry|Radius']
    
@dataclass
class EllipseROI(ROI):
    CenterX:float
    CenterY:float
    RadiusX:float
    RadiusY:float
    Rotation:float=0

    _meta_keys = ['Layer|Ellipse|Geometry|CenterX',
                  'Layer|Ellipse|Geometry|CenterY',
                  'Layer|Ellipse|Geometry|RadiusX',
                  'Layer|Ellipse|Geometry|RadiusY',
                  ("Rotation",'Layer|Ellipse|Rotation')]

@dataclass
class ClosedPolyLineROI(ROI):
    Points:list[tuple[float,float]]
    
    _meta_keys = ['Layer|ClosedPolyline|Geometry|Points']

    @classmethod
    def create(cls,*args:str,**kwargs:str):
        raise NotImplemented

@dataclass
class ClosedBezierROI(ROI):
    Points:list[tuple[float,float]]

    _meta_keys = ['Global Layer|ClosedBezier|Geometry|Points']

    @classmethod
    def create(cls,*args:str,**kwargs:str):
        raise NotImplemented

ROITypes:list[type[ROI]] = [RectROI,CircleROI,EllipseROI,ClosedPolyLineROI,ClosedBezierROI]

def groupkeys(keys:Collection[str])->dict[int,str]:
    res:dict[int,str] = {}
    for k in keys:
        if "#" in k:
            res[int(k.split("#")[1])] = k
        else:
            res[0] = k #this is the special case with no number. Explicitly checked for with extract ROIs.

    if len(res) < len(keys):
        raise ValueError("Duplicate key numbers found in collection!")

    return res       



def read_czi(image):
    with ImageReader(image) as reader:
        rd = JWrapper(reader.rdr.o)
        length = rd.getSizeT()

        ims = []

        for i in range(length):
            ims.append(reader.read(t=i))
        
        return np.array(ims)
    


def extract_ROIs(image):
    ###Some notes for future people.
    ## This file is based off of extractZeissRois, a function in **MATLAB** because of COURSE it is, which attempts to extract zeiss ROIs from image metadata.
    ## extractZeiss ROIs uses the hashtable returned by bfopen in matlab (specifcially, where bfimage = bfopen(path), it uses bfimage{2}). By looking at the code of bfopen,
    ## we can see that bfimage{2} equates to reader.getSeriesMetadata(). We can do that here!
    ## OK so for some reason it's not in the series metadata... BUT it is in the global metadata, so whatever

    with ImageReader(image) as reader:
        # from IPython import embed; embed()

        meta = JWrapper(reader.rdr.getGlobalMetadata())

        #but: meta is now a javabridge object which is a fucking PAIN IN THE ASS to deal with. Thank god JWrapper exists...
        keys:list[str] = list(map(to_string, iterate_collection(meta.keySet().o)))

        # from IPython import embed; embed()

        ROI_matches:list[ROI] = []

        # from IPython import embed; embed()

        for ROI_type in ROITypes:
            requireds = [
                [k for k in keys if key in k] #love python
                for key in ROI_type.required_keys()
            ]

            optionals = [
                [k for k in keys if key[1] in k] #love python
                for key in ROI_type.optional_keys()
            ]
            # results[0].append("hehe")
            assert all([a==b for a,b in itertools.pairwise(map(len,requireds))]),f"Ragged size of keys found in metadata for ROI {ROI_type}: {dict(zip(ROI_type.meta_keys(),results))}"
            if len(requireds[0]) > 0:
                numRois = len(requireds[0])

                requireds = list(map(groupkeys,requireds))
                optionals = list(map(groupkeys,optionals))

                #so the logic here is a little weird, but I want to be as permissive as possible. This will attempt to fetch each ROI in order, from #1 to #numRois. sometimes, however,
                #the first roi doesn't get a #1. I don't want to accidentally conflate anything so I label anything without a number as 0 in groupkeys. Thus, I must check both 0 and 1
                #for the potential first Roi.
                firsted = False
                for i in range(numRois+1):
                    if firsted and i == 1:
                        continue
                    try:
                        req = [meta.get(r[i]) for r in requireds]
                    except KeyError as e:
                        if i > 1 or (i == 1 and not firsted):
                            raise e
                        else:
                            continue
                    
                    opt = dict([(n[0],meta.get(o[i])) for n,o in zip(ROI_type.optional_keys(),optionals) if i in o])

                    ROI_matches.append(ROI_type.create(*req,**opt))

                    firsted = True
                    
            else:
                continue
        
        return ROI_matches


def draw_ROI(im:np.ndarray,r:ROI,thickness:float,color:tuple[float,float,float]):
    if isinstance(r,RectROI):
        if r.Rotation == 0:
            im = cv.rectangle(im,(int(r.Left),int(r.Top)),(int(r.Left+r.Width),int(r.Top+r.Height)),color,thickness=thickness)
        else:
            r = cv.RotatedRect((r.Left+r.Width/2,r.Top + r.Height/2),(r.Width,r.Height),r.Rotation)
            pts = cv.boxPoints(r)
            im = cv.drawContours(im,[pts],0,color,thickness=thickness)
    elif isinstance(r,EllipseROI):
        im = cv.ellipse(im,(int(r.CenterX),int(r.CenterY)),(int(r.RadiusX),int(r.RadiusY)),r.Rotation,0,360,color,thickness=thickness)
    elif isinstance(r,CircleROI):
        im = cv.circle(im,(r.CenterX,r.CenterY),int(r.Radius),color,thickness=thickness)
    else:
        raise NotImplemented

    return im
        
