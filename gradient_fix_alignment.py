from ast import parse
from copy import copy
import itertools
import sys
import json_tricks as json
from math import ceil
import os
from pathlib import Path
import shutil

from typing import Any, Callable, DefaultDict, Iterable, Literal, NamedTuple, overload
import warnings
import inquirer
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.image import AxesImage
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Polygon
from matplotlib.text import Text
from pyparsing import Sequence
# from imageio.v3 import imread
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from tifffile import TiffFile
from skimage.transform import rescale,resize
from mpl_interactions import zoom_factory,panhandler
from mpl_point_clicker import clicker
from json_tricks import dump as jdump, load as jload
import tifffile
# import xtiff #tifffile wrapper specifically for multichannel / stacks
# from utils import inquire
# from utils.bftools import get_omexml_metadata
from utils.metadata import Metadata, MetamorphMetadata, absolute_coordinate_to_pixels, pixel_to_absolute_coordinates,get_pixel_scale

import numpy as np
import cv2 as cv
from cv2.typing import MatLike
from tqdm import tqdm
from nptyping import NDArray,Shape,Float

from skimage.io import imread,imsave

from utils.ome import OMEMetadata

## ensures the clicker can only have (at most) one point. it can also have zero!
def singlify_clicker(c:clicker):
    def on_point_added(pos,cls):
        d = dict(c._positions)
        d.update({cls:[pos]})
        c.set_positions(d)
        c._update_points(cls)
    c.on_point_added(on_point_added)
    return c
    
def has_point(c:clicker,cls:Any=None):
    if not cls:
        return all([len(p) > 0 for p in list(c._positions.values())])
    else:
        return len(c._positions[cls]) > 0
    
class CalibData(NamedTuple):
    phase_pixels:tuple[float,float]
    phase_absolute:tuple[float,float]
    tirf_pixels:tuple[float,float]
    tirf_absolute:tuple[float,float]

def add_confirmsave[**P,R](fun:Callable[P,R]) -> Callable[P,R]:
    def f(self,*args,save:bool=False,**kwargs):
        r = fun(*args,**kwargs)
        if save:
            self.save()
        return r
    return f #type:ignore
        

def clear_cache_on_call[T:Callable](fun:T)->T:
    def f(self,*args,**kwargs):
        res = fun(self,*args,**kwargs)
        self.transform_cache = {}
        return res
    return f 

#dum lol
class BiTuple[K,V](tuple[K,V]):
    @overload
    def other(self,t:K)->V: ...
    @overload
    def other(self,t:V)->K: ...
    def other(self,t:K|V)->K|V:
        return self[self.index(t)-1]

ImOpt = BiTuple(('tirf','phase'))

class PhaseTIRFCalibration:
    data:dict[str,list[CalibData]] #Default: indexed by tirf filename (will look for it in the parent folder; Calibration/name = calibration images, Images/name = regular images)
    def __init__(self,file:str|Path,method:Literal["affine","perspective"]="affine"):
        self.method = method.lower()
        if self.method == "homography":
            self.method = "perspective"
        if method == "perspective":
            warnings.warn("Perspective transform is Very Bad(TM)! Please use affine instead")
        assert method in ["affine","perspective"]
        self.file = Path(file)
        if not self.file.exists():
            self.file.touch()
            self.data = DefaultDict(lambda: [])
            self.save()
        else:
            with open(self.file,'r') as f:
                self.data = jload(f)
            self.data = DefaultDict(lambda: [],{k:[CalibData(*c) for c in v] for k,v in self.data.items()})

        self.transform_cache:dict[str,tuple[MatLike,tuple[CalibData,...],tuple[CalibData,...]]] = {}

   

    def __getitem__(self,name:str):
        return self.get_calibration_points(name)
    
    def __setitem__(self,name:str,points:Iterable[CalibData]):
        return self.set_calibration_points(name,points)

    def get_calibration_points(self,name:str):
        return self.data[name]

    @clear_cache_on_call
    def add_calibration_point(self,name:str,point:CalibData):
        self.data[name].append(point)

    @clear_cache_on_call
    def remove_calibration_point(self,name:str,point:CalibData):
        self.data[name].remove(point)

    @clear_cache_on_call
    def remove_calib_image(self,name:str):
        del self.data[name]

    @clear_cache_on_call
    def set_calibration_points(self,name:str,points:Iterable[CalibData]):
        self.data[name] = list(points)

    @clear_cache_on_call
    def clear_calibration_points(self,name:str|None=None,confirm_clearall:bool=True): #if None, will clear all data
        if name is None:
            if confirm_clearall:
                if inquirer.confirm("Clear *all* calibration points in calibration? This action cannot be undone!"):
                    self.data = {}
            else:
                self.data = {}

    def save(self):
        for d,k in list(self.data.items()):
            if len(k) == 0:
                del self.data[d]
        with open(self.file,'w') as f:
            jdump(dict(self.data),f)


    def _get_transformation(self,to:Literal['phase','tirf']="phase"):
        if to not in ImOpt:
            raise ValueError()
        
        if to in self.transform_cache:
            if not isinstance(self.transform_cache[to],tuple):
                del self.transform_cache[to]
            else:
                return self.transform_cache[to]
        
        dst = f"{to}_absolute"
        src = f"{ImOpt.other(to)}_absolute"

        items = list(self.data.items())
            
        calib_points:list[tuple[tuple[float,float],tuple[float,float]]] = sum([
            [(getattr(c,dst),getattr(c,src)) for c in l]
            for k,l in items
        ],[])
        order:list[tuple[str,int]] = sum([
            [(k,i) for i,c in enumerate(l)]
            for k,l in items
        ],[])

        if (len(calib_points) < 3): #requires >= 3 points to define a transformation
            return None,None,None


        cpoints = np.array(calib_points)
        method = {"perspective":cv.findHomography,"affine":cv.estimateAffine2D}[self.method]
        M, inliers = method(cpoints[:,1].reshape(-1, 1, 2).copy(),cpoints[:,0].reshape(-1, 1, 2).copy(),method=cv.LMEDS)
        matchesMask:Iterable[int] = np.where(inliers)[0]
        full_matrix = np.array([M[0],M[1],[0,0,1]]) if M.shape == (2,3) else np.array(M)
        inlier_points = tuple(self.data[k][i] for (k,i) in (order[m] for m in matchesMask))
        outlier_points = tuple(self.data[k][i] for (k,i) in (order[m] for m in range(len(order)) if m not in matchesMask))

        self.transform_cache[to] = full_matrix,inlier_points,outlier_points
        return full_matrix,inlier_points,outlier_points
    
    def get_transformation_matrix(self,to:Literal['phase','tirf']='phase'):
        return self._get_transformation(to)[0]
    
    def get_inliers(self,to:Literal['phase','tirf']):
        return self._get_transformation(to)[1]
    
    def get_outliers(self,to:Literal['phase','tirf']):
        return self._get_transformation(to)[2]



    def transform_points(self,to:Literal['phase','tirf'],points:np.ndarray):
        M = self.get_transformation_matrix(to)
        pts = np.concatenate([points,[[1]]*points.shape[0]],axis=1)
        dst = np.array([np.matmul(M,pt)[:2] for pt in pts])
        return dst

    ### ONLY FUNCTION WHICH RELIES ON ARBITRARY PIXEL -> COORDINATE TRANSFORMAITON
    def transform_pixels(self,to:Literal['phase','tirf']='phase',*points:tuple[float,float],meta:Metadata)->Sequence[NDArray[Shape["2"],Float]]:
        if to == "phase":
            abs_points,units = pixel_to_absolute_coordinates(meta,points)
        else:
            abs_points = np.array(points) #phase coords = phase pixels
        
        # print(abs_points.shape)
        dst = self.transform_points(to,abs_points)
        # print(dst)

        if to == "tirf":
            dst = absolute_coordinate_to_pixels(meta,dst)
        # print(dst)

        return dst
        



    

def main(force_calibration:bool=False,continue_calibration:bool=False):

    ###TIRF/PHASE CALIBRATION PATHS
    if False:
        parent_folder = Path(r"C:\Users\Harrison Truscott\OneDrive - University of North Carolina at Chapel Hill\Bear Lab\optotaxis calibration\data\tirf calibration local\2024.2.23 OptoITSN Fix Test 1")
        parent_folder = parent_folder/"3 notches"

        phase_parent_folder = parent_folder
        phase_folder = phase_parent_folder/"Phase"
        gradient_folder = phase_parent_folder/"Gradient"

        phase_tiled = phase_folder/"fused.tif"
        gradient_tiled = gradient_folder/"fused.tif"

        TIRF_folder = parent_folder/("TIRF")
        calibration_folder = TIRF_folder/"Calibration"
        tirf_images = TIRF_folder/"Images"
        # tirf_images = TIRF_folder/"Export_test"

        calib_file = (calibration_folder/"calibration.json")

    ###4x/20x CALIBRATION PATHS
    if True:
        parent_folder = Path.home()/(r"OneDrive - University of North Carolina at Chapel Hill\Bear Lab\optotaxis calibration\data\Gradient Analysis\2025.2.16 OptoPLC S345F Steep Gradient Photoactivation")

        phase_parent_folder = parent_folder
        phase_folder = phase_parent_folder/"Phase Post 4x"
        gradient_folder = phase_parent_folder/"Gradient Post 4x"

        phase_tiled = phase_folder/"tiling.tif"
        gradient_tiled = gradient_folder/"tiling.tif"

        TIRF_folder = parent_folder
        calibration_folder = TIRF_folder/"Calibration"
        tirf_images = TIRF_folder
        # tirf_images = TIRF_folder/"Phase Post 20x"
        # tirf_images = TIRF_folder/"Export_test"

        calib_file = (calibration_folder/"calibration.json")

    ### Goal: Align Phase/gradient images with TIRF images
    ## This can be done with some (>3) number of known points in common between the two images
    ## the starting point is 5 indexing points, which are part of the phase tile and have been
    ## individually imaged be TIRF (drawn on the bottom of the cover slip) 
    ## However, the alignment will not be perfect. 
    ## The optimal algorithm would be iterative: given some initial points, calculate the alignment,
    ## and look at one of the cells to align. If it's imperfect, align the cells, then recalculate. 
    ## Eventually, this method should give quite a few reference points. However, it is almost certainly overkill.
    ## For now, manually entered x,y coordinates in the phase image will be referenced with each of the calibration images



    def edit_calibration_interactive(calib_image:str|Path):
        ##GOAL:
        # - open phase image
        # - open calibration image
        # - allow user to place some (equal!) number of points between them
        # -- addition, removal, and editing
        # - allow user to finish calibration
        raise NotImplementedError("Interaction is too hard")


    def edit_calibration_cmdline(cell_num:int):
        ##GOAL
        # - open phase image
        # - open calibratio nimage
        # - allow user to enter pairs of pixel coordinates: one from phase, one from TIRF
        # -- addition, removal, and editing
        # - allow user to finish calibration


        fig,(pax,tax) = plt.subplots(1,2,num="Calibration")
        #resize figure to full width and 80% height


        pax:Axes;tax:Axes
        pax.cla()
        pax.set_title("Phase")
        phase = imread(phase_tiled)
        grad = imread(gradient_tiled)
        tirf:np.ndarray = None
        phase_im = pax.imshow(phase)
        grad_im = pax.imshow(grad)
        grad_im.set_visible(False)


        calib_image:str|Path = ""
        calib_meta:Metadata = None
        calib_name:str = ""
        tirf_im:AxesImage|None = None

        def get_tirf_image(name:str|Path):
            calib_image = Path(name)
            ##make calib image absolute
            if not calib_image.is_absolute() and (TIRF_folder/calib_image).exists():
                #look in calibration folder before local/absolute
                calib_image = TIRF_folder/calib_image
            return calib_image;

        def get_meta(file:str|Path):
            file = Path(file);
            if (cm := file.with_suffix(".xml")).exists():
                return OMEMetadata(file=file,xml=cm);
            else:
                t = TiffFile(file)
                if (t.metaseries_metadata):
                    try:
                        return MetamorphMetadata(t)
                    except Exception as e:
                        warnings.warn(e)
                        pass
                elif t.ome_metadata:
                    try:
                        return OMEMetadata(file);
                    except:
                        pass
            return OMEMetadata(file);
    
        def get_calib_name(file:str|Path):
            file = Path(file);
            try:
                #as_posix is platform-agnostic and easy
                calib_name = (file.relative_to(TIRF_folder)).as_posix()
            except ValueError as v:
                print(v)
                calib_name = (file).as_posix() 
            return calib_name


        def switch_image(cimage:str|Path,zoom:bool=True,custom_name:str|None=None,custom_title:str|None=None):
            clear_test_points()
            clear_bboxes('tirf')
            nonlocal calib_image,calib_meta,calib_name,tirf_im,tirf
            
            calib_image = get_tirf_image(cimage);
            
            calib_meta = get_meta(calib_image)
            
            ##make calib name relative to parent folder if possible
            if custom_name:
                print("using custom calibration key:",custom_name)
                calib_name = custom_name
            else:
                calib_name = get_calib_name(calib_image)
                
            tirf = imread(calib_image)
            
            if tirf_im:
                tirf_im.remove()
            tax.set_title(custom_title if custom_title else calib_name)
            tirf_im = tax.imshow(tirf,cmap="Greys")
            tirf_clicker.set_positions({0:[]})
            tirf_clicker._update_points(0)
            plot_calibdata()

            if zoom:
                zoom_bbox()
                    

        def get_calib_images():
            return list(calibration_folder.glob("*.tif"))
        
        def get_bright_cellnums():
            bcells = (TIRF_folder/"brightcells.txt");
            if not bcells.exists():
                raise FileNotFoundError("brightcells.txt does not exist in tirf folder");
            with open(bcells,"r") as f:
                return list(set([int(i.strip()) for i in f.readlines() if i.strip() != ''])); #only return unique cellnums
        
        tirf_type_num_map:dict[Literal["bf","tirf","epi"],str] = {"bf":"01","tirf":"02","epi":"03"}
        def get_tirf_images(type:Literal["bf","tirf","epi"]="tirf"):
            # return list((tirf_images/type).glob("*tif")) #old folder structure
            res = [];
            for dir in tirf_images.glob("cell*"):
                num = dir.parts[-1].split("cell")[1];
                res.append(dir/f"cell{num}_{tirf_type_num_map[type]}.tif");
            return res;
    
        def get_all_tirf_nums():
            res:list[int] = [];
            for f in tirf_images.glob("p_w1Cy5_s*_t*.TIF"):
                f = str.removeprefix(f.name,"p_w1Cy5_s").removesuffix(".TIF")
                pos,frame = f.split("_t")
                res.append((pos,frame))
            return res
            # for dir in tirf_images.glob("cell*"):
            #     num = dir.parts[-1].split("cell")[1];
            #     res.append(int(num));
            # return res;

        # def get_all_tirf_images(): #silly name, but returns list of dicts associated
        #     res:list[dict[Literal["bf","tirf","epi"],Path]] = [];
        #     for dir in tirf_images.glob("cell*"):
        #         num = dir.parts[-1].split("cell")[1];
        #         res.append({type:dir/f"cell{num}_{tnum}.tif" for type,tnum in tirf_type_num_map.items()});
        #     return res

        def get_cell_num(num:int,type:Literal["bf","tirf","epi"]="tirf"):
            # return tirf_images/type/f"cell{num}.tif"
        
            # return tirf_images/f"cell{num}"/f"cell{num}_{tirf_type_num_map[type]}.tif"
        
            key = num
            if type == "tirf":
                if not isinstance(key,Sequence):
                    key = (key,97)
                return tirf_images/f"Multiwave/p_w1Cy5_s{key[0]}_t{key[1]}.TIF"
            else:
                if isinstance(key,Sequence):
                    key = key[0]
                return tirf_images/f"Phase Post 20x/p_s{num}.tif"
        
        def switch_cell_num(num:int,type:Literal["bf","tirf","epi"]="bf",**kwargs):
            print(f"Images/cell{num}")
            print(num)
            return switch_image(get_cell_num(num,type),**kwargs,custom_name=f"Images/cell{num}",custom_title=f"cell{num} {type}")

        ppoints = pax.scatter([],[],color='orange')
        tpoints = tax.scatter([],[],color='orange')

        markers = itertools.cycle(map(MarkerStyle,['o','v','^','<','>','*','s','_','.','1','X']))

        points:list[tuple[tuple[float,float],tuple[float,float]]] = [] #tuple [phase, tirf]
        texts:list[tuple[Text,Text]] = []


        ## Add interactive clickers for convenience
        tirf_clicker = singlify_clicker(clicker(tax,[0],markers='+',colors=['red']))
        phase_clicker = singlify_clicker(clicker(pax,[0],markers='+',colors=['red']))
        #clickers add a legend which is really not useful for one class
        pax.get_legend().remove()
        tax.get_legend().remove()


        @overload
        def get_click_curr(src:Literal['tirf','phase'],allow_none=True)->tuple[float,float]|None: ...
        @overload
        def get_click_curr(src:Literal['tirf','phase'],allow_none=False)->tuple[float,float]: ...
        def get_click_curr(src:Literal['tirf','phase'],allow_none:bool=True):
            click = {"tirf":tirf_clicker,"phase":phase_clicker}[src]
            poss = click.get_positions()[0]
            if len(poss) == 0:
                if allow_none:
                    return None
                else:
                    raise ValueError(f"Image {src} has no point selected, cannot use value \"current\". Please select a point or input a value manually")
            else:
                return poss[0]


        def update_scatter():
            nonlocal markers, ppoints, tpoints
            # print(points)
            if len(points) == 0:
                #clear points
                ppoints.remove()
                ppoints = pax.scatter([],[],color='orange')
                tpoints.remove()
                tpoints = tax.scatter([],[],color='orange')
            else:
                array = np.array(points)
                markers,m1 = itertools.tee(markers)
                paths = [m.get_path() for m in itertools.islice(m1,len(points))]
                sizes = [4]*len(paths)
                ppoints.set_offsets(array[:,0])
                ppoints.set_paths(paths)
                ppoints.set_sizes(sizes)
                tpoints.set_offsets(array[:,1])
                tpoints.set_paths(paths)
                tpoints.set_sizes(sizes)
            fig.canvas.draw_idle()


        def make_calibdata(*points:tuple[tuple[float,float],tuple[float,float]]):
            res:list[CalibData] = []
            tirf_pos,_ = pixel_to_absolute_coordinates(calib_meta,[p[1] for p in points])
            for (pp,tp),ta in zip(points,tirf_pos):
                res.append(CalibData(tuple(pp),tuple(pp),tuple(tp),tuple(ta)))
            return res

        def add_point(phase:tuple[float,float]|Literal['current']='current',tirf:tuple[float,float]|Literal['current']='current'):
            """Add new calibration point. Keyword arguments phase and tirf can be given as (x,y) coordinates, or if the default value of "current" is used,
            the program will use the last point clicked on the corresponding image. 
            """
            if phase == 'current':
                phase = get_click_curr('phase',allow_none=False)
            if tirf == 'current':
                tirf = get_click_curr('tirf',allow_none=False)
            # print(points)
            points.append((phase,tirf))
            # print(points)
            pt = pax.text(*phase,str(len(points)))
            tt = tax.text(*tirf,str(len(points)))
            texts.append((pt,tt))
            update_scatter()
            Calibration.add_calibration_point(calib_name,make_calibdata(points[-1])[0])
            # print(points)


        def remove_point(num:int):
            """Remove calibration point {num}."""
            idx = num-1
            p = points[idx]
            del points[idx]
            texts[idx][0].remove()
            texts[idx][1].remove()
            texts.remove(texts[idx])
            for i in range(idx,len(texts)):
                (pt,tt) = texts[i]
                pt.set_text(str(i+1))
                tt.set_text(str(i+1))
            update_scatter()
            Calibration.remove_calibration_point(calib_name,make_calibdata(p)[0])
        
        def edit_point(num:int,phase:tuple[float,float]|Literal['current']|None=None,tirf:tuple[float,float]|Literal['current']|None=None):
            """Edit existing calibration point number {num}. Like add_point, {phase} and {tirf} can be provided as coordinates or with "current" to use the currently selected points.
            If phase or tirf is None (default), that point won't be edited. At least one of phase or tirf must be not None."""
            if phase == 'current':
                phase = tuple(get_click_curr('phase'))
            if tirf == 'current':
                tirf = tuple(get_click_curr('tirf'))
            if tirf is None and phase is None:
                return
            idx = num-1
            p = (phase or points[idx][0],tirf or points[idx][1])
            points[idx] = p
            texts[idx][0].set_position(p[0])
            texts[idx][1].set_position(p[1])
            update_scatter()
            cp = copy(Calibration.get_calibration_points(calib_name))
            cp[idx] = make_calibdata(p)[0]
            Calibration.set_calibration_points(calib_name,cp)

        #should only be used in cases of loading or error
        def remake_texts():
            """Remake each text label. Should only be used in cases of loading or error."""
            nonlocal texts
            [(t[0].remove(),t[1].remove()) for t in texts]
            texts = [(pax.text(*ppos,str(i+1)),tax.text(*tpos,str(i+1))) for i,(ppos,tpos) in enumerate(points)]
            fig.canvas.draw_idle()


        def plot_calibdata():
            nonlocal points
            points = [(c.phase_pixels,c.tirf_pixels) for c in Calibration[calib_name]]
            update_scatter()
            remake_texts()


        def test_phase_point(*point:tuple[float,float]):
            if len(point) == 0:
                point = tuple([get_click_curr('phase',allow_none=False)])
            return _test_point(point,source='phase')
        
        def test_tirf_point(*point:tuple[float,float]):
            if len(point) == 0:
                point = tuple([get_click_curr('tirf',allow_none=False)])
            return _test_point(point,source='tirf')


        test_points:list[tuple[PathCollection,PathCollection]] = []

        def _test_point(points:Iterable[tuple[float,float]],source:Literal['tirf','phase']):
            dest = ImOpt.other(source)
            transformed = Calibration.transform_pixels(dest,*points,meta=calib_meta)
            axes = {"phase":pax,"tirf":tax}
            for p,t in zip(points,transformed):
                spoint = axes[source].scatter(p[0],p[1],color='blue',marker='x')
                dpoint = axes[dest].scatter(t[0],t[1],color='blue',marker='x')
                test_points.append((spoint,dpoint))
            fig.canvas.draw_idle()
        
        def clear_test_points():
            nonlocal test_points
            for spoint,dpoint in test_points:
                spoint.remove()
                dpoint.remove()
            pax.relim();
            tax.relim();
            test_points = []
            fig.canvas.draw_idle()

        bboxes:dict[Literal['tirf','phase'],list[tuple[Polygon,Text]]] = {'tirf':[],'phase':[]}
        
        def get_bbox(dest:Literal['tirf','phase']='phase',sourceim:np.ndarray|tuple[int,int]|None=None,meta:Metadata|None=None):
            if meta is None: meta = calib_meta;
            if Calibration.get_transformation_matrix(dest) is None: return None #not enough points for transformation
            source = ImOpt.other(dest)
            shape:tuple[int,...]
            if sourceim is None:
                shape = {'phase':phase,'tirf':tirf}[source].shape;
            elif isinstance(sourceim,tuple):
                shape = sourceim;
            else:
                shape = sourceim.shape;
            corners = [(0,0),(shape[1]-1,0),(shape[1]-1,shape[0]-1),(0,shape[0]-1)]
            transformed = Calibration.transform_pixels(dest,*corners,meta=meta)
            return Polygon(transformed,closed=False);

        def draw_bbox(dest:Literal['tirf','phase']='phase',color="yellow",thickness="3",image=None,label:str|None = None,do_label=True,
                      meta:Metadata|None=None,sourceim:np.ndarray|tuple[int,int]|None=None):
            if (meta is None): meta = calib_meta;
            p = get_bbox(dest,sourceim=sourceim,meta=meta)
            if p is None:
                return (None,None)
            p = Polygon(p.get_xy(),closed=True,color=color,linewidth=3,fill=False)
            destax = {'phase':pax,'tirf':tax}[dest]
            destax.add_artist(p)
            label = label or calib_name
            if do_label:
                t = destax.annotate(label,p.get_xy()[0])
            else:
                t = None;
            bboxes[dest].append((p,t))
            fig.canvas.draw_idle()
            return (p,t)
        
        def draw_all_bboxes(dest:Literal['tirf','phase'],images:list[Path|str|int]|None=None,**kwargs):
            if images is None:
                images = list(get_all_tirf_nums());
            res = []
            for im in tqdm(get_tirf_images()):
                try:
                    p = Path(im)
                    is_path = True
                except:
                    is_path = False
                if (not is_path):
                    impath = get_tirf_image(get_cell_num(im,'tirf'));
                else:
                    impath = get_tirf_image(im);
                
                label:str
                if not is_path:
                    label = f"cell{im}";
                else:
                    from IPython import embed; embed()
                    label = get_calib_name(impath)
                meta = get_meta(impath);
                shape = imread(impath).shape[:2];
                res.append(draw_bbox(dest,sourceim=(shape[0],shape[1]),meta=meta,label=label));
            return res;


        def clear_bboxes(dest:Literal['tirf','phase']):
            for p,t in bboxes[dest]:
                p.remove()
                if t:
                    t.remove()
            bboxes[dest] = []
            fig.canvas.draw_idle()

        def toggle_gradient():
            grad_im.set_visible(not grad_im.get_visible())
            phase_im.set_visible(not phase_im.get_visible())
            fig.canvas.draw_idle()

        def set_zoom_home():
            toolbar = fig.canvas.manager.toolbar
            toolbar.update() # Clear the axes stack
            toolbar.push_current()  # save the current status as home
            
        def zoom_bbox():
            bbox = get_bbox('phase')
            if bbox is None:
                fig.canvas.draw();
                return
            bounds = bbox.get_extents().expanded(1.1,1.1)
            pax.set_xlim(bounds.x0,bounds.x1)
            pax.set_ylim(bounds.y1,bounds.y0) #backwards because image
            # set_zoom_home();
            fig.canvas.draw()

        def unzoom(axis:Literal["phase","tirf","both"]="both"):
            if axis in ["tirf","both"]:
                tax.autoscale();
            if axis in ["phase","both"]:
                pax.autoscale();
            fig.canvas.draw_idle();


        def calibrate_image_export(export_folder:str|Path,
                                   dest_size:int|tuple[int,int]|None=None,
                                   dest_position:tuple[int,int]=(0,0),
                                   fillvalue:int|Callable[[float,float],int]=100,
                                   arr:np.ndarray|None=None,
                                   upscale:bool=False,
                                   smoothing:int|None=None):
            
            """Perform calibration of the warping process (specifically to see how warping the gradient quantitatively affects pixel values).
            Generates creates a dummy image for the gradient, warps it to the tirf coordinate system, and calculates average intensity and 
            total intensity differences.

            Note that just like get_warped_image, we want our destination image to be axis-aligned, 
            so the slice we take of our starting image will not be.
            
            To do so: we first create the destination bounding box based on the position and dest_size (position is the center of the rectangle).
            Then, we unwarp that bounding box into the pixel space of the starting image.
            If arr is provided, the bounding box will be applied straight to arr; if the bounding box is outside of the bounds of arr, and suppress_oob is false, 
            an error will be thrown.

            If arr is not provided, a new array will be created as source using fillvalue. if fillvalue is a function of position (x,y), will use np.fromfunction.
            NOTE: the new array will be shifted from the actual origin for space reasons, e.g. the created arr[0,0] != phase (0,0). 
            However, the fillvalue function will be evaluated using the actual phase coordinates.
            If size is also not provided, it will use the current bbox in TIRF (size and position).

            Finally, we want to calculate some stats of the image. The Post-warp image is easy, because the whole image is relevant. However,
            the pre-warp image is tricky - the nature of the non-axis-aligned polygonal mask 

            Populates the export folder with three files:
            -Pre_warp.tif: A dummy image pre-gradient
            -Pre_warp_clipped.tif: Dummy image after applying the anti-aliased polygonal mask
            -Post_warp.tif: Dummy image after warping
            -warp_stats.json: json file with the following: 
            {
                "Filename":{"Pre":"name","Pre-clipped":"name","Post":"name"},
                "Average":{"Pre":num,"Pre-clipped":num,"Post":num"},
                "Integrated":{"Pre":num,"Pre-clipped":num,"Post":num},
                "Misc":{"Pre-warp-clip-area":num}
            };

            The function also returns a dictionary containing the data in warp_stats.

            """
            export_folder = Path(export_folder);
            if (not export_folder.is_absolute()):
                export_folder = parent_folder/export_folder;
            
            src_bbox:Polygon;
            if dest_size is None:
                src_bbox = get_bbox("phase"); ##NOTE: at first this got the tirf bbox and converted back to phase but the tirf bbox uses the entire phase image lmao
            else:
                if not isinstance(dest_size,tuple):
                    dest_size = (int(dest_size),int(dest_size));
                boundss = [(dest_position[i]-dest_size[i]/2,dest_position[i]+dest_size[i]/2) for i in (0,1)];
                
                corners = list(itertools.product(*boundss)); 
                corners[3],corners[2] = corners[2],corners[3] #second pair needs to be in reverse so that the polygon goes around in a circle 
                dest_bbox = Polygon(corners,closed=False); #closed = false to ensure only 4
                from IPython import embed;
                assert len(dest_bbox.get_xy()) == 4, embed();
            
                src_bbox = Polygon(Calibration.transform_pixels("phase",*dest_bbox.get_xy(),meta=calib_meta),closed=False);
                #NOTE: the absolute position of the dest_bbox is going to be wonky, but it won't matter because get_warped_image doesn't care about 
            
            
            

            src_offset:tuple[float,float] = (0,0); #difference between (0,0) in src pixel space (e.g. arr[0,0] = true_image[src_offset])
            if arr is None:
                src_size = src_bbox.get_extents().size;
                src_size = (abs(int(src_size[0])),abs(int(src_size[1])));
                src_size = np.add(src_size,(2,2)); #pad for off-by-one potential (doesn't really matter if oversized);
                src_center = np.average(src_bbox.get_xy(),axis=0);

                src_offset = src_center - src_size/2; 
                #note: this means that in the constructed array case, the offset is not an integer. This is actually a good thing,
                #because when the warp_image function does its slicing, it won't need to slice between pixels (or at least not as much)
            
                if not isinstance(fillvalue,Callable):
                    closure = fillvalue;
                    def f(x:float,y:float):
                        return closure;
                    fillvalue = f;
                func = fillvalue;
                fillvalue = lambda x,y: func(x+src_offset[0],y+src_offset[1]);
                arr = np.fromfunction(np.vectorize(fillvalue),src_size);
                assert arr is not None
            else:
                arr,src_offset = slice_bbox(arr,src_bbox); ##something about this line or the one below is breaking - the bbox is not in the sliced array anymore
            pre_bbox = src_bbox;
            src_bbox = Polygon(src_bbox.get_xy() - src_offset,closed=False); #0-align

            out = get_warped_image(arr.astype(float),src_bbox,upscale,smoothing);
            
            #stats
            ##NOTE: The post stats are easy because the whole image is relevant. If the source image
            clip_mask = np.zeros_like(arr,dtype=np.uint8); #for some godforsaken reason, opencv doesn't do antialiasing unless 8bit

            #similarly, the astype(np.int32) is really important and idk why
            clip_mask = cv.fillConvexPoly(clip_mask,src_bbox.get_xy().astype(np.int32),255,cv.LINE_AA); 

            #since we filled with 255, need to scale to an image 0-1
            clip_mask = clip_mask.astype(float)/255

            clipped = clip_mask * arr;
        

            pre_path = Path("Pre_warp.tif").resolve();
            imsave(pre_path,arr,check_contrast=False);
            clip_path = Path("Pre_warp_clipped.tif").resolve();
            imsave(clip_path,clipped,check_contrast=False);
            post_path = Path("Post-war.tif").resolve();
            imsave(post_path,out,check_contrast=False);
        
            pre_key,clip_key,post_key="Pre","Pre-clipped","Post"

            data:dict[str,dict[str,Any]] = {
                "Filename":{
                    pre_key:str(pre_path),clip_key:str(clip_path),post_key:str(post_path),
                },
                "Average":{
                    pre_key:arr.mean(),clip_key:clipped.sum()/clip_mask.sum(),post_key:out.mean()
                },
                "Integrated":{
                    pre_key:arr.sum(),clip_key:clipped.sum(),post_key:out.sum()
                },
                "Misc":{
                    "Pre-warp-clip-area":clip_mask.sum()
                }
            }
            with open("warp_stats.json","w") as json_file:
                json.dump(data,json_file);
        
            return data;

        


        ## these export functions are sort of inconsistent since they were made at different times
        def export_images_stacked(dest_folder:str|Path="stacked",
                                  gradient_smoothing:int|None=10,
                                  include_bf:bool=True,
                                  calibrate:Path|str|None="calibration",
                                  calibrate_image_size:int|tuple[int,int]|None=None,
                                  cells:list[int]|None=None):
            """Populated output folder with tiff stacks with the following channels:
                1: Gradient, warped to match the TIRF
                2: TIRF
                3: EPI
                4 (if include_bf is true, default): BF
                
                Automatically upscales on gradient image warp because dimensions need to match
                
                If calibrate is not None, will run calibrate_image_export into given folder, relative to 
                dest_folder if not absolute. Will use a image with size calibrate_image_size, or the size of currently selected
                tirf image if not specified. By default uses a fillvalue of 100.
                """
            
            dest_folder = parent_folder/dest_folder; #if dest_folder is absolute this leaves it unchanged
            print(f"exporting stacked images to {dest_folder}")
            dest_folder.mkdir(exist_ok=True);
            if cells is None:
                cells = get_all_tirf_nums();

            if calibrate is not None:
                calibrate = dest_folder/calibrate
                if calibrate_image_size is None:
                    calibrate_image_size = tirf.shape;
                elif (isinstance(calibrate_image_size,int)):
                    calibrate_image_size = (calibrate_image_size,calibrate_image_size);
                
                calibrate_image_export(calibrate,calibrate_image_size,fillvalue=100,upscale=True,smoothing=gradient_smoothing);    
            
            def rescale(im:np.ndarray):
                # return im
                # # print(im.dtype);
                return rescale_intensity(im.astype(float),in_range="dtype",out_range=np.uint8).astype(np.uint8);

            with plt.ioff():
                for cell_num in tqdm(cells):
                    switch_cell_num(cell_num);
                    
                    images = [get_warped_image("gradient",upscale=True,smoothing=gradient_smoothing,meta=calib_meta)]  \
                            + [imread(get_cell_num(cell_num,ctype)) for ctype in (['tirf','epi'] + ['bf'] if include_bf else [])];
                    # images[3] = rescale_intensity(images[3],out_range=np.uint8) #rescale for visuals
                    # images[:3] = [(i/16.0).astype(np.uint8) for i in images[:3]] #12bit -> 8bit for tirf & metamorph

                    images = np.array(images);
                    images = images.astype(np.uint16);
                    outname = dest_folder/f"cell{cell_num}_stack.tiff";

                    tifffile.imwrite(outname,images,photometric="minisblack",shape=(4,2048,2048));
                    # xtiff.to_tiff(images,outname,channel_names="CYX"); #this means multichannel tiff


        def export_images(dest_folder:str|Path):
            """Populates output folder with four folders:
                --"TIRF": Images/TIRF copied image-for-image
                --"BF": copied Images/BF for each BF with a corresponding TIRF image with the same name
                --"Phase": for each TIRF image, extract the bounding box and inverse transform to match TIRF. 
                    upscaled to have the same dimensions as the TIRF image. same name as original
                --"Gradient": same as "Phase", but from the tiled gradient image instead.
                """
            tirf_out = Path(dest_folder)/"TIRF"
            bf_out = Path(dest_folder)/"BF"
            phase_out = Path(dest_folder)/"Phase"
            grad_out = Path(dest_folder)/"Gradient"
            for p in [tirf_out,bf_out,phase_out,grad_out]:
                os.makedirs(p,exist_ok=True)

            with plt.ioff():
                for im in tqdm(get_tirf_images("tirf")):
                    switch_image(im,zoom=False)
                    name = im.name
                    shutil.copy(im,tirf_out/name)
                    BF_im = tirf_images/"BF"/name
                    if BF_im.exists():
                        shutil.copy(BF_im,bf_out/name)

                    phase_warp = get_warped_image('phase',upscale=True)
                    imsave(phase_out/name,phase_warp)

                    grad_warp = get_warped_image('gradient',upscale=True)
                    imsave(grad_out/name,grad_warp)

            return dest_folder

        def export_images_transpose(dest_folder:str|Path):
            """Like export_images, but folders are cell# instead of image type.
            For each image in Image/TIRF, creates a folder with the same name (minus the extension)
            Each folder contains (up to) four files:
                --"tirf.tif": Images/TIRF copied image-for-image
                --"bf.tif": copied Images/BF for each BF with a corresponding TIRF image with the same name
                --"phase.tif": for each TIRF image, extract the bounding box and inverse transform to match TIRF. 
                    upscaled to have the same dimensions as the TIRF image. same name as original
                --"gradient.tif": same as "Phase", but from the tiled gradient image instead.
                """
            with plt.ioff():
                for im in tqdm(get_tirf_images("tirf")):
                    switch_image(im,zoom=False)
                    name = im.name
                    folder = Path(dest_folder)/(im.stem)
                    os.makedirs(folder,exist_ok=True)
                    shutil.copy(im,folder/"tirf.tif")
                    BF_im = tirf_images/"BF"/name
                    if BF_im.exists():
                        shutil.copy(BF_im,folder/"dest.tif")

                    phase_warp = get_warped_image('phase',upscale=True)
                    imsave(folder/"phase.tif",phase_warp)

                    grad_warp = get_warped_image('gradient',upscale=True)
                    imsave(folder/"gradient.tif",grad_warp)

            return dest_folder
        
        def remove_calibration_points(confirm:bool=True):
            names = [name for name in Calibration.data.keys() if name.startswith("Calibration/")];
            confirm_str = "Are you sure you want to remove all calibration points? This will remove all points from the following images with prefix 'Calibration/': \n";
            confirm_str += "\n".join(names);

            if (not confirm or inquirer.confirm(confirm_str)):
                for n in names:
                    Calibration.remove_calib_image(n);
            else:
                print("Images not removed.");

        
        def slice_bbox(to_slice:np.ndarray,bbox:Polygon):
            """Returns:
            -sliced array
            -origin of sliced array, e.g. slicing coordinates of the final array's [0,0]
            """
            extents = bbox.get_extents()
            slice_extents = [[int(extents.ymin),ceil(extents.ymax)],[int(extents.xmin),ceil(extents.xmax)]]
            sliced = to_slice[slice_extents[0][0]:slice_extents[0][1],slice_extents[1][0]:slice_extents[1][1]];
            origin = np.array([slice_extents[1][0],slice_extents[0][0]]);
            return sliced,origin 

        ### Realistically, there is a maximum accuracy that you can get with whole-field-of-view calibration. Ideally each image should have a 
        ### fine-tuning step for precise calibration of phase and TIRF. the Gradient is noisy enough though that it's probably fine for gradient analysis
        def get_warped_image(src:Literal['phase','gradient']|np.ndarray,
                             bbox:Polygon|None=None,
                             upscale:bool|tuple[int,int]=False,
                             smoothing:int|None=None,
                             meta:Metadata|None=None):
            ###GOAL: Extract the tirf FOV from phase
            ### Difficult because the tirf->phase bounding box is not necessarily axis-aligned, so I need to both do a warp with the transformation matrix
            ### And extract the axis-aligned components
            ### PROCEDURE:
            ## Acquire TIRF image's bounding box on phase (polygon)
            ## Get slice of phase image contained that contains the bounding box
            ## calculate offset between TIRF bounding box's top-left (point that will become zero) and slice top left (current image zero)
            ## Adjust transformation matrix by a translation by the offset
            ## apply transformation to image, using TIRF image bounds as size
            ## If specified, apply a blur to the image based on the smoothing value
            ### this should return the proper image... hopefully
            if meta is None:
                meta = calib_meta;

            if not bbox:
                bbox = get_bbox('phase',meta=meta);
                assert bbox is not None
                        
            if isinstance(src,np.ndarray):
                to_slice = src
            elif src == 'phase':
                to_slice = phase
            elif src == 'gradient':
                to_slice = grad
            else:
                raise ValueError(src)
            sliced, origin = slice_bbox(to_slice,bbox);
            offset = np.subtract(origin,bbox.get_xy()[0]); #slice top - bounds top; applied to (bounds top - slice top), e.g. coordinates within the slice, will be zero

            pre_translate = np.array([
                [1, 0, offset[0]],
                [0, 1, offset[1]],
                [0, 0, 1]
            ])

            tirf_offset = -Calibration.transform_points("tirf",np.array([[0,0]]))[0]
            post_translate = np.array([
                [1, 0, tirf_offset[0]],
                [0, 1, tirf_offset[1]],
                [0, 0, 1]
            ])

            points_matrix = Calibration.get_transformation_matrix("tirf")

            composed = post_translate @ points_matrix @ pre_translate


            #could also use pixel_to_absolute_coordinates on the tirf image, but this ensures the shape is calculated using the same matrix as the image
            dest_corners = Calibration.transform_points("tirf",bbox.get_xy())
            dest_shape = (int(np.max(dest_corners[:,0])-np.min(dest_corners[:,0])),int(np.max(dest_corners[:,1])-np.min(dest_corners[:,1])))

            warped = cv.warpPerspective(sliced,composed,dest_shape)            
            
            if upscale:
                dest_shape = tirf.shape
                if not isinstance(upscale,bool):
                    dest_shape = upscale;

                ##upscale: make the warped image the same size as the TIRF image
                scale = get_pixel_scale(meta=meta)[0]
                assert all([t*s - w < 2 for t,s,w in zip(dest_shape,scale,warped.shape)]),(dest_shape,scale,warped.shape)
                #assert that the difference in size is close to that predicted by the pixel scale!

                warped = resize(warped,dest_shape,preserve_range=True)

            if smoothing:
                warped = cv.GaussianBlur(warped,(0,0),sigmaX=smoothing);
            
            warped = warped.astype(to_slice.dtype)

            return warped

        def show():
            with plt.ion():
                plt.show()

        def save():
            Calibration.save()

        commands = [
            show,
            save,
            add_point,
            edit_point,
            update_scatter,
            remove_point,
            switch_image,
            switch_cell_num,
            toggle_gradient,
            zoom_bbox,
            unzoom,
            set_zoom_home,
            draw_bbox,
            draw_all_bboxes,
            clear_bboxes,
            get_bbox,
            slice_bbox,
            test_phase_point,
            test_tirf_point,
            clear_test_points,
            remake_texts,
            calibrate_image_export,
            export_images,
            export_images_stacked,
            export_images_transpose,
            make_calibdata,
            plot_calibdata,
            remove_calibration_points,
            # get_all_tirf_images,
            get_all_tirf_nums,
            get_tirf_image,
            get_tirf_images,
            get_calib_images,
            get_bright_cellnums,
            get_cell_num,
            get_calib_name,
            get_warped_image,
        ]

        commandnames = [c.__name__ for c in commands]
        
        switch_cell_num(cell_num);
        unzoom()
        set_zoom_home()
        zoom_bbox()

        calib_images = get_calib_images()
        images = get_tirf_images()

        from IPython import embed; 
        embed(header="Welcome to the correlative alignment python commandline utility! Use show() to show the window and start editing, and commandnames for a list of commands. For more details on a particular command, use help(command).")
        exit()


    Calibration = PhaseTIRFCalibration(calib_file)
    edit_calibration_cmdline(1)


    
    exit()

    calib_points:list[tuple[tuple[float,float],tuple[float,float]]]
    if force_calibration or not calib_file.exists():
        if continue_calibration and calib_file.exists():
            raise NotImplementedError("iterative calibration is a little silly")
        else:
            with plt.ion():
                ptile = imread(phase_tiled)
                pf,pax = plt.subplots()
                pf.suptitle("Fused Phase")
                pax.imshow(ptile)
                zoom_factory(pax)
                ph = panhandler(pf)

                images = [*filter(lambda x: x.lower().endswith("tif"),os.listdir(calibration_folder))]
                markers = list(itertools.islice(itertools.chain(['o','x','v','^','+','<','>','*','s','_','.','1'],itertools.cycle('X')),len(images)))
                phase_clicker = singlify_clicker(clicker(pax,images,markers=list(itertools.islice(markers,len(images)))))
                tmarkers = iter(markers)

                calib_points:list[tuple[tuple[float,float],tuple[float,float]]] = [] #phase,tirf
                
                for im in images:
                    # print(im)
                    tf,tax = plt.subplots()
                    tf.suptitle(im)
                    calib_im = imread(calibration_folder/im)
                    tax.imshow(calib_im)

                    tirf_clicker = singlify_clicker(clicker(tax,[im],markers=[next(tmarkers)]))
                    phase_clicker._current_class = im
                    phase_clicker._update_legend_alpha()
                    # print(tirf_clicker._positions)
                    while not (has_point(phase_clicker,cls=im) and has_point(tirf_clicker)):
                        plt.waitforbuttonpress(timeout=0.3)

                    input("When satisfied with chosen points, press enter")

                    ppos = phase_clicker._positions[im][0]
                    tpos = tirf_clicker._positions[im][0]

                    phase_pos_pix:tuple[float,float] = tuple(ppos)
                    tirf_pos_pix:tuple[float,float] = tuple(tpos)

                    ##but, need tirf pos in absolut coordinates
                    ##NOTE: IMPORTANT CONVENTION - ASSUME METADATA POSITIONING IS TOP-LEFT [0,0]

                    tirf_pos_absolute,tirf_pos_units = pixel_to_absolute_coordinates(pos=tirf_pos_pix,file=calibration_folder/im)

                    calib_points.append((phase_pos_pix,tuple(tirf_pos_absolute)))
                    
                    print(f"Calibration point added: Phase={phase_pos_pix}, TIRF={tirf_pos_absolute} w/ units {tirf_pos_units}")

                    plt.close(tf)
                    del tirf_clicker

            calib_data = {"phase":[c[0] for c in calib_points],"tirf":[c[1] for c in calib_points]}
            jdump(calib_data,str(calib_file))
    else:
        calib_data = jload(str(calib_file))
        calib_points = [((a[0],a[1]),(b[0],b[1])) for a,b in zip(calib_data["phase"],calib_data["tirf"])]
        ## extra paranoid typing

    
    print(calib_points)

    cpoints = np.array(calib_points)

    ### CONSTRUCT THE TRANSFORMATION FROM TIRF TO PHASE


    #The .copy() is very important; makes the array contiguous in memory, which is important becuase opencv moment
    M, inliers = cv.estimateAffine2D(cpoints[:,1].reshape(-1, 1, 2).copy(),cpoints[:,0].reshape(-1, 1, 2).copy())
    matchesMask = inliers.ravel().tolist()
    print(M,matchesMask)


    # from IPython import embed; embed()
    # exit()    



    def box_image(source:Path,color='red',thickness=1):
        image = imread(source)
        h,w = image.shape
        corners = [ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]
        pts,_ = pixel_to_absolute_coordinates(corners,file=source)
        
        full_matrix = np.array([M[0],M[1],[0,0,1]])
        pts = np.concatenate([pts,[[1],[1],[1],[1]]],axis=1)

        dst = [np.matmul(full_matrix,pt)[:2] for pt in pts]

        return Polygon(dst,closed=True,color=color,linewidth=thickness,fill=False)

        # dest = cv.polylines(dest,[np.int32(dst)],True,color,thickness, cv.LINE_AA)
    

    # ##test: overlay calibration images
    images = os.listdir(calibration_folder)
    calib_polys = []
    for im in tqdm(images):
        if not im.lower().endswith("tif"): continue
        path = calibration_folder/im
        calib_polys.append((box_image(path),box_image(path)))


    ptile = imread(phase_tiled)
    gtile = imread(gradient_tiled)

    pf,pax = plt.subplots()
    pf.suptitle("Phase")
    pax.imshow(ptile)

    gf,gax = plt.subplots()
    gf.suptitle("Gradient")
    gax.imshow(gtile)

    def add_box(polys:tuple[Artist,Artist]|Iterable[tuple[Artist,Artist]]):
        if isinstance(polys,tuple) and isinstance(polys[0],Artist):
            polys = [polys]
        for p,g in polys:
            pax.add_artist(p)
            pax.draw_artist(p)
            gax.add_artist(g)
            gax.draw_artist(g)
        pf.canvas.draw_idle()
        gf.canvas.draw_idle()

    
    def remove(polys:tuple[Artist,Artist]|Iterable[tuple[Artist,Artist]]):
        if isinstance(polys,tuple) and isinstance(polys[0],Artist):
            polys = [polys]
        for p,g in polys:
            p.remove()
            g.remove()
        pf.canvas.draw_idle()
        gf.canvas.draw_idle()

    add_box(calib_polys)

    def draw_box(path,color='yellow'):
        box = (box_image(path,color=color),box_image(path,color=color))
        add_box(box)
        return box
    
    def draw_cell(num):
        path = rf"F:\Lab Data\23.11.20 Gradient Fix Test #5 2.5notch\Images\TIRF\cell{num}.tif"
        b = draw_box(path)
        t = (pax.annotate(str(num),b[0].get_xy()[0]),gax.annotate(str(num),b[1].get_xy()[0]))
        return b,t

    
    from IPython import embed; embed()
            
            



    


                    


if __name__ == "__main__":
    # typer.run(main)
    main()