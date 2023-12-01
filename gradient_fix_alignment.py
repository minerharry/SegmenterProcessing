from copy import copy
import functools
import inspect
import itertools
import os
from pathlib import Path
from typing import Any, Callable, DefaultDict, Iterable, Literal, NamedTuple
from bs4 import BeautifulSoup
import inquirer
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Polygon
from matplotlib.text import Text
from skimage.io import imread
import matplotlib.pyplot as plt
from tifffile import TiffFile
import typer
from mpl_interactions import zoom_factory,panhandler
from mpl_point_clicker import clicker
from json_tricks import dump as jdump, load as jload
from utils import inquire
from utils.ome import pixel_to_absolute_coordinates

import numpy as np
import cv2 as cv
from cv2.typing import MatLike
from tqdm import tqdm

## ensures the clicker ca
def singlify_clicker(c:clicker):
    def on_point_added(pos,cls):
        c.set_positions(dict(c._positions,**{cls:[pos]}))
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

class PhaseTIRFCalibration:
    data:dict[str,list[CalibData]] #indexed by tirf filename (will look for it in the calibration folder)
    def __init__(self,file:str|Path):
        self.file = Path(file)
        if not self.file.exists():
            self.file.touch()
            self.data = DefaultDict(lambda: [])
            self.save()
        else:
            with open(self.file,'r') as f:
                self.data = jload(f)
            self.data = DefaultDict(lambda: [],{k:[CalibData(*c) for c in v] for k,v in self.data.items()})

        self.transform_cache:dict[str,MatLike] = {}

   

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
        with open(self.file,'w') as f:
            jdump(self.data,f)


    def get_transformation(self,to:Literal['phase','tirf']="phase"):
        opt = ['phase','tirf']
        if to not in opt:
            raise ValueError()
        
        if to in self.transform_cache:
            if not isinstance(self.transform_cache[to],MatLike):
                del self.transform_cache[to]
            else:
                return self.transform_cache[to]
        
        src = f"{to}_absolute"
        dst = f"{opt[opt.index(to)-1]}_absolute"
            
        calib_points:list[tuple[tuple[float,float],tuple[float,float]]] = sum([
            [(getattr(c,dst),getattr(c,src)) for c in l]
            for k,l in self.data.items()
        ],[])
        cpoints = np.array(calib_points)
        M, inliers = cv.estimateAffine2D(cpoints[:,1].reshape(-1, 1, 2).copy(),cpoints[:,0].reshape(-1, 1, 2).copy())
        matchesMask = inliers.ravel().tolist()
        return M,matchesMask
    
    def transform_pixels(self,to:Literal['phase','tirf']='phase',*points:tuple[float,float],meta:BeautifulSoup|str=None,file:TiffFile|str|Path=None):
        abs_points,units = pixel_to_absolute_coordinates(points,meta=meta,file=file)
        



    

def main(force_calibration:bool=False,continue_calibration:bool=False):
    parent_folder = Path(r"F:\Lab Data\2023.11.14 Gradient Fixing Test 5\2.5 notches")
    phase_folder = parent_folder/"Phase"
    gradient_folder = parent_folder/"Gradient"

    phase_tiled = phase_folder/"fused.tif"
    gradient_tiled = gradient_folder/"fused.tif"

    TIRF_folder = Path(r"F:\Lab Data\23.11.20 Gradient Fix Test #5 2.5notch")
    calibration_folder = TIRF_folder/"Calibration"
    tirf_images = TIRF_folder/"Images"

    calib_file = (calibration_folder/"calibration.calib")

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
        raise NotImplementedError("Interaction is hard")


    def edit_calibration_cmdline(calib_image:str|Path):
        ##GOAL
        # - open phase image
        # - open calibratio nimage
        # - allow user to enter pairs of pixel coordinates: one from phase, one from TIRF
        # -- addition, removal, and editing
        # - allow user to finish calibration
        calib_image = Path(calib_image)
        if not calib_image.is_absolute() and (calibration_folder/calib_image).exists():
            #look in calibration folder before local
            calib_image = calibration_folder/calib_image
        calib_name = calib_image.name


        fig,(pax,tax) = plt.subplots(1,2,num="Calibration")
        pax:Axes;tax:Axes
        pax.set_title("Phase")
        tax.set_title(calib_name)

        phase = imread(phase_tiled)
        tirf = imread(calib_image)

        pax.imshow(phase)
        tax.imshow(tirf)

        ppoints = pax.scatter([],[],color='orange')
        tpoints = tax.scatter([],[],color='orange')

        markers = itertools.cycle(map(MarkerStyle,['o','x','v','^','+','<','>','*','s','_','.','1','X']))

        points:list[tuple[tuple[float,float],tuple[float,float]]] = []
        texts:list[tuple[Text,Text]] = []

        


        def update_scatter():
            nonlocal markers
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
            tirf_pos,_ = pixel_to_absolute_coordinates([p[1] for p in points],file=calib_image)
            for (pp,tp),ta in zip(points,tirf_pos):
                res.append(CalibData(tuple(pp),tuple(pp),tuple(tp),tuple(ta)))
            return res


        def add_point(phase_pos:tuple[float,float],tirf_pos:tuple[float,float]):
            points.append((phase_pos,tirf_pos))
            pt = pax.text(*phase_pos,str(len(points)))
            tt = tax.text(*tirf_pos,str(len(points)))
            texts.append((pt,tt))
            update_scatter()
            Calibration.add_calibration_point(calib_name,make_calibdata(points[-1])[0])


        def remove_point(num:int):
            idx = num-1
            p = points[idx]
            points.remove(p)
            texts[idx][0].remove()
            texts[idx][1].remove()
            texts.remove(texts[idx])
            for i in range(idx,len(texts)):
                (pt,tt) = texts[i]
                pt.set_text(str(i+1))
                tt.set_text(str(i+1))
            update_scatter()
            Calibration.remove_calibration_point(calib_name,make_calibdata(p)[0])
        
        def set_point(num:int,phase:tuple[float,float]|None=None,tirf:tuple[float,float]|None=None):
            idx = num-1
            p = (phase or points[idx][0],tirf or points[idx][1])
            points[idx] = p
            texts[idx][0].set_position(p[0])
            texts[idx][1].set_position(p[1])
            update_scatter()
            cp = copy(Calibration.get_calibration_points(calib_name))
            cp[idx] = make_calibdata(p)[0]
            Calibration.set_calibration_points(calib_name,cp)

        from IPython import embed; embed()
        exit()


    Calibration = PhaseTIRFCalibration(calib_file)
    edit_calibration_cmdline("hdots_inner_BF.tif")


    
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
                    print(im)
                    tf,tax = plt.subplots()
                    tf.suptitle(im)
                    calib_im = imread(calibration_folder/im)
                    tax.imshow(calib_im)

                    tirf_clicker = singlify_clicker(clicker(tax,[im],markers=[next(tmarkers)]))
                    phase_clicker._current_class = im
                    phase_clicker._update_legend_alpha()
                    print(tirf_clicker._positions)
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