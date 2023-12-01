from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence
from isort import stream
import matplotlib
from pathlib import Path
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse,Circle as CirclePatch
import numpy as np
from tqdm import tqdm
from centersfitting.fit_inner_ellipse import adaptive_fit_ellipse, brute_interior_ellipse, contour_adaptive_fitellipse, fit_interior_ellipse, fit_mask_interior_ellipse
from centersfitting.iterative_weighted_center import iterative_weighted_center, iterative_weighted_centers
from libraries.centers import get_centers,generate_annotated_image, getcircle, getellipse
from libraries.smoothing import moving_average
from utils.parse_tracks import FijiManualTrack
from matplotlib import pyplot as plt
from tifffile import TiffFile,TiffFrame,TiffPageSeries


### GOOD CELLS TO FOLLOW AS TESTS
#53: mov3 #3, #5

def addPoint(scat:Line2D, new_point):
    old_off = scat.get_xydata()
    new_off = np.concatenate([old_off,np.array(new_point, ndmin=2)])
    # old_c = scat.get_color()
    # if c is None:
    #     c = old_c[-1]
    # new_c = np.concatenate([old_c, np.array(matplotlib.colors.to_rgba(c), ndmin=2)])

    xrange = np.min(new_off[:,1]),np.max(new_off[:,1])
    yrange = np.min(new_off[:,0]),np.max(new_off[:,0])
    xtra = 0.1*(xrange[1]-xrange[0])
    ytra = 0.1*(yrange[1]-yrange[0])


    scat.set_data(new_off.transpose())
    # scat.set_facecolors(new_c)

    scat.axes.figure.canvas.draw_idle()
    # scat.update_scalarmappable()
    return np.array([[yrange[0]-ytra,yrange[1]+ytra],[xrange[0]-xtra,xrange[1]+xtra]])

def addPoints(points:Iterable[tuple[Line2D,Any]]): #returns [0]lim,[1]lim
    bboxes = np.array([addPoint(path,point) for path,point in points])
    bbox2 = [[np.min(bboxes[:,i,0],axis=0),np.max(bboxes[:,i,1],axis=0)] for i in [0,1]]
    return bbox2

# def get_ellipses(images:Iterable[np.ndarray]):
    
class Center(ABC):
    name:str #should be unique
    shortname:str #only used for display purposes
    color:str

    def __init__(self,color=None,shortname=None,name=None,**kwargs) -> None:
        if color: self.color = color
        if name: self.name = name
        if shortname: self.shortname = shortname
        if len(kwargs) > 0:
            raise Exception(f"unexpected keyword arguments: {kwargs}");
        if getattr(self,"shortname",None) is None:
            self.shortname = self.name

    def __hash__(self) -> int:
        return hash(self.name)

    @abstractmethod
    def get_fit_data(self,mask:np.ndarray,frame:int,series:TiffPageSeries,id:int)->dict[str,Any]: ...

    def init_artists(self,axes:Axes):
        self.marker = axes.scatter([0],[0],color=self.color,label=self.shortname);

    def update_annotations(self,data:dict[str,Any]): 
        self.marker.set_offsets([data["center"][0],data["center"][1]])
        
# class IterCenter

class ManualCenter(Center):
    name="manual center"
    shortname="manual"
    color="red"
    def load(self,track:dict[int,tuple[int,int]]):
        self.track = track
    def get_fit_data(self, mask: np.ndarray, frame: int, series: TiffPageSeries,id:int) -> dict[str,Any]:
        return {"center":self.track[frame+1]}


class Circle(Center):
    name="interior circle"
    shortname="in-circle"
    color="green"

    def init_artists(self, axes: Axes):
        super().init_artists(axes)
        self.circle = CirclePatch([0,0],0,color=self.color,fill=False)
        axes.add_artist(self.circle)

    def get_fit_data(self, mask: np.ndarray, frame: int, series: TiffPageSeries, id: int) -> dict[str, Any]:
        (y,x),radius = getcircle(mask);
        return {"center":(x,y),"radius":radius};
        
    def update_annotations(self, data: dict[str, Any]):
        super().update_annotations(data)
        # print(data)
        self.circle.set_center(data["center"])
        self.circle.set_radius(data["radius"])


class IterCenter(Center):
    color="blue" 
    name="iterative weighted center"
    shortname = "itercenter"
    def __init__(self,iters=10,**kwargs) -> None:
        super().__init__(**kwargs)
        self.niters = iters
    def get_fit_data(self, mask: np.ndarray, frame: int, series: TiffPageSeries, id: int) -> dict[str, Any]:
        centers = np.array(iterative_weighted_centers(mask,iters=self.niters))[:,::-1]
        return {"centers":centers,"center":centers[-1]}
    def init_artists(self, axes: Axes):
        alpha_arr = np.concatenate([np.linspace(0,0.8,self.niters)[1:],[1]])
        # alpha_arr = np.take(alpha_arr,[*range(1,self.niters),self.niters+1])
        sizes = np.linspace(5,20,self.niters)
        r, g, b = to_rgb(self.color)
        # r, g, b, _ = to_rgba(color)
        color = [(r, g, b, alpha) for alpha in alpha_arr]
        self.markers = axes.scatter([0]*self.niters,[0]*self.niters,c=color[::-1],marker='o',label=self.shortname,s=sizes)
    def update_annotations(self, data: dict[str, Any]):
        self.markers.set_offsets(data["centers"][::-1])
        


class _DrawEllipse(Center):
    def init_artists(self, axes: Axes):
        super().init_artists(axes)
        self.ellipse = Ellipse([0,0],0,0,0,color=self.color,fill=False);
        axes.add_artist(self.ellipse)

    def update_annotations(self, data: dict[str, Any]):
        super().update_annotations(data)
        self.ellipse.set_center(data["center"])
        self.ellipse.set_width(data["major"])
        self.ellipse.set_height(data["minor"])
        self.ellipse.set_angle(data["angle"])


class InteriorEllipse(_DrawEllipse):
    color="blue"
    name="interior ellipse"
    shortname="in-ellipse"
    def get_fit_data(self, mask: np.ndarray, frame: int, series: TiffPageSeries, id: int) -> dict[str, Any]:
        ellipse = contour_adaptive_fitellipse(mask);# fit_mask_interior_ellipse(mask)
        semimajor,semiminor = ellipse[[0,1]]
        center = ellipse[[2,3]]
        angle = ellipse[4]
        return {"major":semimajor*2,"minor":semiminor*2,"center":center,"angle":angle*180/np.pi}

class ContourEllipse(_DrawEllipse):
    color="yellow";
    name="contour ellipse"
    shortname="ellipse"
    def get_fit_data(self, mask: np.ndarray, frame: int, series: TiffPageSeries, id: int) -> dict[str, Any]:
        center,(major,minor),angle = getellipse(mask);
        return {"major":major,"minor":minor,"center":center,"angle":angle};

class UnionIntersectionEllipse(_DrawEllipse): pass

class gCenters(Center,ABC):
    center:str
    def __init__(self,**kwargs) -> None:
        if getattr(self,"name",None) is None:
            self.name = self.center
        super().__init__(**kwargs)

    def get_fit_data(self, mask: np.ndarray, frame: int, series: TiffPageSeries,id:int) -> dict[str, Any]:
        # print(self.center)
        center = get_centers(mask,self.center,[id],False)[0]
        return {"center":center}
    
class AppMed(gCenters): center = "approximate-medoid";shortname="appmed";color="purple"
class Centroid(gCenters): center = "centroid";color="orange"

class SmoothedCenter(Center):
    color="blue"
    def __init__(self,source:Center,series:TiffPageSeries|None=None,width=3,power=3,**kwargs) -> None:
        self.name = "smoothed " + source.name
        self.shortname = "smooth-" + source.shortname
        super().__init__(**kwargs)
        self.width = width;
        self.power = power;
        self.source = source
        if series:
            self.set_series(series);
    
    def set_series(self,series:TiffPageSeries):
        self.raw = np.ndarray([len(series),2])
        for idx,image in enumerate(series):
            arr = image.asarray()
            if not np.any(arr):
                self.raw[idx] = self.raw[idx-1]
                continue
            self.raw[idx] = self.source.get_fit_data(arr,idx,series,np.unique(arr)[1])["center"]
        self.data = np.array([(moving_average**self.power)(self.raw[:,i],self.width) for i in [0,1]]).T; #do each axis separately

    def get_fit_data(self, mask: np.ndarray, frame: int, series: TiffPageSeries, id: int) -> dict[str, Any]:
        return {"raw":self.raw[frame],"center":self.data[frame]}

        


def dist(t1,t2):
    return np.sqrt(np.sum(np.subtract(t1,t2)**2))


if __name__ == "__main__":
    maskspath = r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\temp\tracks_masks\2023.4.2 OptoTiam Exp 53_movie{0}_track{1}.TIF"
    # files = [maskspath/"2023.4.2 OptoTiam Exp 53_movie3_track2.TIF",maskspath/"2023.4.2 OptoTiam Exp 53_movie3_track3.TIF",maskspath/"2023.4.2 OptoTiam Exp 53_movie3_track5.TIF"];
    movie = 3
    tracks = [
            ((3,3),("control3",1)), #(movie,trackn) for automatic,manual
            ((3,5),("control3",12)),
            ((1,2),("control1",1)),
            ((1,22),("control1",5)), #wrong??
            ((1,28),("control1",8))][1:] #trackfile doesn't exist???
    trackFile = r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.4.2 OptoTiam Exp 53 $manual\manual tracks\{0} in pixels per frame.csv"
    

    # centertypes = [("CoM"]    

    Manual = ManualCenter()
    Smoothed = SmoothedCenter(AppMed())
    centers:list[Center] = [
            Manual,
            # Centroid(),
            AppMed(),
            # ContourEllipse(),
            # # InteriorEllipse(),
            # IterCenter(),
            # Circle()
            Smoothed
        ]

    for (movie,trackn),(mname,mantrackn) in tracks:
        
        file = maskspath.format(movie,trackn)
        tiff = TiffFile(file)
        manTrack = FijiManualTrack(trackFile.format(mname));

        f = plt.figure()
        gs = f.add_gridspec(2,2)
        plot_ax = f.add_subplot(gs[0, 0])
        plot_ax.set_title("Center Posisions")
        plot_ax.set_xlabel("X position (pixels)")
        plot_ax.set_ylabel("Y position (pixels)")
        disp_ax = f.add_subplot(gs[0, 1])
        disp_ax.set_title("Cell Mask")
        disp_ax.set_xticks([])
        disp_ax.set_yticks([])
        dev_ax = f.add_subplot(gs[1, 0:2])
        dev_ax.set_title("Devations from manual center")
        dev_ax.set_ylabel("Distance (pixels)")
        dev_ax.set_xlabel("Time (frames)")
        dev_ax.set_xlim(0,len(tiff.series[0]))

        plt.subplots_adjust(wspace=0.7, hspace=0.5)
        # comb1 = f.add_subplot(gs[1, 0:3])
        # comb2 = f.add_subplot(gs[1, 3:6])
        # fig,plots = plt.subplots(2,2)
        # (plot_ax,disp_ax),(dev_ax,spare_ax) = plots

        # plot_ax = plt.figure("plot").gca()
        paths = {center:plot_ax.plot([],[],label=center.shortname,color=center.color)[0] for center in centers}
        # plot_ax.legend()


        # disp_ax = plt.figure("display").gca()
        [center.init_artists(disp_ax) for center in centers]
        disp_ax.legend(bbox_to_anchor=(-0.04, 1), loc="upper right")
        
        Manual.load(manTrack[mantrackn])
        Smoothed.set_series(tiff.series[0]);

        # dev_ax = plt.figure("deviations").gca()
        reference = Manual
        devs = {center:dev_ax.plot([],[],label=center.shortname,color=center.color)[0] for center in centers if center is not Manual}
        
        def animate(obj):
            (idx,im) = obj
            im = im.asarray()
            cy,cx = np.where(im)
            try:
                bbox = [[min(cy)-10,max(cy)+10],[min(cx)-10,max(cx)+10]]
            except:
                return
            disp_ax.set_xlim(bbox[1])
            disp_ax.set_ylim(bbox[0])
            id = np.unique(im)[1] #should only be zero and [id]
            disp_ax.imshow(im)

            # plt.figure("display")
            try:
                centerData = {center:center.get_fit_data(im,idx,tiff.series[0],id) for center in centers}
            except KeyboardInterrupt as k:
                raise k
            except Exception as e:
                print(e)
                return
            [center.update_annotations(data) for center,data in centerData.items()]
            plt.draw()

            # plt.figure("plot")
            xlims,ylims = addPoints((path,centerData[center]["center"]) for center,path in paths.items())
            plot_ax.set_ylim(ylims)
            plot_ax.set_xlim(xlims)
            plot_ax.autoscale()



            # plt.figure("deviations")
            xlims,ylims = addPoints((dev,(idx,dist(centerData[Manual]["center"],centerData[center]["center"]))) for center,dev in devs.items())
            dev_ax.set_ylim(ylims)
            dev_ax.autoscale()
            # dev_ax.legend()
            plt.draw()

            # plt.pause(0.9)

        able = enumerate(tqdm(tiff.series[0]))
        doanim = False
        if doanim:
            anim = FuncAnimation(f,animate,able)
            anim.save("allcenters.mp4")
        else:
            for idx,im in able:
                animate((idx,im))
                plt.pause(0.9)

        plt.show()
