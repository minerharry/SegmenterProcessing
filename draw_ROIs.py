from contextlib import contextmanager
import datetime
import io
from pathlib import Path
from typing import Any, Iterable, Literal, MutableMapping, TypeVar
import matplotlib
from matplotlib.offsetbox import AnchoredText
import numpy as np
from tqdm import tqdm
from utils.CZI import ROI, extract_ROIs, draw_ROI, plot_ROI, read_czi
from utils.filegetter import afn, skip_cached_popups, asksaveasfilename,adir
from utils.ome import OMEMetadata
from utils.rescale import rescale,rescale_intensity
from matplotlib.colors import XKCD_COLORS,hex2color
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

def getcolor(color:tuple[float,float,float]|str):
    if isinstance(color,str): color = hex2color(colors[color])
    return color
    # color = (color[0]*255,color[1]*255,color[2]*255)

colors = {n.replace('xkcd:', ''): c for n, c in XKCD_COLORS.items()}
def prettify_movie(stack:np.ndarray,color:tuple[float,float,float]|str,channel:int|None=None,im_hists:tuple[float,float]|None=(0.05,0.99),out_dtype:np.dtype=np.uint8):
    if channel is None: assert stack.ndim != 4
    else: stack = stack[...,channel]
    if isinstance(color,str): color = hex2color(colors[color])

    if im_hists:
        stack = rescale(stack,*im_hists)
    
    out = stack[...,None] * color

    out = rescale_intensity(out,(0,1),out_dtype)

    return out

def stack_draw_ROI(pretty:np.ndarray,roi:ROI,thickness:float,color:tuple[float,float,float]|str):
    if not pretty.shape[-1]==3:
        raise ValueError("Input stack must be an RGB image (HxWx3) or stack (TxHxWx3)")
    if isinstance(color,str): color = hex2color(colors[color])

    color = (color[0]*255,color[1]*255,color[2]*255)

    squeeze = False
    if pretty.ndim == 3:
        pretty = np.array([pretty])
        squeeze = True

    res = []
    for slice in pretty:
        res.append(draw_ROI(slice,roi,thickness,color));

    if squeeze:
        return res[0]
    else:
        return np.array(res)

def frameless_plot(figwidth,figheight,dpiscale=100):
    fig = plt.figure(frameon=False,figsize=(figwidth/dpiscale,figheight/dpiscale),dpi=dpiscale)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig,ax

class raster_matplotlib:
    def __init__(self,im:np.ndarray,include_alpha=False):
        self.im = im
        self.al = include_alpha
        
    def __call__(self):
        return self._raster_matplotlib(self.im,self.al)
    
    def ctx(self):
        return self()

    @contextmanager
    def _raster_matplotlib(self,im,include_alpha=False):
        w,h = im.shape[-2],im.shape[-3]

        # print(w,h)
        fig,ax = frameless_plot(w,h,dpiscale=100)

        ax.imshow(im)

        yield ax

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw', bbox_inches='tight',pad_inches=0)
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                            newshape=(int(fig.get_tightbbox().bounds[3]*fig.get_dpi()), int(fig.get_tightbbox().bounds[2]*fig.get_dpi()), -1))
        io_buf.close()

        plt.close(fig)
        if include_alpha:
            self.result=img_arr
        else:
            self.result=img_arr[...,:3]

# def add_matplotlib(im,meta:OMEMetadata,include_alpha=False):
    

    

# Color = str|tuple[float,float,float]
# def make_czi_movie(file:Path|str,im_color:Color,roi_colors:Iterable[Color],stack_channel:int|None=None,scale:int|None=None,extra_rois:Iterable[tuple[ROI,Color]]=[]):
#     rois = extract_ROIs(file)
#     im = read_czi(file)

#     color_im = prettify_movie(im,im_color,channel=stack_channel)

#     for roi,color in zip(rois,roi_colors):
#         color_im = stack_draw_ROI(color_im,roi,)
    
# def process_singleROI_movie(f:Path|str):
#     r = (extract_ROIs(f))
#     im = read_czi(f)
    
#     p = prettify_movie(im,'green',1)
#     roid = stack_draw_ROI(p,r[0],3,'red')

#     meta = OMEMetadata(f)

#     scaled = [add_scalebar(r,meta) for r in roid]

#     return scaled

T = TypeVar("T")

def subdict(d:MutableMapping[T,Any],fields:Iterable[T],strict=False):
    if strict:
        return {k:d[k] for k in fields}
    else:
        return {k:d[k] for k in fields if k in d}

@contextmanager
def illustrator_compatible():
    oldparams = subdict(matplotlib.rcParams,('pdf.fonttype','ps.fonttype'))

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    oldpltparams = subdict(plt.rcParams,("pgf.preamble"))

    # plt.rcParams.update({
    # "pgf.preamble": [
    #      "\\usepackage{arev}",
    #     "\\usepackage[T1]{fontenc}"]
    # })

    yield

    matplotlib.rcParams.update(oldparams)
    plt.rcParams.update(oldpltparams)
    



PositionOption = Literal['best','upper right','upper left','lower left','lower right','right','center left','center right','lower center','upper center','center']

if __name__ == "__main__":
    # with skip_cached_popups():
    f = afn();
    plc = True
    double = False

    meta = OMEMetadata(f)

    font_scale = 1.0



    relative_starttime = None #if this movie is sequential (e.g. forward/backward ROI), set this to the inital start time of the earlier movie. 
    #print(f"start time for movie {Path(f).name}: {meta.acquisition_date}") #get the iso formatted time of this movie
    
    # relative_starttime = datetime.datetime.fromisoformat('2024-10-09T19:59:06.903') #10.9 cell1_grad1
    # relative_starttime = datetime.datetime.fromisoformat('2024-09-25T01:07:30.593') #9.24 cell2_grad1

    acquisition_start = datetime.datetime.fromisoformat(meta.acquisition_date)
    if relative_starttime:
        starttime = (acquisition_start - relative_starttime).seconds
    else:
        starttime = 0


    times,timeunit = meta.get_plane_timestamps()
    assert timeunit == 's'
    timedeltas = [float(time) + starttime for time in times]

    rois = (extract_ROIs(f))
    im = read_czi(f)

    # crop:None|tuple[tuple[int|None,int|None],tuple[int|None,int|None]] = None
    crop = ((None,None),(None,128*5)) #xcrop, ycrop
    
    if crop is not None:
        c:list[tuple[int,int]] = []
        for i,(low,high) in enumerate(crop):
            axis = [-2,-3] #x:axis 1 of crop = third-to-last axis of image,y: axis 2 of crop = fourth-to-last axis of image
            if low is None:
                low = 0
            if high is None:
                high = im.shape[axis[i]]
            c.append((low,high))
        crop = (c[0],c[1])
    # font_scale *= 5/8 #make cropped text smaller
    
    p = prettify_movie(im,'neon green',0 if plc else 1)
    roi_colors:list[tuple[ROI,int,str|tuple[float,float,float]]] = [
        (rois[0],3,'red'),   #first ROI
        (rois[1],3,'yellow') #second ROI
    ]
    

    raster = False
    if raster: #rasters to image and saves to video
        for R,T,C in roi_colors:
            p = stack_draw_ROI(p,R,T,C)

        def add_m(i,im):
            r = raster_matplotlib(im) #this got... less pretty with needing to return a result
            with r() as ax:
                scale = ScaleBar(meta.PhysicalSizeX,meta.PhysicalSizeXUnit,color='w',box_alpha=0,fixed_value=25,fixed_units=meta.PhysicalSizeXUnit,font_properties={"size":30*font_scale});
                ax.add_artist(scale)

                min = int(timedeltas[i]/60)
                sec = timedeltas[i] % 60
                timestr = f"{min}:{sec:02.2f}"

                ax.set_xlim(crop[0])
                ax.set_ylim(crop[1])

                ax.text(10,im.shape[0]-10,timestr,horizontalalignment="left",verticalalignment="bottom",bbox=dict(facecolor="black",alpha=0.7),fontdict=dict(color="w",size=40*font_scale))
            return r.result;
                

        scaled = [add_m(i,r) for i,r in enumerate(tqdm(p,desc="adding annotations"))]

        out = asksaveasfilename(title="Save Movie");
        fps = 6 if double else 4
        from mediapy import write_video

        write_video(out,tqdm(scaled,desc="writing video"),fps=fps,qp=2)


    else:
        # outf = Path("output/images/ROIs/")/Path(f).parent.name/Path(f).name
        # outf.mkdir(parents=True,exist_ok=True)

        frames:None|list[int] = None #set to list of specific frame #s to only do those frames

        outf = Path(adir(title="Folder to save frames"))

        with illustrator_compatible(): #set matplotlib saving params and such
            for i,r in enumerate(tqdm(p)):
                if frames is not None and i not in frames:
                    continue
                w,h = r.shape[-2],r.shape[-3]
                if crop:
                    w = crop[0][1] - crop[0][0]
                    h = crop[1][1] - crop[1][0]

                f,ax = frameless_plot(w,h,dpiscale=100)
                
                #show image
                ax.imshow(r)

                #draw ROIs
                for R,T,C in roi_colors:
                    plot_ROI(ax,R,T,getcolor(C))

                
                
                #draw scalebar
                scaleloc:PositionOption = "upper right"
                scale = ScaleBar(meta.PhysicalSizeX,meta.PhysicalSizeXUnit,color='w',box_alpha=0,fixed_value=25,fixed_units=meta.PhysicalSizeXUnit,font_properties={"size":30*font_scale},location=scaleloc);
                ax.add_artist(scale)

                #draw timestamp
                #valid strings: ['best','upper right','upper left','lower left','lower right','right','center left','center right','lower center','upper center','center']
                textloc:PositionOption = "lower left"
                min = int(timedeltas[i]/60)
                sec = timedeltas[i] % 60
                timestr = f"{min}:{sec:02.2f}"
                T = AnchoredText(timestr,textloc,pad=0.2,prop=dict(color="w",fontproperties=dict(size=30*font_scale))) #anchor text to the corner of the frame so zooming (cropping) keeps it there
                #these set the background properties, you can just remove them if you want. I don't think the set_alpha works with postscript export
                T.patch.set_color("black");
                T.patch.set_alpha(0.7)
                ax.add_artist(T)

                out = outf/f"frame{i}.eps"

                ax.set_xlim(crop[0])
                ax.set_ylim((crop[1][1],crop[1][0])) #do y limit backwards to match image coordinates

                from IPython import embed; embed()
                plt.show()
                f.savefig(out,bbox_inches='tight',pad_inches=0)
                plt.close(f)

    # import matplotlib.pyplot as plt

    # from IPython import embed; embed()
    
    # i = imread(f)
