from contextlib import contextmanager
from datetime import timedelta
import datetime
import io
from pathlib import Path
from typing import Iterable
import numpy as np
from tqdm import tqdm
from utils.CZI import ROI, extract_ROIs, draw_ROI, read_czi
from utils.filegetter import afn, skip_cached_popups, asksaveasfilename
from utils.ome import OMEMetadata
from utils.rescale import rescale,rescale_intensity
from matplotlib.colors import XKCD_COLORS,hex2color
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

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

        fig = plt.figure(frameon=False,figsize=(w/100,h/100),dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

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


if __name__ == "__main__":
    # with skip_cached_popups():
    f = afn();
    plc = True
    double = False

    meta = OMEMetadata(f)

    font_scale = 1.0



    relative_starttime = None #if this movie is sequential (e.g. forward/backward ROI, set this to the inital start time minus the final starttime)    
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

    if not double: #make all movies 5m
        last_frame = np.min(np.where(np.array(timedeltas) > 300)[0])
        assert timedeltas[last_frame-1] <= 300

    r = (extract_ROIs(f))
    im = read_czi(f)

    if not double:
        im = im[:last_frame]

    if not double and plc: #plc single image is much bigger, crop
        #going for 5/8 the height
        im = im[:,:128*5] #top 3/8, center 3/4
        font_scale *= 5/8
    
    p = prettify_movie(im,'neon green',0 if plc else 1)
    roid = stack_draw_ROI(p,r[0],3,'red')
    # roid = stack_draw_ROI(roid,r[0],3,'yellow')

    

    

    def add_m(i,im):
        r = raster_matplotlib(im) #this got... less pretty with needing to return a result
        with r() as ax:
            scale = ScaleBar(meta.PhysicalSizeX,meta.PhysicalSizeXUnit,color='w',box_alpha=0,fixed_value=25,fixed_units=meta.PhysicalSizeXUnit,font_properties={"size":30*font_scale});
            ax.add_artist(scale)

            min = int(timedeltas[i]/60)
            sec = timedeltas[i] % 60
            timestr = f"{min}:{sec:02.2f}"

            ax.text(10,im.shape[0]-10,timestr,horizontalalignment="left",verticalalignment="bottom",bbox=dict(facecolor="black",alpha=0.7),fontdict=dict(color="w",size=40*font_scale))
        return r.result;
            

    scaled = [add_m(i,r) for i,r in enumerate(tqdm(roid,desc="adding annotations"))]

    out = asksaveasfilename();
    fps = 6 if double else 4
    from mediapy import write_video

    write_video(out,tqdm(scaled,desc="writing video"),fps=fps,qp=2)

    # import matplotlib.pyplot as plt

    # from IPython import embed; embed()
    
    # i = imread(f)