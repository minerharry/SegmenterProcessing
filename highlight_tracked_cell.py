from pathlib import Path
import plotly
import plotly.express as px
from skimage.io import imread
from skimage.exposure import rescale_intensity
from dash import dcc
from dash import no_update
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Input,Output,DashProxy,MultiplexerTransform
from utils.paging import Index
from utils.filegetter import afn,adir
from utils.parse_tracks import QCTracks, QCTracksArray
from utils.plotly_utils import Subplotter, Toolbar, CallbackIndexer, opacify_colorscale, opacify_image
from utils.trackmasks import get_trackmasks
import plotly.graph_objects as go
from tifffile import TiffFile
import numpy as np


def colorize(l:list[str]):
    return tuple(zip(np.linspace(0,1,len(l)),l))

bg_coloraxis = go.layout.Coloraxis(colorscale=colorize(plotly.colors.sequential.Greys))
fg_coloraxis = go.layout.Coloraxis(colorscale=opacify_colorscale(colorize(plotly.colors.sequential.Reds),0.4))


movie_folder = Path(adir(title="Movie Folder",key="mfolder"))
tmfolder,tracksmasks = get_trackmasks(afn(title="Tracks Masks Zip",key="trackmasks",filetypes=[("Trackmasks Zip file","*.zip")]))

app = DashProxy(prevent_initial_callbacks=True, transforms=[MultiplexerTransform()])

plotter = Subplotter()
graph = dcc.Graph(figure=plotter.figure,id="fig", responsive=True)
slider = Toolbar(graph,aio_id="slider_toolbar",slider=True)
pager = Toolbar(slider.html,aio_id="paging_toolbar",paging=True)

currtrack = (-1,-1)
currframe = -1
currtiff:TiffFile = None
startframe = -1



def updatetrack(movie:int,track:int,frame:int|None=None):
    print(f"updating track to ({movie},{track})")
    global currtiff,startframe,currtrack
    currtiff = TiffFile(tracksmasks[movie][track])
    startframe = currtiff.shaped_metadata[0]["startframe"]
    currtrack = (movie,track)
    plotter.update([],layout={"title":f"Movie {movie}, Track {track}","coloraxis":bg_coloraxis,"coloraxis2":fg_coloraxis})
    return updateframe(frame or startframe)

def updateframe(frame:int):
    print("updating frame to frame",frame)
    global currframe
    currframe = frame
    idx = frame - startframe 
    mask = currtiff.series[0][idx].asarray()
    image = imread(movie_folder/f"p_s{currtrack[0]}_t{frame}.TIF")

    im1 = px.imshow(image,aspect='equal').data[0]
    im2 = px.imshow(mask,aspect='equal').data[0].update(coloraxis="coloraxis2")

    plotter.update([(im1,im2)],axis_specs=({"scaleratio":1},None),subplot_kwargs=dict(shared_xaxes = "all", shared_yaxes = "all"));
    return plotter.figure,(startframe,startframe+len(currtiff.series[0]))


def changetrack(selection:tuple[int,int],callback_args:tuple[int,int]|None=None):
    print("changing track")
    movie,track = selection
    frame = no_update
    nrange = [no_update,no_update]
    if currtrack != selection:
        fig,nrange = updatetrack(movie,track)
        frame = currframe
    elif currframe != frame:
        assert callback_args is not None
        fig = updateframe(callback_args[1]);
        frame = no_update
    else:
        raise PreventUpdate
    return fig,frame,*nrange

signatures = tuple(zip(pager.paging_signatures(),[slider.slider_signature(drag=False)]*2))

tracks = [(1, 7), (1, 39), (1, 40), (1, 106), (1, 78), (1, 110), (1, 52), (2, 52), (3, 39), (3, 74), (3, 50), (3, 52), (4, 9), (4, 77), (4, 83), (4, 88), (4, 41), (4, 42), (4, 55), (4, 58), (4, 59), (4, 62), (5, 0), (5, 2), (5, 131), (5, 67), (5, 6), (5, 14), (5, 271), (5, 273), (5, 21), (5, 149), (5, 214), (5, 153), (5, 155), (5, 44), (5, 306), (5, 51), (5, 180), (5, 244), (6, 195), (6, 197), (6, 71), (6, 75), (6, 86), (6, 91), (6, 98), (6, 100), (6, 37), (6, 167), (6, 171), (6, 45), (6, 185), (6, 122), (6, 190), (7, 0), (7, 3), (7, 9), (7, 19), (7, 30), (7, 31), (7, 117), (8, 3), (8, 117), (8, 231), (8, 25), (8, 234), (9, 3), (9, 67), (9, 262), (9, 9), (9, 16), (9, 273), (9, 149), (9, 278), (9, 23), (9, 90), (9, 91), (9, 33), (9, 353), (9, 300), (9, 47), (10, 5), (10, 197), (10, 70), (10, 11), (10, 141), (10, 77), (10, 79), (10, 209), (10, 19), (10, 214), (10, 23), (10, 91), (10, 29), (10, 31), (10, 98), (10, 99), (10, 102), (10, 40), (10, 105), (10, 113), (10, 117), (10, 254)]
index = Index(None,changetrack,custom_range=tracks)
caller = CallbackIndexer(app,[Output("fig","figure"),slider.slider_outsignature(drag=False),*slider.slider_range_outsignatures()],signatures,index)
print(index.current)
# current = tracks[0]
graph.figure,slider.slider.value,slider.slider.min,slider.slider.max = changetrack(index.current)

app.layout = pager.html
app.run(debug=True,use_reloader=False)