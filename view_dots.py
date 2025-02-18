from functools import cache
import itertools
from pathlib import Path
import re
from typing import Any
from dash import dcc
from dash_extensions.enrich import DashProxy,Input,Output,BlockingCallbackTransform
from dash import Patch
import joblib
import numpy as np
from ordered_set import OrderedSet
from tqdm import tqdm as tqdm
from libraries.movie_reading import FramedMovieSequence
from libraries.parse_moviefolder import get_movie
from utils.associations import get_full_associations
from utils.filegetter import afn,adir
from utils.parse_tracks import QCTracks, QCTracksDict
from utils.plotly_utils import Toolbar
import plotly.express as px
import plotly.graph_objects as go
from utils.trackmasks import get_trackmasks,read_trackmask
from utils.outlines import get_mask_outlines

dnames = [afn(key="track1_unscaled",title="Automatic QC Tracks (Unscaled)"),afn(key="track2_unscaled",title="Manual QC Tracks (Unscaled)")]
qs = [QCTracks(d) for d in dnames]
d1 = QCTracksDict(qs[0])
d2 = QCTracksDict(qs[1])
images = get_movie(adir(key="images",title="Images Folder"))


experiment = int(re.match(r".*?(\d+)$",str(Path(dnames[0]).parent)).group(1))
# print(experiment)
# from IPython import embed; embed()

smoothing = "smoothed" if "smoothed" in dnames[0] else "raw" if "raw" in dnames[0] else "noqc"
# assert smoothing is not None
# assert smoothing in dnames[0] and smoothing in dnames[1]
assoc_path = f"associations/assoc_results_{experiment}_{smoothing}.txt"

associations,inclusions,remainders = get_full_associations(qs[0],qs[1],names=("Automatic","Manual"),savepath=assoc_path)


movie_num = 1
movie:FramedMovieSequence = images[movie_num]

graph = dcc.Graph(id="graph")
app = DashProxy(transforms=[BlockingCallbackTransform(timeout=2)])
slider = Toolbar(graph,slider=True,slider_range=range(1,len(movie)+1))

framed_points:dict[int,tuple[tuple[list[float],list[float],list[str]],tuple[list[float],list[float],list[str]]]] = {f:(([],[],[]),([],[],[])) for f in movie.frames} #x,y,label


memory = joblib.Memory(location="caches/dots")

for (tid,track) in tqdm(d1[movie_num].items(),desc="framing track1 points"):
    for frame,pos in track.items():
        framed_points[frame][0][0].append(pos[0])
        framed_points[frame][0][1].append(pos[1])
        framed_points[frame][0][2].append(str(tid))

for (tid,track) in tqdm(d2[movie_num].items(),desc="framing track2 points"):
    for frame,pos in track.items():
        framed_points[frame][1][0].append(pos[0])
        framed_points[frame][1][1].append(pos[1])
        framed_points[frame][1][2].append(str(tid))

framed_associations:dict[int,list[tuple[tuple[float,float],tuple[float,float]]]] = {f:[] for f in movie.frames} #list of pairs of associated points in each frame

for tid1,tid2 in tqdm(associations[movie_num],desc="storing associations..."):
    t1,t2 = d1[movie_num][tid1],d2[movie_num][tid2]
    frames = set(t1.keys()).intersection(t2.keys())
    for frame in frames:
        framed_associations[frame].append((t1[frame],t2[frame]))

print(framed_associations)
path = afn(key="trackmasks",title="Track Masks (auto)")
@memory.cache
def get_outlines(movie_num:int,trackmask_path:str):
    _,trackmasks = get_trackmasks(trackmask_path)
    framed_outlines:dict[int,dict[int,np.ndarray]] = {f:{} for f in movie.frames}
    for tid,trackmask in tqdm(trackmasks[movie_num].items(),desc="outlining..."):
        frames,series = read_trackmask(trackmask)
        for frame,im in zip(frames,series):
            os = get_mask_outlines(im.asarray())
            if len(os) > 0:
                outline = os[0] #should only be one!!
                framed_outlines[frame][tid] = outline
    return framed_outlines
framed_outlines = get_outlines(movie_num,path)


track_order = OrderedSet()

@app.callback(
    Output("graph","figure"),
    slider.slider_signature(drag=False),
    blocking=True
)
@cache
def update_graph(frame:int,layout=False):
    print(framed_points[frame][0][0])
    print(framed_points[frame][0][1])
    print(framed_points[frame][1][0])
    print(framed_points[frame][1][1])
    if framed_points[frame][0][0]:
        p1 = (px.scatter(x=framed_points[frame][0][0],y=framed_points[frame][0][1],text=framed_points[frame][0][2]).data[0].update(marker={"color":"rgba(255,0,0,0.5)","size":10},textposition="middle right"),)
    else:
        p1 = tuple()
    if framed_points[frame][1][0]:
        p2 = (px.scatter(x=framed_points[frame][1][0],y=framed_points[frame][1][1],text=framed_points[frame][1][2]).data[0].update(marker={"color":"rgba(0,0,255,0.5)","size":10},textposition="middle right"),)
    else:
        p2 = tuple()
    
    if framed_associations[frame]:
        x = []
        y = []
        for (x1,y1),(x2,y2) in framed_associations[frame]:
            x.extend([x1,x2,None])
            y.extend([y1,y2,None])
        p3 = (px.line(x=x,y=y).data[0].update(line={"color":"rgba(255,0,255,0.8)"}),)

    data_list = []
    if (outs := framed_outlines[frame]):
        data = {(trackid,track_order.add(trackid)):trackdata for trackid,trackdata in outs.items()};
        data_list:list[Any] = [None]*len(track_order)
        for (tid,index),d in data.items():
            d = go.Scatter(x=d[:,0],y=d[:,1],hovertext=[f"track {tid}"]*d.shape[0],fill="toself",showlegend=False)
            data_list[index] = d
    data_list = tuple(data_list)

    image = movie[frame]
    image = px.imshow(image,color_continuous_scale="sunset")
    image.data[0].hoverinfo = "none"

    figure_data = image.data + p1 + p2 + p3 + data_list

    if layout:
        return go.Figure(dict(data=figure_data,layout=image.layout.update(title=f"movie {movie_num}",transition={
                'duration': 500,
                'easing': 'cubic-in-out'
        })))
    else:
        p = Patch()
        p["data"]=figure_data
        return p

graph.figure = update_graph(1,layout=True)

app.layout = slider.html
app.run()