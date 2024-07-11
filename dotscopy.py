import itertools
from typing import DefaultDict
from dash import dcc
from dash_extensions.enrich import DashProxy,Input,Output
from dash import Patch
from libraries.movie_reading import FramedMovieSequence
from libraries.parse_moviefolder import get_movie
from utils.filegetter import afn,adir
from utils.parse_tracks import QCTracksDict
from utils.plotly_utils import Toolbar
import plotly.express as px
import plotly.graph_objects as go

d1 = QCTracksDict(afn(key="track1_unscaled",title="Automatic QC Tracks"))
d2 = QCTracksDict(afn(key="track2_unscaled",title="Manual QC Tracks"))
images = get_movie(adir(key="images",title="Images Folder"))



graph = dcc.Graph(id="graph")
app = DashProxy()
slider = Toolbar(graph,slider=True,slider_range=range(1,len(images[1])+1))
pager = Toolbar(slider.html,slider=True,slider_range=range(1,len(images)+1))

framed_points:dict[int,dict[int,tuple[tuple[list[float],list[float],list[str]],tuple[list[float],list[float],list[str]]]]] = {m: {f:(([],[],[]),([],[],[])) for f in images[m].frames} for m in images.keys()} #x,y,label

for movie_num in images.keys():
    for (tid,track) in d1[movie_num].items():
        for frame,pos in track.items():
            framed_points[movie_num][frame][0][0].append(pos[0])
            framed_points[movie_num][frame][0][1].append(pos[1])
            framed_points[movie_num][frame][0][2].append(str(tid))

    for (tid,track) in d2[movie_num].items():
        for frame,pos in track.items():
            framed_points[movie_num][frame][1][0].append(pos[0])
            framed_points[movie_num][frame][1][1].append(pos[1])
            framed_points[movie_num][frame][1][2].append(str(tid))


@app.callback(
    *slider.slider_range_outsignatures(),
    slider.slider_outsignature(),
    pager.slider_signature(drag=False)
)
def update_movie(movie_num:int):
    return 1,len(images[movie_num])+1,1

@app.callback(
    Output("graph","figure"),
    slider.slider_signature(drag=False),
    pager.slider_signature(state=True)
)
def update_graph(frame:int,movie_num,layout=False):
    im = images[movie_num][frame];
    p1 = px.scatter(x=framed_points[movie_num][frame][0][0],y=framed_points[movie_num][frame][0][1],labels=framed_points[movie_num][frame][0][2]).data[0].update(marker={"color":"red","size":10})
    p2 = px.scatter(x=framed_points[movie_num][frame][1][0],y=framed_points[movie_num][frame][1][1],labels=framed_points[movie_num][frame][1][2]).data[0].update(marker={"color":"blue","size":10})
    # print(p1)
    # from IPython import embed; embed() 
    f = px.imshow(images[movie_num][frame],color_continuous_scale="sunset")
    if layout:
        return go.Figure(dict(data=(p1,)+(p2,)+(f.data[0].update(hoverinfo="skip"),),layout=f.layout))
    else:
        p = Patch()
        p["data"]=(p1,)+(p2,)+(f.data[0].update(hoverinfo="skip"),)
        return p

graph.figure = update_graph(1,1,layout=True)

app.layout = pager.html
app.run()