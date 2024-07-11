import copy
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import plotly
from plotly.basedatatypes import BaseTraceType
from fourier import plot_fourier_tracks
from libraries.analysis import analyze_experiment_tracks
from utils.filegetter import afn
import numpy as np
from utils.paging import Index
from utils.parse_tracks import QCTracks, QCTracksArray
from utils.associations import AssociateTracks
from dash import dcc
from dash_extensions.enrich import DashProxy, Output, Input, State, MultiplexerTransform,clientside_callback
from dash import html
from tifffile import TiffFile, TiffPageSeries

from utils.plotly_utils import CallbackIndexer, Subplotter, Toolbar,figureSpec
from utils.trackmasks import get_trackmasks, read_trackmask

# while True:

autracks = QCTracks(autrackloc := Path(afn(title="automatic",key="autotrack")))
mantracks = QCTracks(mantrackloc := Path(afn(title="manual",key="mantrack")))

# autfolder = autrackloc.parent
# manfolder = mantrackloc.parent

_,trackmasks = get_trackmasks(afn(title="track masks folder (auto keyed)",key="trackmasks"))

autanalysis = analyze_experiment_tracks(autracks,'approximate-medoid')
mananalysis = analyze_experiment_tracks(mantracks,'approximate-medoid')

associations = sum([[(m,) + a for a in AssociateTracks(autracks[m],mantracks[m],bar=True)] for m in set(autracks.keys()).intersection(mantracks.keys())],[])

print(associations)

selection = associations[1]
print(selection)

# exit()
autrackarr,mantrackarr = QCTracksArray(autracks),QCTracksArray(mantracks)

def fmi(x,y):
    lengths = np.sqrt(np.sum(np.diff(np.stack([x,y],axis=1), axis=0)**2, axis=1)) # Length between corners
    total_length = np.sum(lengths)
    print(total_length)

    fx = (x[-1]-x[0])/total_length;
    fy = (y[-1]-y[0])/total_length;
    return (fx,fy)


def plot_tracks(selection:tuple[int,int,int]):
    # print(type([k for k in selection]))
    print(selection)
    # print(type)
    
    movie,s1,s2 = selection
    print("plotting track selection:",selection)
    t1,t2 = np.array(autrackarr[movie][s1]),np.array(mantrackarr[movie][s2])
    # print(np.min(t1),np.max(t1))
    # print(np.min(t2),np.max(t2))
    
    l1 = px.line(x=t1[:,0],y=t1[:,1],markers=True).data[0].update(name=f"auto ({s1})",showlegend=True,line_color="red",marker_symbol=["star"]+["circle"]*(len(t1)-2)+["square"])
    l2 = px.line(x=t2[:,0],y=t2[:,1],markers=True).data[0].update(name=f"manual ({s2})",showlegend=True,line_color="black",marker_symbol=["star"]+["circle"]*(len(t2)-2)+["square"])
    # xs = np.concatenate([t2[:,0],t1[:,0]])
    # ys = np.concatenate([t2[:,1],t1[:,1]])

    
    
    info:list = []
    for name,t,s,anal in (("Automatic",t1,s1,autanalysis),("Manual",t2,s2,mananalysis)):
        f = anal.FMI[movie][s]
        length = anal.trackLength[movie][s]
        time = anal.trackTime[movie][s]
        info.append(html.Div(
                [
                    html.Span(f"{name} Track Stats:"),
                    html.Br(),
                    html.Span(f"FMI.x: {f[0]}, FMI.y: {f[1]}"),
                    html.Br(),
                    html.Span(f"Track Length: {length} um"),
                    html.Br(),
                    html.Span(f"Track Duration: {time} minutes"),
                ],
            style={"display": "table-cell"}
        ))
        

    return [(l1,),(copy.deepcopy(l1),l2),(copy.deepcopy(l2),)],["Automatic"," Automatic vs Manual","Manual"],html.Div(info,style={"display": "table-row"},)
    
    

def plot_fourier_association(selection:tuple[int,int,int]):
    movie,s1,s2 = selection
    print("plotting association:",selection)
    t1,t2 = np.array(autrackarr[movie][s1]),np.array(mantrackarr[movie][s2])
    return plot_fourier_tracks(t1,t2,("auto","manual"),plot_diff=True,plot_div=True)


if __name__ == "__main__":
    app = DashProxy(prevent_initial_callbacks=True,transforms=[MultiplexerTransform()])

    words = html.Div(id="words",style={"width": "100%", "display": "table"})
    fig = dcc.Graph(id="figure")
    fig_store = dcc.Store(id="figure_store")
    trackmasks_store = dcc.Store(id="trackmasks_store")
    slider_toolbar = Toolbar(fig,slider=True)
    content = html.Div([slider_toolbar.html,words,trackmasks_store,fig_store])
    plotter = Subplotter()
    paging_bar = Toolbar(content,paging=True)
    app.layout = paging_bar.html
    
    currmask:TiffPageSeries = None

    def updateplot(selection:tuple[int,int,int]):
        global currmask
        print("seel",selection)
        fig,names,children = plot_tracks(selection)
        movie, s1, s2 = selection;
        plotter.update(fig,layout={"title":f"Movie {movie}, Track {s1}/{s2}"},subtitles=names,subplot_kwargs=dict(shared_xaxes = "all", shared_yaxes = "all"));

        maskfile = trackmasks[movie][s1]
        frames,currmask = read_trackmask(maskfile)
        
        image_frames = []
        for f in frames:
            data = px.imshow(currmask[f-frames[0]].asarray(),binary_string=True).data[0]
            image_frames.append(data)

        from pympler.asizeof import asizeof
        print(asizeof(image_frames))

        return (plotter.figure,children,(frames[0],frames[-1],image_frames))


    @app.callback(
        slider_toolbar.slider_outsignature(),
        slider_toolbar.slider_range_outsignatures(),
        Input("trackmasks_store","data"),
        prevent_initial_call=False
    )
    def update_slider(data:tuple[int,int,np.ndarray]):
        return (data[0],data[0],data[1])



    # clientside_callback(
    #     """
    #     function(frame,maskdata,figure) {
    #         d = figure["data"]
    #         d.unshift(maskdata[2][frame-maskdata[0]])
    #         return {"data":d,
    #                 "layout":figure["layout"]}
    #     }
    #     """,
    @app.callback(
        Output("figure","figure"),
        slider_toolbar.slider_signature(),
        State("trackmasks_store","data"),
        State("figure_store","data"),
        prevent_initial_call=False
    )
    def update_maskfigure(frame:int,maskdata:tuple[int,int,list],figure):
        return {"data":[maskdata[2][frame-maskdata[0]],*figure["data"]],
                "layout":figure["layout"]}


    # @app.callback(
    #     Output("fourier_fig","figure"),
    #     Output("words","children"),
    #     slider_toolbar.slider_signature()
    # )
    # def updateframe(slider_value):

    print(associations)

    index = Index(None,updateplot,custom_range=associations);
    CallbackIndexer(app,
            [Output("figure_store","data"),
            Output("words","children"),
            Output("trackmasks_store","data")],paging_bar.paging_signatures(),index);


    
    print("seeeeeeee:",index.current)
    # pft = plot_tracks(index.current)
    fig_store.data,words.children,data = updateplot(index.current)
    trackmasks_store.data = data
    # slider_toolbar.slider.
    # print(fig.figure)
    # from IPython import embed; embed() 

    # full_fig = fig.figure.full_figure_for_development()
    # from IPython import embed
    # embed()

    app.run(debug=True,use_reloader=False)