import plotly.express as px
import numpy as np
from scipy.fft import fft, fftfreq

from utils.plotly_utils import figureSpec

def fourier_tracks(array:np.ndarray,N:int|None=None):
    lines = (array[:,0],array[:,1])
    normlines = []
    ##normalize
    for line in lines:
        line = (line-np.min(line))/(np.max(line)-np.min(line))*2-1
        normlines.append(line)

    if N is None:
        N = len(array)
    res = []
    for line in normlines:
        y = line
        yf = np.abs(fft(y,N))[:N//2] #only get positive frequencies
        xf = fftfreq(N)[:N//2]
        assert len(yf) == len(xf)
        res.append((xf,yf))
    return res


def plot_fourier_arrays(x:np.ndarray,y1:np.ndarray,y2:np.ndarray,names:tuple[str,str],plot_diff=False,plot_div=False):#->list[tuple[go.Figure|BaseTraceType,...]]:
    lines = [y1,y2];
    names = list(names)
    colors = ["red","blue"]
    if plot_diff:
        lines.append(y2-y1)
        names.append("diff")
        colors.append("green")
    if plot_div:
        lines.append(y2/y1)
        names.append("div")
        colors.append("purple")

    res = []
    for name,y,c in zip(names,lines,colors):
        res.append((px.line(x=x,y=y).data[0].update(name=name,line_color=c,showlegend=True)))
    res.append(px.line(x=x,y=[0]*len(x)).data[0].update(line_color='grey',line_dash="dash",name="zero"))

    return tuple(res)

def plot_fourier_tracks(t1:np.ndarray,t2:np.ndarray,names:tuple[str,str],**kwargs)->figureSpec:
    N = max(len(t1),len(t2))
    (xx1,xy1),(yx1,yy1) = fourier_tracks(t1,N)
    (xx2,xy2),(yx2,yy2) = fourier_tracks(t2,N)
    assert (xx1 == xx2).all()
    assert (yx1 == yx2).all()
    
    fx = plot_fourier_arrays(xx1,xy1,xy2,(f"{names[0]}x",f"{names[1]}x"),**kwargs)
    fy = plot_fourier_arrays(yx1,yy1,yy2,(f"{names[0]}y",f"{names[1]}y"),**kwargs)
    return [fx,fy]