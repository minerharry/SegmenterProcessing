import statistics
from math import sqrt
from typing import Callable, Literal

from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
import numpy as np

#confidence interval code from https://stackoverflow.com/a/70949996/13682828


def plot_data(position:float,values:ArrayLike,
        type:Literal["linear","violin"]="linear",orientation:Literal["vertical","horizontal","x","y"]="vertical",
        **kwargs):
    
    order = 1 if orientation in ("vertical","y") else -1
    match type:
        case "linear":
            plot_data_linear(position,values,orientation=orientation,**kwargs)
        case "violin":
            plot_data_violin(position,values,orientation=orientation,**kwargs)

from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10, size=1):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs(size)

def plot_data_linear(position:float,values:ArrayLike,
        orientation:Literal["vertical","horizontal","x","y"]="vertical",
        values_color:str='black',value_marker:None|str|MarkerStyle=None,value_markersize=1.5,
        plot_mean:bool=True,mean_color:str='#f44336',
        values_width:float=0,
        ax:plt.Axes|None=None,
        **kwargs):
    pl = ax or plt
    order = 1 if orientation in ("vertical","y") else -1
    if plot_mean:
        pl.plot(*[position, np.mean(values)][::order], 'o', color=mean_color)
    
    positions = np.array([position]*len(values))
    if values_width != 0:
        positions = positions + get_truncated_normal(mean=0,sd=values_width/6,low=-values_width/2,upp=values_width/2,size=len(positions))*values_width
    
    pl.scatter(*[positions,values][::order],color=values_color,marker=value_marker,s=value_markersize);

def plot_data_violin(position:float,values:ArrayLike,
        orientation:Literal["vertical","horizontal","x","y"]="vertical",
        plot_mean:bool=True, plot_extrema:bool=True,
        plot_median:bool=False,quantiles:None|ArrayLike=None,
        values_width:float=0.5,
        ax:plt.Axes|None=None,
        **kwargs):
    pl = ax or plt
    if kwargs:
        print(f"WARNING: function {plot_data_violin} received unexpected keyword arguments {kwargs}")
    pl.violinplot([values],[position],vert=orientation in ("vertical","y"),
        showmeans=plot_mean, showmedians=plot_median, showextrema=plot_extrema,
        quantiles=quantiles,widths=[values_width])


def plot_confidence_interval(position, values:list[float]|list[list[float]], z=1.96, 
        orientation:Literal["vertical","horizontal","x","y"] = "vertical",
        interval_color='#2187bb', cross_width=0.25, 
        plot_mean=True, plot_values=True,mean_color:str='#f44336', values_width:float=0.4,
        plot_significance=False,significance_center=0,significance_marker="*",significance_space=None,significance_color="black",
        ax:plt.Axes|None=None,
        **kwargs):
    pl = ax or plt

    values:np.ndarray = np.array(values)
    if len(values.shape) > 1:
        shaped_values = values
        values = np.reshape(values,shape=(-1,));
    else:
        shaped_values = np.array([values])
    
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)

    confidence_interval = z * stdev / sqrt(len(values))

    left = position - cross_width / 2
    top = mean - confidence_interval
    right = position + cross_width / 2
    bottom = mean + confidence_interval

    if significance_space is None:
        significance_space = confidence_interval/6

    significant = 0
    if top > significance_center:
        significant = -1
    elif bottom < significance_center:
        significant = 1

    order = 1 if orientation in ("vertical","y") else -1

    if plot_values:
        positions = np.linspace(position - values_width/2,position + values_width/2,len(shaped_values),endpoint=True) if len(shaped_values) > 1 else [position]
        for p,v in zip(positions,shaped_values):
            plot_data(p,v,orientation=orientation,plot_mean=plot_mean, mean_color=mean_color,ax=ax,values_width=0.9*values_width/len(shaped_values),
                **kwargs)
    elif plot_mean:
        pl.plot(*[position, mean][::order], 'o', color=mean_color,)

    
    pl.plot(*[[position, position], [top, bottom]][::order], color=interval_color, linewidth=2)
    pl.plot(*[[left, right], [top, top]][::order], color=interval_color, linewidth=2)
    pl.plot(*[[left, right], [bottom, bottom]][::order], color=interval_color, linewidth=2)
    if plot_significance and significant != 0:
        print(significance_space)
        if significant == 1:
            print(top+significance_space)
            pos = [position,top-significance_space][::order]
        else:
            print(bottom-significance_space)
            pos = [position,bottom+significance_space][::order]
        pl.plot(*pos, color=significance_color,marker=significance_marker)


    


    return mean, confidence_interval


plot_CI = plot_confidence_interval

if __name__ == "__main__":

    plt.yticks([1, 2, 3, 4], ['FF', 'BF', 'FFD', 'BFD'])
    plt.title('Confidence Interval')
    plot_confidence_interval(1, [10, 11, 42, 45, 44],orientation="horizontal",plot_significance=True)
    plot_confidence_interval(2, [10, 21, 42, 45, 44],orientation="horizontal")
    plot_confidence_interval(3, [20, 2, 4, 45, 44],orientation="horizontal")
    plot_confidence_interval(4, [30, 31, 42, 45, 44],orientation="horizontal")
    plt.show()