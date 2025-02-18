import statistics
from math import sqrt
from typing import Callable, Literal

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

def plot_data_linear(position:float,values:ArrayLike,
        orientation:Literal["vertical","horizontal","x","y"]="vertical",
        values_color:str='black',value_marker:None|str|MarkerStyle=None,
        plot_mean:bool=True,mean_color:str='#f44336',
        **kwargs):
    order = 1 if orientation in ("vertical","y") else -1
    if plot_mean:
        plt.plot(*[position, np.mean(values)][::order], 'o', color=mean_color)
    plt.scatter(*[[position]*len(values),values][::order],color=values_color,marker=value_marker);

def plot_data_violin(position:float,values:ArrayLike,
        orientation:Literal["vertical","horizontal","x","y"]="vertical",
        plot_mean:bool=True, plot_extrema:bool=True,
        plot_median:bool=False,quantiles:None|ArrayLike=None,
        width:float=0.5,
        **kwargs):
    if kwargs:
        print(f"WARNING: function {plot_data_violin} received unexpected keyword arguments {kwargs}")
    plt.violinplot([values],[position],vert=orientation in ("vertical","y"),
        showmeans=plot_mean, showmedians=plot_median, showextrema=plot_extrema,
        quantiles=quantiles,widths=[width])


def plot_confidence_interval(position, values, z=1.96, 
        orientation:Literal["vertical","horizontal","x","y"] = "vertical",
        interval_color='#2187bb', cross_width=0.25,
        plot_mean=True, plot_values=True,mean_color:str='#f44336',
        plot_significance=False,significance_center=0,significance_marker="*",significance_space=None,significance_color="black",
        **kwargs):
    
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
    plt.plot(*[[position, position], [top, bottom]][::order], color=interval_color, linewidth=2)
    plt.plot(*[[left, right], [top, top]][::order], color=interval_color, linewidth=2)
    plt.plot(*[[left, right], [bottom, bottom]][::order], color=interval_color, linewidth=2)
    if plot_significance and significant != 0:
        print(significance_space)
        if significant == 1:
            print(top+significance_space)
            pos = [position,top-significance_space][::order]
        else:
            print(bottom-significance_space)
            pos = [position,bottom+significance_space][::order]
        plt.plot(*pos, color=significance_color,marker=significance_marker)


    if plot_values:
        plot_data(position,values,orientation=orientation,plot_mean=plot_mean, mean_color=mean_color,
            **kwargs)
    elif plot_mean:
        plt.plot(*[position, mean][::order], 'o', color=mean_color,)


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