from typing import Callable
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.axes
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

def input_rectangle(ax:matplotlib.axes.Axes,show_fig:Figure|None|Callable[[],None]=None):
    """NOTE: WILL OPEN SHOW_FIG (or call the function) IN A BLOCKING MANNER TO DETERMINE USER INPUT. THIS MEANS THAT THE PASSED FIGURE (OR ALL FIGURES IF NOT PASSED) WILL BE *CLOSED*!"""
    selector = RectangleSelector(ax)

    if show_fig is None:
        show = plt.show
    elif callable(show_fig):
        show = show_fig
    else:
        show = show_fig.show
    
    def on_select


    show()
    

