

from multiprocessing.sharedctypes import Value
import numpy as np


class LogScale:
    """
    Math and inspiration from https://stackoverflow.com/a/63158920/13682828
    Given a linear scale whose values range from x0 to x1, and a logarithmic scale whose values range from y0 to y1, the mapping between x and y (in either direction) is given by the relationship shown in equation 1:

    x - x0    log(y) - log(y0)
    ------- = -----------------      (1)
    x1 - x0   log(y1) - log(y0)

    where,

    x0 < x1
    { x | x0 <= x <= x1 }

    y0 < y1
    { y | y0 <= y <= y1 }
    y1/y0 != 1   ; i.e., log(y1) - log(y0) != 0
    y0, y1, y != 0

    factor = (x1 - x0) / (log(y1) - log(y0))
    y = exp((x - x0) / factor + log(y0))
    x = (log(y) - log(y0))*factor + x0
    """

    def __init__(self,linear_range:tuple[float,float],log_range:tuple[float,float]):
        if log_range[0] <= 0 or log_range[1] <= 0:
            raise ValueError("Logarithmic bounds must be greater than zero")
        assert linear_range[1] > linear_range[0]
        assert log_range[1] > log_range[0]
        self.linrange = linear_range
        self.logrange = log_range
        linden = linear_range[1] - linear_range[0]
        logden = np.log(log_range[1]) - np.log(log_range[0])
        self.factor = linden/logden

    def linear_to_log(self,x:float|np.ndarray):
        assert np.all(self.linrange[0] <= x) and np.all(self.linrange[1] >= x),x

        return np.exp((x - self.linrange[0])/self.factor + np.log(self.logrange[0]))
    
    def log_to_linear(self,y:float|np.ndarray):
        assert np.all(self.logrange[0] <= y) and np.all(self.logrange[1] >= y),y

        return (np.log(y)-np.log(self.logrange[0]))*self.factor + self.linrange[0]
            
