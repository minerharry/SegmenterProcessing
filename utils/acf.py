## Initial code from Elizabeth M. Davis, Tim Elston Lab, UNC, https://github.com/emdavis2/cell-migration-analysis/blob/main/functions/acf_functions.py

from typing import Sequence
import numpy as np
import tidynamics


def autocorrelate(dataset:Sequence[Sequence[float]]|Sequence[float],normalize:bool=True):
    """
    AUTOCORRELATION: Calculate the correlation between a dataset and itself, and see how that correlation changes as you increase the time lag. 
    See writeup (ACF.docx) for complex math details. Generally, you simply take the dot product of the dataset with itself shifted
    Generally, you also normalize such that correlation(lag=0) = 1. This can be disabled with normalize=False
    Lags are from 0+: a lag of -2 is the the same as the negative of the lag of 2.
    
    Dataset can either be a list of vectors or a list of scalars; anything that can be dotted/multiplied. 
    Numpy's method for correlate is actually rather slow; using tidynamics instead, which also does dot products in-house

    Returns: List of autocorrelations corresponding to lag times 0, 1, 2, 3, ... N
    """

    data = np.array(dataset)

    correlation = tidynamics.acf(data)
            
    if normalize: correlation = correlation / correlation[0]

    return correlation




###OLD CODE
## above acf is equivalent to xcorr_vector and xcorr (generalized both). To recreate xcorr_direction,


# xcorr_direction is used when calculating acf for velocity angle
# def xcorr_direction(dfraw, min_track_length):
#   df = dfraw.dropna()
#   norms1 = (df.iloc[:,0]**2 + df.iloc[:,1]**2)**0.5 
#   norms2 = (df.iloc[:,2]**2 + df.iloc[:,3]**2)**0.5
#   v1x = np.asarray(df.iloc[:,0]/norms1)
#   v1y = np.asarray(df.iloc[:,1]/norms1)
#   v2x = np.asarray(df.iloc[:,2]/norms2)
#   v2y = np.asarray(df.iloc[:,3]/norms2)

#   length = len(df)
#   poslagsmean=[]
#   neglagsmean=[]
#   Nposlags=[]
#   Nneglags=[]
#   for lag in range(length):
#     poslags =  v2x[lag:length]*v1x[0:length-lag] + v2y[lag:length]*v1y[0:length-lag]
#     neglags =  v2x[0:length-lag]*v1x[lag:length] + v2y[0:length-lag]*v1y[lag:length]
#     poslagsmean.append(np.nanmean(poslags))
#     neglagsmean.append(np.nanmean(neglags))
#     Nposlags.append(sum(~np.isnan(poslags)))
#     Nneglags.append(sum(~np.isnan(neglags)))

#   return np.asarray(poslagsmean[0:min_track_length-4]), np.asarray(Nposlags[0:min_track_length-4]), np.asarray(neglagsmean[0:min_track_length-4]), np.asarray(Nneglags[0:min_track_length-4])


# # xcorr_vector is used when calculating acf for polarity vector, polarity angle, and velocity
# def xcorr_vector(*v, min_track_length):
#   v1x = v[0]
#   v1y = v[1]
#   v2x = v[2]
#   v2y = v[3]

#   length = len(v[0])
#   poslagsmean=[]
#   neglagsmean=[]
#   Nposlags=[]
#   Nneglags=[]
#   for lag in range(length):
#     poslags =  v2x[lag:length]*v1x[0:length-lag] + v2y[lag:length]*v1y[0:length-lag]
#     neglags =  v2x[0:length-lag]*v1x[lag:length] + v2y[0:length-lag]*v1y[lag:length]
#     poslagsmean.append(np.nanmean(poslags))
#     neglagsmean.append(np.nanmean(neglags))
#     Nposlags.append(sum(~np.isnan(poslags)))
#     Nneglags.append(sum(~np.isnan(neglags)))

#   return np.asarray(poslagsmean[0:min_track_length-4])/poslagsmean[0], np.asarray(Nposlags[0:min_track_length-4]), np.asarray(neglagsmean[0:min_track_length-4])/neglagsmean[0], np.asarray(Nneglags[0:min_track_length-4])

if __name__ == "__main__":
    # data = [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(11,11),(12,12),(13,13),(14,14),(15,15)]
    data = list(range(10))
    from IPython import embed; embed()

# # xcorr is used when calculating abs-skew, speed, speed_x, and speed_y
# def xcorr(dfraw, min_track_length):
#   df = dfraw.dropna()
#   v1 = np.asarray(df.iloc[:,0])  
#   v2 = np.asarray(df.iloc[:,1])
#   v1 = v1 - v1.mean()
#   v2 = v2 - v2.mean()
#   length = len(df)
#   poslagsmean=[]
#   neglagsmean=[]
#   Nposlags=[]
#   Nneglags=[]
#   for lag in range(length):
#     poslags =  v2[lag:]*v1[0:length-lag] 
#     neglags =  v2[0:length-lag]*v1[lag:] 
#     poslagsmean.append(np.nanmean(poslags))
#     neglagsmean.append(np.nanmean(neglags))
#     Nposlags.append(sum(~np.isnan(poslags)))
#     Nneglags.append(sum(~np.isnan(neglags)))

#   return np.asarray(poslagsmean[0:min_track_length-4])/poslagsmean[0], np.asarray(Nposlags[0:min_track_length-4]), np.asarray(neglagsmean[0:min_track_length-4])/neglagsmean[0], np.asarray(Nneglags[0:min_track_length-4])