from operator import itemgetter
from typing import Literal
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from utils.filegetter import afn,asksaveasfilename, skip_cached_popups
from csv import DictReader
from scipy.optimize import curve_fit
from skimage.exposure import rescale_intensity
# from dtw import dtw,dtwPlot,dtwPlotAlignment,warp,sakoeChibaWindow
from imageio.v3 import imread,imwrite

def read_csv(file,x=None):    
    res = []
    resx = []
    with open(file,"r") as f:
        d = DictReader(f)
        for r in d:
            res.append(r["Gray_Value"])
            if x:
                resx.append(r[x])
    if x:
        return resx,res
    else:
        return res
    



#this just to make sure names are standardized. kinda silly.
Key = Literal["single","double","quad"]
key:Key = "single"
assert key in Key.__args__,f"Please add key \"{key}\" to the Key literal before continuing"

##We have two inputs: the idealized gradient ("Ground Truth") and our measured gradient ("measured"). neither the x nor the y units apply, so we need to calculate: 
# an x scale, a y scale, and an x-offset. Since we know the ground truth should have a y-offset of 0, the min of the gradient dataset is subtracted from itself.
with skip_cached_popups():
    from_csv = False
    if from_csv:
        grad_groundtruth = afn(key=f"Ground Truth (csv): {key}",defaultextension=".csv",filetypes=[("CSV Files",["*.csv"])])
        GT = np.array(list(map(float,read_csv(grad_groundtruth))))

        grad_measured = afn(key=f"Measured (csv): {key}",defaultextension=".csv",filetypes=[("CSV Files",["*.csv"])])
        M = np.array(list(map(float,read_csv(grad_measured))))
    else:
        grad_groundtruth = afn(key=f"Ground Truth (Image): {key}")
        im_groundtruth = imread(grad_groundtruth).T #groundtruth images are horizontal, let's fix that
        
        grad_measured = afn(key=f"Measured (Image): {key}")
        im_measured = imread(grad_measured)

        GT = np.mean(im_groundtruth,axis=1) #take a full-width region for analysis
        M = np.mean(im_measured,axis=1)[5:-5] #skip the ends because they're 0 for some reason



M = M - M.min()
M_x = np.arange(len(M))
GT = GT - GT.min()
GT_x = np.arange(len(GT))

# from IPython import embed; embed()

##we're gonna do this in two stages. First, we're going to manually specify an (approximate) aligned min-max of the datasets. The user will then edit this code file and put in the numbers.
##That way things are saved for next time for consistency, and I don't have to worry about annoying extra files and such. I suppose I could make this a self-modifying program,
##but, uh, no

#where associated values from each should be stored. First entry is ground truth, second is measured. Doesn't actually have to be maxes of both, just values that should be (proportionally) mapped to each other
maxdict:dict[Key,tuple[float,float]] = {"single":(60000.0, 3980.0),"double":(60000.0, 2450.0),"quad":(60000, 2340.0)}

if key not in maxdict:
    #plot values
    plt.figure("Ground Truth")
    plt.plot(GT_x,GT)
    plt.figure("Measured")
    plt.plot(M_x,M)
    

    with plt.ion():
        plt.show()

    g = input("Please enter max value for ground truth plot: ")
    g = float(g)

    m = input("Please enter max value for measured plot: ")
    m = float(m)

    print(f"specified associated y-values for key {key}: Ground truth = {g}, Measured = {m}. Please add an entry into maxdict in fit_gradients.py of the form \"{key}\":{(g,m)}");
    exit(0)


gmax, mmax = maxdict[key]

GT = GT / gmax
M = M / mmax


#returns the value of the ground truth image given corresponding coordinates in the measured image
def adjusted_M(x,xscale,offset):
    newx = (x - offset)*xscale
    outy = np.interp(newx,GT_x,GT)
    outy[newx < 0] = 0
    outy[newx > len(GT)] = 0
    return outy
    
#returns the groundtruth-adjusted *x-value* given an x coordinate in the measured image (just skips the np.interp lookup)
def adjusted_X(x,xscale,offset):
    return (x - offset)*xscale

#returns the measured-adjusted *x-value* given an x coordinate in the *ground-truth* image. Used for converting coordinates to crop the measured image
def unadjusted_X(x,xscale,offset):
    return x/xscale + offset

def lsq(parameters):
    return np.sum(np.square(adjusted_M(M_x,*parameters)-M));


# key = "single"
titles:dict[Key,str] = {"single":"Single Blade Exponential","double":"Double Blade Exponential","quad":"Quad Blade Linear"}
solutions = {("single",True):(0.06522871278534673, -1445.1020938294241),("double",True):(13, 0.04, -170.0),("quad",True):(13.5, 0.04, -135.0),
             ("single",False):(0.051834735825629856, 408.89448713168025),("double",False):(0.07182337314629508, 4807.604496876261),("quad",False):(0.07142857142857142, 3836.995960572921)}

X = M_x
Y = M

##pad M_x left and right with zeros so that a partial alignment doesn't work
nx = np.arange(-len(M_x),0)
px = np.arange(len(M_x),len(M_x)*2)
X = np.concatenate([nx,X,px])
Y = np.concatenate([np.zeros(nx.shape),Y,np.zeros(px.shape)])


use_solutions = True
if use_solutions and (key,from_csv) in solutions:
     params = solutions[key,from_csv]
else:
    #problem: if we don't bound xscale, it shrinks the GT until the interpolation skips the peaks
    #solution: bound xscale at, say, 50% the length ratio and 150% the length ratio
    length_ratio = len(GT)/len(M)
    scale_bounds = (0.2*length_ratio,5*length_ratio)

    offset_guesses = 30

    offset_center = 0
    offset_center = 4800

    paramList = []
    paramScores = []
    for guess in tqdm(np.linspace(offset_center-len(M),offset_center+len(M),offset_guesses,endpoint=True)):
        res = curve_fit(adjusted_M,X,Y,(length_ratio,guess),maxfev=50000,full_output=True,bounds=([scale_bounds[0],-len(M)/length_ratio],[scale_bounds[1],len(M)/length_ratio]))#,x_scale=(10,60000/4000,100))
        paramList.append(res)
        paramScores.append(np.sum(np.square(res[2]['fvec'])))

    best = min(zip(paramList,paramScores),key=itemgetter(1))[0]
    print("param sweep found best values:",best)
    params,pcov,*res = curve_fit(adjusted_M,X,Y,best[0],maxfev=50000,full_output=True,bounds=([scale_bounds[0],-len(M)],[scale_bounds[1],len(M)]))#,x_scale=(10,60000/4000,100))
   


xscale,offset = params


def plt_params(xscale,offset):
    print("plotting parameters:",(xscale,offset))
    import matplotlib.pyplot as plt
    plt.close('all')
    px,py = X,Y #represent minimization domain
    px,py = M_x,M #represent ground-truth domain

    adj_Gx = adjusted_X(M_x,xscale,offset) #plot against adjusted X so x-values are in the ground truth space
    adj_Gy = adjusted_M(M_x,xscale,offset)

    mm_factor = 900 / GT_x.shape[0] # millimeters per GT pixel
    adj_Gx = adj_Gx*mm_factor

    plt.figure(figsize=(10,5),dpi=200)
    plt.plot(adj_Gx,adj_Gy,label="Theoretical")
    plt.plot(adj_Gx,M,label="Observed")
    plt.title(titles[key])
    plt.legend(loc="upper right")

    plt.xlabel("Position (mm) Relative to Theoretical")
    plt.ylabel("Intensity")
    plt.show()

def get_aligned_measure_crop(xscale,offset):
    assert not from_csv
    #returns the measured image, cropped to the same range as the gradient.
    start,end = unadjusted_X(np.array([0,len(GT)]),xscale,offset);

    return im_measured[int(start):int(end)]

from IPython import embed; embed()

if not from_csv and False:
    im = get_aligned_measure_crop(xscale,offset)
    im = rescale_intensity(im,(0,im.max()))
    plt.imshow(im); plt.figure(); plt.imshow(im_groundtruth); plt.show()
    out = asksaveasfilename(key=f"Cropped Gradient Image: {key}")
    imwrite(out,im)


