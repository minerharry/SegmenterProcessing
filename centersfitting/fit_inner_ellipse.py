"""Black box code: sourced from https://math.stackexchange.com/questions/4220411/find-ellipse-of-maximum-area-which-is-entirely-contained-within-the-area-defined
and ported to python by ChatGPT"""


import functools
import re
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize
from skimage.segmentation import find_boundaries

try:
    from centersfitting.mask_union_steps import centroid
    from libraries.centers import getcircle,getellipse
    from utils.point_in_polygon import is_inside_sm
except:
    import ultraimport
    getcircle,getellipse = ultraimport('__dir__/../libraries/centers.py',('getcircle','getellipse')) #type:ignore
    from mask_union_steps import centroid
    is_inside_sm = ultraimport('__dir__/../utils/point_in_polygon.py','is_inside_sm'); #type:ignore


# print(data)
# exit()
# Define the data
# data = np.array([[1, 2], [3, 4], [4, 6], [7, 8], [9, 10]])

def normalize_data(data:np.ndarray,params:tuple[tuple[float,float],float]|None=None): #data is array of n-dimensional points
    #if params provided, will normalize new data to given params
    if not params:
        center = np.sum(data,axis=0)/data.shape[0]
    else:
        center = params[0]

    # Center the data
    data = data - center

    # print(data)

    # Separate the data into X and Y
    X = data[:, 0]
    Y = data[:, 1]


    if not params:

        # Compute the minimum and maximum values of X and Y
        xmin, xmax = np.min(X), np.max(X)
        ymin, ymax = np.min(Y), np.max(Y)

        # Compute dX and dY
        dX = xmax - xmin
        dY = ymax - ymin

        # Compute dXY
        dXY = max(dX, dY)

    else:
        dXY = params[1]

    # Scale X and Y by dXY
    Xs = X/dXY
    Ys = Y/dXY

    # Combine Xs and Ys into a single array
    datas = np.array([Xs, Ys]).T
    # print(datas)
    return center,dXY,datas

def normalize_ellipse(ellipse:np.ndarray,center:tuple[float,float],dXY:float):
    ellipse = ellipse.copy()
    ellipse[2:4] -= center
    ellipse[0:4] /= dXY
    return ellipse

def denormalize_ellipse(ellipse:np.ndarray,center:tuple[float,float],dXY:float):
    ellipse = ellipse.copy()
    ellipse[0:4] *= dXY #scale position, axes up
    ellipse[2:4] += center #shift center by ellipse center
    return ellipse


# F: conic section function (solution is f=0)
def f(x, y, params):
    a, b, x0, y0, t = params
    return b**2 * ((x - x0) * np.cos(t) + (y - y0) * np.sin(t))**2 + a**2 * ((x - x0) * np.sin(t) - (y - y0) * np.cos(t))**2 - a**2 * b**2

# constraint: conic of the input points must be positive, meaning the ellipse (conic=0) must be outside the points
def outside_restraint(datas,params):
    return f(datas[:, 0], datas[:, 1],params).reshape(-1) - 1e-8

# Define the objective function: to be minimized
#returns negative ellipse area, so maximizes area
def obj(params):
    a, b = params[:2]
    return -a*b

dum = (0,0)
extra_ellipses=[] ##all normalized
def fit_interior_ellipse(data:np.ndarray,normalized_in=False,normalized_out=False,fixed_center=None,extra_constraints:None|list=None)->np.ndarray:
    """
    Input: array of x,y data points
    Output: tuple of (semimajor, semiminor, xcenter, ycenter, major angle)"""
    
    if normalized_in:
        normalized_out = True
        datas = data
    else:
        # Normalize data, get normalization parameters
        center,dXY,datas = normalize_data(data)

    # Set mu
    mu = 1

    # Set the bounds for the parameters
    # bounds = [(0, mu), (0, mu), (-dXY/20, dXY/20), (-dXY/20, dXY/20), (0, np.pi)]
    if fixed_center is None:
        bounds = [(0, mu), (0, mu), (-mu/2, mu/2), (-mu/2, mu/2), (0, np.pi)]
    else:
        bounds = [(0, mu), (0, mu), (fixed_center[0],)*2, (fixed_center[1],)*2, (0, np.pi)]

    
    # Set the initial guess for the parameters
    x0 = np.array([mu, mu, *(fixed_center if fixed_center is not None else (0,0)), np.pi/2])


    # Define the constraints as a dictionary
    cons = [{'type': 'ineq', 'fun': functools.partial(outside_restraint,datas)}]

    if extra_constraints is not None:
        cons += extra_constraints

    
    # Compute the solution
    sol = minimize(obj, x0, bounds=bounds, constraints=cons)


    if normalized_out: #we're done
        return sol.x
    else: #denormalize
        return denormalize_ellipse(sol.x,center,dXY);

from scipy.spatial import distance_matrix
def medoid(x,y):
    dist_matrix = distance_matrix(np.vstack((x, y)).transpose(), np.vstack((x, y)).transpose())
    imin = np.argmin(np.sum(dist_matrix, axis=0))
    return (x[imin],y[imin])

def mindist(pt,pts):
    return np.min(np.linalg.norm(pts-pt,axis=1))

def fit_mask_interior_ellipse(mask:np.ndarray,**kwargs):
    if False: np.save("temp3.data.npy",mask)
    startpos,startradius = getcircle(mask)
    mask[startpos] = 4
    boundary = find_boundaries(mask, mode='inner').astype(np.uint8)
    yb,xb = np.where(boundary==1)
    data = np.transpose([xb,yb])
    startpos = centroid(mask)
    ellipse = fit_interior_ellipse(data,**kwargs);
    return ellipse


def brute_interior_ellipse(mask:np.ndarray,normalized_out=False):
    points = np.argwhere(mask)[:,::-1]
    best = None
    bestscore = 0
    boundary = find_boundaries(mask, mode='inner').astype(np.uint8)
    bound_points = np.argwhere(boundary)[:,::-1]
    center,dXY,norm_boundary = normalize_data(bound_points)
    _,_,norm_interior = normalize_data(points,(center,dXY))
    for p in tqdm(norm_interior):
        ellipse = fit_interior_ellipse(norm_boundary,normalized_in=True,fixed_center=p);
        if (score := obj(ellipse)) < bestscore:
            bestscore = score
            best = ellipse
    assert best is not None
    return best if normalized_out else denormalize_ellipse(best,center,dXY);

def adaptive_fit_ellipse(mask:np.ndarray,normalized_out=False,constraints=[]):
    boundary =  np.argwhere(find_boundaries(mask, mode='inner').astype(np.uint8))[:,::-1]
    center,dXY,norm_boundary = normalize_data(boundary);
    solution = fit_interior_ellipse(norm_boundary,normalized_in=True,extra_constraints=constraints)
    if not is_inside_sm(norm_boundary,solution[2:4]):
        print("solution ineffective: brute forcing")
        extra_ellipses.append((solution,'blue'))
        # return solution
        ##ERROR CASE: BRUTE FORCE REQUIRED
        solution = brute_interior_ellipse(mask,normalized_out=True)
    if False: np.save("temp4.data.npy",mask)
    return solution if normalized_out else denormalize_ellipse(solution,center,dXY);
        

def inellipse_constraint(ellipse,params):
    return (1e-8 - f(params[2],params[3],ellipse));

def contour_adaptive_fitellipse(mask:np.ndarray,normalized_out=False,ellipse:tuple[tuple[float,float],tuple[float,float],float]|None=None):
    #format directly from fitellipse
    (cx,cy),(major,minor),angle = ellipse or getellipse(mask)    
    angle *= np.pi/180 #degrees->radians
    semimajor = major/2; semiminor = minor/2; #major to semimajor

    boundary =  np.argwhere(find_boundaries(mask, mode='inner').astype(np.uint8))[:,::-1]
    center,dXY,norm_boundary = normalize_data(boundary);

    ##I think center should be correct? might have to swap x/y
    ellipse_params = np.array([semimajor,semiminor,cx,cy,angle])
    ellipse_params = normalize_ellipse(ellipse_params,center,dXY);
    extra_ellipses.append((ellipse_params,'yellow'))
    # return ellipse_params
    constraint = {'type':'ineq','fun':functools.partial(inellipse_constraint,ellipse_params)};
    return adaptive_fit_ellipse(mask,normalized_out=normalized_out,constraints = [constraint]);




        
def draw_ellipse(axes:Axes,ellipse:np.ndarray,color=None,acolors=['blue','orange'],draw_axes=True):
    semimajor,semiminor = ellipse[[0,1]]
    center = ellipse[[2,3]]
    angle = ellipse[4]
    d = {"major":semimajor*2,"minor":semiminor*2,"center":center,"angle":angle*180/np.pi}
    artist = Ellipse(d["center"],d["major"],d["minor"],d["angle"],color=color,fill=False,linewidth=2);
    axes.add_artist(artist)

    sol = ellipse
    soldict = {"semimajor":sol[0],"semiminor":sol[1],"x":sol[2],"y":sol[3],"theta":sol[4]}
    if draw_axes:
        axes.plot([soldict["x"],soldict["x"]+soldict["semimajor"]*np.cos(soldict["theta"])],[soldict["y"],soldict["y"]+soldict["semimajor"]*np.sin(soldict["theta"])],color=acolors[0])
        axes.plot([soldict["x"],soldict["x"]+soldict["semiminor"]*(np.cos(soldict["theta"]+np.pi/2))],[soldict["y"],soldict["y"]+soldict["semiminor"]*np.sin(soldict["theta"]+np.pi/2)],color=acolors[1])
    axes.scatter(soldict['x'],soldict['y'],color=color,marker='o')

    


if __name__ == "__main__":
    points = """45.66172222222225 	18.841511212833733
    45.66144444444447 	18.841603609265974
    45.661166666666695 	18.84167023720114
    45.66088888888892 	18.841708840666733
    45.66061111111114 	18.841714721198922
    45.66033333333336 	18.8416792127312
    45.66005555555558 	18.841586913726573
    45.659907959603544 	18.8415
    45.659777777777805 	18.841432324588876
    45.65952133520022 	18.84122222222222
    45.65950000000003 	18.841205864567236
    45.65925240431259 	18.840944444444442
    45.65922222222225 	18.84091291427614
    45.65903462855747 	18.840666666666664
    45.65894444444447 	18.840543213692058
    45.658848018922384 	18.84038888888889
    45.658686482591534 	18.84011111111111
    45.6586666666667 	18.840075891391127
    45.65854057470615 	18.83983333333333
    45.65841385937155 	18.839555555555553
    45.658388888888915 	18.839497076739338
    45.65829635419802 	18.839277777777777
    45.65819317225286 	18.839
    45.65811111111114 	18.838743972287034
    45.658103781273304 	18.83872222222222
    45.65801729926895 	18.83844444444444
    45.657943630825045 	18.838166666666666
    45.65788130483856 	18.837888888888887
    45.65783333333336 	18.837633786630168
    45.65782843965894 	18.83761111111111
    45.657777930516 	18.83733333333333
    45.65773797205481 	18.837055555555555
    45.65770846520107 	18.836777777777776
    45.65768983794373 	18.836499999999997
    45.65768315829199 	18.836222222222222
    45.657690315425526 	18.835944444444443
    45.657714290504025 	18.835666666666665
    45.65775954170573 	18.835388888888886
    45.65783252126027 	18.83511111111111
    45.65783333333336 	18.835108788071842
    45.65792541129527 	18.834833333333332
    45.65806060225678 	18.834555555555553
    45.65811111111114 	18.834474038899852
    45.65823358922209 	18.834277777777775
    45.658388888888915 	18.834082623685447
    45.658457173026996 	18.834
    45.6586666666667 	18.833784998750474
    45.658731945343874 	18.83372222222222
    45.65894444444447 	18.83353762228444
    45.65906160654751 	18.833444444444442
    45.65922222222225 	18.83332268424614
    45.65945121712184 	18.833166666666664
    45.65950000000003 	18.833133439986426
    45.659777777777805 	18.83295712908462
    45.659898642372674 	18.83288888888889
    45.66005555555558 	18.832794864416265
    45.66033333333336 	18.832647295512253
    45.66040951888557 	18.83261111111111
    45.66061111111114 	18.832503587959618
    45.66088888888892 	18.832375253663077
    45.66099358303661 	18.83233333333333
    45.661166666666695 	18.832250897046737
    45.66144444444447 	18.83213848859703
    45.66169543834408 	18.832055555555552
    45.66172222222225 	18.832044342721858
    45.66200000000003 	18.831951946289568
    45.6622777777778 	18.83188531835439
    45.662555555555585 	18.83184671488883
    45.66283333333336 	18.831840834356605
    45.663111111111135 	18.831876342824323
    45.66338888888892 	18.831968641828965
    45.663536484840996 	18.832055555555552
    45.66366666666669 	18.83212323096666
    45.663923109244294 	18.83233333333333
    45.663944444444475 	18.832349690988305
    45.66419204013188 	18.83261111111111
    45.66422222222225 	18.832642641279445
    45.66440981588704 	18.83288888888889
    45.664500000000025 	18.83301234186348
    45.664596425522106 	18.833166666666664
    45.664757961852935 	18.833444444444442
    45.66477777777781 	18.833479664164482
    45.664903869738325 	18.83372222222222
    45.66503058507293 	18.834
    45.66505555555558 	18.834058478816264
    45.66514809024647 	18.834277777777775
    45.66525127219165 	18.834555555555553
    45.66533333333336 	18.834811583268447
    45.66534066317122 	18.834833333333332
    45.66542714517556 	18.83511111111111
    45.66550081361947 	18.835388888888886
    45.66556313960594 	18.835666666666665
    45.66561111111114 	18.83592176892534
    45.66561600478556 	18.835944444444443
    45.665666513928514 	18.836222222222222
    45.66570647238968 	18.836499999999997
    45.665735979243436 	18.836777777777776
    45.665754606500776 	18.837055555555555
    45.66576128615252 	18.83733333333333
    45.66575412901897 	18.83761111111111
    45.665730153940466 	18.837888888888887
    45.665684902738775 	18.838166666666666
    45.66561192318422 	18.83844444444444
    45.66561111111114 	18.83844676748368
    45.665519033149245 	18.83872222222222
    45.66538384218772 	18.839
    45.66533333333336 	18.8390815166557
    45.66521085522241 	18.839277777777777
    45.66505555555558 	18.83947293187009
    45.66498727141749 	18.839555555555553
    45.66477777777781 	18.83977055680508
    45.664712499100624 	18.83983333333333
    45.664500000000025 	18.84001793327112
    45.66438283789701 	18.84011111111111
    45.66422222222225 	18.840232871309414
    45.66399322732267 	18.84038888888889
    45.663944444444475 	18.840422115569137
    45.66366666666669 	18.840598426470944
    45.66354580207184 	18.840666666666664
    45.66338888888892 	18.84076069113929
    45.663111111111135 	18.840908260043292
    45.663034925558904 	18.840944444444442
    45.66283333333336 	18.841051967595934
    45.662555555555585 	18.84118030189247
    45.66245086140788 	18.84122222222222
    45.6622777777778 	18.841304658508818
    45.66200000000003 	18.8414170669585
    45.66174900610049 	18.8415
    45.66172222222225 	18.841511212833733"""
    # print([re.split("\\s+",line) for line in points.split("\n")])

    data = np.array([[float(s.strip()) for s in re.split("\\s+",line)] for line in re.split("\\s*\n\\s*",points)])
    mask = np.load("temp4.data.npy")
    boundary = find_boundaries(mask, mode='inner').astype(np.uint8)
    yb,xb = np.where(boundary==1)
    data = np.transpose([xb,yb])
    # data[:,1] *= 2

    fig, ax = plt.subplots()
    center,dXY,normed = normalize_data(data);
    if False:
        sol = adaptive_fit_ellipse(mask,normalized_out=True)
        soldict = {"semimajor":sol[0],"semiminor":sol[1],"x":sol[2],"y":sol[3],"theta":sol[4]}

        # Compute the value of f at the solution
        x, y = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000))
        f0 = f(x, y,sol)

        ax.set_aspect('equal')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        pltpoints = normed

    else:
        sol = adaptive_fit_ellipse(mask)
        soldict = {"semimajor":sol[0],"semiminor":sol[1],"x":sol[2],"y":sol[3],"theta":sol[4]}


        # Compute the value of f at the solution
        x, y = np.meshgrid(np.linspace(center[0]-dXY, center[0]+dXY, 1000), np.linspace(center[1]-dXY, center[1]+dXY, 1000))
        f0 = f(x, y,sol)

        ax.set_aspect('equal')
        ax.set_xlim(center[0]-dXY, center[0]+dXY)
        ax.set_ylim(center[1]-dXY, center[1]+dXY)

        pltpoints = data

        extra_ellipses = [(denormalize_ellipse(ellipse,center,dXY),color) for ellipse,color in extra_ellipses]




    #F is the conic! the solution ellipse is f=0. 
    # Plot the solution
    ax.plot(pltpoints[:, 0], pltpoints[:, 1], 'ro', markersize=5)
    # levels = np.linspace(np.min(f0),np.max(f0),30)
    levels = []
    ax.contour(x, y, f0, levels, colors='k')

    draw_ellipse(ax,sol,color='purple')
    for ellipse,color in extra_ellipses:
        draw_ellipse(ax,ellipse,color,draw_axes=False)

    plt.show()