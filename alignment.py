import functools
import itertools
from math import ceil, floor, prod
from typing import Callable, Iterable, Literal, Sequence, Tuple, TypeVarTuple
import numpy as np
from scipy.fft import fft2,fftfreq,fftshift,ifft2 #TODO: rFFFT?
from skimage.io import imread, imshow
import cv2
from tqdm import tqdm
from moisan2011 import per,rper


def grid_reduce[V:np.dtype,*T](f:Callable[[*T],V],*vectors:Sequence|np.ndarray)->np.ndarray[None,V]:
    meshed = np.meshgrid(*vectors)
    with np.nditer([None,*meshed]) as it:
        x:Tuple[*T]
        for y,*x in tqdm(it,total=prod(meshed[0].shape)): 
            y[...] = f(*x);
        return it.operands[0];


@functools.cache
def sector_mask(shape:tuple[int,int],centre:tuple[float,float],radius:float|tuple[float,float],angle_range:tuple[float,float],ang_in:Literal['radians','degrees']='radians'): #for polar binning
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    if not isinstance(radius,tuple):
        radius = (0,radius)

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    # print(cx,cy)
    # print(shape)
    if (ang_in == 'degrees'):
        tmin,tmax = np.deg2rad(angle_range)
    else:
        tmin,tmax = angle_range

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = (r2 <= radius[1]*radius[1]) * (radius[0]*radius[0] <= r2)

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask,r2,theta


def analyze_angles(image:np.ndarray,center:tuple[int,int]|None=None):
    if center is None:
        center = (image.shape[0]//2,image.shape[1]//2); #use center of image as default (**SHOULD BE CENTER OF CELL / OBJECT!! FOR FOCAL ADHESIONS!!**)

def make_diagonal(angle:float,im_size:tuple[int,int],line_thickness:int=2,line_spacing:int|None=None,color=255):

    if (angle > np.pi/2 or angle < -np.pi/2):
        angle = (angle + np.pi/4) % np.pi/2 - np.pi/4

    print(angle)

    #simple and slow. draw a sufficiently large *diagonal* rectangle of diagonal lines, then crop to desired size
    period = line_spacing;
    if line_spacing is None:
        period = line_thickness*2;
    cos,sin = np.cos(angle),np.sin(angle)
    xoffset = int(period*sin) #this is bad but it makes line spacing consistent
    yoffset = int(period*cos) 

    w,h = im_size
    im = np.zeros((w,h),dtype=np.uint8);

    margin = 6
    w += margin #margin
    h += margin

    if angle < 0:

        x1 = w*sin*sin - margin//2
        y1 = w*sin*cos

        x2 = w - h*sin*cos - margin//2
        y2 = h*sin*sin

    else:
        x1 = -h*sin*cos - margin//2
        y1 = h*sin*sin

        x2 = w - w*sin*sin - margin//2
        y2 = - w*sin*cos

    print(f"x1: {x1}, y1: {y1}")
    print(f"x2: {x2}, y2: {y2}")

    # im = cv2.line(im,(y1,x1),(y2,x2),color,line_thickness,cv2.LINE_AA);
    H = abs(h*cos) + abs(w*sin);
    N = floor(H/np.sqrt(xoffset**2 + yoffset**2))+1
    res = [],[],[],[]
    for _ in range(N):
        res[0].append(x1)
        res[1].append(y1)
        res[2].append(x2)
        res[3].append(y2)
        im = cv2.line(im,(int(x1),int(y1)),(int(x2),int(y2)),color,line_thickness//2,cv2.LINE_AA);
        x1 += xoffset; x2 += xoffset;
        y1 += yoffset; y2 += yoffset;

    return im,res,xoffset,yoffset


@functools.cache
def get_radii(shape:tuple[int,int]):
    invfreqs = [1/fftshift(fftfreq(shape[i])) for i in (0,1)]

    def get_wavelength(iyfreq,ixfreq):
        if (ixfreq == np.inf):
            res = iyfreq
        elif (iyfreq == np.inf):
            res = ixfreq
        else:
            res = iyfreq*ixfreq/np.linalg.norm([ixfreq,iyfreq]);
        if res == np.nan:
            from IPython import embed; embed()
        return res
        

    
    return abs(grid_reduce(get_wavelength,invfreqs[1],invfreqs[0]));


def analyze_frequencies(im:np.ndarray,wavelength:float|tuple[float,float],wavelength_range:float=0.1,angle_bins:int|Iterable[float]=20):
    
    # fft = fftshift(fft2(im));
    fft = im;
    
    if (isinstance(angle_bins,int)):
        angle_bins = np.linspace(0,np.pi,angle_bins,endpoint=True);
    angle_bins = list(angle_bins);
    
    
    radii = get_radii(im.shape);

    if (not isinstance(wavelength,tuple)):
        wavelength = (wavelength*(1-wavelength_range),wavelength*(1+wavelength_range));

    rad_mask = (wavelength[0] <= radii) * (wavelength[1] >= radii);


    res = []
    masks = []
    for i in tqdm(range(len(angle_bins)),leave=False):
        angle_mask,_,_ = sector_mask(
            im.shape,
            (im.shape[0]/2,im.shape[1]/2),
            max(im.shape),
            (angle_bins[i],angle_bins[(i+1)%len(angle_bins)])
            );
        angle_mask_2,_,_ = sector_mask(
            im.shape,
            (im.shape[0]/2,im.shape[1]/2),
            max(im.shape),
            (angle_bins[i]+np.pi,angle_bins[(i+1)%len(angle_bins)]+np.pi)
            );
        
        angle_mask = angle_mask | angle_mask_2
        # print(np.sum(angle_mask))
        # plt.imshow(angle_mask);
        # plt.show()
        # print(np.max(angle_mask))
                                 
        total_mask = rad_mask*angle_mask
        masks.append(total_mask)
        val = np.sum(total_mask*fft)/np.sum(total_mask);
        print(np.isnan(val))
        res.append(val);
        
    return res,masks,angle_bins,rad_mask;


def diag_cos(shape:tuple[int,int],xwave:float,ywave:float,amplitude:float,make_positive:bool=False,center:tuple[float]|None=None):
    if center is None: center = (shape[0]/2,shape[1]/2);
    mag = np.dot([xwave,ywave],[xwave,ywave])/2/np.pi; #(xwave**2 + ywave**2)/2pi

    f = np.fromfunction(lambda y,x: np.cos(xwave/mag*(x-center[1])+ywave/mag*(y-center[0])),shape)
    if (make_positive):
        f += 1
        amplitude /= 2
    return f*amplitude


def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


def make_radial_image(im_size:tuple[int,int],
                      radii:Iterable[float]|float,
                      angles:Iterable[float]|float,
                      rotations:Iterable[float]|float,
                      lengths:Iterable[float]|float,
                      widths:Iterable[float]|float,
                      center:tuple[float,float]|None=None,
                      color:float|tuple[int,int,int]=1.0,
                      n_ellipses:int=100):
    """Ellipse angle will be angle + rotation. pass 0 to rotations to make each ellipse perfectly radial"""
    im = np.zeros(im_size);
    any_sequence = False
    
    if not isinstance(radii,Iterable):
        radii = itertools.cycle([radii])
    else:
        any_sequence = True
    
    if not isinstance(angles,Iterable):
        angles = itertools.cycle([angles])
    else:
        any_sequence = True

    if not isinstance(rotations,Iterable):
        rotations = itertools.cycle([rotations])
    else:
        any_sequence = True

    if not isinstance(widths,Iterable):
        widths = itertools.cycle([widths])
    else:
        any_sequence = True

    if not isinstance(lengths,Iterable):
        lengths = itertools.cycle([lengths])
    else:
        any_sequence = True

    if not any_sequence:
        radii = itertools.islice(radii,n_ellipses)
    

    if center is None:
        center = (im_size[0]/2,im_size[1]/2)

    for radius,angle,rotation,length,width in zip(radii,angles,rotations,lengths,widths):
        print("plotting")
        ell_center = int(np.cos(angle)*radius+center[1]),int(np.sin(angle)*radius+center[0])
        ell_angle = angle + rotation

        im = cv2.ellipse(im,ell_center,(length,width),np.rad2deg(ell_angle),0,360,color,thickness=-1,lineType=cv2.LINE_AA);

    return im;

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    if False:
        n = 70
        random = np.random.random
        rad = random(n)*170+100
        ang = random(n)*np.pi/8 #ANGULAR DISTRIBUTION
        rot = random(n)*np.pi/8-np.pi/16 #small angle variation
        leng = np.random.randint(5,20,n)
        wid = np.random.randint(1,3,n)

        diag = make_radial_image((600,600),rad,ang,rot,leng,wid);
        wavelengths = [(4,16),4,6,8,12,16]

    if False:
        size = (300,300)
        func = lambda x,y: np.linalg.norm((size[0]/2-x,size[1]/2-y),axis=0)
        im = np.fromfunction(func,size,dtype=float)
        print(im.shape)
        im = im < 30
        diag = im

        wavelengths = [10,20,30,45,60,90]


    if False:
        w1 = 8; a1 = np.pi/3
        w2 = 20; a2 = -2*np.pi/5
        diag = diag_cos((400,400),np.cos(a1)*w1,np.sin(a1)*w1,10) \
             + diag_cos((400,400),np.cos(a2)*w2,np.sin(a2)*w2,7)
        wavelengths = [4,8,12,16,20,24]
    

    if True:
        stage = 3
        if stage == 1:
            diag = imread('s1_frame16.tif',as_gray=True)
        else:
            diag = imread('frame1.tif',as_gray=True)
        
        # diag = imread('VASP_frame.tif')
        wavelengths = [(3,8),3,4,6,8,12]


    do_per_filter = True;
    if do_per_filter:
        orig_diag = diag
        diag,s = rper(diag)
        fig, (ax1,ax2, ax3) = plt.subplots(1,3)
        ax1.set_title("pre-decomposition")
        ax1.imshow(orig_diag)
        ax2.set_title("post-decomposition")
        ax2.imshow(diag)
        ax3.set_title("smooth residual")
        ax3.imshow(s)
        # diag = np.abs(diag)
    
    fft = fft2(diag);
    fft = fftshift(fft);

    fig, (ax1,ax2) = plt.subplots(1,2);
    ax1.set_title("pre-fft")
    ax1.imshow(diag)
    # ax1.scatter(pts1[0],pts1[1])
    # ax1.scatter(pts1[2],pts1[3])
    ax2.set_title("fft")
    ax2.imshow(np.log(abs(fft)**2))
    # ax2.scatter(pts1[0],pts1[1])
    # ax2.scatter(pts1[2],pts1[3])

    center = fft.shape[1]/2,fft.shape[0]/2;

    # offset = np.multiply((np.sqrt(2),np.sqrt(2)),wavelength)
    # offnorm = np.linalg.norm(offset);
    # offset = np.divide(offset,offnorm);
    # offset *= 100/offnorm; 
    # ax2.plot((center[0],offset[0]+center[0]),(center[1],offset[1]+center[1]));

    # fig,ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # hist,masks,bins = analyze_frequencies(diag,wavelength,angle_bins=30,wavelength_range=0.2)
    # ax.bar(bins,np.abs(hist)**2,align='edge',width=np.diff(bins)[0]);

    fig,ax = plt.subplots(2,3,subplot_kw={'projection': 'polar'},)
    fig.tight_layout()
    
    fig4,ax4 = plt.subplots(2,3,subplot_kw={'projection':'polar'})
    fig4.tight_layout()

    fig2,ax2 = plt.subplots(2,3)
    fig2.tight_layout()

    fig3,ax3 = plt.subplots(2,3)
    fig3.tight_layout()

    axs = *ax[0],*ax[1]
    ax2s = *ax2[0],*ax2[1]
    ax3s = *ax3[0],*ax3[1]
    ax4s = *ax4[0],*ax4[1]

    for ax,ax2,ax3,ax4,wave in tqdm(list(zip(axs,ax2s,ax3s,ax4s,wavelengths))):
        hist,masks,bins,rad_mask = analyze_frequencies(fft,wave,angle_bins=30,wavelength_range=0.2)
        print(bins)
        
        # bins = np.rad2deg(bins)
        ax.set_title(wave)
        power = np.abs(hist)**2

        ax.bar(bins,power,align='edge',width=np.diff(bins)[0]);
        #draw perpendicular lines on ax4
        perp_angles = np.add(bins,np.pi/2)
        perp_angles = np.concatenate([perp_angles,perp_angles + np.pi])
        perp_mags = np.concatenate([power,power])
        # perp_mags = np.log(perp_mags)
        ax4.set_title(wave)
        ax4.bar(perp_angles,perp_mags,width=np.diff(bins)[0],color='orange');
        
        ax2.set_title(wave)
        ax2.imshow(masks[4]*np.log(np.abs(fft)**2))

        #inverse the radial mask
        rad_masked = fft*rad_mask
        ifft = ifft2(fftshift(rad_masked))
        ax3.set_title(wave)
        ax3.imshow(np.abs(ifft))
        


        # print(np.average([np.sum(m) for m in masks]))
    

    from IPython import embed; embed()


    




