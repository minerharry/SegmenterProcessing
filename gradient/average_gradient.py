import cv2 as cv
import scipy.interpolate
from skimage.io import imread, imshow, show
import matplotlib.pyplot as plt
import numpy as np

img = imread( "gradients/down1.tif");
xdiff = 40;
# img = np.array([np.linspace(i,i+xdiff,40) for i in range(40)]);
# img[:,20:] = 0

# imshow(img);

# smooth_size = 51
# smooth_filter = np.ones((smooth_size,smooth_size),np.float64)/(smooth_size**2);
# img = cv.filter2D(img,-1,smooth_filter);

img = cv.GaussianBlur(img,(15,15),0);

sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=11)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=11)

# print(sobelx.shape);

fig = plt.figure()
imfig = fig.add_subplot(2,4,1)
imfig.imshow(img,cmap="gray");
imfig.set_title("down3");
imAxes = imfig.axes;

xfig = fig.add_subplot(2,4,2)
xfig.imshow(sobelx,cmap="gray");
xfig.set_title("sobelx");
xAxes = xfig.axes;

yfig = fig.add_subplot(2,4,3)
yfig.imshow(sobely,cmap="gray");
yfig.set_title("sobely");
yAxes = yfig.axes;




xDat = np.average(img,0);
xX = np.arange(img.shape[1])
xCurve = fig.add_subplot(2,4,5);
plt.plot(xDat);

yDat = np.average(img,1);
yX = np.arange(img.shape[0])
yCurve = fig.add_subplot(2,4,6);
plt.plot(yDat);

### calculate windowed (averaged across )



xInterpolation = scipy.interpolate.CubicSpline(xX,xDat)
smallXX = np.arange(0,img.shape[1],0.1)
xCurve.plot(smallXX,xInterpolation(smallXX),"k--")

yInterpolation = scipy.interpolate.CubicSpline(yX,yDat)
smallYY = np.arange(0,img.shape[0],0.1)
yCurve.plot(smallYY,yInterpolation(smallYY),"k--")

crossRadius = 5
def crossMultiply(x,y):
    print(x);
    print(y);
    return np.sqrt(np.multiply(xInterpolation(x),yInterpolation(y)));
    #return np.sqrt(np.multiply(np.average(xDat[int(x-crossRadius):int(x+crossRadius)]),np.average(yDat[int(y-crossRadius):int(y+crossRadius)])));

def crossAverage(x,y):
    return np.average((xInterpolation(x),yInterpolation(y)),0);

def crossSumSq(x,y):
    return np.sqrt(np.power(xInterpolation(x),2) + np.power(yInterpolation(y),2));

fig.add_subplot(2,4,7);
crossmultiply = np.fromfunction(crossMultiply,img.shape);
plt.imshow(crossmultiply,cmap="gray");


radius = 5
def squareIntensity(x,y):
    x = int(x);
    y = img.shape[0] - int(y);
    return np.average(img[max(y-radius,0):min(y+radius,img.shape[0]),max(x-radius,0):min(x+radius,img.shape[1])]);

# def 


imdiff = img - crossmultiply;
fig.add_subplot(2,4,8);
plt.imshow(imdiff);
diff = np.average(np.abs(imdiff))/np.std(img);
print(diff);


grad_out = fig.add_subplot(2,4,4);

def g(a,x1,x2):
    x1 = max(x1,0);
    x2 = min(x2,a.shape[0]-1);
    return (a[x2]-a[x1])/(x2-x1);

def print_event(event):
    # print(event);
    # print(event.x,event.y,event.xdata,event.ydata);
    try:
        if event.inaxes == imAxes:
            imx = int(event.xdata);
            imy = int(event.ydata);
            # print(imx,imy);
            # print(xDat[imx],yDat[imy]);
            # print("xgrad");
            # xgrad = g(sobelx,imx-radius,imx+radius);
            # print("ygrad");
            # ygrad = g(sobely,imy-radius,imy+radius);
            print("x:",xInterpolation(event.xdata,1),"y:",yInterpolation(event.ydata,1),"midpoint:",yInterpolation(event.ydata));
    except Exception as e:
        print(e);
        pass;

cid = fig.canvas.mpl_connect('button_press_event', print_event)


plt.show();
