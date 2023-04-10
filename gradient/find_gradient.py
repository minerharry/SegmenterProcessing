import numpy as np
from skimage.io import imread, imshow,show;
from skimage.exposure import rescale_intensity
import cv2 as cv
import matplotlib.pyplot as plt

image = "C:/Users/Harrison Truscott/Documents/Gradient Calibration/down3.tif"
im2 = imread("C:/Users/Harrison Truscott/Documents/Gradient Calibration/mid1.tif")

im1 = imread(image);
# img = rescale_intensity(img);
# laplacian = cv.Laplacian(img,cv.CV_64F,ksize=11);
# sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=11);
# sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=11);

smooth_filter = np.ones((5,5),np.float32)/25;

# smoothed_laplacian = cv.filter2D(laplacian,-1,smooth_filter);
# smoothed_sobely = cv.filter2D(sobely,-1,smooth_filter);

def grads(img:np.ndarray):
    images = { 
            "laplacian": cv.Laplacian(img,cv.CV_64F,ksize=11),
            "sobelx": cv.Sobel(img,cv.CV_64F,1,0,ksize=11),
            "sobely": cv.Sobel(img,cv.CV_64F,0,1,ksize=11),
        };

    for n,im in list(images.items()):
        images[f"smoothed_{n}"] = cv.filter2D(im,-1,smooth_filter);
    return images;


im1_ims = grads(im1);
im2_ims = grads(im2);

# print(im1.shape);
im1_liney = [];
for y in im1:
    # print(np.average(im1[y]));
    im1_liney.append(np.average(y));
    # im1_linex.append(y);

im2_liney = [];
for y in im2:
    im2_liney.append(np.average(y));
    
    # im2_linex.append(y);


print("avg y 1",np.average(im1_ims["smoothed_sobely"]))
print("avg y 2",np.average(im2_ims["smoothed_sobely"]))

print(im1.shape);
plt.subplot(2,4,1),plt.imshow(im1,cmap = 'gray')
plt.title('Down1'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,2),plt.imshow(im1_ims["smoothed_sobelx"],cmap = 'gray')
plt.title('Smoothed Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,3),plt.imshow(im1_ims["smoothed_sobely"],cmap = 'gray')
plt.title('Smoothed Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,4),plt.plot(im1_liney)
plt.title('Linescan'), plt.xticks([]), plt.yticks([])


plt.subplot(2,4,5),plt.imshow(im2,cmap = 'gray')
plt.title('Mid1'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,6),plt.imshow(im2_ims["smoothed_sobelx"],cmap = 'gray')
plt.title('Smoothed Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,7),plt.imshow(im2_ims["smoothed_sobely"],cmap = 'gray')
plt.title('Smoothed Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,8),plt.plot(im2_liney)
plt.title('Linescan'), plt.xticks([]), plt.yticks([])
plt.show()