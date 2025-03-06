from skimage.io import imread, imshow,show;
from skimage.exposure import rescale_intensity
import cv2 as cv
import matplotlib.pyplot as plt

image = "C:/Users/Harrison Truscott/Documents/Gradient Calibration/down3.tif"

img = imread(image);
# img = rescale_intensity(img);
laplacian = cv.Laplacian(img,cv.CV_64F,ksize=11);
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=11);
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=11);

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()