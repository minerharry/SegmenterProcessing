import imaplib
import shutil
import numpy as np
from skimage.io import imread,imsave;
from skimage.exposure import rescale_intensity;
import os
from pathlib import Path
iteration_testing_folder = Path("C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/optotaxis calibration/data/segmentation_iteration_testing")
iteration = 3
round = 1;

processing_folder_name = "processing";
x_slices = 5;
y_slices = 5;

auto_rescale = True;
context_bounds = [0]*4; #negative y, negative x, positive y, positive x; all values positive
crop = [0]*4; #same order as above; part of image that will never be used to make the slices integers

context_bounds = [32,42]*2;
# crop = [2,0,2,0];

trainFolder = iteration_testing_folder/f"iter{iteration}/round{round}"
processFolder = iteration_testing_folder/processing_folder_name;

mask_class = 5;

if __name__ == "__main__":
    ##check if processing directories empty:
    mFolder = processFolder/'training_masks'
    if (any(os.scandir(mFolder)) != 0):
        while True:
            doDelete = input(f"Warning: mask input directory {mFolder}\nmust be empty; delete? (y/n), \'cancel\' to cancel\n",);
            if doDelete == "y":
                for im in os.scandir(mFolder):
                    f = os.path.join(mFolder,im)
                    if os.path.isdir(f): shutil.rmtree(f); #just in case
                    else: os.remove(f);
                break;
            elif doDelete == "n":
                break;
            elif doDelete.lower() == "cancel":
                exit();
    iFolder = processFolder/'training_images'
    if (any(os.scandir(iFolder)) != 0):
        while True:
            doDelete = input(f"Warning: image input directory {iFolder}\nmust be empty; delete?  (y/n), \'cancel\' to cancel\n",);
            if doDelete == "y":
                for im in os.scandir(iFolder):
                    f = os.path.join(iFolder,im)
                    if os.path.isdir(f): shutil.rmtree(f); #just in case
                    else: os.remove(f);
                break;
            elif doDelete == "n":
                break;
            elif doDelete.lower() == "cancel":
                exit();
    names = None;
    for type in ["masks","images"]:
        inFolder = trainFolder/type;
        outFolder = processFolder/("training_" + type);
        names = os.listdir(inFolder) if names is None else names;
        for name in names:
            im = imread(inFolder/name);
            if auto_rescale and type == "images":
                im = rescale_intensity(im);
            assert isinstance(im,np.ndarray);
            if type == "masks" and len(im.shape) >= 3: #idk when it would be greater than three but /shrug
                im=im[:,:,0]>0;
            sliced = False
            if x_slices > 1 or y_slices > 1:
                M = (im.shape[0]-context_bounds[0]-context_bounds[2]-crop[0]-crop[2])/y_slices;
                N = (im.shape[1]-context_bounds[1]-context_bounds[3]-crop[1]-crop[3])/x_slices;

                if int(M) != M or int(N) != N:
                    raise Exception(f"ERROR: Mask with size {im.shape[:2]} cannot be sliced into {x_slices} columns and {y_slices} rows\nwith context bounds of {context_bounds}; {M} and {N} not integers");
                else:
                    M = int(M)
                    N = int(N)
                    sliced = True;
                    if type == "images":
                        im = (im/256).astype('uint8');
                        im = np.stack((im,im,im),axis=2);
                    elif mask_class is not None:
                        im[im>0]=mask_class;
                    tiles = [im[y-context_bounds[0]:y+M+context_bounds[2],x-context_bounds[1]:x+N+context_bounds[3]] 
                            for y in range(context_bounds[0]+crop[0],im.shape[0]-crop[0]-crop[2]-context_bounds[0]-context_bounds[2],M) 
                            for x in range(context_bounds[1]+crop[1],im.shape[1]-crop[1]-crop[3]-context_bounds[1]-context_bounds[3],N)];
                    imBounds = [[y-context_bounds[0],y+M+context_bounds[2],x-context_bounds[1],x+N+context_bounds[3]] 
                            for y in range(context_bounds[0]+crop[0],im.shape[0]-crop[0]-crop[2]-context_bounds[0]-context_bounds[2],M) 
                            for x in range(context_bounds[1]+crop[1],im.shape[1]-crop[1]-crop[3]-context_bounds[1]-context_bounds[3],N)];

                    # print((context_bounds[0]+crop[0],context_bounds[1]+crop[1]))
                    # print(imBounds);
                    for num,m in enumerate(tiles):
                        imsave(outFolder/(os.path.splitext(name)[0] + f"-{num}" + ".TIF"), m,check_contrast=False);
            else:
                imsave(outFolder/(os.path.splitext(name)[0] + ".TIF"), im,check_contrast=False)
