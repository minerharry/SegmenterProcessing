import os
from skimage.io import imread,imsave
from prepare_run import *
import numpy as np;
import shutil;

inImages = iteration_testing_folder + f"/iter{iteration}/round{round+1}/images/" #processing the images from the next round for starting masks
outImages = processFolder + "segmentation_images/";
completeMasks = processFolder + "segmentation_output_masks/";
maskOutputFolder = iteration_testing_folder + f"/iter{iteration}/round{round+1}/input/";
maskSuffix = "";#"_mask";

# skipImageCopy = True;

if __name__ == "__main__":
    ##check if processing directories empty:
    if (any(os.scandir(outImages)) != 0):
        while True:
            doDelete = input(f"Warning: image processing directory {outImages}\nmust be empty; delete? (y/n), \'cancel\' to cancel\n",);
            if doDelete == "y":
                for im in os.scandir(outImages):
                    try: shutil.rmtree(os.path.join(outImages,im)); #just in case
                    except OSError: os.remove(os.path.join(outImages,im));
                break;
            elif doDelete == "n":
                break;
            elif doDelete.lower() == "cancel":
                exit();
        
    if (any(os.scandir(completeMasks)) != 0):
        while True:
            doDelete = input(f"Warning: mask output directory {completeMasks}\nmust be empty; delete?  (y/n), \'cancel\' to cancel\n",);
            if doDelete == "y":
                for im in os.scandir(completeMasks):
                    try: shutil.rmtree(os.path.join(completeMasks,im)); #just in case
                    except OSError: os.remove(os.path.join(completeMasks,im));
                break;
            elif doDelete == "n":
                break;
            elif doDelete.lower() == "cancel":
                exit();
        
    names = os.listdir(inImages);
    print(len(names));
    maskNames = [];
    for name in names:
        im = imread(inImages+name);
        print(name);
        if auto_rescale:
            im = rescale_intensity(im);
        assert isinstance(im,np.ndarray);
        sliced = False
        if x_slices > 1 or y_slices > 1:
            # print(im.shape);
            M = (im.shape[0]-context_bounds[0]-context_bounds[2]-crop[0]-crop[2])/y_slices;
            N = (im.shape[1]-context_bounds[1]-context_bounds[3]-crop[1]-crop[3])/x_slices;
            # print(M,N);
            # print(-context_bounds[1]-context_bounds[3]-crop[1]-crop[3])
            # print(im.shape);

            if int(M) != M or int(N) != N:
                raise Exception(f"ERROR: Mask with size {im.shape[:2]} cannot be sliced into {x_slices} columns and {y_slices} rows\nwith context bounds of {context_bounds}; {M} and {N} not integers");
            else:
                M = int(M)
                N = int(N)
                im = (im/256).astype('uint8');
                im = np.stack((im,im,im),axis=2);
                sliced = True;
                tiles = [im[y-context_bounds[0]:y+M+context_bounds[2],x-context_bounds[1]:x+N+context_bounds[3]] 
                        for y in range(context_bounds[0]+crop[0],im.shape[0]-crop[0]-crop[2]-context_bounds[0]-context_bounds[2],M) 
                        for x in range(context_bounds[1]+crop[1],im.shape[1]-crop[1]-crop[3]-context_bounds[1]-context_bounds[3],N)];
                imBounds = [[y-context_bounds[0],y+M+context_bounds[2],x-context_bounds[1],x+N+context_bounds[3]] 
                        for y in range(context_bounds[0]+crop[0],im.shape[0]-crop[0]-crop[2]-context_bounds[0]-context_bounds[2],M) 
                        for x in range(context_bounds[1]+crop[1],im.shape[1]-crop[1]-crop[3]-context_bounds[1]-context_bounds[3],N)];

                # print((context_bounds[0]+crop[0],context_bounds[1]+crop[1]))
                # print(imBounds);
                outMasks = [name,[]];
                for num,m in enumerate(tiles):
                    outMasks[1].append(os.path.splitext(name)[0] + f"-{num}" + ".TIF");
                    imsave(outImages+os.path.splitext(name)[0] + f"-{num}" + ".TIF", m,check_contrast=False)
                maskNames.append(outMasks.copy());
        else:   
            imsave(outImages+os.path.splitext(name)[0] + ".TIF", im,check_contrast=False)

        # # print(name);
        # im = imread(inImages+name);
        # if x_slices > 1 or y_slices > 1:
        #     M = (im.shape[0]-context_bounds[0]-context_bounds[2]-crop[0]-crop[2])/x_slices
        #     N = (im.shape[1]-context_bounds[1]-context_bounds[3]-crop[1]-crop[3])/y_slices;
        #     if int(M) != M or int(N) != N:
        #         raise Exception(f"ERROR: Mask with size {im.shape[:2]} cannot be sliced into {x_slices} columns and {y_slices} rows\nwith context bounds of {context_bounds}; {M} and {N} not integers");
        #     else:
        #         M = int(M)
        #         N = int(N)
        #         tiles = [im[x-context_bounds[0]:x+M+context_bounds[0]+context_bounds[2],y-context_bounds[1]:y+N+context_bounds[1]+context_bounds[3]] 
        #                 for x in range(context_bounds[0]+crop[0],im.shape[0]-crop[0]-crop[2]-context_bounds[0]-context_bounds[2],M) 
        #                 for y in range(context_bounds[1]+crop[1],im.shape[1]-crop[1]-crop[3]-context_bounds[1]-context_bounds[3],N)];
        #         imBounds = [[x-context_bounds[0],x+M+context_bounds[0]+context_bounds[2],y-context_bounds[1],y+N+context_bounds[1]+context_bounds[3]] 
        #                 for x in range(context_bounds[0]+crop[0],im.shape[0]-crop[0]-crop[2]-context_bounds[0]-context_bounds[2],M) 
        #                 for y in range(context_bounds[1]+crop[1],im.shape[1]-crop[1]-crop[3]-context_bounds[1]-context_bounds[3],N)];

        #         # print(imBounds);
        #         outMasks = [name,[]];
        #         for num,m in enumerate(tiles):
        #             if auto_rescale:
        #                 try:
        #                     m = rescale_intensity(m);
        #                 except:
        #                     print(tiles);
        #                     print(imBounds);
        #                     print(m);
        #                     print(num);
        #                     raise Exception();
        #             imsave(outImages+os.path.splitext(name)[0] + f"-{num}" + ".TIF", m, check_contrast = False);
        #             outMasks[1].append(os.path.splitext(name)[0] + f"-{num}" + ".TIF");
        #         # print(len)
        #         maskNames.append(outMasks.copy());
        #         # print(len(maskNames))
        # else:
        #     if auto_rescale:
        #         im = rescale_intensity(im);
        #     imsave(outImages+os.path.splitext(name)[0] + ".TIF", im)
        #     maskNames.append(os.path.splitext(name)[0] + ".TIF");
    while True:
        s = input("Waiting for remote segmentation, press enter to continue or type \'CANCEL\' to cancel\nPlease make sure that google drive for desktop has finished syncing\n")
        if s.lower() == "cancel":
            exit();
        elif s == "":
            break;
    print(maskNames);
    for n in maskNames:
        try:
            if isinstance(n,list):
                names = n[1];
                baseName = n[0];
                stitchMasks = [imread(completeMasks+os.path.splitext(name)[0] + maskSuffix + os.path.splitext(name)[1]) for name in names];
                for i,m in enumerate(stitchMasks):
                    # print(i);
                    x = i // y_slices;
                    y = i % x_slices;
                    # print(x,y);
                    imBounds = [crop[0]+context_bounds[0] if x != 0 else 0,m.shape[0]-crop[2]-context_bounds[2] if x != x_slices-1 else m.shape[0],crop[1]+context_bounds[1] if y!= 0 else 0 ,m.shape[1]-crop[3]-context_bounds[3] if y != y_slices - 1 else m.shape[1]];
                    stitchMasks[i] = m[imBounds[0]:imBounds[1],imBounds[2]:imBounds[3]];
                    # print(stitchMasks[i].shape);
                stitched = np.concatenate([np.concatenate(stitchMasks[i*x_slices:(i+1)*x_slices],axis=1) for i in range(y_slices)]);
                print(stitched.shape);
                imsave(maskOutputFolder + os.path.splitext(baseName)[0]+".TIF",stitched,check_contrast=False);
        except:
            print(f"error: {n[0]} does not exist")
    

