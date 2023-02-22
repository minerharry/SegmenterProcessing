import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.io import imread,imsave
from skimage.exposure import rescale_intensity
from skimage.transform import resize

x_slices = 5;
y_slices = 5;

auto_rescale = True;
auto_downscale = True;
imperfect_resize = False;
context_bounds = [0]*4; #negative y, negative x, positive y, positive x; all values positive
crop = [0]*4; #same order as above; part of image that will never be used to make the slices integers

context_bounds = [32,42]*2; 
# crop = [2,0,2,0];

final_size = [256,336];

inFolder = "C:/Users/Harrison Truscott/Downloads/s/t1"
outFolder = "C:/Users/Harrison Truscott/Downloads/s/t2"


inFolder = Path(inFolder);
outFolder = Path(outFolder);
if not os.path.exists(outFolder):
    os.makedirs(outFolder);
type = "images";

mask_class = 5;
names = os.listdir(inFolder);
for name in tqdm(names):
    im = imread(inFolder/name);
    if auto_rescale and type == "images":
        im = rescale_intensity(im);
    assert isinstance(im,np.ndarray);
    if type == "masks":
        if len(im.shape) >= 3: #idk when it would be greater than three but /shrug
            im=im[:,:,0];
    sliced = False
    if x_slices > 1 or y_slices > 1:
        M = (im.shape[0]-context_bounds[0]-context_bounds[2]-crop[0]-crop[2])/y_slices;
        N = (im.shape[1]-context_bounds[1]-context_bounds[3]-crop[1]-crop[3])/x_slices;

        if int(M) != M or int(N) != N:
            raise Exception(f"ERROR: Mask with size {im.shape[:2]} cannot be sliced into {x_slices} columns and {y_slices} rows\nwith context bounds of {context_bounds}; {M} and {N} not integers");
        else:
            M = int(M)
            N = int(N)
            if (type == "images"):
                im = (im/256).astype('uint8');
                im = np.stack((im,im,im),axis=2);
            elif mask_class is not None:
                im[im>0]=mask_class;
            tiles:list[np.ndarray] = [im[y-context_bounds[0]:y+M+context_bounds[2],x-context_bounds[1]:x+N+context_bounds[3]] 
                    for y in range(context_bounds[0]+crop[0],im.shape[0]-crop[0]-crop[2]-context_bounds[0]-context_bounds[2],M) 
                    for x in range(context_bounds[1]+crop[1],im.shape[1]-crop[1]-crop[3]-context_bounds[1]-context_bounds[3],N)];
            # imBounds = [[y-context_bounds[0],y+M+context_bounds[2],x-context_bounds[1],x+N+context_bounds[3]] 
            #         for y in range(context_bounds[0]+crop[0],im.shape[0]-crop[0]-crop[2]-context_bounds[0]-context_bounds[2],M) 
            #         for x in range(context_bounds[1]+crop[1],im.shape[1]-crop[1]-crop[3]-context_bounds[1]-context_bounds[3],N)];
            # return tiles;

            # print((context_bounds[0]+crop[0],context_bounds[1]+crop[1]))
            # print(imBounds);
            for num,m in enumerate(tiles):
                if auto_downscale:
                    if imperfect_resize or ((m.shape[0]/final_size[0]).is_integer() and (m.shape[1]/final_size[1]).is_integer()):
                        if len(m.shape) == 3:
                            stype = m.dtype;
                            m = resize(m,[final_size[0],final_size[1],m.shape[2]],anti_aliasing=True).astype(stype);
                        else:
                            stype = m.dtype;
                            m = resize(m,final_size,anti_aliasing=True).astype(stype);
                    else:
                        raise Exception(f"Error: Image with shape {m.shape} not integer resizeable to final size {final_size}. If you want to force resize anyway, please set imperfect_resize to True");
                imsave(outFolder/(os.path.splitext(name)[0] + f"-{num}.TIF"), m,check_contrast=False);
    else:
        if auto_downscale:
            if imperfect_resize or ((im.shape[0]/final_size[0]).is_integer() and (im.shape[1]/final_size[1]).is_integer()):
                if len(im.shape) == 3:
                    stype = im.dtype;
                    im = resize(im,[final_size[0],final_size[1],im.shape[2]],anti_aliasing=True).astype(stype);
                else:
                    stype = im.dtype;
                    im = resize(im,final_size,anti_aliasing=True).astype(stype);
            else:
                raise Exception(f"Error: Image with shape {im.shape} not integer resizeable to final size {final_size}. If you want to force resize anyway, please set imperfect_resize to True");
        imsave(outFolder/(os.path.splitext(name)[0] + ".TIF"), im,check_contrast=False)