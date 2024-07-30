from operator import itemgetter
import os
from pathlib import Path

import PIL
import PIL.Image
import cv2
from imageio.v3 import imread, imwrite
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap
import numpy as np

from roifile import ImagejRoi
from skimage.measure import label
from tqdm import tqdm

PathLike = str|Path

class IJRoi(ImagejRoi):
    def to_contours(self):
        coords:list[np.ndarray] = self.coordinates(multi=True);
        contours = tuple(cont[:,np.newaxis,:] for cont in coords);
        return contours


ColorMappable = Colormap | str
def masks_to_outlines(
        masks:list[PathLike|np.ndarray|IJRoi], 
        cmap:ColorMappable|None=None, 
        outline_thickness:int=2, 
        contour_alpha:float=1.0, 
        im_size:tuple[int,int]|None=None,
        warn_multiple:bool=True):
    
    contours = []
    combined:np.ndarray|None = None;
    colors:Colormap = get_cmap(cmap);
    for i,mask in enumerate(tqdm(masks)):
        r,g,b,a = colors(i/len(masks)); #ignore alpha
        color = r,g,b

        cont_shape = im_size

        # from IPython import embed; embed()
        if isinstance(mask,PathLike):
            print(mask)
            try:
                mask = imread(mask);
            except:
                mask = IJRoi.fromfile(mask)
        if (isinstance(mask,np.ndarray)):
            print(mask.dtype)
            print(np.unique(mask))

            if len(mask.shape) == 3:
                assert np.all((mask[:,:,0] == mask[:,:,1]) & (mask[:,:,1] == mask[:,:,2]))
                mask = mask[:,:,0]
            
            #assure only one mask
            labeled,n_labels = label(mask,return_num=True); #type:ignore
            if (n_labels > 1):
                
                sizes:list[tuple[int,int]] = [(l,np.sum(labeled == l)) for l in np.unique(labeled) if l != 0]
                biggest = max(sizes,key=itemgetter(1));
                if (warn_multiple):
                    tqdm.write(f"warning: multiple objects detected in frame {i}. selecting the largest... (label,size): {biggest}")
                mask = (labeled == biggest[0]).astype(np.uint8)*255
                # from IPython import embed; embed()

            # assert n_labels == 1; #exactly one object in the image

            #load mask, get outline
            print(mask.shape)
            contour,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
            print(contour)
        
            if cont_shape is None:
                cont_shape = mask.shape
        else:
            contour = mask.to_contours();
            if (cont_shape is None):
                raise ValueError("Must provide shape of contour image if ROIs provided")

        contour_image = np.zeros((*cont_shape[:2],3));
        contour_image = cv2.drawContours(contour_image,contour,0,color,outline_thickness);

        if combined is None:
            combined = np.zeros(contour_image.shape);

        nonzero = np.nonzero(np.count_nonzero(contour_image,axis=2));
        combined[nonzero] = combined[nonzero]*(1-contour_alpha) + contour_image[nonzero]*contour_alpha;

        contours.append(contour_image);

    return combined,contours;


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.filegetter import adir

    # path = r"C:\Users\Harrison Truscott\OneDrive - University of North Carolina at Chapel Hill\Bear Lab\Mitch Morphodynamics Panels\pre movie\maskframes"
    # path = r"C:\Users\Harrison Truscott\Downloads\RoiSetMeCell Ctrl E3 Pos8"
    path = adir(title="masks folder")
    files = os.listdir(path)

    #cellprofiler output
    files = sorted(files,key=lambda f: int(f.split("frame")[1].split(".")[0]))
    
    #roi output
    # files = sorted(files,key=lambda f: int(f.split("-")[0]))

    path = Path(path);

    # n_outlines = 10
    # frames = list(map(int,np.linspace(0,len(files)-1,n_outlines)))

    outline_space=4
    frames = range(0,len(files),outline_space)
    chosen = [files[f] for f in frames]
    print(f"using frames: {chosen}")
    im,contours = masks_to_outlines([path/f for f in chosen],outline_thickness=2,cmap="jet",im_size=(1024,1024),warn_multiple=False);

    from skimage.exposure import rescale_intensity
    outtype = np.uint8
    scaled = (rescale_intensity(im)*np.iinfo(outtype).max).astype(outtype)

    out = path.parent/f"{path.parent.name}_contours_combined.png"
    imwrite(out,scaled)

    import matplotlib.pyplot as plt

    plt.imshow(scaled); plt.show()
    # from IPython import embed; embed()

