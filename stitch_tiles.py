import functools
# import itertools
import gc
import math
import os
from pathlib import Path
from typing import Collection, Container, Iterable, NamedTuple, Sequence
from imageio.v3 import imread,imwrite

from skimage.transform import rescale
from skimage.transform import downscale_local_mean

import numpy as np
from scipy.special import softmax
from tifffile import TiffFile
from tqdm import tqdm
from metamorph_tiling import Tiling
# from parsend import parseND

from ast import literal_eval as leval
# import bidict

from parsend import parseND
from stack_nd import SizedGenerator, StageDimension, TimeDimension, WaveDimension, iter_nd, sorted_groups
from stack_tiffs import write_series

class Rect(NamedTuple): 
    xmin:float
    xmax:float
    ymin:float
    ymax:float

    @property
    def width(self):
        return self.xmax - self.xmin
    
    @property
    def height(self):
        return self.ymax - self.ymin
    
    @property
    def size(self):
        return (self.height,self.width)

    def scaled(self,factor:float|tuple[float,float,*tuple[float,...]]):
        if isinstance(factor,Iterable):
            yfactor,xfactor,*_ = factor
        else:
            yfactor,xfactor = factor,factor
        return Rect(self.xmin*xfactor,self.xmax*xfactor,self.ymin*yfactor,self.ymax*yfactor)

    def shifted(self,yoffset:float,xoffset:float):
        return Rect(self.xmin+xoffset,self.xmax+xoffset,self.ymin+yoffset,self.ymax+yoffset)

    def grown(self,offset:float):
        return Rect(self.xmin-offset,self.xmax+offset,self.ymin-offset,self.ymax + offset)    
    
    def intify(self,max_shape:tuple[int,int]):
        return Rect(
            max(math.floor(self.xmin),0),
            min(math.ceil(self.xmax),max_shape[1]),
            max(math.floor(self.ymin),0),
            min(math.ceil(self.ymax),max_shape[0]),
        )

    @classmethod
    def from_center_size(cls,center:tuple[float,float],size:tuple[float,float]): #assumes (y,x)
        ymin,xmin = center[0]-size[0]//2,center[1]-size[1]//2
        ymax,xmax = ymin + size[0], xmin + size[1]
        return cls(xmin,xmax,ymin,ymax);


@functools.cache
def rectangle_phase_field(tile:Rect, #assumes (y,x)tuptu
                          im_size:tuple[int,int,*tuple[int,...]],
                          maxval:float=1, ext_padding=0.2):
    tile = tile.grown(ext_padding) #make sure edge pixels of image included, just in case
    tile = tile.intify(im_size[:2])

    Y,X = np.meshgrid(np.arange(tile.ymin,tile.ymax),np.arange(tile.xmin,tile.xmax),indexing='ij',copy=False,sparse=True)
    X = X.astype(int)
    Y = Y.astype(int)

    field_shape = (Y.shape[0],X.shape[1])


    # print(im_size)
    # print(Y.shape)
    # print(X.shape)
    
    f_x = maxval*4*(X - tile.xmin)*(tile.xmax - X)/(tile.xmax-tile.xmin)**2 #x parabola
    # print(f_x.shape)
    f_y = maxval*4*(Y - tile.ymin)*(tile.ymax - Y)/(tile.ymax-tile.ymin)**2 #y parabola
    # print(f_y.shape)
    minfield = np.min(np.broadcast_arrays(f_x,f_y),axis=0)
    # print(min.shape)
    subfield = np.max(np.broadcast_arrays(minfield,np.zeros(field_shape)),axis=0) #combined "parabaloid (ish)", clamped to >= 0
    # print(field.shape)

    field = np.zeros(im_size[:2],dtype=np.float32)
    field[Y,X] = subfield
    return field

def get_big_objects():
    from pympler.asizeof import asizeof
    print(asizeof(all=True, above=1024, cutoff=10, stats=1))

def situate_image(tile:Rect,
                  im_size:tuple[int,int,*tuple[int,...]],
                  image:np.ndarray):
    
    topleft = int(tile.ymin),int(tile.xmin)

    im = np.zeros(im_size,dtype=image.dtype)

    im[topleft[0]:topleft[0]+image.shape[0],topleft[1]:topleft[1]+image.shape[1]] = image
    
    return im


def stitch_tiles(tiling:Tiling|Iterable[tuple[int,int]],
                 images:Sequence[np.ndarray|Path|str],
                 size_per_pixel:tuple[float,float]=(1,1),
                 upscale:int|tuple[int,int]|None=4,
                 downscale:bool=True):

    if len(tiling) != len(images):
        raise ValueError("Number of tiles does not equal number of images!")

    
    #get image shapes. For memory reasons, don't keep the images around. We will use a generator to make sure only one image is loaded at a time
    iminfo_generator = (imread(im) if not isinstance(im,np.ndarray) else im for im in images)
    iminfo = ((im.shape,im.dtype) for im in iminfo_generator)
    im_shapes,im_dtypes = zip(*iminfo) #unzip list of tuples
    del iminfo, iminfo_generator

    #read images generator. this will be passed to upscale generator as well
    to_stitch_generator = (imread(im) if not isinstance(im,np.ndarray) else im for im in images)

    if upscale is None:
        upscale = 1
    #perform "upscaling" to allow for between-pixel image placement
    if not isinstance(size_per_pixel,tuple): size_per_pixel = (size_per_pixel,size_per_pixel)
    if not isinstance(upscale,tuple): upscale = (upscale,upscale)

    extra_channels = (len(im_shapes[0]) - len(upscale))
    extra_channels_shape = im_shapes[0][-extra_channels:] if extra_channels > 0 else tuple()
    if extra_channels > 0:
        upscale:tuple[int,...] = upscale + (1,)*extra_channels
    upscaled_generator = (rescale(im,upscale,preserve_range=True).astype(im.dtype) for im in to_stitch_generator)
    im_shapes = [(upscale[0]*im_shape[0],upscale[1]*im_shape[1]) for im_shape in im_shapes]

    #extract the (upscaled-space) bounding boxes from the tiles and image sizes, get global minimum + maximum
    unshifted_im_rects:list[Rect] = []
    minx,miny,maxx,maxy = math.inf,math.inf,-math.inf,-math.inf
    for (y,x),im_shape in zip(tiling,im_shapes):
        if isinstance(y,str):
            assert isinstance(x,Iterable)
            x,y = x #tiling is rerversed
        else:
            assert not isinstance(x,Iterable)
        y *= upscale[0] / size_per_pixel[0]
        x *= upscale[1] / size_per_pixel[1]

        rect = Rect.from_center_size((y,x),(im_shape[0],im_shape[1]));
        minx = min(rect.xmin,minx)
        miny = min(rect.ymin,miny)
        maxx = max(rect.xmax,maxx)
        maxy = max(rect.ymax,maxy)

        unshifted_im_rects.append(rect)
    

    #shift smallest to 0, get size of full image
    im_rects = [rect.shifted(-miny,-minx) for rect in unshifted_im_rects] 
    im_size = (math.ceil(maxy-miny),math.ceil(maxx-minx)) + extra_channels_shape

    #get the images situated in the large image as well as the "phase fields" - image relevance, 1 at the center and 0 outside the image bounds
    stitched_shape = (len(im_rects),*im_size[:2])
    phase_fields = np.ndarray(stitched_shape,dtype=np.float32)
    for i,rect in enumerate(tqdm(im_rects,desc="Calculating image stitching weights",leave=False)):
        phase_fields[i] = (rectangle_phase_field(rect,im_size))
        

    #do pixelwise ~~softmax~~ L1 normalization of the phase fields to get blends for each image
    # normalized_fields = np.linalg.nor(phase_fields,axis=0)
    # from sklearn.preprocessing import normalize

    pixelsums = np.sum(phase_fields,axis=0); 
    pixelsums[pixelsums==0] = 1
    phase_fields /= pixelsums
    del pixelsums
    # from IPython import embed; embed()
    # assert np.allclose(np.sum(normalized_fields,axis=0),np.round(np.sum(normalized_fields,axis=0))),"normalization failure"
    
    #extend normalized_fields to be able to multiply with the extra channels
    phase_fields = np.reshape(phase_fields,phase_fields.shape + (1,)*extra_channels)


    #get situated images
    stitched = np.ndarray(im_size,dtype=phase_fields.dtype)

    for i,(rect,im) in enumerate(zip(tqdm(im_rects,desc="Stitching images",leave=False),upscaled_generator)):
        sit = situate_image(rect,im_size,im)
        broad_im,p = np.broadcast_arrays(sit,phase_fields[i])
        stitched += broad_im*p
        del im,p,broad_im,sit

    stitched = stitched.astype(im_dtypes[0])
    #weight images by normalized phase fields and add to get the stitched image
    # phase_fields *= situated_images
    # ims,weights = np.broadcast_arrays(situated_images,phase_fields)
    # weightfile = None
    # return_weights = True
    # if return_weights:
    #     import tempfile
    #     weightfile = tempfile.TemporaryDirectory()
    #     np.save(Path(weightfile.name)/"weights.npy",weights)

    # weights *= ims #because weights dtype is float
    # stitched = weights.sum(0).astype(ims.dtype)

    # if weightfile:
    #     weights = [np.load(Path(weightfile.name)/"weights.npy")]
    # stitched:np.ndarray = np.ma.average(ims,weights=weights,axis=0).filled(0).astype(ims.dtype)
    # stitched = stitched.astype(situated_images.dtype)
    # del situated_images

    #downscale to original pixel sizes
    print(gc.get_referrers(phase_fields))
    if downscale:
        stitched = downscale_local_mean(stitched,upscale).astype(stitched.dtype)
        im_rects = [rect.scaled(np.reciprocal(upscale)) for rect in im_rects]

    del phase_fields
    return stitched,im_rects,None

rowformat = "{name}; ; ({x}, {y})\n"

image_offset = {"4x":(21500,-16300),"10x":(8593,-6640),"20x":(4332,-3320)} ##width,height of image in microscope units (X,Y)

image_size = (1344,1024) ##width, height of image in pixels

def stitch_nd(nd_loc:str|Path|os.PathLike[str],
              output:str|Path|os.PathLike[str],
              source_exts:Collection[str]=(".tif",".tiff"),
              images_folder:str|Path|os.PathLike[str]="",
              selected_stages:Container[int|str]|None=None,
              mag:str|None=None, #4x, 10x, etc
              size_per_pixel:tuple[float,float]|None=None, #y,x
              total_image_size:tuple[float,float]|None=None, #y,x
              upscale=1,
              ):
    assert len(source_exts) > 0
    ims_folder = Path(nd_loc).parent/images_folder
    filenames = (ims_folder).glob(f"*[{']['.join(source_exts)}]")
    
    if size_per_pixel is None and total_image_size is None:
        if mag is None:
            raise ValueError("At least one of size_per_pixel, total_image_size, or mag must not be None for the program to determine pixel size")
        else:
            total_image_size = image_offset[mag][1],image_offset[mag][0] #y,x        
    
    # all_stitched:np.ndarray|list[np.ndarray]|None = None
    it = iter_nd(filenames,nd=nd_loc,order=(TimeDimension,StageDimension,WaveDimension))
    assert isinstance(it,SizedGenerator)
    def iter_stitch(): #get result in iterator form to pass to series writer
        assert isinstance(it,SizedGenerator)
        for (timepoint,_),stages in tqdm(it):
            images_to_stitch:list[np.ndarray] = []
            stage_positions:list[tuple[int,int]] = []
            assert isinstance(stages,SizedGenerator)
            for (stagenum,stagename),waves in stages:
                if stagenum is None:
                    raise ValueError("No stage positions to stitch!")
                if selected_stages:
                    if stagenum not in selected_stages and stagename not in selected_stages:
                        continue
                stage_image:np.ndarray|None = None
                ##y,x; these are in **metamorph units**
                stage_pos:tuple[int,int]|None = None
                assert isinstance(waves,SizedGenerator)
                for (wavenum,wavename),wave in waves:
                    tiff = TiffFile(wave);
                    im = tiff.asarray()

                    #get position from file
                    if stage_pos is None:
                        meta = tiff.metaseries_metadata;
                        assert meta
                        plane = meta['PlaneInfo'];
                        stage_pos = int(plane['stage-position-y']),int(plane['stage-position-x']);
                    
                    assert len(im.shape) == 2 #all metamorph outputs should be grayscale
                    if stage_image is None:
                        if wavenum is None: #single channel, grayscale image
                            stage_image = im
                        else:
                            stage_image = im[:,:,None] #add channel for stacking
                    else:
                        if wavenum is None:
                            raise ValueError("Cannot have 'None' wavelength and a wave number in the same dimension!")
                        else:
                            stage_image = np.stack([stage_image,im[:,:,None]],axis=-1) #stack channels
                assert stage_image is not None
                assert stage_pos is not None
                images_to_stitch.append(stage_image)
                stage_positions.append(stage_pos)

            pixelsize = size_per_pixel
            if pixelsize is None:
                assert total_image_size is not None
                target_shape = images_to_stitch[0].shape[:2]
                pixelsize = (total_image_size[0]/target_shape[0],total_image_size[1]/target_shape[1])

            #stitch images
            stitched_image,rects,phases = stitch_tiles(stage_positions,images_to_stitch,size_per_pixel=pixelsize,upscale=upscale)

            del rects,phases #make sure no memory leaks
            yield stitched_image
            gc.collect()


    output = (Path(nd_loc)/Path(output)).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    print("saving output to",output)
    if len(it) == 1: #single image, just imwrite
        stich_it = iter_stitch();
        im = next(stich_it)
        try:
            k = next(stich_it)
            raise AssertionError("Too many images returned by nd iterator, this shouldn't be possible.")
        except StopIteration:
            pass
        imwrite(output,im)
    else:
        write_series(output,SizedGenerator(iter_stitch(),len(it)))
    
    list(it)
    it.close()
    list(stich_it)
    stich_it.close()

def stitch_backtile(nd_loc:str|Path|os.PathLike[str],
              output_folder:str|Path|os.PathLike[str],
              source_exts:Collection[str]=(".tif",".tiff"),
              images_folder:str|Path|os.PathLike[str]="",
              mag:str|None=None, #4x, 10x, etc
              size_per_pixel:tuple[float,float]|None=None, #y,x
              total_image_size:tuple[float,float]|None=None, #y,x
              upscale=1):
    
    NDData = parseND(nd_loc)
    positions = [NDData.get(f"Stage{i}") for i in range(1,1+int(NDData.get("NStagePositions")))]

    def get_orig(x:str):
        return x.split("backtile{")[1].split('}')[0]
    
    for backtile,positions in sorted_groups(positions,key=get_orig):
        stitch_nd(nd_loc,
                  Path(output_folder)/f"{backtile}.tif",
                  source_exts=source_exts,
                  images_folder=images_folder,
                  selected_stages=list(positions),
                  mag=mag,
                  size_per_pixel=size_per_pixel,
                  total_image_size=total_image_size,
                  upscale=upscale)
    




if __name__ == "__main__":

    # im = imread('favicon.ico')

    # tiles = (10,20),(30,40)

    # out = stitch_tiles(tiles,[im,im])

    # from IPython import embed; embed();

    par = Path("C:/Users/bearlab/Documents/Data_temp/Harrison/2024.8.7 OptoPLC S345F Protrusion Test")

    for folder,mag in [("Phase","10x"),("Cy5 Pre","20x"),("Halo Pre","20x")]:

        nd = par/folder/"p.nd"
        out = str(par/"stitches"/f"{folder}.tif") + "f" if folder == "Phase" else ""
        if mag == "20x":
            stitch_backtile(nd,par/"stitches"/folder,mag=mag)
        else:
            continue
            stitch_nd(nd,out,mag=mag)