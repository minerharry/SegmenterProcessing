#https://github.com/Parquery/gs-wrap very cool
import os
from pathlib import Path, PosixPath, PurePosixPath
import shutil
import stat
import gsutilwrap
from zipfile import ZipFile


#keyword should be unique for each type of object pulled from the cloud to avoid collisions
def _fetch_image_files(in_path:Path,keyword:str,overwrite:bool)->Path: ##should not be called outside of other helper functions, will always overwrite
  # print(in_path);
  is_file = str(in_path).lower().endswith(('.zip','.tif','.tiff'));
  destination = gcp_transfer_folder/keyword/experiment;
  if not(os.path.exists(destination)):
    os.makedirs(destination);

  if overwrite or len(os.listdir(destination)) == 0:
    if is_file:
      gsutilwrap.copy(f"gs://{in_path}",destination,multithreaded=True,recursive=True);
    else:
      gsutilwrap.rsync(f"gs://{in_path}",destination,multithreaded=True,recursive=True);
  
  if (not is_file):
    return destination; #we're done
  else:
    destination = destination/in_path.name;
  if (in_path.suffix == '.zip'):
    out_path = destination.with_suffix('');
    if (overwrite or not os.path.exists(out_path)):
        with ZipFile(destination,'r') as zip:
            zip.extractall(destination.parent);
    return out_path;
  elif (in_path.suffix.lower().startswith('.tif')):
    raise NotImplementedError("unstacking TIF files not yet supported");
  else:
    raise NameError("Invalid input suffix, input validation should have caught this >:(");  

def fetch_images(force_overwrite=False)->Path:
  global images_changed;
  out = _fetch_image_files(PurePosixPath(gcp_images),'images',images_changed or force_overwrite);
  images_changed = False;
  return out;

def fetch_cell_masks(force_overwrite=False)->Path:
  global cell_masks_changed;
  out = _fetch_image_files(PurePosixPath(gcp_cell_masks),'cellmasks',cell_masks_changed or force_overwrite);
  cell_masks_changed = False;
  return out;

def fetch_nuc_masks(force_overwrite=False)->Path:
  global nuc_masks_changed;
  out = _fetch_image_files(PurePosixPath(gcp_nuc_masks),'nucmasks',nuc_masks_changed or force_overwrite);
  nuc_masks_changed = False;
  return out;

def on_rm_error( func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod( path, stat.S_IWRITE )
    # os.unlink( path )

def cleardir(dir): #clears all files in dir without deleting dir
  for f in os.scandir(dir):
    # f = os.path.join(dir,f)
    if os.path.isdir(f): shutil.rmtree(f,onerror=on_rm_error); #just in case
    else: os.remove(f);


if __name__ == "__main__":

    gsutil_exec_path = "C:/Users/Harrison Truscott/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gsutil.cmd"
    gsutilwrap.gsutil_path = gsutil_exec_path;

    #@markdown where various folders are on disk, you probably never need to change these
    local_folder = "content" #@param {type:"string"};
    local_folder:Path = Path(local_folder);

    #where files and folders are stored when downloaded from GCP
    gcp_transfer_folder = "gcp_transfer" #@param {type:"string"}
    gcp_transfer_folder:Path = Path(gcp_transfer_folder)

    #where files and folders are stored locally that should be cleared between operations
    temp_folder = "temp" #@param {type:"string"}
    temp_folder:Path = Path(temp_folder)

    if not os.path.exists(gcp_transfer_folder):
        os.mkdir(gcp_transfer_folder);

    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder);
    

    #@markdown Experiment Name (will be incorporated into output file and folder names)
    experiment = "2022.12.20 ITSNAIOopto Better Segmentation" #@param {type:"string"}


    #@markdown Analysis folder: all output analysis data will be output to gs://{analysis output folder}/{experiment name}
    analysis_output_folder = "optotaxisbucket/Segmentation Analysis" #@param {type:"string"}
    analysis_output_folder:Path = Path(analysis_output_folder)
    gcp_analysis_output_folder:Path = PurePosixPath(analysis_output_folder/experiment);
    local_analysis_output_folder:Path = gcp_transfer_folder/analysis_output_folder.name/experiment;
    if not os.path.exists(local_analysis_output_folder):
        os.makedirs(local_analysis_output_folder);
    del analysis_output_folder;

    if "gcp_images" in locals():
        prev_im = gcp_images
    else:
        prev_im = None;

    if "gcp_cell_masks" in locals():
        prev_cm = gcp_cell_masks
    else:
        prev_cm = None;

    if "gcp_nuc_masks" in locals():
        prev_nm = gcp_nuc_masks
    else:
        prev_nm = None;

    #@markdown For any of the following paths, if you put {experiment} in the string, it'll replace that with the name of the experiment for convenience:

    #@markdown Location of experiment images on the GCP bucket. Can be a folder, a .zip file, or a tif stack; If a zip file, should contain a folder of the same name as the zip file. Exclude the 'gs://' from the path. 
    gcp_images_s:str = "optotaxisbucket/movies/2022.12.20 ITSNAIOopto/2022.12.20 ITSNAIOopto" #@param {type:"string"}
    gcp_images_s = gcp_images_s.format(experiment=experiment).replace("gs://","");
    # print(gcp_images);
    gcp_images:Path = PurePosixPath(gcp_images_s);

    #@markdown Location of segmented cell-area masks on the GCP bukcet. Can be a folder, a .zip file, or a tif stack; If a zip file, should contain a folder of the same name as the zip file. Exclude the 'gs://' from the path.
    gcp_cell_masks_s:str = "optotaxisbucket/movie_segmentation/{experiment}/segmentation_output_masks/Cell.zip" #@param {type:"string"}
    gcp_cell_masks_s = gcp_cell_masks_s.format(experiment=experiment).replace("gs://","");
    gcp_cell_masks:Path = PurePosixPath(gcp_cell_masks_s);

    #@markdown Location of segmented nucleus-area masks on the GCP bukcet. Can be a folder, a .zip file, or a tif stack; If a zip file, should contain a folder of the same name as the zip file. Exclude the 'gs://' from the path.
    gcp_nuc_masks_s = "optotaxisbucket/movie_segmentation/{experiment}/segmentation_output_masks/Nucleus.zip" #@param {type:"string"}
    gcp_nuc_masks_s = gcp_nuc_masks_s.format(experiment=experiment).replace("gs://","");
    gcp_nuc_masks:Path = PurePosixPath(gcp_nuc_masks_s);

    images_changed,cell_masks_changed,nuc_masks_changed = [False,False,False];
    if gcp_images != prev_im:
        images_changed = True;
    if gcp_cell_masks != prev_cm:
        cell_masks_changed = True;
    if gcp_nuc_masks != prev_nm:
        nuc_masks_changed = True;
    
    do_test = False;
    if do_test:
        test = gcp_images;
        if (not str(test).lower().endswith((".zip",".tif",".tiff"))): #directory
            test = str(test) + '/*'
            valid = gsutilwrap.stat(f"gs://{test}");
            print(valid);
            if (not valid): #images dir does not exist
                raise Exception(f"Error: Images dir gs://{gcp_images} does not exist in bucket. To ignore this error simply run the succeeding cells.")

        test = gcp_cell_masks;
        if (not str(test).lower().endswith((".zip",".tif",".tiff"))): #directory
            test = str(test) + '/*'
            valid = gsutilwrap.stat(f"gs://{test}");
            print(valid);
            if (not valid): #cell dir does not exist
                raise Exception(f"Error: Cell masks dir gs://{gcp_cell_masks} does not exist in bucket. To ignore this error simply run the succeeding cells.")

        test = gcp_nuc_masks;
        if (not str(test).lower().endswith((".zip",".tif",".tiff"))): #directory
            test = str(test) + '/*'
            valid = gsutilwrap.stat(f"gs://{test}");
            print(valid);
            if (not valid): #nuc dir does not exist
                raise Exception(f"Error: Nucleus masks dir gs://{gcp_nuc_masks} does not exist in bucket. To ignore this error simply run the succeeding cells.")
        print("Successfully verified inputs - all folders and files exist in the bucket")

    fetch_cell_masks();