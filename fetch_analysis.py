import gsutilwrap
from gsutilwrap import rsync
gsutilwrap.gsutil_path = "C:/Users/Harrison Truscott/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gsutil.cmd"

out_folder = "C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/optotaxis calibration/data/Segmentation Analysis"

rsync("gs://optotaxisbucket/Segmentation Analysis",out_folder,multithreaded=True,recursive=True);
print("analysis sucessfully fetched");