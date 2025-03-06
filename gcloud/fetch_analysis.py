from pathlib import Path
import gsutilwrap
from gsutilwrap import rsync

def fetch_analysis():
    gsutilwrap.gsutil_path = str(Path.home()/"AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gsutil.cmd")

    out_folder = str(Path.home()/"OneDrive - University of North Carolina at Chapel Hill/Bear Lab/optotaxis calibration/data/Segmentation Analysis")

    rsync("gs://optotaxisbucket/Segmentation Analysis",out_folder,multithreaded=True,recursive=True);
    print("analysis sucessfully fetched");

def push_analysis():
    gsutilwrap.gsutil_path = str(Path.home()/"AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gsutil.cmd")

    out_folder = str(Path.home()/"OneDrive - University of North Carolina at Chapel Hill/Bear Lab/optotaxis calibration/data/Segmentation Analysis")

    rsync(out_folder,"gs://optotaxisbucket/Segmentation Analysis",multithreaded=True,recursive=True);
    print("analysis sucessfully fetched");

if __name__ == "__main__":
    fetch_analysis();