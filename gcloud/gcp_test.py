import glob
import os 
from google.cloud import storage
import subprocess
import time
import tqdm

GCS_CLIENT = storage.Client()
def upload_from_directory(directory_path: str, dest_bucket_name: str, dest_blob_name: str):
    rel_paths = glob.glob(directory_path + '/**', recursive=True)
    bucket = GCS_CLIENT.get_bucket(dest_bucket_name)
    for local_file in tqdm(rel_paths):
        remote_path = f'{dest_blob_name}/{"/".join(local_file.split(os.sep)[1:])}'
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)

bucketname = "optotaxisbucket";
destination_blob_name = "temp_output\\";
test_in = "C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/optotaxis calibration/data/segmentation_iteration_testing/processing/segmentation_images";

start = time.time();
# t = (f'gsutil -m cp -r gs:\\\\{bucketname}\\{destination_blob_name} \"{test_in}\"');
# print(t);
# os.system(t);
upload_from_directory(test_in,bucketname,destination_blob_name);
end = time.time();
print("elapsed:",end-start);