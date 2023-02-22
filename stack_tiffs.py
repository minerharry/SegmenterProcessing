import glob
import tifffile
import fnmatch
import re
import os
from fastprogress import progress_bar



def stack_files(inputs,output):
    with tifffile.TiffWriter(output) as stack:
        for filename in progress_bar(inputs): #case 
            stack.save(
                tifffile.imread(filename), 
                photometric='minisblack', 
                contiguous=True
            )

if __name__ == "__main__":
    # experiment = "2022.3.3 Migration Test 7";
    # if not os.path.exists(f"C:\\Users\\Harrison Truscott\\OneDrive - University of North Carolina at Chapel Hill\\Bear Lab\\optotaxis calibration\\data\\segmentation_iteration_testing\\movie_stacks\\{experiment}\\"):
        # os.mkdir(f"C:\\Users\\Harrison Truscott\\OneDrive - University of North Carolina at Chapel Hill\\Bear Lab\\optotaxis calibration\\data\\segmentation_iteration_testing\\movie_stacks\\{experiment}\\");
    series = 4;
    # parent_input = f"G:\\Other computers\\USB and External Devices\\USB_DEVICE_1643752484\\{experiment}";
    parent_input = f"C:\\Users\\Harrison Truscott\\Downloads\\mov4_manual";
    basename = "p1";
    input_files = glob.glob(f"{parent_input}\\{basename}_s{series}_*.TIF");
    # print(input_files);
    input_files.sort(key=lambda s: int(re.match(f".*{basename}_s{series}_t([0-9]*)\\.TIF",s).group(1)));
    
    # print(input_files);
    input_files = input_files[80:]
    out_type = "images";
    # output_file = f"C:\\Users\\Harrison Truscott\\OneDrive - University of North Carolina at Chapel Hill\\Bear Lab\\optotaxis calibration\\data\\segmentation_iteration_testing\\movie_stacks\\{experiment}\\s{series}_{out_type}.tiff";
    output_file = "C:\\Users\\Harrison Truscott\\Downloads\\mov4_manual\\mov4.tiff"
    stack_files(input_files,output_file);