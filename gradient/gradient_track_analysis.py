from pathlib import Path
from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
from gcloud.fetch_analysis import fetch_analysis

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'
        
def format_path(path:Union[str,Path],collection:Union[bool,None]=None,movie:Union[int,None]=None,experiment:Union[str,None]=None,**kwargs):
  is_path = isinstance(path,Path);
  path = str(path);
  map = SafeDict();
  if collection is not None:
    map["collection"] = "using" if collection else "no";
  if movie is not None:
    map["movie"] = str(movie);
  if experiment is not None:
    map["experiment"] = experiment;
  map.update(kwargs);
  result = path.format_map(map);
  if is_path:
    result = Path(result);
  return result;
  

analysis_folder = "C:/Users/Harrison Truscott/OneDrive - University of North Carolina at Chapel Hill/Bear Lab/optotaxis calibration/data/Segmentation Analysis/{experiment}";
flattened_gradient_path = "{analysis_folder}/gradient_track_flattened_analysis.csv";

fetch_analysis();

experiments = ["2023.1.31 OptoITSN Test 42","2023.2.1 OptoITSN Test 43","2023.2.3 OptoITSN Test 44"];

#dataframe columns: "movie","frame","trackid","label",centerx,centery,"gradient.x","gradient.y","gradient intensity","velocity.x","velocity.y","speed","persistence","dTheta"

collated_data = [];


for exp in experiments:
    gradient_data = pd.read_csv(format_path(flattened_gradient_path,analysis_folder=format_path(analysis_folder,experiment=exp)));
    # print(gradient_data);
    collated_data.append(gradient_data);

total_data = pd.concat(collated_data);

print(total_data.iloc[:10])
print(total_data.columns)
# exit()

intensity_fig = plt.subplots(2,3);

int_vs_yvel = intensity_fig[1][0][0];
int_vs_yvel.plot(total_data['gradient intensity'],total_data['velocity.y'],marker='.',linestyle="None")
int_vs_yvel.set_title("Intensity vs Y Velocity")

int_vs_xvel = intensity_fig[1][0][1];
int_vs_xvel.plot(total_data['gradient intensity'],total_data['velocity.x'],marker='.',linestyle="None")
int_vs_xvel.set_title("Intensity vs X Velocity")

int_vs_speed = intensity_fig[1][0][2];
int_vs_speed.plot(total_data['gradient intensity'],total_data['speed'],marker='.',linestyle="None")
int_vs_speed.set_title("Intensity vs Speed")

int_vs_per = intensity_fig[1][1][0];
int_vs_per.plot(total_data['gradient intensity'],total_data['persistence'],marker='.',linestyle="None")
int_vs_per.set_title("Intensity vs Persistence")

int_vs_dtheta = intensity_fig[1][1][1];
int_vs_dtheta.plot(total_data['gradient intensity'],total_data['dTheta'],marker='.',linestyle="None")
int_vs_dtheta.set_title("Intensity vs dTheta")

int_hist = intensity_fig[1][1][2];
int_hist.hist(total_data['gradient intensity'],bins=20);
int_hist.set_yscale("log")
int_hist.set_title("Intensity Histogram")


ysteepness_fig = plt.subplots(2,3);

ysteep_vs_yvel = ysteepness_fig[1][0][0];
ysteep_vs_yvel.plot(total_data['gradient.y'],total_data['velocity.y'],marker='.',linestyle="None")
ysteep_vs_yvel.set_title("Y Steepness vs Y Velocity")

ysteep_vs_xvel = ysteepness_fig[1][0][1];
ysteep_vs_xvel.plot(total_data['gradient.y'],total_data['velocity.x'],marker='.',linestyle="None")
ysteep_vs_xvel.set_title("Y Steepness vs X Velocity")

ysteep_vs_speed = ysteepness_fig[1][0][2];
ysteep_vs_speed.plot(total_data['gradient.y'],total_data['speed'],marker='.',linestyle="None")
ysteep_vs_speed.set_title("Y Steepness vs Speed")

ysteep_vs_per = ysteepness_fig[1][1][0];
ysteep_vs_per.plot(total_data['gradient.y'],total_data['persistence'],marker='.',linestyle="None")
ysteep_vs_per.set_title("Y Steepness vs Persistence")

ysteep_vs_dtheta = ysteepness_fig[1][1][1];
ysteep_vs_dtheta.plot(total_data['gradient.y'],total_data['dTheta'],marker='.',linestyle="None")
ysteep_vs_dtheta.set_title("Y Steepness vs dTheta")

ysteep_hist = ysteepness_fig[1][1][2];
ysteep_hist.hist(total_data['gradient.y'],bins=20);
ysteep_hist.set_yscale("log")
ysteep_hist.set_title("Y Steepness Histogram")

plt.show();