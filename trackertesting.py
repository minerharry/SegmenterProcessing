from __future__ import annotations
import builtins
from enum import Enum
import math

from matplotlib import pyplot as plt
from libraries.centroidtracker import CentroidTracker
from utils.parse_tracks import QCTracks, QCTracksArray
import pandas as pd
from pathlib import Path


fold = r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\Segmentation Analysis\2023.5.17 OptoTiam Exp 61"
folder = Path(fold)


track_params = pd.read_pickle(folder/"Analysis Parameters"/"tracking_parameters.pkl")
dfracsize = track_params["dfracsize"]
# max_tracked_speed = track_params["max_tracked_speed"]
max_tracked_speed = 30
print(max_tracked_speed)
centroidtype = track_params['centroidtype']
max_track_disappeared_time = track_params['max_disappeared_time']
class SpeedType(Enum): 
    linear = "linear";
    MSD = "MSD";
    def __str__(self): 
        return self.value;
    def __eq__(self, __o: object) -> bool:
       if isinstance(__o,str):
          return __o.__eq__(self.value)
       return super().__eq__(__o)
untracked_speed_type = track_params['untracked_speed_type']
do_cell_collection = track_params['do_cell_collection']
cell_collection_time = track_params['cell_collection_time']
min_track_persistence_time = track_params['min_track_persistence_time']

if untracked_speed_type == SpeedType.linear:
  max_untracked_speed = track_params['max_untracked_speed'];
elif untracked_speed_type == SpeedType.MSD:
  max_untracked_diffusivity = track_params['max_untracked_diffusivity']


class Cell:
  def __init__(self,frame:pd.DataFrame,can_match=True):
    self.frame = frame;
    self.min_dist = None;
    self.can_match = can_match
  
  @staticmethod
  def cell_distance(t0:Cell,t1:Cell)->float:
    #   print(t0.frame,t1.frame)
      t0_f = t0.frame.iloc[0];
      t1_f = t1.frame.iloc[0];
      out = math.sqrt((t0_f[centroidtype+'x']-t1_f[centroidtype+'x'])**2 + (t0_f[centroidtype+'y']-t1_f[centroidtype+'y'])**2);
      if (t1.min_dist is None or out < t1.min_dist):
        t1.min_dist = out;
      return out;

  def __str__(self):
    return f"Cell <{builtins.id(self)}>:\nframe: \n" + str(self.frame) + ",\nminimum distance:" + str(self.min_dist)
  
  def __eq__(self, other: Cell) -> bool:
    return self.frame.equals(other.frame)

  @property
  def pos(self)->tuple[float,float]:
     return (self.frame.iloc[0][centroidtype+'x'],self.frame.iloc[0][centroidtype+'y'])

  @staticmethod
  def cells_filter(t0:Cell,t1:Cell,disappeared_time:int,dist:float)->bool:
    if not t0.can_match or not t1.can_match:
       return False
    t0_f = None;
    t1_f = None;
    good_area = None;
    try:
      t0_f = t0.frame.iloc[0];
      t1_f = t1.frame.iloc[0];
      # raise Exception();  
      good_area:bool = float(t1_f['area']) > (float(t0_f['area'])*(1-dfracsize)) and float(t1_f['area']) < (float(t0_f['area'])*(1+dfracsize));
      print("good area:",good_area)
    except Exception as e:
      print(t0,t1);
      print(t0_f,t1_f);
      print(t0_f['area'],t1_f['area']);
      print(type(t0_f['area']),type(t1_f['area']));
      print(good_area);
      raise e;

    if disappeared_time == 0:
        result = dist < max_tracked_speed;
        print("tracked dist valid:",result)
    else:
        if untracked_speed_type == SpeedType.linear:
          result = dist < max_untracked_speed*disappeared_time;
        elif untracked_speed_type == SpeedType.MSD:
          result = dist**2 < max_untracked_diffusivity*disappeared_time;
        else:
          raise Exception("unrecognized speed type");
        print("untracked dist valid:",result)

    if not(good_area) and result:
        result = False;

    return result

# print(list(p1.iterrows()))

if __name__ == "__main__":
    track = QCTracksArray(folder/"qc_tracks_raw.pkl")
    t1 = track[1][19][89]
    t2 = track[1][19][89]

    features = pd.read_csv(folder/"cell_features.csv")
    mov1 = features[features['movie']==1]
    p1 = mov1[mov1['frame']==89]
    p1 = [p1.iloc[[i]] for i in range(len(p1))]
    # p1.pop(3)
    # p1.pop(7)
    p2 = mov1[mov1['frame']==90]
    p2 = [p2.iloc[[i]] for i in range(len(p2))]
    # p1 = [ti[(ti['frame']==89)] for ti in tracks[1].values() if 89 in ti['frame'].values]
    # p2 = [ti[(ti['frame']==90)] for ti in tracks[1].values() if 90 in ti['frame'].values]


    tracker = CentroidTracker[Cell](
        Cell.cell_distance,
        frame_filter=Cell.cells_filter,
        maxDisappearedFrames=0,
        minPersistenceFrames=0)
    c1 = [Cell(p) for p in p1]
    c2 = [Cell(p) for p in p2]
    # c2.append(Cell(p2[0],can_match=False))
    d1 = tracker.update(c1).copy()
    d2 = tracker.update(c2).copy()
    print(d1)
    print(d2)
    # d01,d02 = pd.read_pickle("p_out.pkl")
    # print(d01,d02)
    # print(type(d01))
    # print(d01==d1,d02==d2)
    # print(d1.keys(),d01.keys())
    # pd.to_pickle((d1,d2),"p_out.pkl")

    # cellind1 = c1.index(d1[20])
    # cellind2 = c2.index(d2[20])
    # print(cellind1,cellind2)
    
    
    plt.scatter([c.pos[0] for c in d1.values()],[c.pos[1] for c in d1.values()],color='red',marker='.')
    plt.scatter([c.pos[0] for c in d2.values()],[c.pos[1] for c in d2.values()],color='blue',marker='.')
    for id,cell in d1.items():
       plt.text(cell.pos[0],cell.pos[1],str(id),verticalalignment='bottom',color='red')
    for id,cell in d2.items():
       plt.text(cell.pos[0],cell.pos[1],str(id),verticalalignment='top',color='blue')
    plt.show()