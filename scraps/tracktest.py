from libraries.centroidtracker import CentroidTracker
import numpy as np
import random

objects = [1,2,3,4,5,6,7,8];
o = [1,2,3,5,6,7,8,20]
objects2 = [o[i] + 0.05*i for i in range(len(o))];
# random.shuffle(objects2);
print(objects,objects2);

tracker = CentroidTracker[float](lambda a,b: abs(a-b),frame_filter = lambda c1,c2,time,dist: dist < 10,maxDisappeared=3);
with np.printoptions(suppress=True):
    print(tracker.update(objects));
    for _ in range(5):
        print(tracker.update(objects2));
    print(tracker.disappeared[3])