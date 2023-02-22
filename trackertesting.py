from __future__ import annotations
from centroidtracker import CentroidTracker

class StupidClass:
    def __init__(self,val,filter_out=True):
        self.val = val;
        self.filter_o= filter_out;

    @staticmethod
    def distance(c1:StupidClass,c2:StupidClass):
        return abs(c1.val-c2.val);
    
    @staticmethod
    def filter(c1:StupidClass,c2:StupidClass,stupid,stupi2d):
        return c2.filter_o;

    def __str__(self):
        return str(self.val) + " " + str(self.filter_o);
    
    def __repr__(self):
        return str(self);

tracker = CentroidTracker(2,StupidClass.distance,StupidClass.filter);
obj1 = [StupidClass(1),StupidClass(2),StupidClass(3)];
tracker.update(obj1);
print(tracker.objects);
print(tracker.disappeared);
obj2 = [StupidClass(1),StupidClass(2),StupidClass(3)];
tracker.update(obj2);
print(tracker.objects);
print(tracker.disappeared);
obj3 = [StupidClass(1),StupidClass(2,False),StupidClass(3)];
tracker.update(obj3);
print(tracker.objects);
print(tracker.disappeared);