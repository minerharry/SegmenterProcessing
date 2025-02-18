from abc import ABC, abstractmethod
from typing import DefaultDict

import cv2

from scraps.cv2_utils import getCentroid

class StatisticsTracker(ABC):
    @abstractmethod
    def movie_start(self):
        pass;

    #contours_dict: id-saved contours for persistence across frames
    @abstractmethod
    def track_frame(self,frame_num,image,mask,contours_dict,colors_dict):
        return image;

    @abstractmethod
    def movie_end(self):
        pass;

class CentroidStatisticsTracker(StatisticsTracker):
    def movie_start(self):
        self.frame = -1;
        self.tracked_centroids = DefaultDict(lambda: [[-1,-1]]*self.frame); #dict of id:history; [-1,-1] is untracked
        
    def track_frame(self,frame_num,image,mask,contours_dict,colors_dict):
        self.frame = frame_num;
        for id,contour in contours_dict.items():
            centr = getCentroid(contour);
            self.tracked_centroids[id].append(centr);
        for id in self.tracked_centroids.keys():
            if id not in contours_dict.keys():
                self.tracked_centroids[id].append([-1,-1]);

        for id,centrs in self.tracked_centroids.items():
            if centrs[-1] != [-1,-1]:
                image = cv2.drawMarker(image,centrs[-1],colors_dict[id]);
        return image;
        

    def movie_end(self):
        # print(self.tracked_centroids);
        pass;