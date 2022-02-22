import re
from typing import DefaultDict
import cv2
from abc import ABC,abstractmethod
import os
from centroidtracker import CentroidTracker
import random
from cv2_utils import getCentroid
from statistics_trackers import CentroidStatisticsTracker, StatisticsTracker



def track_cells():
    inputImageFolder = "D:\\2022.1.20 Migration Test 1\\"; #"input\\images\\"
    inputMaskFolder = "C:\\Users\\Harrison Truscott\\Documents\\GitHub\\SegmenterGui\\export\\test3\\"; #"input\\masks\\"
    outputImageFolder = "output/images/";

    maskNames = os.listdir(inputMaskFolder);
    imageNames = os.listdir(inputImageFolder);

    def compare_ending_nums(key):
        base = os.path.splitext(key)[0];
        return int(re.match('.*?([0-9]+)$', base).group(1));

    maskNames.sort(key=compare_ending_nums);


    #for easy maskname->corresponding image
    #note: unsupported behavior for multiple images w/ same basename & diff extensions
    imageDict = {os.path.splitext(name)[0]:name for name in imageNames};

    ct = CentroidTracker(maxDisappeared=4); #maximum frames for wich something can be disappeared

    contourColors = DefaultDict(lambda: (random.randrange(0,256),random.randrange(0,256),random.randrange(0,256)));

    statTrackers:list[StatisticsTracker] = [CentroidStatisticsTracker()];
    for stat in statTrackers:
        stat.movie_start();

    #TODO: how to ensure to play the movie in order?
    for i,maskName in enumerate(maskNames):
        mask = cv2.imread(inputMaskFolder+maskName,flags=cv2.IMREAD_GRAYSCALE);
        
        
        ##Track contours through centroid tracking
        ##NOTE: check for accuracy - is a more advanced algorithm needed?

        #get contours in the image
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
        print("contours:",len(contours))
        #get centroids - is a dict so that we can id the contours based on output from the tracker
        centroids = {getCentroid(contour):contour for contour in contours};

        #update tracker with current centroids, label the contours with ids
        objs = ct.update(list(centroids.keys()));
        print("tracked objects:",len(objs));
        labeledContours = {id:centroids[centr] for id,centr in objs.items() if centr in centroids};

        imageName = (imageDict[os.path.splitext(maskName)[0]]);
        image = cv2.imread(inputImageFolder+imageName);
        print(image.shape);

        for stat in statTrackers:
            stat.track_frame(i,image,mask,labeledContours,contourColors);

        # print(labeledContours);

        for id,c in labeledContours.items():
            cv2.drawContours(image=image,contours=[c],contourIdx=-1,color=contourColors[id]);
        cv2.imwrite(outputImageFolder+imageName,image);


        # cv2.imshow('blah',image);
        # cv2.waitKey(0);

    for stat in statTrackers:
        stat.movie_end();



if __name__ == "__main__":
    track_cells();