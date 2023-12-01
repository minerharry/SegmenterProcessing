from __future__ import annotations
import math
from pathlib import Path
import re
from typing import DefaultDict
import cv2
from abc import ABC
import os
import libraries.parsend as parsend

import numpy as np
from fastprogress import progress_bar,master_bar
from tqdm import tqdm
from libraries.centroidtracker import CentroidTracker
import random
from skimage.io import imread, imshow
from skimage.exposure import rescale_intensity;
from scraps.cv2_utils import getCentroid
from libraries.parsend import sorted_dir
from scraps.statistics_trackers import CentroidStatisticsTracker, StatisticsTracker

# T = TypeVar('T')

class DummyClass:

    def __init__(self,contour,centroid):
        self.contour = contour;
        self.val = centroid;

    @staticmethod
    def distance(a:list[DummyClass],b:list[DummyClass])->float:
        return math.sqrt((a[0].val[0]-b[0].val[0])**2 + (a[0].val[1]-b[0].val[1])**2);

    def __str__(self):
        return str(self.val);

    @staticmethod
    def filter_frame(a:list[DummyClass],b:list[DummyClass],timeDisappeared:int,distance:float)->bool:
        # print(f"checking objects {a}, {b}");
        # print(f"distance: {DummyClass.distance(a,b)}");
        # return DummyClass.distance(a,b) < 50;
        return (distance < 500 if timeDisappeared == 0 else distance < 300*timeDisappeared);




def track_cells():
    inputImageFolder = Path("G:/Other computers/USB and External Devices/USB_DEVICE_1643752484/2022.3.8 Migration Test 8"); #"input//images//"
    inputMaskFolder = Path("C:/Users/Harrison Truscott/Downloads/movie_segmentation_2022.3.8 Migration Test 8_segmentation_output_masks_Cell/Cell"); #"input//masks//"
    outputImageFolder = Path("output/images/");
    if not os.path.exists(outputImageFolder):
        os.makedirs(outputImageFolder);

    maskNames = parsend.grouped_dir(os.listdir(inputMaskFolder));
    # print(maskNames[:10]);
    imageNames = parsend.sorted_dir(os.listdir(inputImageFolder));

    # def compare_ending_nums(key):
    #     try:
    #         base = os.path.splitext(key)[0];
    #         return int(re.match('.*?([0-9]+)$', base).group(1));
    #     except Exception as e:
    #         print(key);
    #         raise e;

    #for easy maskname->corresponding image
    #note: unsupported behavior for multiple images w/ same basename & diff extensions
    imageDict = {os.path.splitext(name)[0]:name for name in imageNames};


    contourColors = DefaultDict(lambda: (random.randrange(0,256),random.randrange(0,256),random.randrange(0,256)));

    statTrackers:list[StatisticsTracker] = [CentroidStatisticsTracker()];

    m = master_bar(maskNames[2:]);
    for movie in m:
        ct = CentroidTracker[list[DummyClass]](maxDisappeared=4,distance_key=DummyClass.distance,frame_filter=DummyClass.filter_frame); #maximum frames for wich something can be disappeared
        for stat in statTrackers:
            stat.movie_start();
        #TODO: how to ensure to play the movie in order?
        for i,maskName in enumerate(progress_bar(movie,parent=m)):
            # print(maskName);
            mask = imread(inputMaskFolder/maskName);

            ##Track contours through centroid tracking
            ##NOTE: check for accuracy - is a more advanced algorithm needed?

            #get contours in the image
            contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
            # print("contours:",len(contours))
            #get centroids - is a dict so that we can id the contours based on output from the tracker
            # print([cv2.contourArea(c) for c in contours]);
            centroids = [[DummyClass(contour,getCentroid(contour)),] for contour in contours if cv2.contourArea(contour) != 0];

            #update tracker with current centroids, label the contours with ids
            objs = ct.update(list(centroids));
            # print("tracked objects:",len(objs));
            labeledContours = {id:obj[0].contour for id,obj in objs.items()};

            imageName = (imageDict[os.path.splitext(maskName)[0]]);
            image = cv2.imread(str(inputImageFolder/imageName));
            image = rescale_intensity(image);
            # print(image.shape);

            for stat in statTrackers:
                stat.track_frame(i,image,mask,labeledContours,contourColors);


            for id,c in labeledContours.items():
                cv2.drawContours(image=image,contours=[c],contourIdx=-1,color=contourColors[id]);
            # cv2.imshow("title",image);
            # cv2.waitKey(0);
            cv2.imwrite(str((outputImageFolder/imageName).absolute()),image);


            # cv2.imshow('blah',image);
            # cv2.waitKey(0);

        for stat in statTrackers:
            stat.movie_end();



if __name__ == "__main__":
    track_cells();