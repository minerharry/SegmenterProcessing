# https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

# import the necessary packages
from typing import Callable, Generic, List, TypeVar, Union, Dict
from scipy.spatial import distance as dist
import numpy as np

T = TypeVar('T');
class CentroidTracker(Generic[T]):
    class Dum: ##I hate this so much
        def __init__(self,val):
            self.val = val;
            
    def __init__(self, distance_key:Callable[[T,T],float],frame_filter:Union[None,Callable[[T,T,int,float],bool]]=None,maxDisappearedFrames=50,minPersistenceFrames=20):
        # initialize the next unique object ID along with three ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid, number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.distance_key = distance_key;
        self.frame_filter = frame_filter;
        self.nextObjectID = 0
        self.objects:Dict[int,T] = dict()
        self.disappeared:Dict[int,int] = dict()
        self.active_time:Dict[int,int] = dict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappearedFrames

        # store how long an object needs to be actively tracked
        # to keep its reference active when it "disappears"
        self.minPersistence = minPersistenceFrames

    def register(self, centroid:T):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.active_time[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, inputCentroids:List[T], allow_new = True):

        # check to see if the list of input object centroids is empty
        if len(inputCentroids) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()).copy():
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared or self.active_time[objectID] < self.minPersistence:
                    self.deregister(objectID)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects


        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])


        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            # print(np.array(objectCentroids).shape, np.array(inputCentroids).shape)
            if self.distance_key is None:
                D = dist.cdist(np.array(objectCentroids), np.array(inputCentroids));
            else:
                # print(np.array([[i] for i in inputCentroids],dtype='obj').shape);
                # print(np.array([self.Dum(i) for i in inputCentroids]).reshape(-1,1).shape);
                D = dist.cdist(np.array([self.Dum(o) for o in objectCentroids]).reshape(-1,1), np.array([self.Dum(i) for i in inputCentroids]).reshape(-1,1),metric=lambda x,y: self.distance_key(x[0].val,y[0].val));

            biggest = np.max(D) + 2;
            if self.frame_filter:
                filtered_out = [i for i in np.ndindex(*D.shape) if not(self.frame_filter(
                    objectCentroids[i[0]],
                    inputCentroids[i[1]],
                    self.disappeared[objectIDs[i[0]]],
                    D[i]
                    ))];
                try:
                    for i in filtered_out:
                        D[i] = biggest;
                except IndexError as e:
                    print(filtered_out);
            

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = np.min(D,axis=1).argsort()


            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list

            cols = np.argmin(D,axis=1,)[rows]


            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            falseConversions = False;
            for (row, col) in zip(rows, cols):
                # print(D[row,col],row,col);
                if (D[row,col].item()) == biggest:
                    # print("bigelow");
                    falseConversions = True;
                    continue;
                # if we have already examined either the row or
                # column value before, ignore it - that object has already been matched
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, reset the disappeared counter,
                # and increment the active time for that object
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.active_time[objectID] += 1;
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, aka the existing and new objects respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # loop over the unused row indexes
            for row in unusedRows:
                # grab the object ID for the corresponding row
                # index and increment the disappeared counter
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1

                # check to see if the number of consecutive
                # frames the object has been marked "disappeared"
                # for warrants deregistering the object
                # also deregister if the object has not been actively
                # for enough time to warrant keeping it
                if self.disappeared[objectID] > self.maxDisappeared or self.active_time[objectID] < self.minPersistence:
                    self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            if allow_new:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects
