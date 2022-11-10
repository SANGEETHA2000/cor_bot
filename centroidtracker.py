# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bbox = OrderedDict()  # CHANGE

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.max_disappeared = max_disappeared

        # store the maximum distance between centroids to associate
        # an object -- if the distance is larger than this maximum
        # distance we'll start to mark the object as "disappeared"
        self.max_distance = max_distance

    def register(self, centroid, input_rect):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.next_object_id] = centroid
        self.bbox[self.next_object_id] = input_rect  # CHANGE
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bbox[object_id]  # CHANGE
        
    def rects_ifempty(self, rects):
        # loop over any existing tracked objects and mark them
        # as disappeared
        for object_id in list(self.disappeared.keys()):
            self.disappeared[object_id] += 1

            # if we have reached a maximum number of consecutive
            # frames where a given object has been marked as
            # missing, deregister it
            self.disappeared[object_id] > self.max_disappeared ? self.deregister(object_id)

        # return early as there are no centroids or tracking info
        # to update
        # return self.objects
        return self.bbox
    
    def loop_boundingbox(self, rects, input_centroids, input_rects):
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (c_x, c_y)
            input_rects.append(rects[i])  # CHANGE
            return input_centroids, input_rects
        
    def if_no_track_objects(self, input_centroids, input_rects):
        for i in range(0, len(input_centroids)):
            self.register(input_centroids[i], input_rects[i])  # CHANGE
            
    def check_disappeared(self, unused_rows, object_ids):
        # loop over the unused row indexes
        for row in unused_rows:
            # grab the object ID for the corresponding row
            # index and increment the disappeared counter
            object_id = object_ids[row]
            self.disappeared[object_id] += 1

            # check to see if the number of consecutive
            # frames the object has been marked "disappeared"
            # for warrants deregistering the object
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
                
    def register_centroid(self, input_centroids, input_rects, unused_cols):
        for col in unused_cols:
            self.register(input_centroids[col], input_rects[col])
            
    def if_track_objects(self, input_centroids, input_rects):
        # grab the set of object IDs and corresponding centroids
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # compute the distance between each pair of object
        # centroids and input centroids, respectively -- our
        # goal will be to match an input centroid to an existing
        # object centroid
        D = dist.cdist(np.array(object_centroids), input_centroids)

        # in order to perform this matching we must (1) find the
        # smallest value in each row and then (2) sort the row
        # indexes based on their minimum values so that the row
        # with the smallest value as at the *front* of the index
        # list
        rows = D.min(axis=1).argsort()

        # next, we perform a similar process on the columns by
        # finding the smallest value in each column and then
        # sorting using the previously computed row index list
        cols = D.argmin(axis=1)[rows]

        # in order to determine if we need to update, register,
        # or deregister an object we need to keep track of which
        # of the rows and column indexes we have already examined
        used_rows = set()
        used_cols = set()

        # loop over the combination of the (row, column) index tuples
        for (row, col) in zip(rows, cols):
            # if we have already examined either the row or
            # column value before, ignore it or
            # if the distance between centroids is greater than the maximum distance, do not associate the two centroids to the same object
            row in used_rows or col in used_cols or (D[row, col] > self.max_distance) ? continue

            # otherwise, grab the object ID for the current row, set its new centroid, and reset the disappeared counter
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.bbox[object_id] = input_rects[col]  # CHANGE
            self.disappeared[object_id] = 0

            # indicate that we have examined each of the row and
            # column indexes, respectively
            used_rows.add(row)
            used_cols.add(col)

        # compute both the row and column index we have NOT yet examined
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)

        # in the event that the number of object centroids is equal or greater than the number of input centroids
        # we need to check and see if some of these objects have potentially disappeared
        # otherwise, if the number of input centroids is greater than the number of existing object centroids we need to
        # register each new input centroid as a trackable object
        D.shape[0] >= D.shape[1] ? check_disappeared(unused_rows, object_ids) : register_centroid(input_centroids, input_rects, unused_cols)

    def update(self, rects):
        # check to see if the list of input bounding box rectangles is empty
        len(rects) == 0 ? return rects_ifempty(self, rects)
            
        # initialize an array of input centroids for the current frame
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_rects = []
        
        # loop over the bounding box rectangles
        input_centroids, input_rects = loop_boundingbox(rects, input_centroids, input_rects)

        # if we are currently not tracking any objects take the input centroids and register each of them
        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object centroids
        len(self.objects) == 0 ? if_no_track_objects(self, input_centroids, input_rects) : if_track_objects(self, input_centroids, input_rects)

        # return the set of trackable objects
        # return self.objects
        return self.bbox
