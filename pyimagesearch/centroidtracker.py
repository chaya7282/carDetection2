# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
from pyimagesearch.Rect_Comperator import rectComperator
import numpy as np
import cv2
import statistics
from math import hypot
import math
from  pyimagesearch.Single_Track import Single_Track
class CentroidTracker:
	def __init__(self, maxDisappeared=50, maxDistance=50):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.bboxes = OrderedDict()
		self.trackers = OrderedDict()
		self.KalmanFilter =  OrderedDict()

		self.Paths= OrderedDict()
		self.DeadPaths = OrderedDict()
		self.maxDisappeared = maxDisappeared
		self.maxDistance = maxDistance


	def register(self, frame,centroid ,rect,score):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		tracker = cv2.TrackerKCF_create()
		kalmanFilter =
		tracker.init(frame, (rect[0], rect[1], rect[2]-rect[0] ,rect[3]-rect[1] ))
		cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2, 1)
		self.trackers[self.nextObjectID] = tracker
		self.disappeared[self.nextObjectID] = 0
		self.bboxes[self.nextObjectID] = rect
		self.Paths[self.nextObjectID]= Single_Track(centroid,score,rect)

		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]
		del self.bboxes[objectID]
		del self.trackers[objectID]
		self.DeadPaths[objectID]=self.Paths[objectID]
		del self.Paths[objectID]

	def update(self,frame, rects, scores,status):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			
			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them

		if (len(self.objects) == 0) and (status == "Detecting"):
			self.register(frame,inputCentroids[0], rects[0],scores[0])


		if  0 < len(self.objects):
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			for row in range(len(self.bboxes)):
				for col in range(len(rects)):
					D[row,col]= 1- rectComperator.get_iou(rects[col],self.bboxes[objectIDs[row]])


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
			# or deregister an object we need to keep track o30f which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()



				# loop overrow the combination of the (row, column) index
			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore itcol,row
				if row in usedRows or col in usedCols:
					continue



				# if the distance between centroids is greater than
				# the maximum distance, do not associate the two
				# centroids to the same object[objectID]
				if D[row, col] > self.maxDistance:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0
				self.bboxes[objectID] = [rects[col][0],rects[col][1] ,rects[col][2] ,rects[col][3]]
				self.trackers[objectID] = cv2.TrackerKCF_create()
				self.trackers[objectID].init(frame, (rects[col][0],  rects[col][1],  rects[col][2]-rects[col][0], rects[col][3]- rects[col][1]))
				cv2.rectangle(frame, (rects[col][0],  rects[col][1]), (rects[col][2],  rects[col][3]), (0, 0, 255), 2, 1)

				self.Paths[objectID].addPointToTrack(inputCentroids[col],  scores[col],rects[col])
				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			for row in range(len(objectCentroids)):
				for col in range(len(inputCentroids)):
					if D[row][col] < self.maxDistance:
						usedCols.add(col)


			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)


			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared

				# loop over the unused row indexes
				#for row in unusedRows:
				# grab the object ID for the corresponding row
				# index and increment the disappeared counter
			for row in unusedRows:
				# grab the object ID for the corresponding row
				# index and increment the disappeared counter
				objectID = objectIDs[row]
				#self.disappeared[objectID] = self.disappeared[objectID] + 1
				self.Paths[objectID].addPointToTrack(self.objects[objectID], 0, self.bboxes[objectID])
				# check to see if the number of consecutive
				# frames the object has been marked "disappeared"
				# for warrants deregistering the object


				# otherwise, if the number of input centroids is greater
			# than the20 number of existing object centroids we need to
			# register each new input centroid as a trackable object

			if status =="Detecting":
				for col in unusedCols:
					self.register(frame,inputCentroids[col],rects[col],scores[col])

	# return the set of trackable objects
		return self.objects

	def track(self, frame):
		frame2= frame.copy()
		invalidTracks = set()
		for (objectID, tracker) in self.trackers.items():

			ret, pos = self.trackers[objectID].update(frame)
			#  pos = tracker.get_position()
			if self.disappeared[objectID] > self.maxDisappeared:
				invalidTracks.add(objectID)
			if ret == False:
				self.disappeared[objectID] = self.disappeared[objectID] + 1
				continue


			roi = (pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3])
			(startX, startY, endX, endY) = [int(n) for n in roi]
			#cv2.rectangle(frame2, (startX, startY), (endX, endY ), (255, 0, 0), 2, 1)
			# add the bounding box coordinates to the rectangles list
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			area = pos[2]*pos[3]
			self.objects[objectID] =np.array ([cX, cY])
			self.bboxes[objectID]= [startX, startY, endX, endY]


			cv2.imwrite("trackTest.jpg",frame2)

		for objectID in invalidTracks:
			self.deregister(objectID)

	def removeFromDeadList(self, objectID):
		del self.DeadPaths[objectID]

	def IsInDeathList(self, objectID):
		to = self.DeadPaths.get(objectID, None)
		if to is None:
			return False
		else:
			return True

	def newDeadPath(self):
		if len(self.DeadPaths):
			return True
		else:
			return False