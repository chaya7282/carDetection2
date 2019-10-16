# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject

import numpy as np
import argparse

import time
import dlib
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import os
# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt",  type=str,default=" XXX",
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model",  type=str,default=" XXX",
                help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,default = "sub-1504614469486.mp4",
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,default= "/home/ubuntu/PycharmProjects/custom_vehicle_training/out.avi",
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=5,
                help="# of skip frames between detections")
ap.add_argument("-t", "--draw-trajectory", type=bool, default=False,
                help="# of skip frames between detections")
args = vars(ap.parse_args())

# What model to download.
MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Download Model
# uncomment if you have not download the model yet
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

frame_width = None
frame_height =  None

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")

    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.


frame_width = int(vs.get(3))
frame_height = int(vs.get(4))

# initialize the video writer (we'll instantiate later if need be)
writer = None
# if we are supposed to be writing a video to disk, initialize
# the writer

if not args.get("output", False):
    print("[INFO] starting video stream...")
    writer = None
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening output video file...")
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    writer = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1, (frame_width, frame_height))


# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
car_count =0;
# start the frames per second throughput estimator

(ret, frame2) = vs.read()
trackers = []

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')



        # loop over frames from the video stream
        while vs.isOpened():

            # grab the next frame and handle if we are reading from either
            # VideoCapture or VideoStream
            ret = False
            time.sleep(2.0)

            if 2000 <  totalFrames:
                break
            # frame = frame[1] if args.get("input", False) else frame
            (ret, frame) = vs.read()
            # if we are viewing a video and we did not grab a frame then we
            # have reached the end of the video
            if args["input"] is not None and frame is None:
                break

            # if the frame dimensions are empty, set them
            if W is None or H is None:
                H, W , _ = frame.shape

            # if we are supposed to be writing a video to disk, initialize
            # the writer

            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
            status = "Waiting"
            rects = []
            # check to see if we should run a more computationally expensive
            # object detection method to aid our tracker
            if totalFrames % args["skip_frames"] == 0:
                # set the status and initialize our new set of object trackers
                status = "Detecting"
                trackers =[]
                input_frame = frame\
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                              detection_classes, num_detections],
                             feed_dict={image_tensor: image_np_expanded})
                boxes = np.squeeze(boxes)
                classes = np.squeeze(classes).astype(np.int32)
                scores = np.squeeze(scores)

                for i in range(0,len(boxes)):
                    # extract the confidence (i.e., probability) associated
                    # with the prediction
                    confidence = scores[i]

                    # filter out weak detections by requiring a minimum
                    # confidence
                    if confidence > args["confidence"]:

                        # if the class label is not a person, ignore it
                        if categories[classes[i]-1]['name'] != "car":
                            continue

                        # compute the (x, y)-coordinates of the bounding box
                        # for the object
                        box = boxes[i] * np.array([H, W, H, W])

                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs

                        (ymin, xmin, ymax, xmax)= box.astype(np.int32)

                        # construct a dlib rectangle object from the bounding
                        # box coordinates and then start the dlib correlation
                        # tracker
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle( xmin, ymin, xmax, ymax)
                        tracker.start_track(frame, rect)

                        # add the tracker to our list of trackers so we can
                        # utilize it during skip frames
                        trackers.append(tracker)
                        rects.append((xmin, ymin, xmax, ymax))


                # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
            else:
                # loop over the trackers
                for tracker in trackers:
                    # set the status of our system to be 'tracking' rather
                    # than 'waiting' or 'detecting'
                    status = "Tracking"

                    # update the tracker and grab the updated position
                    tracker.update(frame)
                    pos = tracker.get_position()

                    # unpack the position object
                    roi = [pos.left(), pos.top(), pos.right(), pos.bottom()]
                    (startX, startY,endX,endY) = [int(n) for n in roi]
                   # add the bounding box coordinates to the rectangles list
                    rects.append((startX, startY, endX, endY))

                # draw a horizontal line in the center of the frame -- once an
                # object crosses this line we will determine whether they were
                # moving 'up' or 'down'
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

                # use the centroid tracker to associate the (1) old object
                # centroids with (2) the newly computed object centroids

            objects = ct.update(rects)

            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # check to see if a trackable object exists for the current
                # object ID
                to = trackableObjects.get(objectID, None)

                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid)

                else:
                    # the difference between the y-coordinate of the *current*
                    # centroid and the mean of *previous* centroids will tell
                    # us in which direction the object is moving (negative for
                    # 'up' and positive for 'down')
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    # check to see if the object has been counted or not
                    if not to.counted:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        if  centroid[1] < H // 2:
                             to.pos_up = True

                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object
                        elif  centroid[1] > H // 2:
                             to.pos_down = True

                        if to.pos_up  and to.pos_down:
                            car_count += 1
                            to.counted = True;
                    if args.get("draw-trajectory"):
                        pts= to.centroids
                        # loop over the set of tracked points
                        for k in range(1, len(pts)):
                            # if either of the tracked points are None, ignore
                            # them
                            if pts[k - 1] is None or pts[k] is None:
                                continue

                            # otherwise, compute the thickness of the line and
                            # draw the connecting lines

                            cv2.line(frame, (pts[k - 1][0],pts[k - 1][1]),  (pts[k ][0],pts[k ][1]), (0, 0, 255), 2)




                # store the trackable object in our dictionary
                trackableObjects[objectID] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("car_count", car_count),
                ("Status", status),
                ("Frame number", totalFrames)
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

         #   plt.imshow(frame, cmap='gray')
         #   plt.show()

# check to see if we should write the frame to disk
            if writer is not None:
                writer.write(frame)

           # increment the total number of frames processed thus far and
            # then update the FPS counter
            totalFrames += 1
            print(totalFrames)

        # check to see if we need to release the video writer pointer
        if writer is not None:
            writer.release()

        # if we are not using a video file, stop the camera video stream
        if not args.get("input", False):
            vs.stop()

        # otherwise, release the video file pointer
        else:
            vs.release()



# close any open windows
