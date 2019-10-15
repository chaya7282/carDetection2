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
ap.add_argument("-i", "--input", type=str,default = "sub-1504619634606.mp4",
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.965,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=1,
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




# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")

    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

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

# start the frames per second throughput estimator

(ret, frame2) = vs.read()

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
                rgb = frame
                trackers = []

                input_frame = frame
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
                        box = boxes[i]

                    #    box = box * np.array([W, H, W, H])
                        ymin = int(box[0] *H)
                        xmin =int( box[1]*W)
                        ymax = int(box[2]*H)
                        xmax= int(box[3]*W)
                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs



                        # construct a dlib rectangle object from the bounding
                        # box coordinates and then start the dlib correlation
                        # tracker
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle( xmin, ymin, xmax, ymax)
                        tracker.start_track(rgb, rect)

                        # add the tracker to our list of trackers so we can
                        # utilize it during skip frames
                        trackers.append(tracker)
                        rects.append((xmin, ymin, xmax, ymax))





                for j in range(0, len(rects)):
                    cv2.rectangle(frame, (rects[j][0],  rects[j][1]), ( rects[j][2], rects[j][3]), (255, 0, 0))

                plt.imshow(frame, cmap='gray')
                plt.show()
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
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

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
                ("Up", totalUp),
                ("Down", totalDown),
                ("Status", status),
                ("Frame number", totalFrames)
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # check to see if we should write the frame to disk
            if writer is not None:
                writer.write(frame)

            # show the output frame

            # cv2.imshow("Frame", frame))
            plt.imshow(frame, cmap='gray')
            cv2.imwrite("test_2.jpg",frame)
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
