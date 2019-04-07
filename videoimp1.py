# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import math
from PIL import Image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-y", "--yolo", required=True)
ap.add_argument("-c", "--confidence", type=float, default=0.5)
ap.add_argument("-t", "--threshold", type=float, default=0.3)
args = vars(ap.parse_args())
x1=100
y1=500
x2=1000
y2=520
r=0
list1=[]
boundaries =[([17, 15,100], [50, 56, 200])]
lower_white = np.array([100, 15, 17], dtype=np.uint8)
upper_white = np.array([200, 56, 50], dtype=np.uint8)
'''
def my_function():
        
    _, frame = vs.read()
    # define range of white color in HSV
    # change it according to your need !
 

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(frame, lower_white, upper_white)
    # Bitwise-AND mask and original image
    
    r = cv2.bitwise_and(frame, frame, mask = mask).astype(np.uint8)
    mask = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(mask,100,200)
    _, cnts,_ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts=sorted(cnts, key = cv2.contourArea)
    # loop over our contours to find the best possible approximate contour of number plate
    count = 0
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
      
        if len(approx) == 4:
            rect = cv2.minAreaRect(cnt)       #I have used min Area rect for better result
            x,y,w,h = cv2.boundingRect(cnt)
        
            if (w>5) and (h>5) and (w < 250) and (h<500):
            
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                #cv2.line(frame, (x-100,y), (x+100, y), (0, 0xFF, 0), 1)
                cx=int(x+w/2)
                cy=int(y+h/2)
                list1.append(cy)
                cv2.circle(frame,(cx, cy), 4, (0, 0, 255))
    p=max(list1,key=list1.count)
    return p;
'''
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
lower_white = np.array([0,0,225], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)
# initialize the video stream, pointer to output video file, and
# frame dimensions
r=0
vs = cv2.VideoCapture("videos/testing8.mp4")


writer = None
(W, H) = (None, None)


# loop over frames from the video file stream
while True:
        # read the next frame from the file
        #p=my_function()
        p=900
        (grabbed, frame) = vs.read()
        centerx=0
        centery=0
        w1=0
        h1=0        
        f=0
        r+=1
        
        cv2.line(frame, (0,p), (1500, p), (0, 0xFF, 0), 5)
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
                break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
                (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        # loop over each of the layer outputs
        for output in layerOutputs:
                #cv2.line(frame, (x1,y1), (x2, y2), (0, 0xFF, 0), 2)
                # loop over each of the detections
                for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > args["confidence"]:
                                # scale the bounding box coordinates back relative to
                                # the size of the image, keeping in mind that YOLO
                                
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                #print(centerY)                       
                                if(centerY>=p-10 and centerY<p+10):
                                        cv2.line(frame, (0,p), (1000, p), (0, 0, 0xFF), 5)
                                        centerx=centerX
                                        centery=centerY
                                        w1=width
                                        h1=height
                                        f=1
                                        
                                else:
                                        cv2.line(frame, (0,p), (1000, p), (0, 0xFF, 0), 5)
                                      
                                # use the center (x, y)-coordinates to derive the top
                                # and and left corner of the bounding box
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))
                                
                                # update our list of bounding box coordinates,
                                # confidences, and class IDs
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)                  

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                args["threshold"])

        # ensure at least one detection exists
        if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                        
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        
                        # draw a bounding box rectangle and label on the frame
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                confidences[i])
                        
                        if (LABELS[classIDs[i]]=="traffic light"):
                                x_new=x+int(w/2);
                                y_new=y+int(h/2);
                                #cv2.circle(frame,(x_new, y_new), 4, (0,255,0))
                                hue = frame[y:y+h, x:x+w]
                                #cv2.imwrite('mask{}.jpg'.format(r),hue)
                                hue = cv2.cvtColor(hue, cv2.COLOR_BGR2HSV)
                                
                                if(hue.shape[0]!=0):
                                        mask1 =cv2.inRange(hue, (0, 100, 100), (10, 255, 255))
                                        mask2 =cv2.inRange(hue, (160, 100, 100), (179, 255, 255))
                                        mask=mask1+mask2
                                        #mask =cv2.bitwise_and(mask,mask, mask=mask)
                                        cv2.imwrite('mas{}.jpg'.format(r),mask)
                                        
                                        if cv2.countNonZero(mask) <= 10:
                                                print ("No red detected")
                                        else:
                                                
                                       
                                                
                                                print("Red Detected")
                                                if f==1 :
                                                        cv2.line(frame, (0,p), (1000, p), (0, 0, 0xFF), 5)
                                                        cv2.imwrite('images/{}.jpg'.format(r),frame[int(centery-(h1/2)):int(centery-(h1/2))+h1, int(centerx-(w1/2)):int(centerx-(w1/2))+w1])
                                                        print("maada")        
                                               
                                        centerx=0
                                        centery=0
                                        w1=0
                                        h1=0       
                                              
                        

        # check if the video writer is None
        if writer is None:
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 30,
                        (frame.shape[1], frame.shape[0]), True)

                # some information on processing single frame
                

        # write the output frame to disk
        writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")

vs.release()
