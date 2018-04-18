"""
Canny Edge Detection
Goal
In this chapter, we will learn about

Concept of Canny edge detection
OpenCV functions for that : cv2.Canny()
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def nothing(x):
    pass


cap = cv2.VideoCapture('IMG_0125.MOV')

cv2.namedWindow('Canny Edges', cv2.WINDOW_AUTOSIZE)

vid_size = (540, 960)

# Create trackbars for modifying edge thresholds
cv2.createTrackbar('Low_Thresh', 'Canny Edges', 100, 400, nothing)
cv2.createTrackbar('High_Thresh', 'Canny Edges', 200, 500, nothing)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output/canny_edge_output.avi', fourcc, 30.0, (vid_size[0], int(vid_size[1] * 2/3)))

while cap.isOpened():

    ret, frame = cap.read()

    if ret:

        # Change to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Rotate the frame
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Resize frame before displaying
        frame = cv2.resize(frame, vid_size, fx=0, fy=0, interpolation=cv2.INTER_AREA)

        # Get frame dimensions
        rows, cols = frame.shape

        # Region of Image of interest
        roi = frame[(rows//3):rows, :]

        # Get values from trackbar positions
        low_thresh = cv2.getTrackbarPos('Low_Thresh', 'Canny Edges')
        high_thresh = cv2.getTrackbarPos('High_Thresh', 'Canny Edges')

        edges = cv2.Canny(roi, low_thresh, high_thresh)
        # edges2 = cv2.Canny(frame, 100, 200, L2gradient=True)

        cv2.imshow('Original', frame)
        cv2.imshow('Canny Edges', edges)
        # cv2.imshow('Canny Edges L2', edges2)

        # Resize frame before displaying
        # edges = cv2.resize(edges, (540, 640), fx=0, fy=0, interpolation=cv2.INTER_AREA)

        # Write to file
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        out.write(edges)

        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
