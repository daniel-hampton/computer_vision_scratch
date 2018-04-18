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


cap = cv2.VideoCapture(1)

cv2.namedWindow('Canny Edges', cv2.WINDOW_AUTOSIZE)

# Create trackbars for modifying edge thresholds
cv2.createTrackbar('Low_Thresh', 'Canny Edges', 100, 400, nothing)
cv2.createTrackbar('High_Thresh', 'Canny Edges', 200, 500, nothing)

while cap.isOpened():

    ret, frame = cap.read()

    if ret:

        # Get values from trackbar positions
        low_thresh = cv2.getTrackbarPos('Low_Thresh', 'Canny Edges')
        high_thresh = cv2.getTrackbarPos('High_Thresh', 'Canny Edges')

        edges = cv2.Canny(frame, low_thresh, high_thresh, L2gradient=True)
        # edges2 = cv2.Canny(frame, 100, 200, L2gradient=True)

        cv2.imshow('Original', frame)
        cv2.imshow('Canny Edges', edges)
        # cv2.imshow('Canny Edges L2', edges2)

        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()