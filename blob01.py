"""
Blob Detection Using OpenCV ( Python, C++ )
FEBRUARY 17, 2015 BY SATYA MALLICK 152 COMMENTS

OpenCV Blob Detection Example
This tutorial explains simple blob detection using OpenCV.

What is a Blob ?
A Blob is a group of connected pixels in an image that share some common property ( E.g grayscale value ).
In the image above, the dark connected regions are blobs, and the goal of blob detection is to identify and mark these regions.

SimpleBlobDetector Example
OpenCV provides a convenient way to detect blobs and filter them based on different characteristics. Let’s start with the simplest example

https://www.learnopencv.com/blob-detection-using-opencv-python-c/
"""
import numpy as np
import cv2

# read image
img = cv2.imread('blob.jpg', cv2.IMREAD_GRAYSCALE)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change Thresholds
params.minThreshold = 10
params.maxThreshold = 100

# Filter by Area
params.filterByArea = True
params.minArea = 1500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.5

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(img)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
print(keypoints)
if len(keypoints) > 0:
    print(type(keypoints[0]))
    print(len(keypoints))
    print(keypoints[0].pt)

cv2.namedWindow('Keypoints', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Keypoints', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
