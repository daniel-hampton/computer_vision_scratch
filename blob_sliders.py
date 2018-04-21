"""
Trackbars for adjustment have been added

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


def nothing(x):
    pass


# read image
img = cv2.imread('moth.jpg', cv2.IMREAD_GRAYSCALE)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change Thresholds
params.minThreshold = 0
params.maxThreshold = 200

# Filter by Area
params.filterByArea = True
params.minArea = 50

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.18

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.1

window_name = 'Keypoints'
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

# create trackbars for color change
area = 'Min Area'
min_thresh = 'Min Threshold'
max_thresh = 'Max Threshold'
circ = 'Min Circularity'
inertia = 'Min Intertia Ratio'
convex = 'Min Convexity'

cv2.createTrackbar(area, window_name, int(params.minArea), 2000, nothing)
cv2.createTrackbar(min_thresh, window_name, int(params.minThreshold), 5000, nothing)
cv2.createTrackbar(max_thresh, window_name, int(params.maxThreshold), 5000, nothing)
cv2.createTrackbar(circ, window_name, int(params.minCircularity * 100), 100, nothing)  # convert from 0 to 1 range
cv2.createTrackbar(inertia, window_name, int(params.minInertiaRatio * 100), 100, nothing)  # convert from 0 to 1 range
cv2.createTrackbar(convex, window_name, int(params.minConvexity * 100), 100, nothing)  # convert from 0 to 1 range

while True:
    # get current values of trackbars
    params.minArea = cv2.getTrackbarPos(area, window_name)
    params.minThreshold = cv2.getTrackbarPos(min_thresh, window_name)
    params.maxThreshold = cv2.getTrackbarPos(max_thresh, window_name)
    params.minCircularity = cv2.getTrackbarPos(circ, window_name) / 100  # must be between 0 and 1
    params.minInertiaRatio = cv2.getTrackbarPos(inertia, window_name) / 100  # must be between 0 and 1
    params.minConvexity = cv2.getTrackbarPos(convex, window_name) / 100  # must be between 0 and 1

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(img)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow(window_name, img_with_keypoints)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


# Show keypoints
print(keypoints)
if len(keypoints) > 0:
    print(type(keypoints[0]))
    print(len(keypoints))
    print(keypoints[0].pt)

# Save resulting image
# cv2.imwrite('cloth_holes_detection.png', img_with_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()
