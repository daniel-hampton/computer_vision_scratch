"""
Applying simple blob detection to video file.

Trackbars for adjustment have been added

Blob Detection Using OpenCV ( Python, C++ )
FEBRUARY 17, 2015 BY SATYA MALLICK 152 COMMENTS

OpenCV Blob Detection Example
This tutorial explains simple blob detection using OpenCV.

What is a Blob ?
A Blob is a group of connected pixels in an image that share some common property ( E.g grayscale value ).
In the image above, the dark connected regions are blobs,
and the goal of blob detection is to identify and mark these regions.

SimpleBlobDetector Example
OpenCV provides a convenient way to detect blobs and filter them based on different characteristics.

https://www.learnopencv.com/blob-detection-using-opencv-python-c/
"""
import numpy as np
import cv2


def nothing(x):
    pass


# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change Thresholds
params.minThreshold = 0
params.maxThreshold = 200

# Filter by Area
params.filterByArea = True
params.minArea = 100
# params.maxArea = 500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.2

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.2

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5

window_name = 'Video'
cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)

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

# Define video capture object
cap = cv2.VideoCapture('IMG_0125.MOV')
frame_counter = 0

# Define the codec and create VideoWriter object.
vid_size = (540, 960)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/blob_vid_output.avi', fourcc, 30.0, vid_size)

while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_counter += 1
    print('Frame Count: {}, Current Frame: {}, Frame Counter: {}'.format(cv2.CAP_PROP_FRAME_COUNT,
                                                                         cv2.CAP_PROP_POS_FRAMES,
                                                                         frame_counter))

    # Play the video again when it reaches the end.
    # if frame_counter == 214:
    #     frame_counter = 0
    #     cap = cv2.VideoCapture('IMG_0125.MOV')
    #     continue

    if ret is True:

        # get current values of trackbars
        params.minArea = cv2.getTrackbarPos(area, window_name)
        params.minThreshold = cv2.getTrackbarPos(min_thresh, window_name)
        params.maxThreshold = cv2.getTrackbarPos(max_thresh, window_name)
        params.minCircularity = cv2.getTrackbarPos(circ, window_name) / 100  # must be between 0 and 1
        params.minInertiaRatio = cv2.getTrackbarPos(inertia, window_name) / 100  # must be between 0 and 1
        params.minConvexity = cv2.getTrackbarPos(convex, window_name) / 100  # must be between 0 and 1

        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create(params)

        # Rotate the frame
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Change to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame before displaying
        frame = cv2.resize(frame, vid_size, fx=0, fy=0, interpolation=cv2.INTER_AREA)

        # Detect blobs
        keypoints = detector.detect(frame)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        img_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Add text counter
        font = cv2.FONT_HERSHEY_SIMPLEX
        # print(img_with_keypoints.shape)
        output_string = 'Number of Blobs: {}'.format(len(keypoints))
        cv2.putText(img_with_keypoints, output_string,
                    (10, int(0.9 * img_with_keypoints.shape[0])),
                    font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Write the video to file
        out.write(img_with_keypoints)

        # Display the frame.
        cv2.imshow(window_name, img_with_keypoints)

        # Check for ESC key to exit
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

    else:
        break

# Show keypoints
try:
    print(keypoints)
    if len(keypoints) > 0:
        print(type(keypoints[0]))
        print(len(keypoints))
        print(keypoints[0].pt)
except NameError as err:
    print(err)

# When everything is done, release all.
cap.release()
out.release()
cv2.destroyAllWindows()
