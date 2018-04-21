"""
Demo blob detection on fabric with light and dark holes
"""
import numpy as np
import cv2


def nothing(x):
    pass


# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change Thresholds
# 120 was pretty good for dark holes, 130 also good but looks a little more sensitive
# Light holes: max 176, min 165, step 10.
params.minThreshold = 0
params.maxThreshold = 120
params.thresholdStep = 10

# Filter by Area
params.filterByArea = True
params.minArea = 100  # default was 100
# params.maxArea = 500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.2

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.2

# Filter by Color
params.filterByColor = True  # This seems to allow detection of both light and dark blobs
params.blobColor = 0

# Filter by distance between
params.minDistBetweenBlobs = -100

window_name = 'Video'
cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)

# create trackbars for color change
area = 'Min Area'
min_thresh = 'Min Threshold'
max_thresh = 'Max Threshold'
circ = 'Min Circularity'
inertia = 'Min Intertia Ratio'
convex = 'Min Convexity'

# cv2.createTrackbar(area, window_name, int(params.minArea), 2000, nothing)
# cv2.createTrackbar(min_thresh, window_name, int(params.minThreshold), 5000, nothing)
# cv2.createTrackbar(max_thresh, window_name, int(params.maxThreshold), 5000, nothing)
# cv2.createTrackbar(circ, window_name, int(params.minCircularity * 100), 100, nothing)  # convert from 0 to 1 range
# cv2.createTrackbar(inertia, window_name, int(params.minInertiaRatio * 100), 100, nothing)  # convert from 0 to 1 range
# cv2.createTrackbar(convex, window_name, int(params.minConvexity * 100), 100, nothing)  # convert from 0 to 1 range

# Define video capture object
cap = cv2.VideoCapture('fabric.mp4')
frame_counter = 0

# Define the codec and create VideoWriter object.
vid_size = (1280, 720)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/fabric_blob_output.avi', fourcc, 30.0, vid_size)

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
        # Redefine params and detect dark blobs at start of loop
        params.minThreshold = 0
        params.maxThreshold = 120
        params.thresholdStep = 10
        params.blobColor = 0

        # Get dimensions of frame
        height, width = frame.shape[:2]

        # Define rectangular Region of Interest (ROI)
        x0 = 0.2 * width
        y0 = 0.2 * height
        x1 = 0.8 * width
        y1 = 0.2 * height
        x2 = 0.8 * width
        y2 = 0.8 * height
        x3 = 0.2 * width
        y3 = 0.8 * height
        roi = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], np.int32)
        cv2.polylines(frame, [roi], True, (0, 255, 0), 1, cv2.LINE_AA)

        # Create mask from ROI
        mask = np.zeros((height, width), np.uint8)
        mask = cv2.fillPoly(mask, [roi], 1)

        frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

        # get current values of trackbars
        # params.minArea = cv2.getTrackbarPos(area, window_name)
        # params.minThreshold = cv2.getTrackbarPos(min_thresh, window_name)
        # params.maxThreshold = cv2.getTrackbarPos(max_thresh, window_name)
        # params.minCircularity = cv2.getTrackbarPos(circ, window_name) / 100  # must be between 0 and 1
        # params.minInertiaRatio = cv2.getTrackbarPos(inertia, window_name) / 100  # must be between 0 and 1
        # params.minConvexity = cv2.getTrackbarPos(convex, window_name) / 100  # must be between 0 and 1

        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector_create(params)

        # Rotate the frame
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Change to grayscale
        frame_masked = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2GRAY)

        # Resize frame before displaying
        # frame = cv2.resize(frame, vid_size, fx=0, fy=0, interpolation=cv2.INTER_AREA)

        # Detect blobs
        keypoints = detector.detect(frame_masked)

        # Redefine params and detect light blobx
        params.minThreshold = 165
        params.maxThreshold = 176
        params.thresholdStep = 10
        params.blobColor = 255
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints2 = detector.detect(frame_masked)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        # Draw dark blob keypoints
        img_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Draw light blob keypoints
        img_with_keypoints = cv2.drawKeypoints(img_with_keypoints, keypoints2, np.array([]), (0, 0, 255),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Add text counter
        font = cv2.FONT_HERSHEY_SIMPLEX
        # print(img_with_keypoints.shape)
        output_string = 'Holes Detected: {}'.format(len(keypoints) + len(keypoints2))
        # Draw black background for text
        cv2.rectangle(img_with_keypoints, (0, int(0.91 * height)), (width//4, int(0.91 * height - 40)), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(img_with_keypoints, output_string,
                    (10, int(0.9 * img_with_keypoints.shape[0])),
                    font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Resize img_with_keypoints before recording to file. Output video will fail if it does not match vid_size
        img_with_keypoints = cv2.resize(img_with_keypoints, vid_size, fx=0, fy=0, interpolation=cv2.INTER_AREA)

        # Write the video to file
        out.write(img_with_keypoints)

        # Display the frame.
        cv2.imshow(window_name, img_with_keypoints)

        # Check for ESC key to exit
        k = cv2.waitKey(1) & 0xFF
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
