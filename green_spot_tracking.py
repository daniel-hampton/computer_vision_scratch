"""
Detect green spots as points and draw a polygon from them as it moves through camera field
"""

import cv2
import numpy as np

ip = "192.168.1.235"
# cap = cv2.VideoCapture('rtsp://{0}/11'.format(ip))
cap = cv2.VideoCapture(1)

cv2.namedWindow('Mask', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)

# target_green_rgb = np.array([0, 160, 0])
# green_hsv = cv2.cvtColor(target_green_rgb, cv2.COLOR_RGB2HSV)
# green_hsl = np.array([60, 127, 127])

# Define range of green color in HSV
h = int(151 * (179 / 359))
s = int(53 * (255 / 100))
v = int(78 * (255 / 100))

lower_green = np.array([h - 10, s - 20, v - 50])
upper_green = np.array([h + 10, s + 20, v + 50])

while cap.isOpened():

    ret, frame = cap.read()

    if ret:
        # Do something

        height, width = frame.shape[:2]

        print('Resolution: {} x {}'.format(width, height))

        # Convert colorspace to HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # frame_hsv = cv2.GaussianBlur(frame_hsv, (5, 5), 0)
        # frame_hsv = cv2.medianBlur(frame_hsv, 3)
        # blur = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

        mask = cv2.inRange(frame_hsv, lower_green, upper_green)

        # Remove noise through morphological transforms, opening and closing
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of spots.
        ret_image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Print number of contours
        print(len(contours))

        # Get list of centroids of contours
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))
        centroids = np.asarray(centroids, np.int32)
        print(centroids)
        if len(contours) > 2:
            hull = cv2.convexHull(centroids, returnPoints=True)

            # draw ROI from centroids
            cv2.polylines(frame, [hull], True, (0, 0, 255), 1, cv2.LINE_AA)

            mask_roi = np.zeros((height, width), np.uint8)
            mask_roi = cv2.fillPoly(mask_roi, [hull], 255)
            cv2.imshow('Mask ROI', mask_roi)
        else:
            hull = []
            print('Contours 2 or less')



        result_frame = cv2.bitwise_and(frame_hsv, frame_hsv, mask=mask)

        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_HSV2BGR)

        # Display Frames
        cv2.imshow('Mask', mask)
        cv2.imshow('Video', frame)

        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
