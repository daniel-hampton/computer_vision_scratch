"""
Detect green spots as points and draw a polygon from them as it moves through camera field
"""

import cv2
import numpy as np

ip = "192.168.1.235"
cap = cv2.VideoCapture('rtsp://{0}/11'.format(ip))
# cap = cv2.VideoCapture(1)

cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)

# target_green_rgb = np.array([0, 160, 0])
# green_hsv = cv2.cvtColor(target_green_rgb, cv2.COLOR_RGB2HSV)
# green_hsl = np.array([60, 127, 127])

# Define range of green color in HSV
h = int(207 * (179 / 359))
s = int(88 * (255 / 100))
v = int(81 * (255 / 100))

lower_green = np.array([27, 25, 25])
upper_green = np.array([70, 255, 255])

while cap.isOpened():

    ret, frame = cap.read()

    if ret:
        # Do something

        height, width = frame.shape[:2]

        print('Resolution: {} x {}'.format(width, height))

        # Convert colorspace to HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # frame_hsv = cv2.GaussianBlur(frame_hsv, (5, 5), 0)
        # blur = cv2.medianBlur(frame, 3)
        # blur = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

        mask = cv2.inRange(frame_hsv, lower_green, upper_green)

        result_frame = cv2.bitwise_and(frame_hsv, frame_hsv, mask=mask)

        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_HSV2BGR)

        # cv2.imshow('Mask', mask)

        # cv2.imshow('Mask', mask)
        cv2.imshow('Video', result_frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
