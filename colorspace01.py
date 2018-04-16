"""
Goal
In this tutorial, you will learn how to convert images from one color-space to another, like BGR \leftrightarrow Gray, BGR \leftrightarrow HSV etc.
In addition to that, we will create an application which extracts a colored object in a video
You will learn following functions : cv2.cvtColor(), cv2.inRange() etc.
"""

from pprint import pprint
import numpy as np
import cv2


# flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
# pprint(flags)

# cap = cv2.VideoCapture('cuphead.mp4')
cap = cv2.VideoCapture(1)

while cap.isOpened():

    # take each frame
    ret, frame = cap.read()

    if ret:

        # convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range of blue color in HSV
        h = int(207 * (179 / 359))
        s = int(88 * (255 / 100))
        v = int(81 * (255 / 100))
        lower_blue = np.array([h - 10, 50, 180])  # raised v value to exclude darker blues in background.
        upper_blue = np.array([h + 10, 255, 255])

        # Define range of blue color in HSV
        red = np.uint8([[[5, 21, 195]]])
        hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
        lower_red = np.array([hsv_red[0, 0, 0] - 20, 200, 50])
        upper_red = np.array([hsv_red[0, 0, 0] + 20, 255, 255])

        # Threshold the SHV image to get only blue colors
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Threshold the SHV image to get only red colors
        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        # Combine masks with bitwise-OR
        mask = cv2.bitwise_or(red_mask, blue_mask)

        # Bitwise-AND mask and original image.
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('result', result)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
