"""
Contours : Getting Started
Goal
Understand what contours are.
Learn to find contours, draw contours etc
You will see these functions : cv2.findContours(), cv2.drawContours()
"""

import cv2
import numpy as np

img = cv2.imread('multi-contour.png')
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imggray, 127, 255, cv2.THRESH_BINARY)

image, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours: {}".format(len(contours)))

# cnt = contours[4]
# img = cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # -1 for draw all contours

cv2.imshow('Contours', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
