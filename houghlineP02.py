"""
Hough Line Transform
Goal
In this chapter,
We will understand the concept of Hough Tranform.
We will see how to use it detect lines in an image.
We will see following functions: cv2.HoughLines(), cv2.HoughLinesP()
"""

import cv2
import numpy as np

img = cv2.imread('vlcsnap.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)

for line in lines:
    for x1, y1, x2, y2 in line:

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

img = cv2.resize(img, (540, 960), interpolation=cv2.INTER_AREA)

# cv2.imshow('edges', edges)
cv2.imshow('Hough Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
