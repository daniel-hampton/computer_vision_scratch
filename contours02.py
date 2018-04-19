"""

Contour Features
Goal
In this article, we will learn

To find the different features of contours, like area, perimeter, centroid, bounding box etc
You will see plenty of functions related to contours.
"""

import cv2
import numpy as np

from pprint import pprint

img = cv2.imread('multi-contour.png')
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imggray, 127, 255, cv2.THRESH_BINARY)

image, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours: {}".format(len(contours)))

# Print Image Moments


cnt = contours[2]
cnt2 = contours[0]
M = cv2.moments(cnt)
pprint(M)

# Calc centroid
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
print('{}, {}'.format(cx, cy))

print('Contour area: {:.2f}'.format(cv2.contourArea(cnt)))

print('Contour perimeter: {:.2f}'.format(cv2.arcLength(cnt, True)))

# approximating contour of potentially misshapen objects
epsilon = 0.01 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

print('Approx area: {:.2f}'.format(cv2.contourArea(approx)))
print('Approx perimeter: {:.2f}'.format(epsilon / 0.01))
print('Approx perimeter: {:.2f}'.format(cv2.arcLength(approx, True)))

# Convex hull
hull = cv2.convexHull(cnt)
print('is contour convex: {}'.format(cv2.isContourConvex(cnt)))

# draw a regular bounding rectangle (not rotated for min area)
x, y, w, h = cv2.boundingRect(cnt2)
img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# draw a rotated, min area bounding rectangle
rect = cv2.minAreaRect(cnt2)
box = cv2.boxPoints(rect)
box = np.int0(box)  # Converts array dtype to integer
img = cv2.drawContours(img, [box], 0, (255, 0, 0), 2)

# draw minimum are enclosing circle
(x, y), radius = cv2.minEnclosingCircle(cnt2)
center = (int(x), int(y))
radius = int(radius)
cyan = (238, 244, 66)
img = cv2.circle(img, center, radius, cyan, 2)

# Fitting a straight line to contour
rows, cols = img.shape[:2]
[vx, vy, x, y] = cv2.fitLine(cnt2, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x*vy/vx) + y)  # Y intercept (b in y=mx+b) for the left hand side of the image
righty = int(((-x + cols) * vy/vx) + y)  # Y intercept for right hand side of image. Adjust X by cols (width) of image.
img - cv2.line(img, (cols-1, righty), (0, lefty), (0, 255, 0), 2)


# contours parameter must be a list dtype. List of contours: [cnt]
img = cv2.drawContours(img, contours, -1, (244, 66, 244), 3)

# Draw Convex Hull
img = cv2.drawContours(img, [hull], 0, (0, 255, 0), 3)
# img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # -1 for draw all contours

cv2.imshow('Contours', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
