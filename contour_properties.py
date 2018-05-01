from datetime import datetime

import cv2
import numpy as np
from matplotlib import pyplot as plt

from pprint import pprint

# img = cv2.imread('circle-holes.png')
img = cv2.imread('multi-contour.png')
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imggray, 127, 255, cv2.THRESH_BINARY)

image, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours: {}".format(len(contours)))

# Draw Contours
img = cv2.drawContours(img, contours, 2, (0, 255, 0), 2)

# First contour
cnt = contours[2]

# Aspect Ratio of contour
# It is the ratio of width to height of bounding rect of the object.
x, y, w, h = cv2.boundingRect(cnt)
aspect_ratio = w / h
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2, cv2.LINE_AA)
print('Aspect Ratio: {:.2f}'.format(aspect_ratio))

# Extent of contour
# Extent is the ratio of contour area to bounding rectangle area.
cnt_area = cv2.contourArea(cnt)
x, y, w, h = cv2.boundingRect(cnt)
rect_area = w * h
extent = cnt_area / rect_area
print('Extent: {:.2f}'.format(extent))

# Solidity
# Solidity is the ratio of contour area to its convex hull area.
cnt_area = cv2.contourArea(cnt)
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = cnt_area / hull_area
print('Solidity: {:.2f}'.format(solidity))

# Equivalent Diameter
# Equivalent Diameter is the diameter of the circle whose area is same as the contour area
cnt_area = cv2.contourArea(cnt)
equiv_diameter = np.sqrt(4 * cnt_area / np.pi)
print('Equiv. Diameter: {:.2f}'.format(equiv_diameter))

# Orientation
# Orientation is the angle at which object is directed. Following method also gives the Major Axis and Minor Axis lengths.
(x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
print('Ellipse center: {:.2f}, {:.2f}'.format(x, y))
print('MA: {:.2f}, ma: {:.2f}'.format(MA, ma))
print('Angle: {:.2f}'.format(angle))  # In degrees it seems
# Convert to integers
center = (int(x), int(y))
axes = (int(MA), int(ma))
angle = int(angle)

# Draw ellipse for orientation
cv2.ellipse(img, center, axes, angle, 0, 360, (255, 0, 0), 1, cv2.LINE_AA)

# Mask and Pixel Points
mask = np.zeros(imggray.shape, np.uint8)
cv2.drawContours(mask, [cnt], 0, 255, -1)
start_time = datetime.now()
pixelpoints = np.transpose(np.nonzero(mask))
delta_t = datetime.now() - start_time
print('Numpy method time: {}'.format(delta_t))
# Display the mask image,
cv2.imshow('mask', mask)

start_time = datetime.now()
pixelpoints2 = cv2.findNonZero(mask)
delta_t = datetime.now() - start_time
print('Opencv method time: {}'.format(delta_t))
# cv2.imshow('mask', pixelpoints2)  # Cannot display the list of points, not an image.

# Maximum Value, Minimum Value and their locations
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imggray, mask=mask)
print('min_val: {}'.format(min_val))
print('max_val: {}'.format(max_val))
print('min_loc: {}'.format(min_loc))
print('max_loc: {}'.format(max_loc))

# Extreme Points
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

print('\nExtreme Points\n')
print('Left: {}'.format(leftmost))
print('Right: {}'.format(rightmost))
print('Top: {}'.format(topmost))
print('Bottom: {}'.format(bottommost))

# Draw extreme point circle
cv2.circle(img, leftmost, 10, (0, 0, 255), -1, cv2.LINE_AA)
cv2.circle(img, rightmost, 10, (0, 0, 255), -1, cv2.LINE_AA)
cv2.circle(img, topmost, 10, (0, 0, 255), -1, cv2.LINE_AA)
cv2.circle(img, bottommost, 10, (0, 0, 255), -1, cv2.LINE_AA)

# Mean Color or Mean Intensity
mean_val = cv2.mean(imggray, mask)
print('mean_val: {}'.format(mean_val))

# Create list of centroids and area of contours
# contour_centroids = []  # items with format (x, y, area)
# total_area = 0
# for cont in contours:
#     M = cv2.moments(cont)
#     cx = int(M['m10'] / M['m00'])
#     cy = int(M['m01'] / M['m00'])
#
#     area = cv2.contourArea(cont)
#     total_area += area
#
#     contour_centroids.append((cx, cy, area))

cv2.imshow('Contour Areas', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
