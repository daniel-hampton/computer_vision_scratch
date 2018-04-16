"""

Arithmetic Operations on Images
Goal
Learn several arithmetic operations on images like addition, subtraction, bitwise operations etc.
You will learn these functions : cv2.add(), cv2.addWeighted() etc.
"""

import cv2
import numpy as np

img1 = cv2.imread('building.jpg')
img2 = cv2.imread('opencv-logo.png')

# Put logo in top left corner
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]  # region of image (roi)

# Create a mask of logo and create its inverse mask as well.
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# Take only region of logo from logo image
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst

cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
