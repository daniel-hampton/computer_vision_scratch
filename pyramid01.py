"""
Goal
In this chapter,
We will learn about Image Pyramids
We will use Image pyramids to create a new fruit, “Orapple”
We will see these functions: cv2.pyrUp(), cv2.pyrDown()
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('building.jpg')

lower_reso1 = cv2.pyrDown(img)
lower_reso2 = cv2.pyrDown(lower_reso1)
lower_reso3 = cv2.pyrDown(lower_reso2)

upimage = cv2.pyrUp(lower_reso3)
upimage = cv2.pyrUp(upimage)
upimage = cv2.pyrUp(upimage)

cv2.imshow('Original', img)
cv2.imshow('Pyr Down 1', lower_reso1)
cv2.imshow('Pyr Down 2', lower_reso2)
cv2.imshow('Pyr Down 3', lower_reso3)
cv2.imshow('Upscaled Image', upimage)

cv2.waitKey(0)
cv2.destroyAllWindows()

