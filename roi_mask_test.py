"""
Test code to create a non-rectangular Region of Interest (ROI)
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('vlcsnap.png')

# Get dimensions of image array
rows, cols = img.shape[:2]

# initialize mask as all zeroes (black)
mask = np.zeros((rows, cols), np.uint8)  # mask must be np.unint8 not the default np.float65

# roi of polygon points
myROI = [(0, 985), (692, 786), (1025, 822), (1089, 985), (1089, 1724), (0, 1516)]
cv2.fillPoly(mask, [np.array(myROI)], 1)

# apply the mask to the original image
res = cv2.bitwise_and(img, img, mask=mask)


# Resize image to display
mask = cv2.resize(mask, (540, 960), interpolation=cv2.INTER_AREA)
res = cv2.resize(res, (540, 960), interpolation=cv2.INTER_AREA)

cv2.imshow('Mask', mask)
cv2.imshow('Result', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
