"""
Geometric Transformations of Images
Goals
Learn to apply different geometric transformation to images like translation, rotation, affine transformation etc.
You will see these functions: cv2.getPerspectiveTransform
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('building.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

M = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

