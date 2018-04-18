"""
Geometric Transformations of Images
Goals
Learn to apply different geometric transformation to images like translation, rotation, affine transformation etc.
You will see these functions: cv2.getPerspectiveTransform
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sudoku-original.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

pts1 = np.float32([[55, 63], [367, 51], [26, 385], [390, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img, M, (300, 300))

plt.subplot(121)
plt.imshow(img, 'gray')
plt.title('Input')

plt.subplot(122)
plt.imshow(dst, 'gray')
plt.title('Output')

plt.show()
