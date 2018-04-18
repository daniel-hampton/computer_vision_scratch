"""
Geometric Transformations of Images
Goals
Learn to apply different geometric transformation to images like translation, rotation, affine transformation etc.
You will see these functions: cv2.getPerspectiveTransform
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('moth.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rows, cols, ch = img.shape

print(img.shape)

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 2050]])

M = cv2.getAffineTransform(pts1, pts2)

dst = cv2.warpAffine(img, M, (cols, rows))

plt.subplot(121)
plt.imshow(img)
plt.title('Input')

plt.subplot(122)
plt.imshow(dst)
plt.title('Output')

plt.show()
