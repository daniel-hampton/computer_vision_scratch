"""
Geometric Transformations of Images
Goals
Learn to apply different geometric transformation to images like translation, rotation, affine transformation etc.
You will see these functions: cv2.getPerspectiveTransform
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('small_person.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

res2 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 2)
plt.imshow(res)
plt.title('Enlarged 2x CUBIC')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 3)
plt.imshow(res2)
plt.title('Enlarged 2x LINEAR')
plt.xticks([])
plt.yticks([])

plt.show()
