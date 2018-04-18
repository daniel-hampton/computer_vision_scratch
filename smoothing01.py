"""
Smoothing Images
Goals
Learn to:
Blur imagess with various low pass filters
Apply custom-made filters to images (2D convolution)
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('opencv-logo.png')

kernel = np.ones((5, 5), np.float32) / 25

dst = cv2.filter2D(img, -1, kernel)

plt.subplot(121)
plt.imshow(img)
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(dst)
plt.title('Averaging')
plt.xticks([])
plt.yticks([])

plt.show()
