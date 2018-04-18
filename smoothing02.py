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

# blur = cv2.blur(img, (5, 5))
blur = cv2.GaussianBlur(img, (5, 5), 0)

plt.subplot(121)
plt.imshow(img)
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(blur)
plt.title('Blurred')
plt.xticks([])
plt.yticks([])

plt.show()
