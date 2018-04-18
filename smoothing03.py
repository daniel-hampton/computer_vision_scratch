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

img = cv2.imread('small_person.jpg')
img = img[:, :, ::-1]  # convert colors from BGR to RGB using numpy indexing (faster than cv2.cvtColor() method)

# m = (50, 50, 50)
# s = (50, 50, 50)
# cv2.randn(img, m, s)  # just adds noise (and in this case replaces original image..)

blur1 = cv2.blur(img, (5, 5))
blur2 = cv2.GaussianBlur(img, (5, 5), 0)
blur3 = cv2.medianBlur(img, 3)
blur4 = cv2.bilateralFilter(img, 9, 75, 75)

plt.subplot(231)
plt.imshow(img)
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(232)
plt.imshow(blur1)
plt.title('Blur')
plt.xticks([])
plt.yticks([])

plt.subplot(233)
plt.imshow(blur2)
plt.title('Gaussian Blur')
plt.xticks([])
plt.yticks([])

plt.subplot(234)
plt.imshow(blur3)
plt.title('Median Blur')
plt.xticks([])
plt.yticks([])

plt.subplot(235)
plt.imshow(blur4)
plt.title('Bilateral Filter')
plt.xticks([])
plt.yticks([])


plt.show()
