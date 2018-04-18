"""
Image Thresholding
Goal
In this tutorial, you will learn Simple thresholding, Adaptive thresholding, Otsu’s thresholding etc.
You will learn these functions : cv2.threshold, cv2.adaptiveThreshold etc.
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('sudoku-original.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.medianBlur(img, 5)

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

titles = ['Original Image',
          'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding',
          'Adaptive Gaussian Thresholding',
          ]

images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
