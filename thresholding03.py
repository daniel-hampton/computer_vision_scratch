"""
Image Thresholding
Goal
In this tutorial, you will learn Simple thresholding, Adaptive thresholding, Otsu’s thresholding etc.
You will learn these functions : cv2.threshold, cv2.adaptiveThreshold etc.
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('noise.png', cv2.IMREAD_GRAYSCALE)

# global thresholding
ret1, th1, = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Otsu's thresholding
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]

titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v = 127)',
          'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
          'Gaussian Filtered Image', 'Histogram', "Otsu's Thresholding"]

for i in range(3):
    plt.subplot(3, 3, i*3 + 1)
    plt.imshow(images[i*3], 'gray')
    plt.title(titles[i*3])
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 3, i*3 + 2)
    plt.hist(images[i*3].ravel(), 256)
    plt.title(titles[i*3 + 1])
    plt.xticks([x for x in range(0, 255, 20)])
    plt.yticks([])

    plt.subplot(3, 3, i*3 + 3)
    plt.imshow(images[i*3 + 2], 'gray')
    plt.title(titles[i*3 + 2])
    plt.xticks([])
    plt.yticks([])

plt.show()
