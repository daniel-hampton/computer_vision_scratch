"""
Goal
In this chapter, we will learn to:

Find Image gradients, edges etc
We will see following functions : cv2.Sobel(), cv2.Scharr(), cv2.Laplacian() etc
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = cv2.imread('white_square.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('sudoku-original.jpg', cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x direction gradient
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y direction gradient

images = [img, laplacian, sobelx, sobely]
titles = ['Original', 'Laplacian', 'Sobel X', 'Sobel Y']

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
