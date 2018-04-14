"""
First couple exercises/examples in the opencv Python tutorials.
Goals
Here, you will learn how to read an image, how to display it and how to save it back
You will learn these functions : cv2.imread(), cv2.imshow() , cv2.imwrite()
Optionally, you will learn how to display images with Matplotlib
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read in the image.
img = cv2.imread('moth.jpg')
# Reverse the color scheme from BGR (opencv's read) and RGB (matplotlib's mode)
# so we can display the image correctly using matplotlib.
img2 = img[:, :, ::-1]

# Use Matplotlib to display image.
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.set_title('BGR')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.imshow(img)

ax2.set_title('RGB')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.imshow(img2)

plt.show()

# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.xticks([])
# plt.yticks([])
# plt.title('Image Result')
# plt.show()
