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
img = cv2.imread('moth.jpg', 0)

# Use Matplotlib to display image.
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([])
plt.yticks([])
plt.title('Image Result')
plt.show()
