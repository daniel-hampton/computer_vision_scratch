"""
Testing threshold functions on still image from client
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read in color
img = cv2.imread('vlcsnap.png')

# Get height, width and area of whole image.
height, width = img.shape[:2]
total_area = width * height

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define region of interest, bottom half of image
roi = img_gray[height//2:, :]

# Create threshold binary image
ret, thresh = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)

ret_image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# number of contours
print('Number of contours: {}'.format(len(contours)))
print('Total area: {}'.format(total_area))

# Create list of contour areas sorted descending
areas = [cv2.contourArea(x) for x in contours]
print("type contours: {}".format(type(contours)))
con_areas = list(zip(contours, areas))  # list of tuples of (contour, contour_area)
con_areas = sorted(con_areas, key=lambda x: x[1], reverse=True)

# for cont, area in con_areas[:5]:
    # print(area)


# Draw contours
# todo: write in height - roi_height for draw contours region
cv2.drawContours(img[height//2:, :], [con_areas[0][0]], -1, (0, 255, 0), 2)

text = 'Area: {:02.2f}%'.format(con_areas[0][1] / total_area * 100)

font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, text, (int(0.05 * width), int(0.95 * height)), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

# covert BGR to RGB colors
img = img[:, :, ::-1]

# todo: how do contours handle holes in the object bodies?

# plt.hist(img.ravel(), 256)
# plt.title('Histogram')
# plt.yticks([])

plt.subplot(121)
plt.imshow(thresh, 'gray')
plt.title('Binary Thresh {}'.format(127))
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(img)
plt.title('Contours')
plt.xticks([])
plt.yticks([])

# plt.subplot(133)
# plt.imshow(big_contour)
# plt.title('Largest Contour')
# plt.xticks([])
# plt.yticks([])

plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()
