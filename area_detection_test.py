"""
Testing threshold functions on still image from client
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# todo: figure out contour heirarchy to remove interior black contour areas.

MODE = 'white'  # 'black' will detect black areas instead of light/white areas (inverts threshold within mask)

# Read in color
img = cv2.imread('vlcsnap.png')

# Get height, width and area of whole image.
height, width = img.shape[:2]

# initialize mask as all zeroes (black)
mask = np.zeros((height, width), np.uint8)  # mask must be np.unint8 not the default np.float65

# Define region of interest polygon
# roi of polygon points
myROI = np.array([[0, 985], [692, 786], [1025, 822], [1089, 985], [1089, 1724], [0, 1516]], np.int32)
cv2.fillPoly(mask, [myROI], 1)

# Get total area of region of interest
total_area = cv2.contourArea(myROI)

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply the mask to the grayscale image

gray_roi = cv2.bitwise_and(img_gray, img_gray, mask=mask)

# Create threshold binary image
ret, thresh = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)

# Remove noise through morphological transforms, opening and closing
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Can invert threshold within mask to detect black regions
if MODE == 'black':
    thresh = cv2.bitwise_not(thresh, mask=mask)

ret_image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# number of contours
print('Number of contours: {}'.format(len(contours)))


# Create list of contour areas sorted descending
areas = [cv2.contourArea(x) for x in contours]
con_areas = list(zip(contours, areas))  # list of tuples of (contour, contour_area)
con_areas = sorted(con_areas, key=lambda x: x[1], reverse=True)

# total_area = 0
# for cont, area in con_areas:
#     total_area += area

print('Total area: {}'.format(total_area))


# Draw contours
# roi_h, roi_w = roi.shape[:2]
cv2.drawContours(img, [con_areas[0][0]], -1, (0, 255, 0), 2)
# cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Display text of contour area percentage of total area
text = 'Area: {:02.2f}%'.format(con_areas[0][1] / total_area * 100)

font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, text, (int(0.05 * width), int(0.95 * height)), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

# covert BGR to RGB colors
img = img[:, :, ::-1]

# plt.hist(img.ravel(), 256)
# plt.title('Histogram')
# plt.yticks([])

plt.subplot(131)
plt.imshow(thresh, 'gray')
plt.title('Binary Thresh {}'.format(127))
plt.xticks([])
plt.yticks([])

plt.subplot(132)
plt.imshow(img)
plt.title('Contours')
plt.xticks([])
plt.yticks([])

plt.subplot(133)
plt.imshow(gray_roi, 'gray')
plt.title('Gray ROI')
plt.xticks([])
plt.yticks([])

plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()
