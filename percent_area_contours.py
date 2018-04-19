
import cv2
import numpy as np

from pprint import pprint

img = cv2.imread('multi-contour.png')
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imggray, 127, 255, cv2.THRESH_BINARY)

image, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours: {}".format(len(contours)))

# Draw Contours
img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Create list of centroids and area of contours
contour_centroids = []  # items with format (x, y, area)
total_area = 0
for cont in contours:
    M = cv2.moments(cont)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    area = cv2.contourArea(cont)
    total_area += area

    contour_centroids.append((cx, cy, area))

# Define font for text
font = cv2.FONT_HERSHEY_SIMPLEX

# Write text to each contour
for i, cont in enumerate(contour_centroids):
    text = '#{:02d}\n'.format(i)
    text += 'Area: {:.0f} px^2\n'.format(cont[2])
    text += '% of total: {:02.0f}%'.format(cont[2] / total_area * 100)

    # putText method does nto handle \n characters. Manually workaround
    for num, line in enumerate(text.split('\n')):
        y0 = cont[1]  # starting point for first line of text, y coordinate
        dy = 14  # change in y for starting point of each line of text
        y = y0 + num * dy  # y coordinate for each line of text
        img = cv2.putText(img, line, (cont[0], y), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('Contour Areas', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
