"""
First test script to extract number on metal plate from image
"""

from PIL import Image
import pytesseract
import argparse
import cv2
import os

import numpy as np


def nothing():
    """Blank function for trackbars"""
    pass

# construct argument parser from command line arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", '--image', required=True, help='path to input image to be OCR\'d')
args = vars(ap.parse_args())

# Load the example image, convert to grayscale
image = cv2.imread(args['image'])

if image is None:
    raise TypeError("Image was not loaded correctly")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create small rectangular ROI in the image.
x1 = 747
y1 = 414
x2 = 1175
y2 = 690

roi = np.array([[x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]], np.int32)

# crop the image
cropped_image = gray[y1:y2, x1:x2]

# Define image windows
cv2.namedWindow('Original', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Gray', cv2.WINDOW_KEEPRATIO)

# Display the original image
cv2.imshow('Original', cropped_image)

# create trackbars to adjust canny edge detection parameters
cv2.createTrackbar('threshold1', 'Gray', 60, 300, nothing)
cv2.createTrackbar('threshold2', 'Gray', 200, 400, nothing)

# Loop for adjusting trackbars
while True:
    thresh1 = cv2.getTrackbarPos('threshold1', 'Gray')
    thresh2 = cv2.getTrackbarPos('threshold2', 'Gray')

    edges = cv2.Canny(cropped_image, thresh1, thresh2, apertureSize=3, L2gradient=True)

    cv2.imshow('Gray', edges)

    # OCR the image to pick up digits and print to console
    text = pytesseract.image_to_string(Image.fromarray(edges))
    print('Numbers: {}'.format(text))

    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break
    #
    # if KeyboardInterrupt:
    #     break

cv2.destroyAllWindows()
