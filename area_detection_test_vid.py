"""
Testing threshold functions on video from client
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


cap = cv2.VideoCapture('device_video.mp4')

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
vid_size = (540, 960)
out = cv2.VideoWriter('output/contour_output.avi', fourcc, 30.0, vid_size)

while cap.isOpened():

    ret, img = cap.read()

    if ret:
        # Get height, width and area of whole image.
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # img = cv2.resize(img, (1090, 1920), interpolation=cv2.INTER_AREA)

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
        retval, thresh = cv2.threshold(gray_roi, 110, 255, cv2.THRESH_BINARY)

        # Remove noise through morphological transforms, opening and closing
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Use Adaptive Thresholding
        # thresh = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        ret_image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # number of contours
        print('Number of contours: {}'.format(len(contours)))

        # Create list of contour areas sorted descending
        areas = [cv2.contourArea(x) for x in contours]
        con_areas = list(zip(contours, areas))  # list of tuples of (contour, contour_area)
        con_areas = sorted(con_areas, key=lambda x: x[1], reverse=True)

        # Calc total area of contours
        contour_area_total = 0
        for cont, area in con_areas:
            contour_area_total += area

        print('Total area: {}'.format(total_area))
        print('Total Contour Area: {:.0f}'.format(contour_area_total))

        # Draw contours
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        # Draw ROI polygon
        # cv2.polylines(img, [myROI], True, (0, 0, 255), 2, cv2.LINE_AA)  # must be list of polygon points

        # Resize video for display on monitor
        # [WARNING] DON'T DETERMINE AREA VALUES AFTER THIS POINT. NUMBER OF PIXELS HAS CHANGED.
        img = cv2.resize(img, (540, 960), interpolation=cv2.INTER_AREA)
        # Recalc height, width for writing text
        height, width = img.shape[:2]

        # Write text to image
        text = 'Dry Area: {:02.0f}%'.format(contour_area_total / total_area * 100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Draw black background for text
        cv2.rectangle(img, (0, int(0.97 * height)), (width - 1, int(0.96 * height - 65)), (0, 0, 0), -1, cv2.LINE_AA)
        img = cv2.putText(img, text, (int(0.05 * width), int(0.95 * height)), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

        # covert BGR to RGB colors
        # img = img[:, :, ::-1]

        # Display Image
        cv2.imshow('Video', img)

        # Write video
        out.write(img)

        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
