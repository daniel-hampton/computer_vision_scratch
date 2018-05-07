"""
Testing threshold functions on video from client
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


cap = cv2.VideoCapture(1)
ip = "192.168.1.235"
admin = 'admin'
pwd = 'admin'
# cap = cv2.VideoCapture('http://{0}/videostream.cgi?user={1}&pwd={2}&resolution=32'.format(ip, admin, pwd))
# cap = cv2.VideoCapture('rtsp://{0}/11'.format(ip))

# ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
fps = cap.get(cv2.CAP_PROP_FPS)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
vid_size = (1920, 1080)
out = cv2.VideoWriter('output/webcam_contour_output2.avi', fourcc, 20.0, vid_size)

cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)

while cap.isOpened():

    ret, img = cap.read()

    if ret:
        # Get height, width and area of whole image.
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # img = cv2.resize(img, (1090, 1920), interpolation=cv2.INTER_AREA)

        # Get height, width and area of whole image.
        height, width = img.shape[:2]

        # initialize mask as all zeroes (black)
        mask = np.zeros((height, width), np.uint8)  # mask must be np.unint8 not the default np.float65

        # Define region of interest polygon
        # roi of polygon points
        offset = 0.20  # fractional offset from edge of  screen
        myROI = np.array([[width * offset, height * offset],
                         [(1-offset) * width, height * offset],
                         [(1-offset) * width, (1-offset) * height],
                         [width * offset, (1-offset) * height]], np.int32)
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

        ret_image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Area of ROI
        print('ROI total area: {}'.format(total_area))

        try:
            # Get the level-0 contours indices. Those with Parent hierarchy values of -1
            h0 = np.where(hierarchy[0][:, 3] == -1)

            # Create a list of contours from the h0 contour indices.
            contours_white = [contours[i] for i in h0[0]]

            # Get the level-1 contours indices. The children of the level-0 contours.
            h1 = np.where(hierarchy[0][:, 3] != -1)

            # Create a list of contours from the h1 contour indices.
            contours_black = [contours[i] for i in h1[0]]
            areas_black = [cv2.contourArea(x) for x in contours_black]

            # Create list of contour areas sorted descending
            areas = [cv2.contourArea(x) for x in contours_white]
            con_areas = list(zip(contours_white, areas))  # list of tuples of (contour, contour_area)
            con_areas = sorted(con_areas, key=lambda x: x[1], reverse=True)

            # Calc total area of white contours
            white_contour_area_total = 0
            for cont, area in con_areas:
                white_contour_area_total += area

            # Calc total area of black (holes) contours
            holes_area_total = 0
            for area in areas_black:
                holes_area_total += area

            contour_area_total = white_contour_area_total - holes_area_total

            print('Total Contour Area: {:.0f}'.format(white_contour_area_total))
            print('Number of contours total: {}'.format(len(contours_white)))

        except TypeError as err:
            print('No contours found.')
            print(err)

            # When there are no contours set the area to zero
            contour_area_total = 0

        print('{} x {}'.format(w, h))

        # Draw contours
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        # Draw ROI polygon
        cv2.polylines(img, [myROI], True, (0, 0, 255), 2, cv2.LINE_AA)  # must be list of polygon points

        # Resize video for display on monitor
        # [WARNING] DON'T DETERMINE AREA VALUES AFTER THIS POINT. NUMBER OF PIXELS HAS CHANGED.
        # img = cv2.resize(img, (540, 960), interpolation=cv2.INTER_AREA)
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
        # out.write(img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
