"""
Contour area detection and dry area percentage calculation for filter 04 small platform from GoPro camera.
"""

import os
import cv2
import numpy as np
import pandas as pd

from datetime import datetime
from datetime import timedelta

from matplotlib import pyplot as plt
import matplotlib.dates as mdates


def nothing():
    """ Empty function to supply to opencv trackbar object"""

    pass


ip = "192.168.1.235"
# cap = cv2.VideoCapture('rtsp://{0}/11'.format(ip))  # ip camera
# cap = cv2.VideoCapture(0)  # webcam
input_path = '/home/dh084/Videos/'
# input_path = '/home/dh084/computer_vision_scratch/bartow_video/video_from_dan/'
filename = 'filter04_left_side04.m4v'
cap = cv2.VideoCapture(os.path.join(input_path, filename))  # video file

cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Mask', cv2.WINDOW_KEEPRATIO)

# Set trackbar mode
trackbar_mode = True

# Set default target threshold value
default_thresh = 85
default_high_thresh = 255

# Define lists of data
dry_data = []
time_data = []

# Define current frame time
current_time = datetime.now()

if trackbar_mode is True:
    cv2.createTrackbar('Threshold', 'Video', default_thresh, 255, nothing)
    cv2.createTrackbar('HighThreshold', 'Video', default_high_thresh, 255, nothing)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
vid_size = (1920, 1080)
recording_fps = 30.0
out = cv2.VideoWriter('filter04_left_side04_output.avi', fourcc, recording_fps, vid_size)
# out = cv2.VideoWriter('output/webcam_output.avi', fourcc, recording_fps, vid_size)

image_file = 'output/filter04_right_side04_graph.png'


while cap.isOpened():
    start_time = datetime.now()

    ret, img = cap.read()

    if ret:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)

        # seconds between each frame
        delta_t = 1 / int(fps)

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
        # Box ROI
        myROI = np.array([[width * offset, height * offset],
                          [(1 - offset) * width, height * offset],
                          [(1 - offset) * width, (1 - offset) * height],
                          [width * offset, (1 - offset) * height]], np.int32)

        # Define ROI polygon bounding box
        # myROI = np.array([[727, 711],
        #                   [713, 543],
        #                   [1553, 89],
        #                   [1657, 1001]], np.int32)
        cv2.fillPoly(mask, [myROI], 1)

        # Get total area of region of interest
        total_area = cv2.contourArea(myROI)

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # apply the mask to the grayscale image
        gray_roi = cv2.bitwise_and(img_gray, img_gray, mask=mask)

        # Create threshold binary image
        if trackbar_mode is True:
            thresh_value = cv2.getTrackbarPos('Threshold', 'Video')
            high_thresh = cv2.getTrackbarPos('HighThreshold', 'Video')
        else:
            thresh_value = default_thresh
            high_thresh = default_high_thresh

        # retval, thresh = cv2.threshold(gray_roi, thresh_value, 255, cv2.THRESH_BINARY)
        thresh = cv2.inRange(gray_roi, thresh_value, high_thresh)

        # Remove noise through morphological transforms, opening and closing
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        ret_image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Area of ROI
        print('ROI total area: {}'.format(total_area))

        try:
            # Get the level-0 contours indices. Those with Parent hierarchy values of -1
            # hierarchy format: [NEXT, PREVIOUS, FIRST_CHILD, PARENT]
            # Value of -1 for none. (i.e. -1 for NEXT if there are no next
            # contours in that level of the hierarchy
            h0 = np.where(hierarchy[0][:, 3] == -1)

            # Create a list of contours from the h0 contour indices.
            contours_white = [contours[i] for i in h0[0]]

            # Get the level_1 contours indices. The children of the level_0 contours.
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

        print('{} x {}'.format(width, height))

        # Draw contours
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        # Draw ROI polygon
        cv2.polylines(img, [myROI], True, (0, 0, 255), 2, cv2.LINE_AA)  # must be list of polygon points

        # Resize video for display on monitor
        # [WARNING] DON'T DETERMINE AREA VALUES AFTER THIS POINT. NUMBER OF PIXELS HAS CHANGED.
        # img = cv2.resize(img, (540, 960), interpolation=cv2.INTER_AREA)

        # Rotate image 90 degrees clockwise
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # Recalc height, width for writing text
        height, width = img.shape[:2]

        # Write text to image
        dry_area_percent = contour_area_total / total_area * 100
        text = 'Dry Area: {:02.0f}%'.format(dry_area_percent)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Draw black background for text
        cv2.rectangle(img, (0, int(0.97 * height)), (width - 1, int(0.96 * height - 65)), (0, 0, 0), -1, cv2.LINE_AA)
        img = cv2.putText(img, text, (int(0.05 * width), int(0.95 * height)), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

        # covert BGR to RGB colors
        # img = img[:, :, ::-1]

        # Display Image
        cv2.imshow('Video', img)

        # Display thresh mask
        # thresh = cv2.rotate(thresh, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('Mask', thresh)

        # Code to plot % Dry relative to time
        dry_data.append(dry_area_percent)

        time_data.append(current_time)
        current_time = current_time + timedelta(seconds=delta_t)
        # time_data.append(datetime.now())

        # Write video
        # out.write(img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        loop_time = datetime.now() - start_time
        print(loop_time)
    else:
        break


# Plot % Area vs Time after video.
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

ax1.plot(time_data, dry_data)

ax1.set_xlabel('Time')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate()  # Auto rotates/formats date

ax1.set_ylabel('% Dry')

try:
    ax1.set_title('% Dry Area @ Thresh: {}'.format(thresh_value))
except NameError as err:
    print('thresh_value not defined.')
    print(err)

plt.show()

# fig.savefig(image_file)
cap.release()
cv2.destroyAllWindows()
