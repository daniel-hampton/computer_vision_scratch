"""
Mouse as a Paint-Brush
Goal
Learn to handle mouse events in OpenCV
You will learn these functions : cv2.setMouseCallback()
"""

import numpy as np
import cv2
from pprint import pprint

# Print list of available mouse events in cv2
events = [i for i in dir(cv2) if 'EVENT' in i]
pprint(events)


# mouse callback function
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('image', draw_circle)

while 1:
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
        
cv2.destroyAllWindows()

