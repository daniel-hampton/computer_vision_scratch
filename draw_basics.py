"""
Drawing Functions in OpenCV
Goal
Learn to draw different geometric shapes with OpenCV
You will learn these functions : cv2.line(), cv2.circle() , cv2.rectangle(), cv2.ellipse(), cv2.putText() etc.
"""

import numpy as np
import cv2

# Create a black banner
img = np.zeros((512, 512, 3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

# Draw a rectangle
img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

# Draw a circle
img = cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)

# Draw an ellipse
img = cv2.ellipse(img, (256, 256), (50, 100), 0, 0, 180, 255, -1)

# Draw a polygon
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts.reshape((-1, 1, 2))
print(pts)
img = cv2.polylines(img, [pts], True, (0, 255, 255))

# Add Text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image', img)

# Exit if ESC is pressed. Save and exit if S is pressed.
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s') & 0xFF:
    cv2.imwrite('output/draw_basic.png', img)
    cv2.destroyAllWindows()
