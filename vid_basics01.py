"""

Getting Started with Videos
Goal
Learn to read video, display video and save video.
Learn to capture from Camera and display it.
You will learn these functions : cv2.VideoCapture(), cv2.VideoWriter()
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations  on this frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Set resolution to 1280x720
    cap.set(3, 1280)
    cap.set(4, 720)
    # Print resolution
    print('{0:d}x{1:d}'.format(int(cap.get(3)), int(cap.get(4))))

    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
