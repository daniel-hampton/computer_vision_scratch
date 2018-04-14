"""

Getting Started with Videos
Goal
Learn to read video, display video and save video.
Learn to capture from Camera and display it.
You will learn these functions : cv2.VideoCapture(), cv2.VideoWriter()
"""

import numpy as np
import cv2

cap = cv2.VideoCapture('device_video.mp4')

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret is True:

        # Our operations  on this frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize frames
        gray = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Print resolution
        print('{0:d}x{1:d}'.format(int(cap.get(3)), int(cap.get(4))))

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
