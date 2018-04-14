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

# Define the codec and create VideoWriter object.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/output.avi', fourcc, 20.0, (1280, 720))

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret is True:

        # Resize frames
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Rotate the frame
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Write the rotated frame.
        out.write(frame)

        # Print resolution
        print('{0:d}x{1:d}'.format(int(cap.get(3)), int(cap.get(4))))

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# When everything is done, release all.
cap.release()
out.release()
cv2.destroyAllWindows()
