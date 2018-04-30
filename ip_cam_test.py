
import cv2
import numpy as np
from matplotlib import pyplot as plt

# cap = cv2.VideoCapture(1)  
ip = "192.168.1.235"
admin = 'admin'
pwd = 'admin'
# cap = cv2.VideoCapture('http://{0}/videostream.cgi?user={1}&pwd={2}&resolution=32'.format(ip, admin, pwd))
cap = cv2.VideoCapture('rtsp://{0}/11'.format(ip))

cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)

while cap.isOpened():
    
    ret, frame = cap.read()

    if ret:
        cv2.imshow('Video', frame)

        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()