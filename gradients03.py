"""
Goal
In this chapter, we will learn to:

Find Image gradients, edges etc
We will see following functions : cv2.Sobel(), cv2.Scharr(), cv2.Laplacian() etc
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(1)

while cap.isOpened():

    ret, frame = cap.read()

    if ret:
        # frame = cv2.medianBlur(frame, 3)  # Added this myself but does nto seem to help reduce noise in output.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)  # x direction gradient
        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)  # y direction gradient

        cv2.imshow('Original', frame)
        cv2.imshow('Laplacian', laplacian)
        cv2.imshow('Sobel X', sobelx)
        cv2.imshow('Sobel Y', sobely)

        k = cv2.waitKey(25) & 0xFF
        if k == 27:  # break on ESC key
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

