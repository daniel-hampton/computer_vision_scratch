"""
Canny Edge Detection
Goal
In this chapter, we will learn about

Concept of Canny edge detection
OpenCV functions for that : cv2.Canny()
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(1)

while cap.isOpened():

    ret, frame = cap.read()

    if ret:
        edges = cv2.Canny(frame, 100, 200)
        edges2 = cv2.Canny(frame, 100, 200, L2gradient=True)

        cv2.imshow('Original', frame)
        cv2.imshow('Canny Edges', edges)
        cv2.imshow('Canny Edges L2', edges2)

        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()