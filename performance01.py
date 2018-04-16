"""

Performance Measurement and Improvement Techniques
Goal
In image processing, since you are dealing with large number of operations per second, it is mandatory that your code is not only providing the correct solution, but also in the fastest manner. So in this chapter, you will learn

To measure the performance of your code.
Some tips to improve the performance of your code.
You will see these functions : cv2.getTickCount, cv2.getTickFrequency etc.
Apart from OpenCV, Python also provides a module time which is helpful in measuring the time of execution. Another module profile helps to get detailed report on the code, like how much time each function in the code took, how many times the function was called etc. But, if you are using IPython, all these features are integrated in an user-friendly manner. We will see some important ones, and for more details, check links in Additional Resouces section.
"""
import time
from datetime import datetime

import numpy as np
import cv2

img1 = cv2.imread('building.jpg')

start_time = datetime.now()
# start_time = time.time()

for i in range(5, 49, 2):
    img1 = cv2.medianBlur(img1, i)

duration = datetime.now() - start_time
# duration = time.time() - start_time

print(duration)
