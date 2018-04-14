"""
First couple exercises/examples in the opencv Python tutorials.
Goals
Here, you will learn how to read an image, how to display it and how to save it back
You will learn these functions : cv2.imread(), cv2.imshow() , cv2.imwrite()
Optionally, you will learn how to display images with Matplotlib
"""

import cv2

# Read in the image.
img = cv2.imread('moth.jpg', 0)

# Display image in a window that closes when any key is pressed.
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print dimensions of image.
print(img)
print(img.shape)
print(type(img))
height, width = img.shape[:2]
print('Image is {}x{} pixels'.format(width, height))

# Save the image
cv2.imwrite('output/moth-gray.png', img)
