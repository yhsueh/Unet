import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob


file_name = '210312004.png'
img = cv2.imread(file_name,0)
kernel = np.ones((5,5), np.uint8)
img = cv2.erode(img, kernel, iterations=4)
img = cv2.dilate(img, kernel, iterations=4)
img = cv2.GaussianBlur(img, (9,9), sigmaX=1.5)

circles = cv2.HoughCircles(img,
                           cv2.HOUGH_GRADIENT,
                           dp=2, 
                           minRadius=80,
                           maxRadius=150,
                           minDist=img.shape[0]/16, 
                           param1=10,
                           param2=105)
for (x, y, r) in circles[0,:]:
	print(x,y,r)
	cv2.circle(img, (x, y), int(r), (255, 255, 255), 10)
print(circles.shape)
plt.imsave(file_name, img, cmap='gray')
