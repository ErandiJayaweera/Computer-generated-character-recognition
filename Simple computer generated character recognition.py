#pip install opencv-python
#pip install numpy

import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt

im = cv2.imread("Database.jpg",1)
im2 = im.copy()
gray_im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,bw_im = cv2.threshold(gray_im,127,255,cv2.THRESH_BINARY_INV)
bw_im2 = bw_im.copy()

r,c = bw_im.shape
mask = np.zeros([r+2,c+2],dtype = 'uint8')
cv2.floodFill(bw_im2,mask[0,0],255)
bw_im3 = cv2.bitwise_not(bw_im2)
bw_im4 = cv2.bitwise_or(bw_im,bw_im3)



contours,hi = cv2.findContours(bw_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#36 contours getting with inner and outer boundaries 

for c in contours:
    x,y,w,h = cv2.boundingRect()
    cv2.rectangle(im2,(x,y),(x+w,y+h),(255,0,0),1)
    cv2.imshow("objects",im2)
    cv2.waitKey(500)
# cv2.imshow("BW",bw_im)
# cv2.imshow("fl",bw_im2)


k = cv2.waitKey(0) & 0xFF
if (k==27):
    cv2.destroyAllWindows()
