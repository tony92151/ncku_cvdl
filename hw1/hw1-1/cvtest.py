
import sys
import cv2
import os
import time
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtWidgets, uic
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import time

path = os.getcwd()


imgpath = path + os.sep + "CameraCalibration"
imglist = os.listdir(imgpath)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*11,3), np.float32)
objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
objpoints = []
imgpoints = []
print(imgpath + os.sep + imglist[0])
img = cv2.imread(imgpath + os.sep + imglist[0])
print(img.shape)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(gray.shape)
# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (11,8),None)
print(ret)
# If found, add object points, image points (after refining them)
if ret == True:
    print("trans")
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (11,8), corners2,ret)
    img = cv2.resize(img, (600, 600)) 
    cv2.imshow('img',img)
    cv2.waitKey(0)
    #cv2.imwrite(imgpath+"/new.jpg",img)
    #print("save to :" ,imgpath+"/new.jpg")
    #time.sleep(5)
    #cv2.destroyAllWindows()
    # plt.imshow(img)
    # plt.show()