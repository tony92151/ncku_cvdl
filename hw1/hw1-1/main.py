import sys
import cv2
import os
import time
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtWidgets, uic
import numpy as np

import matplotlib

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

from matplotlib import pyplot as plt

import time
import threading


path = os.getcwd()
qtCreatorFile = path + os.sep + "mainwindow.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile) 

imgpath = path + os.sep + "CameraCalibration"
imglist = os.listdir(imgpath)
imglistd = [i[:-4] for i in imglist]
imglistd = np.sort(np.array(imglistd).astype('int'))
imgSortedList = ["All"]+[format(i)+".bmp" for i in imglistd]

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

        self.imgSortedList = imgSortedList[1:]
        self.selected_img = "All"

        self.objpoints = []
        self.imgpoints = []

    def onBindingUI(self):
        self.bt_find_corners.clicked.connect(self.on_bt_find_corners_click)
        #self.bt_intrinsic.clicked.connect(self.on_bt_find_intrinsic_click)
        self.comboBox.addItems(imgSortedList)
        self.comboBox.activated[str].connect(self.comboBox_onChanged) 
        self.bt_cancel.clicked.connect(self.on_bt_cancel_click)

    def on_bt_find_corners_click(self):
        find_corner_img = self.find_imgs_corner()
        #plt.figure(figsize=(10,10))
        for i in range(len(find_corner_img)):
            # plt.subplot(4,4,i)
            # plt.imshow(find_corner_img[i])
            t = threading.Thread(target = self.diaplay_imgs(find_corner_img[i],i))
            t.start()
        # plt.savefig(path + "/result.jpg")
        # self.diaplay_imgs(path + "/result.jpg")
        



    def on_bt_cancel_click(self):
        sys.exit(app.exec_())

    def comboBox_onChanged(self,text):
        self.selected_img = text

    def diaplay_imgs(self, imgs, idx):
        leng = len(imgs)
        #plt.figure(figsize=(10,10))
        #for i in range(leng):
        cv2.imshow('img'+format(idx+1),imgs)
        cv2.waitKey(0)
        # time.sleep(5)
        cv2.destroyAllWindows()
        #plt.subplot(2,int(leng/2),i)
        #plt.imshow(imgs)

    def find_imgs_corner(self):

        find_corner_img = []

        for i in range(len(self.imgSortedList)):
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.001)

            objp = np.zeros((8*11,3), np.float32)
            objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

            print(imgpath + os.sep + self.imgSortedList[i])
            img = cv2.imread(imgpath + os.sep + self.imgSortedList[i])
            print(img.shape)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            print(gray.shape)

            ret, corners = cv2.findChessboardCorners(gray, (11,8),None)
            print(ret)

            if ret == True:
                self.objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,8),(-1,-1),criteria)
                self.imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (11,8), corners2,ret)
                img = cv2.resize(img, (600, 600)) 
                find_corner_img.append(img)
            else:
                img = cv2.resize(img, (600, 600)) 
                find_corner_img.append(img)
        return find_corner_img
        
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
