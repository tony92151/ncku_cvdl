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
from numpy.linalg import inv

import time
import threading

print(cv2.__version__)


path = os.getcwd()
qtCreatorFile = path + os.sep + "mainwindow.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile) 


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()


    def onBindingUI(self):
        # 1.1 bt_find_diaparity
        self.bt_find_diaparity.clicked.connect(self.on_bt_find_diaparity)
        # 2.1 bt_ncc
        self.bt_ncc.clicked.connect(self.on_bt_ncc)
        # 3.1 bt_ketpoints
        self.bt_ketpoints.clicked.connect(self.on_bt_ketpoints)
        # 3.2 bt_matched_keypoints
        self.bt_matched_keypoints.clicked.connect(self.on_bt_matched_keypoints)
        # OK & cancel
        self.bt_cancel.clicked.connect(self.on_bt_cancel)

    def on_bt_find_diaparity(self):
        input_l = cv2.imread("imL.png", cv2.IMREAD_GRAYSCALE)
        input_r = cv2.imread("imR.png", cv2.IMREAD_GRAYSCALE)
        stereo = cv2.StereoBM_create(16*6, 9)
        disparity = stereo.compute(input_l, input_r)
        #disparity = disparity*1
        #print(input_l)
        #self.diaplay_imgs(disparity)
        print(disparity.max(),disparity.min())
        #plt.savefig("ma.png",cmap="gray")
        plt.imshow(disparity,cmap="gray")
        plt.show()

    def on_bt_ncc(self):
        img = cv2.imread("ncc_img.jpg")
        img_g = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
        template = cv2.imread("ncc_template.jpg", cv2.IMREAD_GRAYSCALE)
        #print(template.shape[::-1])
        h,w = template.shape

        res = cv2.matchTemplate(img_g,template,cv2.TM_CCOEFF_NORMED)

        #print(res)
        threshold = 0.95
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,0), 2)
            #print("Draw")

        img = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB)
        #######################################
        plt.subplot(121)
        plt.imshow(res,cmap = 'gray')
        plt.title('Template matching feature')
        #######################################
        plt.subplot(122)
        plt.imshow(img)
        plt.title('Detected Point')
        #######################################
        plt.show()
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #print(min_val)
        #pass

    def on_bt_ketpoints(self):
        img1 = cv2.imread("Aerial1.jpg")
        img2 = cv2.imread("Aerial2.jpg")

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Initiate SIFT detector
        print("Init SIFT")
        sift = cv2.xfeatures2d.SIFT_create()

        kp1 = sift.detect(img1,None)
        kp1.sort(key=lambda x: -x.size)

        kp2 = sift.detect(img2,None)
        kp2 = sorted(kp2,key=lambda x: -x.size)

        

        img1_display = cv2.drawKeypoints(img1_gray.copy(),
                                        kp1[:6],color=(0,255,0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
                                        outImage = img1_gray)
        img2_display = cv2.drawKeypoints(img2_gray.copy(),
                                        kp2[:6],color=(0,255,0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
                                        outImage = img2_gray)
        cv2.imwrite("FeatureAerial1.jpg",img1_display)
        cv2.imwrite("FeatureAerial2.jpg",img2_display)
        #print("Draw key point")
        #######################################
        plt.subplot(121)
        plt.imshow(img1_display,cmap = 'gray')
        plt.title('FeatureAerial1.jpg')
        #######################################
        plt.subplot(122)
        plt.imshow(img2_display,cmap = 'gray')
        plt.title('FeatureAerial2.jpg')
        #######################################
        plt.show()
        #pass

    def on_bt_matched_keypoints(self):
        img1 = cv2.imread("Aerial1.jpg")
        img2 = cv2.imread("Aerial2.jpg")
        matchImg = np.zeros_like(img1)

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Initiate SIFT detector
        print("Init SIFT")
        sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)

        kpdes1 = sorted(list(zip(kp1, des1)),key = lambda x: -x[0].size)
        kpdes2 = sorted(list(zip(kp2, des2)),key = lambda x: -x[0].size)
        kp1 = [i[0] for i in kpdes1][:6]
        des1 = [i[1] for i in kpdes1][:6]
        kp2 = [i[0] for i in kpdes2][:6]
        des2 = [i[1] for i in kpdes2][:6]
        #print(kp1)

        img1_display = cv2.drawKeypoints(img1_gray.copy(),
                                        kp1,color=(0,255,0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
                                        outImage = img1_gray)
        img2_display = cv2.drawKeypoints(img2_gray.copy(),
                                        kp2,color=(0,255,0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
                                        outImage = img2_gray)

        bf = cv2.BFMatcher_create(normType=cv2.NORM_L2, crossCheck=True)
        matches = bf.match(np.asarray(des1,np.float32), np.asarray(des2,np.float32)) 
        # bf = cv2.BFMatcher()
        # matches = bf.knnMatch(des1, des2, k=2)
        #matches = sorted(matches, key=lambda x: x.distance)
        matchImg = cv2.drawMatches(img1_gray, kp1, img2_gray, kp2, matches, matchImg, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        plt.imshow(matchImg)
        plt.show()
        # plt.subplot(121)
        # plt.imshow(img1_display,cmap = 'gray')
        # plt.title('FeatureAerial1.jpg')
        # #######################################
        # plt.subplot(122)
        # plt.imshow(img2_display,cmap = 'gray')
        # plt.title('FeatureAerial2.jpg')
        # #######################################
        # plt.show()
        pass

    def on_bt_cancel(self):
        sys.exit(app.exec_())

    def diaplay_imgs(self, imgs):
        cv2.imshow('img',imgs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
