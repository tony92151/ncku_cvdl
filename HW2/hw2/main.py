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
        self.diaplay_imgs(disparity)

    def on_bt_ncc(self):
        pass

    def on_bt_ketpoints(self):
        pass

    def on_bt_matched_keypoints(self):
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
