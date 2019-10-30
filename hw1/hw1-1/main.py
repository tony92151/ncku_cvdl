import sys
import cv2
import os
import time
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtWidgets, uic
import numpy as np


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

        self.selected_img = "All"

    def onBindingUI(self):
        self.bt_find_corners.clicked.connect(self.on_bt_find_corners_click)
        self.bt_intrinsic.clicked.connect(self.on_bt_find_intrinsic_click)
        self.comboBox.addItems(imgSortedList)
        self.comboBox.activated[str].connect(self.comboBox_onChanged)  

    def on_bt_find_corners_click(self):
        print(self.selected_img)

    def on_bt_find_intrinsic_click(self):
        pass

    def comboBox_onChanged(self,text):
        self.selected_img = text
        
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
