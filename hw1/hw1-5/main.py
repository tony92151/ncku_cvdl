import sys
import cv2
import os
import time
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from PyQt5 import QtWidgets, uic
import numpy as np


import matplotlib
matplotlib.use('Qt5Agg')

from sys import platform as sys_pf
# if sys_pf == 'darwin':
#     import matplotlib
#     matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
from numpy.linalg import inv

import time
import threading

###########################################################################
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import math
from PIL import Image, ImageOps, ImageFilter
from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
import random
import datetime
import os

print('pytorch version :' , torch.__version__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("devices : ",device)
###########################################################################


path = os.getcwd()
qtCreatorFile = path + os.sep + "mainwindow.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile) 



class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

        self.cifar = training_agent()


    def onBindingUI(self):
        self.bt_show_train_images.clicked.connect(self.on_bt_show_train_images_click)
        self.bt_show_hyperparameter.clicked.connect(self.on_bt_show_hyperparameter_click)

        self.bt_train_1_epoch.clicked.connect(self.on_bt_train_1_epoch_click)
        self.bt_show_training_result.clicked.connect(self.on_bt_show_training_result_click)
        self.bt_interface.clicked.connect(self.on_bt_interface_click)
       
    def on_bt_show_train_images_click(self):
        self.cifar.init_data()
        #print("init")
        t = threading.Thread(target = self.cifar.show_image())
        t.start()

    def on_bt_show_hyperparameter_click(self):
        self.cifar.show_hyperparameter()

    def on_bt_train_1_epoch_click(self):
        if not self.cifar.if_have_data:
            self.cifar.init_data()
        self.cifar.step()

    def on_bt_show_training_result_click(self):
        pass

    def on_bt_interface_click(self):
        pass

##################################################################
##################################################################
##################################################################

class training_agent():
    def __init__(self):
        self.epoch = 0
        self.batch = 32
        self.learning_rate = 0.001
        self.optimizer = 'SGD'

        ##############################
        self.if_have_data = False

        self.train_data = None
        self.train_loader = None

        self.test_data = None
        self.test_loader = None
        ##############################
        self.model = models.resnet18()
        self.model.to(device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.loss_func = nn.CrossEntropyLoss()

        self.loss_plot = []
        self.acc_plot = []

    def init_data(self):
        #td = threading.Thread(target = self.notice())
        t = threading.Thread(target = self.download_data())
        t.start()
        t.join()
        #print(self.train_data)

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data,
                                                        batch_size=self.batch,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data,
                                                        batch_size=self.batch,
                                                        shuffle=False)
        self.if_have_data = True

    def show_image(self):
        #print(len(self.train_data.train_data))
        #print(len(self.train_data.train_labels))
        try:
            plt.figure(12)
            #axes = fig.add_subplot(111)
            for i in range(10):   
                print("plot")
                plt.subplot(2,5,i+1)
                plt.imshow(transforms.ToPILImage()(self.train_loader.dataset[i][0]).convert('RGB'))
                plt.title(self.train_loader.dataset[i][1])
            plt.show(block=False)
            plt.savefig(path+'/images.png')
        except:
            pass
        
    def step(self):
        los = []
        self.model.train()
        self.epoch += 1
        for step, (x, y) in enumerate(self.train_loader):
            data = Variable(x).to(device)
            target = Variable(y).to(device)
            output = self.model(data)
            loss = self.loss_func(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            los.append(loss.detach().numpy()) 
            if step % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, 
                    step * len(data), 
                    len(self.train_loader.dataset),
                    100. * step / len(self.train_loader), 
                    loss.data.item()))
        print("Finish")
        
        self.loss_plot.append(sum(los)/len(los))
        print(self.loss_plot)


    def download_data(self):
        self.train_data = torchvision.datasets.CIFAR10(
            root = path+'/data',
            train = True,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        self.test_data = torchvision.datasets.CIFAR10(
            root='./data/',
            train=False,
            transform=torchvision.transforms.ToTensor())

    def show_hyperparameter(self):
        print("================================")
        print("hyperparameter :")
        print("batch :",self.batch)
        print("learning rate :",self.learning_rate)
        print("optimizer :",self.optimizer)
        print("================================")

    def notice(self):
        msgBox = QMessageBox()
        msgBox.setWindowTitle('Downloading')
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText("Data downloading...")
        msgBox.setStandardButtons(QMessageBox.Ok)
        reply = msgBox.exec()

    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
