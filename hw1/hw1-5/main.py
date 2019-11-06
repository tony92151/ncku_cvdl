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

dict = {'0': "airplain",
        '1': "automobile",
        '2': "bird",
        '3': "cat",
        '4': "deer",
        '5': "dog",
        '6': "frog",
        '7': "horse",
        '8': "ship",
        '9': "truck"}



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
        if not self.cifar.if_have_data:
            self.cifar.init_data()
        #print("init")
        t = threading.Thread(target = self.cifar.show_image())
        t.start()

    def on_bt_show_hyperparameter_click(self):
        self.cifar.show_hyperparameter()

    def on_bt_train_1_epoch_click(self):
        if not self.cifar.if_have_data:
            self.cifar.init_data()
        
        if not True:
            self.cifar.step()
        else:
            for i in range(50):
                self.cifar.step()
            #self.cifar.show_acc_and_loss()

    def on_bt_show_training_result_click(self):
        self.cifar.show_acc_and_loss()

    def on_bt_interface_click(self):
        if not self.cifar.if_have_data:
            self.cifar.init_data()
        self.cifar.predict(10)
        #print(self.cifar.train_length)
        #print(self.cifar.test_length)

##################################################################
##################################################################
##################################################################

class training_agent():
    def __init__(self):
        self.epoch = 0
        self.batch = 64
        self.learning_rate = 0.01
        self.optimizer_show = 'SGD'

        self.train_length = 0
        self.test_length = 0

        ##############################
        self.if_have_data = False

        self.train_data = None
        self.train_loader = None
        
        self.test_data = None
        self.test_loader = None
        ##############################

        self.unorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std= (0.2023, 0.1994, 0.2010))

        self.model = models.resnet18()
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.Linear(in_features=128, out_features=10, bias=True),
            nn.Softmax()
            )
        self.model.to(device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        self.loss_func = nn.CrossEntropyLoss()

        self.loss_plot = []
        self.losstep = []
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
        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)

    def show_image(self):
        plt.close()
        #print(len(self.train_data.train_data))
        #print(len(self.train_data.train_labels))
        try:
            plt.figure("Train images")
            #axes = fig.add_subplot(111)
            for i in range(10):   
                #print("plot")
                plt.subplot(2,5,i+1)
                img = self.unorm(self.train_loader.dataset[i][0])
                plt.imshow(transforms.ToPILImage()(img).convert('RGB'))
                plt.title(dict[format(self.train_loader.dataset[i][1])])
            plt.show(block=False)
            #plt.savefig(path+'/images.png')
        except:
            pass

    def show_loss_plot(self):
        plt.close()
        try:
            plt.figure("Step loss plot") 
            plt.plot(self.losstep)
            plt.title("Step loss plot")
            plt.show(block=False)
            #plt.savefig(path+'/images.png')
        except:
            pass
        pass

    def show_acc_and_loss(self):
        plt.close()
        try:
            plt.figure("Acc Loss")
            plt.subplot(2,1,2)
            plt.plot(self.loss_plot)
            plt.xlabel("Epoch")
            plt.title("Loss")
            plt.subplot(2,1,1)
            plt.plot(self.acc_plot)
            plt.title("Acc (%)")
            
            plt.show(block=False)
            #plt.savefig(path+'/images.png')
        except:
            pass

    def step(self):
        self.train()
        self.test()
        self.show_loss_plot()

    def predict(self,idx):
        self.model.eval()
        img = self.test_loader.dataset[idx][0]
        #print(img)
        img_show = self.unorm(img)
        img = img.unsqueeze(0)
        x = Variable(img).to(device)
        output = self.model(x)
        #print(output[0].shape)
        if torch.cuda.is_available():
            output = output.cpu().detach().numpy()
        else:
            output = output.detach().numpy()

        print(output[0])
        pass

    def train(self):
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
            if torch.cuda.is_available():
                los.append(loss.cpu().detach().numpy()) 
                self.losstep.append(loss.cpu().detach().numpy()) 
            else:
                los.append(loss.detach().numpy()) 
                self.losstep.append(loss.detach().numpy()) 
            if step % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, 
                    step * len(data), 
                    len(self.train_loader.dataset),
                    100. * step / len(self.train_loader), 
                    loss.data.item()))
        self.loss_plot.append(sum(los)/len(los))
        # print(self.loss_plot)

    def test(self):
        self.model.eval()
        avloss = []
        correct = 0
        for step, (x, y) in enumerate(self.test_loader):
            data = Variable(x).to(device)
            target = Variable(y).to(device)
            output = self.model(data)
            loss = self.loss_func(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            if torch.cuda.is_available():
                avloss.append(loss.cpu().detach().numpy()) 
            else:
                avloss.append(loss.detach().numpy()) 
            
            
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    sum(avloss)/len(avloss), 
                    correct, 
                    len(self.test_loader.dataset),
                    100. * correct / len(self.test_loader.dataset)))
        self.acc_plot.append(correct)


    def download_data(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.train_data = torchvision.datasets.CIFAR10(
            root = path+'/data',
            train = True,
            transform=transform,
            download=True,
        )
        self.test_data = torchvision.datasets.CIFAR10(
            root='./data/',
            train=False,
            transform=transform)

    def show_hyperparameter(self):
        print("================================")
        print("hyperparameter :")
        print("batch :",self.batch)
        print("learning rate :",self.learning_rate)
        print("optimizer :",self.optimizer_show)
        print("================================")

    def notice(self):
        msgBox = QMessageBox()
        msgBox.setWindowTitle('Downloading')
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText("Data downloading...")
        msgBox.setStandardButtons(QMessageBox.Ok)
        reply = msgBox.exec()

#https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
