# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!
import sys

sys.path.append('..')

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import *
import SimpleITK as sitk
from utility import *
import matplotlib.pyplot as pyplot
import numpy as np
from cv2 import *
import qtawesome
import torch
from unet import UNet
from numpy import *
import os


class Ui_mainWindow(QMainWindow):

    def __init__(self):

        self.file_time = []
        super().__init__()
        self.x = 0
        self.y = 0
        self.z = 0
        self.impot_flag = False
        self.seg_flag = True# 如果目录下有分割好的文件，设为True
        self.load_flag = False

    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(800, 600)
        # self.centralwidget = QtWidgets.QWidget(mainWindow)
        # self.centralwidget.setObjectName("centralwidget")

#创建四个lable，lable类重写加入了滚轮的信号。向centralwidget添加lable
        self.widget = Lables("正视图")
        #self.widget.setGeometry(QtCore.QRect(10, 10, 690, 690))
        self.widget.setObjectName("widget")
        self.widget.setAutoFillBackground(1)

#在widget里添加lable
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.widget.signal[int].connect(self.change_image_1)

        self.widget_2 = Lables("俯视图")
        self.widget_2.setObjectName("widget_2")
        self.label_2 = QtWidgets.QLabel(self.widget_2)
        self.label_2.setObjectName("label_2")
        self.widget_2.signal[int].connect(self.change_image_2)

        self.widget_3 = Lables("侧视图")
        self.widget_3.setObjectName("widget_3")
        self.label_3 = QtWidgets.QLabel(self.widget_3)
        self.label_3.setObjectName("label_3")
        self.widget_3.signal[int].connect(self.change_image_3)

        self.widget_4 = Lables("暂无")
        self.widget_4.setObjectName("widget_4")
#左侧菜单栏
        self.left_widget = QtWidgets.QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout)  # 设置左侧部件布局为网格

        self.left_close = QtWidgets.QPushButton("")  # 关闭按钮
        self.left_visit = QtWidgets.QPushButton("")  # 空白按钮
        self.left_mini = QtWidgets.QPushButton("")  # 最小化按钮

        self.left_button_1 = QtWidgets.QPushButton("BrainStem")
        self.left_button_1.setObjectName('left_button')
        self.left_button_2 = QtWidgets.QPushButton("Chiasm")
        self.left_button_2.setObjectName('left_button')
        self.left_button_3 = QtWidgets.QPushButton("Mandible")
        self.left_button_3.setObjectName('left_button')
        self.left_button_4 = QtWidgets.QPushButton("OpicNerve")
        self.left_button_4.setObjectName('left_button')
        self.left_button_5 = QtWidgets.QPushButton("Parotid")
        self.left_button_5.setObjectName('left_button')
        self.left_button_6 = QtWidgets.QPushButton("Submandibular")
        self.left_button_6.setObjectName('left_button')

        self.left_layout.addWidget(self.left_mini, 0, 0, 1, 1)
        self.left_layout.addWidget(self.left_close, 0, 2, 1, 1)
        self.left_layout.addWidget(self.left_visit, 0, 1, 1, 1)
        self.left_layout.addWidget(self.left_button_1, 2, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_2, 3, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_3, 4, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_4, 6, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_5, 7, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_6, 8, 0, 1, 3)
#设置上侧菜单
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

#创建按钮
        self.action = QtWidgets.QAction(mainWindow)
        self.action.setObjectName("action")
        self.action.setStatusTip('选择一个dcm文件')

        self.action_2 = QtWidgets.QAction(mainWindow)
        self.action_2.setObjectName("action_2")
        self.action_2.setStatusTip('开始分割')
#把按钮放添加到菜单栏
        self.menu.addAction(self.action)
        self.menu.addAction(self.action_2)
        self.menubar.addAction(self.menu.menuAction())
#连接信号和槽
        self.action.triggered.connect(self.showDialog)
        self.action_2.triggered.connect(self.segment)

#布局
        # grid_layout = QGridLayout()
        # grid_layout.addWidget(self.widget, 0, 0)
        # grid_layout.addWidget(self.widget_2,0,1)
        # grid_layout.addWidget(self.widget_3,1,0)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.widget)
        v_layout.addWidget(self.widget_3)
        v_layout.setSpacing(5)

        right_grid_layout = QGridLayout()
        right_grid_layout.addWidget(self.widget_2, 0,0)

        left_grid_layout = QGridLayout()
        left_grid_layout.addWidget(self.left_widget,0,0)

        main_layout = QGridLayout()
        main_layout.addLayout(left_grid_layout, 0, 0)
        main_layout.addLayout(v_layout, 0, 1)
        main_layout.addLayout(right_grid_layout, 0, 2)

        main_layout.setAlignment(QtCore.Qt.AlignTop)

#设置窗口大小
        self.widget.setFixedSize(680, 350)
        self.widget_2.setFixedSize(690, 690)
        self.widget_3.setFixedSize(680,350)

#美化窗口样式
        self.widget.setStyleSheet('''
            QWidget#widget{
            color:#232C51;
            background:gray;
            border-top:1px solid darkGray;
            border-bottom:1px solid darkGray;
            border-right:1px solid darkGray;

            }
        ''')

        self.widget_2.setStyleSheet('''
               QWidget#widget_2{
               color:#232C51;
               background:gray;
               border-top:1px solid darkGray;
               border-bottom:1px solid darkGray;
               border-right:1px solid darkGray;

               }
        ''')

        self.widget_3.setStyleSheet('''
             QWidget#widget_3{
             color:#232C51;
             background:gray;
             border-top:1px solid darkGray;
             border-bottom:1px solid darkGray;
             border-right:1px solid darkGray;

             }
             ''')

        self.left_close.setFixedSize(15, 15)  # 设置关闭按钮的大小
        self.left_visit.setFixedSize(15, 15)  # 设置按钮大小
        self.left_mini.setFixedSize(15, 15)  # 设置最小化按钮大小

        self.label.setFixedSize(40,15)
        self.label_2.setFixedSize(40, 15)
        self.label_3.setFixedSize(40, 15)

        self.label.setStyleSheet('''
            QLabel#label{
             color:white;
             background:black;
             border-top-right-radius:5px;
             border-bottom-right-radius:5px;
             border-top-left-radius:5px;
             border-bottom-left-radius:5px;
             font-size:15px;
             font-weight:500;
            }  
            ''')

        self.label_2.setStyleSheet('''
                  QLabel#label_2{
                   color:white;
                   background:black;
                   border-top-right-radius:5px;
                   border-bottom-right-radius:5px;
                   border-top-left-radius:5px;
                   border-bottom-left-radius:5px;
                   font-size:15px;
                   font-weight:500;
                  }  
                  ''')

        self.label_3.setStyleSheet('''
                  QLabel#label_3{
                   color:white;
                   background:black;
                   border-top-right-radius:5px;
                   border-bottom-right-radius:5px;
                   border-top-left-radius:5px;
                   border-bottom-left-radius:5px;
                   font-size:15px;
                   font-weight:500;
                  }  
                  ''')

        self.left_close.setStyleSheet('''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.left_visit.setStyleSheet('''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet('''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')

        # 创建一个窗口对象
        layout_widget = QWidget()
        # 设置窗口的布局层
        layout_widget.setLayout(main_layout)

        mainWindow.setCentralWidget(layout_widget)
        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def change_image_1(self, val):
        if self.impot_flag:
            if -1 < self.y + val < self.d_2:
#切片img
                self.label.setText(QtCore.QCoreApplication.translate("mainWindow", str(self.y)))
                print(self.y)
                matrix = self.matrix3D[1:-1, self.y, :]
                print(matrix.shape)
                self.y += val
#切片segment
                if self.y > 0 and self.y < self.d_2 - 1 and self.seg_flag:
                    segment = self.seg[:, self.y-1, :]
                else:
                    segment = np.zeros([self.d_1-2, self.d_3])

                show_image(self.widget, matrix, segment, self.seg_flag)

    def change_image_2(self, val):
        if self.impot_flag:
            if -1 < self.x + val< self.d_1:
#切片img
                self.label_2.setText(QtCore.QCoreApplication.translate("mainWindow", str(self.x)))
                print(self.x)
                matrix = self.matrix3D[self.x, :, :]
                self.x += val
#切片segment
                if self.x>0 and self.x<self.d_1-1 and self.seg_flag:
                    segment = self.seg[self.x-1, :, :]
                else:
                    segment = np.zeros([self.d_2, self.d_3])

                #print(self.x)
                show_image(self.widget_2, matrix, segment, self.seg_flag)

    def change_image_3(self, val):
        if self.impot_flag:
            #print(self.d_1,self.d_2,self.d_3)
            if -1 < self.z + val < self.d_3:

#切片img
                self.label_3.setText(QtCore.QCoreApplication.translate("mainWindow", str(self.z)))
                print(self.z)
                matrix = self.matrix3D[1:-1, :, self.z]
                self.z += val
#切片segment
                if self.z > 0 and self.z < self.d_3 - 1 and self.seg_flag:
                    segment = self.seg[:, self.z - 1, :]
                else:
                    segment = np.zeros([self.d_1-2, self.d_2])

                show_image(self.widget_3, matrix, segment, self.seg_flag)

    def showDialog(self):
        fname = QFileDialog.getOpenFileName()

        if fname[0]:
            #print(fname[0])
            dir = os.path.dirname(fname[0])
            #print(dir)  #某个dcm文件夹地址

#读取array
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dir)
            reader.SetFileNames(dicom_names)
            image_data = reader.Execute()
            #self.matrix3D = sitk.GetArrayFromImage(image_data)
            #print(np.unique(self.matrix3D))
            #print(image_data)

            size = image_data.GetSize()
            space = image_data.GetSpacing()
            origin = image_data.GetOrigin()
            direciton = image_data.GetDirection()
            print('size:',size, ' space: ', space, ' origin:',origin, ' direction: ', direciton)

#开始插值
            resample = sitk.ResampleImageFilter()
            resample.SetInterpolator(sitk.sitkLinear)
            resample.SetDefaultPixelValue(0)
            resample.SetOutputDirection(image_data.GetDirection())
            resample.SetOutputOrigin(image_data.GetOrigin())
            resample.SetOutputSpacing([1, 1, 1])
            size_ = [int(size[0]*space[0]), int(size[1]*space[1]), int(size[2]*space[2])]

            resample.SetSize(size_)
            newimage = resample.Execute(image_data)
            #newimage = Preprocess(newimage)
            self.matrix3D = np.rot90(sitk.GetArrayFromImage(newimage), k=2, axes=(0,2))
            self.matrix3D = Preprocess(sitk.GetImageFromArray(self.matrix3D))

            #print(self.matrix3D.shape)
            shape = self.matrix3D.shape
            #print('reshaped size: ', shape)

            self.d_1 = shape[0]
            self.d_2 = shape[1]
            self.d_3 = shape[2]
            #print("d1  d2  d3", self.d_1, self.d_2, self.d_3)

            self.impot_flag = True

    def segment(self):
#设置边界条件
        if self.matrix3D.max() == 0:
            self.action_2.setStatusTip('请先导入dicom序列')
            return
        if self.seg_flag == True:
            self.seg = ReadSegment(os.getcwd())
            return
        if not self.seg_flag and os.path.exists('segment_1.npy'):
            self.seg = ReadSegment(os.getcwd())
            return

#读入数据
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = UNet(3, 2)
        net.to(device=device)
        net.load_state_dict(torch.load('C:/Users/mamama9503/Desktop/Pytorch-UNet/checkpoints/CP_epoch2.pth'))
        # self.segment3D = np.empty([0, self.d_2, self.d_3])

#分割
        for i in range(1,self.d_3-1):#因为每三层作为一个输入，所以第一层和最后一层没法分割
            input = torch.from_numpy(self.matrix3D[i-1:i+2, :, :].copy()).unsqueeze(0)#切片得到i-1，i，i+1
            input = input.to(device=device, dtype=torch.float32)
            out = torch.sigmoid(net(input))
            out = np.expand_dims(out[:, 1,...].squeeze().detach().cpu().numpy(), axis=0)#从（1，2，512，512）转换成（512，512，1）
            print(i, input.shape, out.shape)

#每张存成一个npy
            np.save('segment_'+str(i), out)

#读取分割文件
        self.seg = ReadSegment(os.getcwd())
        self.seg_flag = True

            # img = input[0][1].cpu().detach().numpy()
            # imshow("img", (((img-img.min())/(img.max()-img.min()))*255.0).astype(np.uint8))
            # imshow("pre", ((out[0,:,:]>0.4)*255.0).astype(np.uint8))
            # waitKey(30)
            # print(out.shape, self.segment3D.shape)

            # self.segment3D = np.append(self.segment3D, out, axis=0)

        #self.segment3D.astype(bool)
        # np.save('segment.npy', self.segment3D)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "医疗图像分割工具"))
        #text_1 = "slice:"+str(self.y)+"/"+str(self.d_1)
        # self.label.setText(_translate("mainWindow", str(self.y)))
        # self.label_2.setText(_translate("mainWindow", str(self.x)))
        # self.label_3.setText(_translate("mainWindow", str(self.z)))
        # self.label_4.setText(_translate("mainWindow", "slice"))
        self.menu.setTitle(_translate("mainWindow", "功能栏"))
        self.action.setText(_translate("mainWindow", "导入"))
        self.action_2.setText(_translate("mainWindow", "分割"))


class Lables(QLabel):
    signal = pyqtSignal(int)

    def __init__(self,parent):
        super().__init__(parent)

    def wheelEvent(self, event):
        val = event.angleDelta().y()/120
        self.signal.emit(val)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
