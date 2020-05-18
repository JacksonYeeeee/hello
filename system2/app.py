# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QMainWindow, QTextEdit, QLabel, QAction, QFileDialog, QApplication, QPushButton, QWidget, QGridLayout
from PyQt5.QtGui import QIcon
from PyQt5 import QtGui
from PyQt5.QtCore import Qt

import numpy as np
import cv2
import pandas as pd
import os

import h5py

import tflearn
from cnn_model import CNNModel

import tensorflow as tf
from tensorflow.python.framework import ops

csvPath = './system2/test_images_tag.csv'
saveDrawPath = './system2/temp/'
csvdata = pd.read_csv(csvPath)

def get_prediction(X_test_image):
    ops.reset_default_graph()
    ## Model definition
    convnet  = CNNModel()
    network = convnet.define_network(X_test_image)
    model = tflearn.DNN(network, tensorboard_verbose=0,\
  		   checkpoint_path='nodule-classifier.ckpt')
    model.load("./models_CNN2/model_conv/nodule-classifier")

    predictions = np.vstack(model.predict(X_test_image[:,:,:,:]))
    label_predictions = np.zeros_like(predictions)
    label_predictions[np.arange(len(predictions)), predictions.argmax(1)] = 1
    return predictions, label_predictions

def line_image(image,x,y,r):
    rgb_im = np.zeros((image.shape[0],image.shape[1],3))
    rgb_im[:,:,0] = image
    rgb_im[:,:,1] = image
    rgb_im[:,:,2] = image
    draw_im = cv2.rectangle(rgb_im,(int(x-r),int(y-r)),(int(x+r),int(y+r)),(250,126,32),2)
    return draw_im

def get_subimage(image, x,y,width=50):
        """
        Returns cropped image of requested dimensiona
        """
        subImage = image[int(y-width/2):int(y+width/2),\
         int(x-width/2):int(x+width/2)]
        return subImage

class MainUi(QMainWindow):

    def __init__(self,imgName=""):
        super().__init__()
        self.imgName=imgName
        self.initUI()

    def initUI(self):
        self.setFixedSize(780, 520)
        self.setWindowTitle("肺结节良恶性诊断")
        self.setWindowIcon(QIcon("./system/0.png"))
        self.setStyleSheet("QMainWindow{background:rgb(240,240,240)}")
        self.main_widget = QWidget()  # 创建窗口主部件
        self.main_layout = QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局
        
        self.left_widget = QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QGridLayout()  # 创建左侧部件的网格布局
        self.left_widget.setLayout(self.left_layout)  # 设置左侧部件布局为网格布局

        self.right_widget = QWidget()  # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QGridLayout()  # 创建右侧部件的网格布局
        self.right_widget.setLayout(self.right_layout)  # 设置右侧部件布局为网格布局

        self.main_layout.addWidget(self.left_widget, 0, 0, 20,20)
        self.main_layout.addWidget(self.right_widget, 0, 20, 20, 20)
        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

        self.label = QLabel()  # 创建label用于显示图片
        self.left_layout.addWidget(self.label, 0, 0, 40, 40)
        self.label.setStyleSheet('border-width: 1px;border-style: solid;border-color: rgb(40, 113, 62);\
                                    background-color: rgb(240,240,240);border-radius: 4px' )
        
        self.label3 = QLabel() #左侧绿条
        self.left_layout.addWidget(self.label3, 0, 0, 4, 40)
        self.label3.setText("        选择CT图像并显示")
        self.label3.setStyleSheet('background-color: rgb(40, 113, 62);\
                                    color:rgb(240,240,240);font-size:20px;font-weight:bold;font-family:宋体')
        
        self.button1 = QPushButton(QIcon(''), '打开图像')  # 打开文件按钮
        self.left_layout.addWidget(self.button1, 2, 4, 8, 12)
        self.button1.setStyleSheet(''' 
                     QPushButton
                     {text-align : center;
                     color:rgb(255, 255, 255);
                     background-color : rgb(111,117,97);
                     font-size:14px;
                     font-weight:bold;
                     border-radius: 2px;
                     height : 36px;
                     border-style: outset;
                     }
                     QPushButton:pressed
                     {text-align : center;
                     background-color : rgb(91,87,77);
                     font-size:14px;
                     font-weight:bold;
                     border-radius: 2px;
                     height : 36px;
                     border-style: outset}''')
        self.button1.clicked.connect(self.showDialog1)
        
        self.button2 = QPushButton(QIcon(''), '开始诊断')  # 诊断按钮
        self.left_layout.addWidget(self.button2, 2, 24, 8, 12)
        self.button2.setStyleSheet(''' 
                     QPushButton
                     {text-align : center;
                     color:rgb(255, 255, 255);
                     background-color : rgb(111,117,97);
                     font-size:14px;
                     font-weight:bold;
                     border-radius: 2px;
                     height : 36px;
                     border-style: outset;
                     }
                     QPushButton:pressed
                     {text-align : center;
                     background-color : rgb(91,87,77);
                     font-size:14px;
                     font-weight:bold;
                     border-radius: 2px;
                     height : 36px;
                     border-style: outset}''')
        self.button2.clicked.connect(self.showDialog2)
        
        self.label5 = QLabel()  #左侧显示图像小区域
        self.left_layout.addWidget(self.label5, 12, 4, 27, 34)
        self.label5.setText("          显示图像")
        self.label5.setStyleSheet('background-color: rgb(240,240,240);\
                                    color:rgba(75,87,62,155);font-size:20px;font-weight:bold;font-family:宋体')
        
        self.label7 = QLabel()  #左侧显示图像路径label
        self.left_layout.addWidget(self.label7, 9, 1,2,38)
        self.label7.setAlignment(Qt.AlignCenter)
        self.label7.resize(120,600)
        self.label7.setStyleSheet('background-color: rgb(255,255,255);border-radius: 2px;\
                                    color:rgb(75,87,62);font-size:14px;font-family:微软雅黑')
        
        self.label2 = QLabel()   #右侧label
        self.label2.setText("\n\n            结果诊断")
        self.right_layout.addWidget(self.label2, 0, 0, 10, 10)
        self.label2.setStyleSheet('border-width: 1px;border-style: solid;border-color: rgb(250, 132, 43);\
                                    background-color: rgb(240,240,240);border-radius: 4px;\
                                    color:rgba(75,87,62,155);font-size:20px;font-weight:bold;font-family:宋体')

        self.label4 = QLabel()  #右侧橙条
        self.right_layout.addWidget(self.label4, 0, 0, 1, 10)
        self.label4.setText("          诊断结果显示")
        self.label4.setStyleSheet('background-color: rgb(250, 132, 43);\
                                    color:rgb(240,240,240);font-size:20px;font-weight:bold;font-family:宋体')
        
        self.label6 = QLabel()  #右侧显示结果小区域
        self.right_layout.addWidget(self.label6, 6, 1, 1, 8)
        self.label6.setStyleSheet('background-color: rgb(240,240,240);\
                                    color:rgba(75,87,62,250);font-size:20px;font-family:宋体')
        
    #定义打开文件夹目录的函数
    def showDialog1(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, '打开图片', "./system2/test_images/", "*.jpg;;*.png;;All Files(*)")
        if imgName:
            #img = QtGui.QPixmap(imgName).scaled(self.label5.width(), self.label5.height())            
            img = QtGui.QPixmap(imgName).scaled(300, 300)
            #self.label5.setScaledContents (True)
            self.label5.setPixmap(img)
            self.label7.setText(imgName)

            self.label2.setText("\n\n            结果诊断")
            self.label2.setStyleSheet('border-width: 1px;border-style: solid;border-color: rgb(250, 132, 43);\
                                    background-color: rgb(240,240,240);border-radius: 4px;\
                                    color:rgba(75,87,62,155);font-size:20px;font-weight:bold;font-family:宋体')
            self.label6.setText(" ")
        self.imgName = imgName
    
    #定义诊断的函数 
    def showDialog2(self):
        print(self.imgName)
        if not self.imgName == "":
            image = cv2.imread(self.imgName,cv2.IMREAD_GRAYSCALE)
            image = np.array(image,dtype=np.float32) 

            imName = self.imgName.split('/')[-1]
            x = csvdata[(csvdata.seriesuid == imName)]['coordX']
            y = csvdata[(csvdata.seriesuid == imName)]['coordY']
            x = x.values[0]
            y = y.values[0]
            draw_img = line_image(image,x,y,50/2)
            cv2.imwrite(saveDrawPath+'0.jpg',draw_img)
            draw_img = QtGui.QPixmap(saveDrawPath+'0.jpg').scaled(300, 300)
            self.label5.setPixmap(draw_img)

            image = get_subimage(image,x,y,width=50)
            image = np.reshape(image/255, (1,50, 50, 1))

            prediction, label_prediction = get_prediction(image)
            tag = label_prediction[0][1].astype('int')
            if tag == 1:
                self.label2.setText("    诊断结果：\n      恶性")
                self.label6.setText("    恶性可能性："+str('%.4f' % (prediction[0][1]*100))+"%")
            else:
                self.label2.setText("    诊断结果：\n      良性")
                self.label6.setText("    良性可能性："+str('%.4f' % (prediction[0][0]*100))+"%")
            self.label2.setStyleSheet('border-width: 1px;border-style: solid;border-color: rgb(250, 132, 43);\
                                    background-color: rgb(240,240,240);border-radius: 4px;\
                                    color:rgba(75,87,62,250);font-size:40px;font-weight:bold;font-family:宋体')
            print(prediction,label_prediction)


def main():
    app = QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
