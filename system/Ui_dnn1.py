# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'f:\bigwork\object\system\dnn.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class Ui_Form(QWidget):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1234, 577)

        # 左侧train group
        self.groupBox_train = QtWidgets.QGroupBox(Form)
        self.groupBox_train.setGeometry(QtCore.QRect(10, 10, 371, 541))
        self.groupBox_train.setTitle("")

        self.label_train1 = QtWidgets.QLabel(self.groupBox_train)
        self.label_train1.setGeometry(QtCore.QRect(0, 0, 371, 51))
        self.label_train1_2 = QtWidgets.QLabel(self.groupBox_train)
        self.label_train1_2.setGeometry(QtCore.QRect(0, 40, 371, 11))

        font = QtGui.QFont()
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.label_train1.setFont(font)

        self.label_train2 = QtWidgets.QLabel(self.groupBox_train)  # 初始学习率
        self.label_train2.setGeometry(QtCore.QRect(50, 110, 81, 20))
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_train)
        self.lineEdit.setGeometry(QtCore.QRect(170, 110, 81, 25))
        self.lineEdit.setFrame(False)
        self.lineEdit.setPlaceholderText(str(0.002))
        self.lineEdit.setValidator(QDoubleValidator(0, 20, 6, self))

        self.labeltrain2_2 = QtWidgets.QLabel(self.groupBox_train) # 学习衰减率
        self.labeltrain2_2.setGeometry(QtCore.QRect(50, 170, 81, 20))
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox_train)
        self.lineEdit_2.setGeometry(QtCore.QRect(170, 170, 81, 25))
        self.lineEdit_2.setFrame(False)
        self.lineEdit_2.setPlaceholderText(str(0.9))
        self.lineEdit_2.setValidator(QDoubleValidator(0, 20, 6, self))

        self.labeltrain2_3 = QtWidgets.QLabel(self.groupBox_train) # 学习衰减延迟
        self.labeltrain2_3.setGeometry(QtCore.QRect(50, 220, 91, 20))
        self.lineEdit_3 = QtWidgets.QLineEdit(self.groupBox_train)
        self.lineEdit_3.setGeometry(QtCore.QRect(170, 220, 81, 25))  
        self.lineEdit_3.setFrame(False)
        self.lineEdit_3.setPlaceholderText(str(100))
        self.lineEdit_3.setValidator(QIntValidator(0, 1000, self))

        self.label_train3 = QtWidgets.QLabel(self.groupBox_train)  # 迭代次数
        self.label_train3.setGeometry(QtCore.QRect(50, 290, 91, 20))
        self.horizontalSlider = QtWidgets.QSlider(self.groupBox_train)
        self.horizontalSlider.setGeometry(QtCore.QRect(170, 290, 120, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(80000)
        self.horizontalSlider.setSingleStep(100)
        self.horizontalSlider.setValue(10000)
        self.horizontalSlider.sliderReleased.connect(self.slider2edit)
        '''self.horizontalSlider.setTickInterval(5000)
        self.horizontalSlider.setTickPosition(QSlider.TicksAbove)'''
        self.slider_text1 = QtWidgets.QLineEdit(self.groupBox_train)
        self.slider_text1.setGeometry(QtCore.QRect(300, 290, 45, 27)) 
        self.slider_text1.setFrame(False)
        self.slider_text1.setValidator(QIntValidator(0, 80000, self))
        self.slider_text1.setPlaceholderText(str(10000))
        self.slider_text1.returnPressed.connect(self.edit2slider)


        self.label_train4 = QtWidgets.QLabel(self.groupBox_train) # 单次迭代数据量
        self.label_train4.setGeometry(QtCore.QRect(50, 350, 91, 20))
        self.horizontalSlider2 = QtWidgets.QSlider(self.groupBox_train)
        self.horizontalSlider2.setGeometry(QtCore.QRect(170, 350, 120, 22))
        self.horizontalSlider2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider2.setMinimum(0)
        self.horizontalSlider2.setMaximum(1000)
        self.horizontalSlider2.setSingleStep(10)
        self.horizontalSlider2.setValue(100)
        self.horizontalSlider2.sliderReleased.connect(self.slider2edit2)
        '''self.horizontalSlider2.setTickInterval(500)
        self.horizontalSlider2.setTickPosition(QSlider.TicksAbove)'''
        self.slider_text2 = QtWidgets.QLineEdit(self.groupBox_train)
        self.slider_text2.setGeometry(QtCore.QRect(300, 350, 45, 27))
        self.slider_text2.setFrame(False)
        self.slider_text2.setValidator(QIntValidator(0, 1000, self))
        self.slider_text2.setPlaceholderText(str(100))
        self.slider_text2.returnPressed.connect(self.edit2slider2)
        
        self.pushButton_train = QtWidgets.QPushButton(self.groupBox_train) # 开始训练
        self.pushButton_train.setGeometry(QtCore.QRect(115, 430, 131, 51))
        

        # 中间 accuracy
        self.groupBox_acc = QtWidgets.QGroupBox(Form)
        self.groupBox_acc.setGeometry(QtCore.QRect(390, 10, 391, 541))
        self.groupBox_acc.setTitle("")

        self.label_acc1 = QtWidgets.QLabel(self.groupBox_acc)
        self.label_acc1.setGeometry(QtCore.QRect(0, 0, 391, 51))
        self.label_acc1_2 = QtWidgets.QLabel(self.groupBox_acc)
        self.label_acc1_2.setGeometry(QtCore.QRect(0, 40, 391, 11))


        self.label_acc2 = QtWidgets.QLabel(self.groupBox_acc)
        self.label_acc2.setGeometry(QtCore.QRect(10, 100, 371, 401))
        self.label_acc2.setText("")
        self.edit_acc = QtWidgets.QTextEdit(self.groupBox_acc)
        self.edit_acc.setGeometry(QtCore.QRect(20, 100, 351, 421))
        self.edit_acc.setPlaceholderText("显示训练准确率数据")
        self.edit_acc.setReadOnly(True)

        # 右边 loss
        self.groupBox_loss = QtWidgets.QGroupBox(Form)
        self.groupBox_loss.setGeometry(QtCore.QRect(790, 10, 391, 541))
        self.groupBox_loss.setTitle("")

        self.label_loss1 = QtWidgets.QLabel(self.groupBox_loss)
        self.label_loss1.setGeometry(QtCore.QRect(0, 0, 391, 51))
        self.label_loss1_2 = QtWidgets.QLabel(self.groupBox_loss)
        self.label_loss1_2.setGeometry(QtCore.QRect(0, 40, 391, 11))

        self.label_loss2 = QtWidgets.QLabel(self.groupBox_loss)
        self.label_loss2.setGeometry(QtCore.QRect(10, 100, 371, 401))
        self.label_loss2.setText("")
        self.edit_loss = QtWidgets.QTextEdit(self.groupBox_loss)
        self.edit_loss.setGeometry(QtCore.QRect(20, 100, 351, 421))
        self.edit_loss.setPlaceholderText("显示训练损失数据")
        self.edit_loss.setReadOnly(True)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_train1.setText(_translate("Form", "数据集训练"))
        self.label_train1.setAlignment(Qt.AlignCenter)
        self.pushButton_train.setText(_translate("Form", "开始训练"))
        self.label_train2.setText(_translate("Form", "初始学习率"))
        self.label_train3.setText(_translate("Form", "迭代次数"))
        self.label_train4.setText(_translate("Form", "单次迭代数据量"))
        self.labeltrain2_2.setText(_translate("Form", "学习衰减率"))
        self.labeltrain2_3.setText(_translate("Form", "学习衰减延迟"))
        self.label_acc1.setText(_translate("Form", "准确率展示"))
        self.label_acc1.setAlignment(Qt.AlignCenter)
        self.label_loss1.setText(_translate("Form", "损失展示"))
        self.label_loss1.setAlignment(Qt.AlignCenter)

        self.setWindowOpacity(0.9) # 设置窗口透明度
        pe = QPalette()
        self.setAutoFillBackground(True)

        self.groupBox_train.setStyleSheet('''QGroupBox{border:1px solid #00AA72;border-radius:6px;}''')
        self.label_train1.setStyleSheet('''QLabel{color:white;background:#00AA72;border-radius:6px;border:1px solid #00AA72;} ''')
        self.label_train1_2.setStyleSheet('''QLabel{background:#00AA72} ''')  

        self.groupBox_acc.setStyleSheet('''QGroupBox{border:1px solid #0772A1;border-radius:6px;}''')
        self.label_acc1.setStyleSheet('''QLabel{color:white;background:#0772A1;border-radius:6px;border:1px solid #0772A1;} ''')
        self.label_acc1_2.setStyleSheet('''QLabel{background:#0772A1} ''')  

        self.groupBox_loss.setStyleSheet('''QGroupBox{border:1px solid #FFA100;border-radius:6px;}''')
        self.label_loss1.setStyleSheet('''QLabel{color:white;background:#FFA100;border-radius:6px;border:1px solid #FFA100;} ''')
        self.label_loss1_2.setStyleSheet('''QLabel{background:#FFA100} ''')  

        self.pushButton_train.setStyleSheet("QPushButton{color:white}"
                                            "QPushButton{background-color:#FA842B}"
                                            "QPushButton:hover{background-color:#E75B12}"
                                            "QPushButton:pressed{background-color:#FA842B}"
                                            "QPushButton{border:2px}"
                                            "QPushButton{border-radius:10px}"
                                            "QPushButton{padding:2px 4px}")

        font = QtGui.QFont()  # 三个个标题
        font.setFamily("宋体")
        font.setBold(True)
        font.setPointSize(14)
        self.label_train1.setFont(font)
        self.label_acc1.setFont(font)
        self.label_loss1.setFont(font)
        font.setBold(False)          # 初始学习率等
        font.setPointSize(11)
        self.label_train2.setFont(font)
        self.labeltrain2_2.setFont(font)
        self.labeltrain2_3.setFont(font)
        self.label_train3.setFont(font)
        self.label_train4.setFont(font)
        font.setFamily('微软雅黑')    # 开始训练
        font.setBold(True)         
        font.setPointSize(15)
        font.setWeight(90)
        self.pushButton_train.setFont(font)
        font.setBold(False)  # 数据打印
        font.setPointSize(10)
        font.setWeight(60) 
        self.edit_acc.setFont(font)
        self.edit_loss.setFont(font)

        self.label_train2.setStyleSheet('''QLabel{color:#373F43}''') # 初始学习率
        self.labeltrain2_2.setStyleSheet('''QLabel{color:#373F43}''')
        self.labeltrain2_3.setStyleSheet('''QLabel{color:#373F43}''')
        self.label_train3.setStyleSheet('''QLabel{color:#373F43}''')
        self.label_train4.setStyleSheet('''QLabel{color:#373F43}''')

        font.setFamily('comforta')  # 初始学习率等的输入
        font.setBold(False)
        font.setPointSize(10)
        font.setWeight(80)
        self.lineEdit.setFont(font)
        self.lineEdit_2.setFont(font)
        self.lineEdit_3.setFont(font)
        self.slider_text1.setFont(font)
        self.slider_text2.setFont(font)
    
    def slider2edit(self):
        val = self.horizontalSlider.value()
        self.slider_text1.setText(str(val))
    
    def edit2slider(self):
        val = int(self.slider_text1.displayText())
        self.horizontalSlider.setValue(val)

    def slider2edit2(self):
        val = self.horizontalSlider2.value()
        self.slider_text2.setText(str(val))
    
    def edit2slider2(self):
        val = int(self.slider_text2.displayText())
        self.horizontalSlider2.setValue(val)

        


