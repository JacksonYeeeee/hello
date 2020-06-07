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

class Ui_Form_DBN(QWidget):
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

        self.label_train2 = QtWidgets.QLabel(self.groupBox_train)  # RBM学习率
        self.label_train2.setGeometry(QtCore.QRect(40, 70, 81, 20))
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_train)
        self.lineEdit.setGeometry(QtCore.QRect(170, 70, 81, 25))
        self.lineEdit.setFrame(False)
        self.lineEdit.setPlaceholderText(str(0.001))
        self.lineEdit.setValidator(QDoubleValidator(0, 20, 6, self))

        self.label_train3_2 = QtWidgets.QLabel(self.groupBox_train)  # RBM迭代次数
        self.label_train3_2.setGeometry(QtCore.QRect(40, 110, 91, 20))
        self.horizontalSlider_2 = QtWidgets.QSlider(self.groupBox_train)
        self.horizontalSlider_2.setGeometry(QtCore.QRect(170, 110, 110, 22))
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setMinimum(0)
        self.horizontalSlider_2.setMaximum(100)
        self.horizontalSlider_2.setValue(10)
        self.horizontalSlider_2.sliderReleased.connect(self.slider2edit3)
        '''self.horizontalSlider.setTickInterval(5000)
        self.horizontalSlider.setTickPosition(QSlider.TicksAbove)'''
        self.slider_text1_2 = QtWidgets.QLineEdit(self.groupBox_train)
        self.slider_text1_2.setGeometry(QtCore.QRect(290, 110, 45, 27)) 
        self.slider_text1_2.setFrame(False)
        self.slider_text1_2.setValidator(QIntValidator(0, 100, self))
        self.slider_text1_2.setPlaceholderText(str(10))
        self.slider_text1_2.returnPressed.connect(self.edit2slider3)

        self.label_train4 = QtWidgets.QLabel(self.groupBox_train) # 单次迭代数据量
        self.label_train4.setGeometry(QtCore.QRect(40, 150, 106, 20))
        self.horizontalSlider2 = QtWidgets.QSlider(self.groupBox_train)
        self.horizontalSlider2.setGeometry(QtCore.QRect(170, 150, 110, 22))
        self.horizontalSlider2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider2.setMinimum(0)
        self.horizontalSlider2.setMaximum(1000)
        self.horizontalSlider2.setSingleStep(10)
        self.horizontalSlider2.setValue(100)
        self.horizontalSlider2.sliderReleased.connect(self.slider2edit2)
        '''self.horizontalSlider2.setTickInterval(500)
        self.horizontalSlider2.setTickPosition(QSlider.TicksAbove)'''
        self.slider_text2 = QtWidgets.QLineEdit(self.groupBox_train)
        self.slider_text2.setGeometry(QtCore.QRect(290, 150, 45, 27))
        self.slider_text2.setFrame(False)
        self.slider_text2.setValidator(QIntValidator(0, 1000, self))
        self.slider_text2.setPlaceholderText(str(100))
        self.slider_text2.returnPressed.connect(self.edit2slider2)

        self.label_train3 = QtWidgets.QLabel(self.groupBox_train)  # DBN迭代次数
        self.label_train3.setGeometry(QtCore.QRect(40, 190, 91, 20))
        self.horizontalSlider = QtWidgets.QSlider(self.groupBox_train)
        self.horizontalSlider.setGeometry(QtCore.QRect(170, 190, 110, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(1000)
        self.horizontalSlider.setValue(100)
        self.horizontalSlider.sliderReleased.connect(self.slider2edit)
        '''self.horizontalSlider.setTickInterval(5000)
        self.horizontalSlider.setTickPosition(QSlider.TicksAbove)'''
        self.slider_text1 = QtWidgets.QLineEdit(self.groupBox_train)
        self.slider_text1.setGeometry(QtCore.QRect(290, 190, 45, 27)) 
        self.slider_text1.setFrame(False)
        self.slider_text1.setValidator(QIntValidator(0, 1000, self))
        self.slider_text1.setPlaceholderText(str(100))
        self.slider_text1.returnPressed.connect(self.edit2slider)

        self.labeltrain2_2 = QtWidgets.QLabel(self.groupBox_train) # DBN学习率
        self.labeltrain2_2.setGeometry(QtCore.QRect(40, 230, 81, 20))
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox_train)
        self.lineEdit_2.setGeometry(QtCore.QRect(170, 230, 81, 25))
        self.lineEdit_2.setFrame(False)
        self.lineEdit_2.setPlaceholderText(str(0.001))
        self.lineEdit_2.setValidator(QDoubleValidator(0, 20, 6, self))

        
        self.pushButton_train = QtWidgets.QPushButton(self.groupBox_train) # 开始训练
        self.pushButton_train.setGeometry(QtCore.QRect(115, 280, 131, 51))

        self.label_train_show1 = QtWidgets.QLabel(self.groupBox_train)
        self.label_train_show1.setGeometry(QtCore.QRect(40, 360, 291, 5))
        self.label_train_show = QtWidgets.QLabel(self.groupBox_train)
        self.label_train_show.setGeometry(QtCore.QRect(40, 365, 291, 31))

        self.label_text_acc = QtWidgets.QLabel(self.groupBox_train)
        self.label_text_acc.setGeometry(QtCore.QRect(40, 420, 60, 31))
        self.label_text_recision = QtWidgets.QLabel(self.groupBox_train)
        self.label_text_recision.setGeometry(QtCore.QRect(155, 420, 60, 31))
        self.label_text_recall = QtWidgets.QLabel(self.groupBox_train)
        self.label_text_recall.setGeometry(QtCore.QRect(270, 420, 60, 31))

        self.label_acc = QtWidgets.QLabel(self.groupBox_train)
        self.label_acc.setGeometry(QtCore.QRect(30, 460, 80, 61))
        self.label_acc.setText('-')
        self.label_recision = QtWidgets.QLabel(self.groupBox_train)
        self.label_recision.setGeometry(QtCore.QRect(145, 460, 80, 61))
        self.label_recision.setText('-')
        self.label_recall = QtWidgets.QLabel(self.groupBox_train)
        self.label_recall.setGeometry(QtCore.QRect(260, 460, 80, 61))
        self.label_recall.setText('-')
        

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
        self.edit_acc.setPlaceholderText("显示RBM预训练过程数据")
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
        self.edit_loss.setPlaceholderText("显示DBN有监督训练过程数据")
        self.edit_loss.setReadOnly(True)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_train1.setText(_translate("Form", "数据集训练"))
        self.label_train1.setAlignment(Qt.AlignCenter)
        self.label_train_show.setText(_translate("Form", "测试集结果"))
        self.label_text_acc.setText(_translate("Form", "准确率"))
        self.label_text_recision.setText(_translate("Form", "精准率"))
        self.label_text_recall.setText(_translate("Form", "召回率"))
        self.label_text_acc.setAlignment(Qt.AlignCenter)
        self.label_text_recision.setAlignment(Qt.AlignCenter)
        self.label_text_recall.setAlignment(Qt.AlignCenter)
        '''self.label_acc.setText(_translate("Form", "89.34%"))
        self.label_recision.setText(_translate("Form", "65.74%"))
        self.label_recall.setText(_translate("Form", "90.00%"))'''
        self.label_acc.setAlignment(Qt.AlignCenter)
        self.label_recision.setAlignment(Qt.AlignCenter)
        self.label_recall.setAlignment(Qt.AlignCenter)
        self.pushButton_train.setText(_translate("Form", "开始训练"))
        self.label_train2.setText(_translate("Form", "RBM学习率"))
        self.label_train3_2.setText(_translate("Form", "RBM迭代次数"))
        self.label_train4.setText(_translate("Form", "单次迭代数据量"))
        self.label_train3.setText(_translate("Form", "DBN迭代次数"))
        self.labeltrain2_2.setText(_translate("Form", "DBN学习率"))
        self.label_acc1.setText(_translate("Form", "无监督预训练（RBM）"))
        self.label_acc1.setAlignment(Qt.AlignCenter)
        self.label_loss1.setText(_translate("Form", "有监督训练（DBN）"))
        self.label_loss1.setAlignment(Qt.AlignCenter)

        self.setWindowOpacity(0.9) # 设置窗口透明度
        pe = QPalette()
        self.setAutoFillBackground(True)

        self.groupBox_train.setStyleSheet('''QGroupBox{border:1px solid #00AA72;border-radius:6px;}''')
        self.label_train1.setStyleSheet('''QLabel{color:white;background:#00AA72;border-radius:6px;border:1px solid #00AA72;} ''')
        self.label_train1_2.setStyleSheet('''QLabel{background:#00AA72} ''')  

        self.label_train_show1.setStyleSheet('''QLabel{background:#00AA72;} ''')
        self.label_train_show.setStyleSheet('''QLabel{color:#00AA72;background:#f0f0f0;} ''')

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
        self.label_train_show.setFont(font)
        font.setBold(False)          # 初始学习率等
        font.setPointSize(11)
        self.label_train2.setFont(font)
        self.labeltrain2_2.setFont(font)
        self.label_train3.setFont(font)
        self.label_train3_2.setFont(font)
        self.label_train4.setFont(font)
        font.setBold(False)          #测试集准确率等 
        font.setPointSize(12)
        self.label_text_acc.setFont(font)
        self.label_text_recall.setFont(font)
        self.label_text_recision.setFont(font)
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
        self.label_train3.setStyleSheet('''QLabel{color:#373F43}''')
        self.label_train3_2.setStyleSheet('''QLabel{color:#373F43}''')
        self.label_train4.setStyleSheet('''QLabel{color:#373F43}''')

        self.label_text_acc.setStyleSheet('''QLabel{color:#373F43}''') #准确率 、 召回率……
        self.label_text_recision.setStyleSheet('''QLabel{color:#373F43}''')
        self.label_text_recall.setStyleSheet('''QLabel{color:#373F43}''')
        self.label_acc.setStyleSheet('''QLabel{color:#373F43}''') 
        self.label_recision.setStyleSheet('''QLabel{color:#373F43}''')
        self.label_recall.setStyleSheet('''QLabel{color:#373F43}''')

        font.setFamily('comforta')  # 初始学习率等的输入
        font.setBold(False)
        font.setPointSize(10)
        font.setWeight(80)
        self.lineEdit.setFont(font)
        self.lineEdit_2.setFont(font)
        self.slider_text1.setFont(font)
        self.slider_text1_2.setFont(font)
        self.slider_text2.setFont(font)
        font.setPointSize(15)
        font.setWeight(90)
        self.label_acc.setFont(font)
        self.label_recision.setFont(font)
        self.label_recall.setFont(font)
    
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

    def slider2edit3(self):
        val = self.horizontalSlider_2.value()
        self.slider_text1_2.setText(str(val))
    
    def edit2slider3(self):
        val = int(self.slider_text1_2.displayText())
        self.horizontalSlider_2.setValue(val)

        


