# -*- coding: utf-8 -*-
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from Ui_dnn1 import Ui_Form  #导入创建的GUI类
from Ui_rnn import Ui_Form_RNN
from Ui_dbn import Ui_Form_DBN
from Ui_cnn import Ui_Form_CNN

from model_dnn import *  #模型
from model_rnn import *
from model_cnn.train import *
from model_dbn2.train import *


class MyFigure(FigureCanvas):
    def __init__(self,width=3, height=2, dpi=100,):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MyFigure,self).__init__(self.fig) 
        self.axes = self.fig.add_subplot(111)

class MainUi(QTabWidget,Ui_Form,Ui_Form_RNN,Ui_Form_DBN):

    def __init__(self):
        super().__init__()
        self.initUI()
        #super(MainUi, self).__init__()

        #创建3个选项卡
        self.tab_dnn=Ui_Form()
        self.tab_rnn=Ui_Form_RNN()
        self.tab_dbn=Ui_Form_DBN()
        self.tab_cnn=Ui_Form_CNN()
        self.addTab(self.tab_dnn, "DNN")
        self.addTab(self.tab_rnn, "RNN")
        self.addTab(self.tab_dbn, "DBN")
        self.addTab(self.tab_cnn, "CNN")

        self.tab_dnn.setupUi(self.tab_dnn)
        self.tab_rnn.setupUi(self.tab_rnn)
        self.tab_dbn.setupUi(self.tab_dbn)
        self.tab_cnn.setupUi(self.tab_cnn)

        self.gridlayout1 = QGridLayout(self.tab_dnn.label_acc2)  # 继承容器groupBox
        self.gridlayout2 = QGridLayout(self.tab_dnn.label_loss2) #清除plt图时使用
        self.gridlayout1_rnn = QGridLayout(self.tab_rnn.label_acc2)  
        self.gridlayout2_rnn = QGridLayout(self.tab_rnn.label_loss2)
        self.gridlayout1_dbn = QGridLayout(self.tab_dbn.label_acc2)  
        self.gridlayout2_dbn = QGridLayout(self.tab_dbn.label_loss2)
        self.gridlayout1_cnn = QGridLayout(self.tab_cnn.label_acc2)  
        self.gridlayout2_cnn = QGridLayout(self.tab_cnn.label_loss2)

        self.button_click()
        

    def initUI(self):
        self.setFixedSize(1200, 590)
        self.setWindowTitle("肺结节良恶性诊断")
        self.setWindowIcon(QIcon("./system/0.png"))
        #self.setWindowFlag(QtCore.Qt.FramelessWindowHint) # 隐藏边框
        #self.setStyleSheet("QMainWindow{background:rgb(250,240,251)}")
        self.main_widget = QWidget()  # 创建窗口主部件
        self.main_layout = QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局

        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(True)
        self.setFont(font)
        self.setAutoFillBackground(True)
        self.setStyleSheet("QTabWidget::pane{border:0px;}\
                            QTabBar::tab:selected {color:#373F43;background-color:#f0f0f0}\
                            QTabBar::tab:!selected {color:#9EA0A1;}")
    
    # 按钮点击事件绑定
    def button_click(self):
        self.tab_dnn.pushButton_train.clicked.connect(self.dnntrain)
        self.tab_rnn.pushButton_train.clicked.connect(self.rnntrain)
        self.tab_dbn.pushButton_train.clicked.connect(self.dbntrain2)
        self.tab_cnn.pushButton_train.clicked.connect(self.cnntrain)
    
    # 检查用户是否输入
    def check_val(self):
        if self.tab_dnn.lineEdit.displayText() == "":
            self.tab_dnn.lineEdit.setText(str(0.002))
        if self.tab_dnn.lineEdit_2.displayText() == "":
            self.tab_dnn.lineEdit_2.setText(str(0.9))
        if self.tab_dnn.lineEdit_3.displayText() == "":
            self.tab_dnn.lineEdit_3.setText(str(100))
        if self.tab_dnn.slider_text1.displayText() == "":
            self.tab_dnn.slider_text1.setText(str(10000))
        if self.tab_dnn.slider_text2.displayText() == "":
            self.tab_dnn.slider_text2.setText(str(100))
    
    def check_val_rnn(self):
        if self.tab_rnn.lineEdit.displayText() == "":
            self.tab_rnn.lineEdit.setText(str(0.0002))
        if self.tab_rnn.lineEdit_2.displayText() == "":
            self.tab_rnn.lineEdit_2.setText(str(0.9))
        if self.tab_rnn.lineEdit_3.displayText() == "":
            self.tab_rnn.lineEdit_3.setText(str(100))
        if self.tab_rnn.slider_text1.displayText() == "":
            self.tab_rnn.slider_text1.setText(str(8000))
        if self.tab_rnn.slider_text2.displayText() == "":
            self.tab_rnn.slider_text2.setText(str(100))
    
    def check_val_dbn(self):
        if self.tab_dbn.lineEdit.displayText() == "":
            self.tab_dbn.lineEdit.setText(str(0.001))
        if self.tab_dbn.lineEdit_2.displayText() == "":
            self.tab_dbn.lineEdit_2.setText(str(0.001))
        if self.tab_dbn.slider_text1_2.displayText() == "":
            self.tab_dbn.slider_text1_2.setText(str(10))
        if self.tab_dbn.slider_text2.displayText() == "":
            self.tab_dbn.slider_text2.setText(str(100))
        if self.tab_dbn.slider_text1.displayText() == "":
            self.tab_dbn.slider_text1.setText(str(100))
    
    def check_val_cnn(self):
        if self.tab_cnn.lineEdit.displayText() == "":
            self.tab_cnn.lineEdit.setText(str(0.0002))
        if self.tab_cnn.lineEdit_2.displayText() == "":
            self.tab_cnn.lineEdit_2.setText(str(0.9))
        if self.tab_cnn.lineEdit_3.displayText() == "":
            self.tab_cnn.lineEdit_3.setText(str(100))
        if self.tab_cnn.slider_text1.displayText() == "":
            self.tab_cnn.slider_text1.setText(str(3000))
        if self.tab_cnn.slider_text2.displayText() == "":
            self.tab_cnn.slider_text2.setText(str(100))
        
    
    # 获取参数
    def get_val(self,tab):
        init_learning_rate = float(tab.lineEdit.displayText())
        decay_rate = float(tab.lineEdit_2.displayText())
        learning_rate_decay_steps = int(tab.lineEdit_3.displayText())
        n_epochs = int(int(tab.slider_text1.displayText()) / 100)
        batch_size = int(tab.slider_text2.displayText())
        return init_learning_rate, decay_rate, learning_rate_decay_steps, n_epochs, batch_size
    
    def get_val_dbn(self):
        rbm_lr = float(self.tab_dbn.lineEdit.displayText())
        rbm_epochs = int(self.tab_dbn.slider_text1_2.displayText())
        batch_size = int(self.tab_dbn.slider_text2.displayText())
        dbn_epochs = int(self.tab_dbn.slider_text1.displayText())
        dbn_lr = float(self.tab_dbn.lineEdit_2.displayText())
        return rbm_lr,rbm_epochs,batch_size,dbn_epochs,dbn_lr
    
    # 训练过程中禁止操作
    def nofocus(self,tab):
        tab.lineEdit.setReadOnly(True)
        tab.lineEdit_2.setReadOnly(True)
        if tab != self.tab_dbn:
            tab.lineEdit_3.setReadOnly(True)
        tab.slider_text1.setReadOnly(True)
        tab.slider_text2.setReadOnly(True)
        tab.horizontalSlider.setEnabled(False)
        tab.horizontalSlider2.setEnabled(False)
        tab.pushButton_train.setEnabled(False)
        if tab == self.tab_dbn:
            tab.slider_text1_2.setReadOnly(True)
            tab.horizontalSlider_2.setEnabled(False)
    
    #训练后允许操作
    def focus(self,tab):
        tab.lineEdit.setReadOnly(False)
        tab.lineEdit_2.setReadOnly(False)
        if tab != self.tab_dbn:
            tab.lineEdit_3.setReadOnly(False)
        tab.slider_text1.setReadOnly(False)
        tab.slider_text2.setReadOnly(False)
        tab.horizontalSlider.setEnabled(True)
        tab.horizontalSlider2.setEnabled(True)
        tab.pushButton_train.setEnabled(True)
        if tab == self.tab_dbn:
            tab.slider_text1_2.setReadOnly(False)
            tab.horizontalSlider_2.setEnabled(True)
    
    # 绘制准确率和损失曲线
    def draw_acc(self,layout,x,y1,y2):
        for i in range(layout.count()):
            layout.itemAt(i).widget().deleteLater()
        F = MyFigure(width=3, height=2, dpi=100)
        F.axes.plot(x, y1, color='#00AA72')
        F.axes.plot(x, y2, color='#FFA100')
        F.axes.legend(["train_acc","val_acc"])
        F.fig.suptitle("accuracy")
        layout.addWidget(F)
    
    def draw_loss(self,layout,x,y1,y2):
        for i in range(layout.count()):
            layout.itemAt(i).widget().deleteLater()
        F = MyFigure(width=3, height=2, dpi=100)
        F.axes.plot(x, y1, color='#00AA72')
        F.axes.plot(x, y2, color='#FFA100')
        F.axes.legend(["train_loss","val_loss"])
        F.fig.suptitle("loss")
        layout.addWidget(F)
    
    def draw_loss_dbn(self,layout,x,y):
        for i in range(layout.count()):
            layout.itemAt(i).widget().deleteLater()
        F = MyFigure(width=3, height=2, dpi=100)
        F.axes.plot(x, y, color='#00AA72')
        F.axes.legend(["pre_train_loss"])
        F.fig.suptitle("pre_train")
        layout.addWidget(F)
    
    def draw_acc_and_loss_dbn(self,layout,x,y1,y2,y3):
        for i in range(layout.count()):
            layout.itemAt(i).widget().deleteLater()
        F = MyFigure(width=3, height=2, dpi=100)
        F.axes.plot(x, y1, color='#00AA72')
        F.axes.plot(x, y2, color='#FFA100')
        F.axes.plot(x, y3, color='#0772A1')
        F.axes.legend(["loss","train_acc","val_acc"])
        F.fig.suptitle("tune_train")
        layout.addWidget(F)
    

    
    # 打印数据集数量
    def printlen(self,tab):
        tab.edit_acc.append("训练数据集个数: 5187")
        tab.edit_acc.append("验证数据集个数: 1297\n------")
        tab.edit_loss.append("训练数据集个数: 5187")
        tab.edit_loss.append("验证数据集个数: 1297\n------")

    
    # dnn 训练
    def dnntrain(self):
        self.tab_dnn.edit_acc.setHidden(False)
        self.tab_dnn.edit_loss.setHidden(False)
        self.tab_dnn.label_acc.setText('-')
        self.tab_dnn.label_recision.setText('-')
        self.tab_dnn.label_recall.setText('-')
        self.check_val()
        QApplication.processEvents()
        self.nofocus(self.tab_dnn)
        
        X_train_images, Y_train_labels, X_val_images, Y_val_labels, X_test_images, Y_test_labels = getdata()

        self.printlen(self.tab_dnn)
        QApplication.processEvents()

        init_learning_rate, decay_rate, learning_rate_decay_steps, n_epochs, batch_size = self.get_val(self.tab_dnn)

        init=tf.compat.v1.global_variables_initializer()
        saver=tf.compat.v1.train.Saver()

        accs = []
        costs = []
        val_accs = []
        val_costs = []

        with tf.compat.v1.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                learning_rate = Exponential_Decay(init_learning_rate, 100 * epoch, learning_rate_decay_steps, decay_rate) #学习率衰减函数
                acc_hand = []
                cost_hand = []
                for i in range(100):
                    X_batch,y_batch=next_batch(X_train_images,Y_train_labels,batch_size)
                    cost,acc,_ = sess.run([loss,accuracy,training_op],feed_dict={X:X_batch,y:y_batch})
                    acc_hand.append(acc)
                    cost_hand.append(cost)
                    QApplication.processEvents()

                cost = np.mean(acc_hand)
                cost = np.mean(cost_hand)
                val_cost,val_acc = sess.run([loss,accuracy],feed_dict={X:X_val_images,y:Y_val_labels})

                self.tab_dnn.edit_acc.append(str(epoch + 1) + " [×100]  train_acc: " + str('%.6f'%acc) + "  val_acc: " + str('%.6f'%val_acc))
                self.tab_dnn.edit_loss.append(str(epoch + 1) + " [×100]  train_loss: " + str('%.6f'%cost) + "  val_loss: " + str('%.6f'%val_cost))
                QApplication.processEvents()

                accs.append(acc)
                costs.append(cost)
                val_accs.append(val_acc)
                val_costs.append(val_cost)
                
            tp, tn, fp, fn = sess.run([tp_op, tn_op, fp_op, fn_op],feed_dict={X:X_test_images,y:Y_test_labels})
            test_acc = (tp+tn)/(tp+tn+fp+fn)
            test_recision = tp/(tp+fp)
            test_recall = tp/(tp+fn)
            self.tab_dnn.label_acc.setText(str('%.2f'%(test_acc*100))+'%')
            self.tab_dnn.label_recision.setText(str('%.2f'%(test_recision*100))+'%')
            self.tab_dnn.label_recall.setText(str('%.2f'%(test_recall*100))+'%')
            QApplication.processEvents()
            #print(test_acc, test_recision, test_recall)

        self.focus(self.tab_dnn)
        self.tab_dnn.edit_acc.setText("")
        self.tab_dnn.edit_loss.setText("")
        self.tab_dnn.edit_acc.setHidden(True)
        self.tab_dnn.edit_loss.setHidden(True)

        self.draw_acc(self.gridlayout1,list(range(len(accs))),accs,val_accs)
        self.draw_loss(self.gridlayout2,list(range(len(costs))),costs,val_costs)
        QApplication.processEvents()
    
    # 训练rnn
    def rnntrain(self):
        self.tab_rnn.edit_acc.setHidden(False)
        self.tab_rnn.edit_loss.setHidden(False)
        self.tab_rnn.label_acc.setText('-')
        self.tab_rnn.label_recision.setText('-')
        self.tab_rnn.label_recall.setText('-')
        self.check_val_rnn()
        QApplication.processEvents()
        self.nofocus(self.tab_rnn)

        train_data_set, train_labels, val_data_set, val_labels, test_data_set, test_labels = get_rnndata()

        self.printlen(self.tab_rnn)
        QApplication.processEvents()
        
        init_learning_rate, decay_rate, learning_rate_decay_steps, n_epochs, batch_size = self.get_val(self.tab_rnn)

        mynet = RNN(TIME_STEPS, INPUT_DIMS, OUTPUT_DIMS, NUM_UNITS, init_learning_rate)
        GlOBAL_STEP = 0 #全局训练次数
        loss_hundred = []
        acc_hundred = []
        val_loss_hundred = []
        val_acc_hundred = []
        for epoch in range(n_epochs):
            costs = []
            accs = []

            learning_rate = Exponential_Decay(init_learning_rate, 100 * epoch, learning_rate_decay_steps, decay_rate) #学习率衰减函数
            mynet.learning_rate = learning_rate

            for i in range(100):
                GlOBAL_STEP += 1

                X_batch, y_batch = next_batch(train_data_set, train_labels,batch_size)

                outputs = mynet.fit(X_batch, y_batch, batch_size, GlOBAL_STEP)

                cost = confidence_loss(y_batch, outputs)
                acc = compute_accuracy(y_batch, outputs)
                costs.append(cost)
                accs.append(acc)

                QApplication.processEvents()

            mean_loss = np.mean(costs)
            mean_acc = np.mean(accs)
            outputs = mynet.predict(val_data_set)
            val_acc = compute_accuracy(val_labels,outputs)
            val_loss = confidence_loss(val_labels,outputs)
            loss_hundred.append(mean_loss)
            acc_hundred.append(mean_acc)
            val_loss_hundred.append(val_loss)
            val_acc_hundred.append(val_acc)

            self.tab_rnn.edit_acc.append(str(epoch + 1) + " [×100]  train_acc: " + str('%.6f'%mean_acc) + "  val_acc: " + str('%.6f'%val_acc))
            self.tab_rnn.edit_loss.append(str(epoch + 1) + " [×100]  train_loss: " + str('%.6f'%mean_loss) + "  val_loss: " + str('%.6f'%val_loss))
            QApplication.processEvents()
        
        outputs = mynet.predict(test_data_set)
        test_acc = compute_accuracy(test_labels,outputs)
        test_recision, test_recall = compute_recision_and_recall(test_labels,outputs)
        self.tab_rnn.label_acc.setText(str('%.2f'%(test_acc*100))+'%')
        self.tab_rnn.label_recision.setText(str('%.2f'%(test_recision*100))+'%')
        self.tab_rnn.label_recall.setText(str('%.2f'%(test_recall*100))+'%')
        QApplication.processEvents()
        
        self.focus(self.tab_rnn)
        self.tab_rnn.edit_acc.setText("")
        self.tab_rnn.edit_loss.setText("")
        self.tab_rnn.edit_acc.setHidden(True)
        self.tab_rnn.edit_loss.setHidden(True)

        self.draw_acc(self.gridlayout1_rnn,list(range(len(acc_hundred))),acc_hundred,val_acc_hundred)
        self.draw_loss(self.gridlayout2_rnn,list(range(len(loss_hundred))),loss_hundred,val_loss_hundred)
        QApplication.processEvents()

    
    def cnntrain(self):
        self.tab_cnn.edit_acc.setHidden(False)
        self.tab_cnn.edit_loss.setHidden(False)
        self.check_val_cnn()
        QApplication.processEvents()
        self.nofocus(self.tab_cnn)

        train_data_set, train_labels, val_data_set, val_labels, test_data_set, test_labels = get_rnndata()

        self.printlen(self.tab_cnn)
        QApplication.processEvents()
        
        init_learning_rate, decay_rate, learning_rate_decay_steps, n_epochs, batch_size = self.get_val(self.tab_cnn)
        train_accuracys = []
        val_accuracys = []
        train_losses = []
        val_losses = []

        mynetwork =CNNModel()

        for epoch in range(n_epochs):  
            learning_rate = Exponential_Decay(init_learning_rate, 100 * epoch, learning_rate_decay_steps, decay_rate) #学习率衰减函数
            mynetwork.update_learning_rate(learning_rate)
            for i in range(2):
                X_batch,y_batch = next_batch(train_data_set,train_labels,batch_size)
                train_pre = onetrain(mynetwork,X_batch,y_batch)
                train_acc = compute_accuracy(y_batch,train_pre)
                train_loss = confidence_loss(y_batch,train_pre)
                self.tab_cnn.edit_acc.append(str(epoch * 2 + i) + "  train_acc: " + str('%.6f'%train_acc))
                self.tab_cnn.edit_loss.append(str(epoch * 2 + i) + "  train_loss: " + str('%.6f'%train_loss))
                QApplication.processEvents()
            
            val_pre = onetest(mynetwork,val_data_set)
            val_acc = compute_accuracy(val_labels,val_pre)
            val_loss = confidence_loss(val_labels,val_pre)
            self.tab_rnn.edit_acc.append(str(epoch * 2) + "  val_acc: " + str('%.6f'%val_acc)+"\n")
            self.tab_rnn.edit_loss.append(str(epoch * 2) + "  val_loss: " + str('%.6f'%val_loss)+"\n")
            QApplication.processEvents()

            train_accuracys.append(train_acc)
            train_losses.append(train_loss)
            val_accuracys.append(val_acc)
            val_losses.append(val_loss)
        
        self.focus(self.tab_cnn)
        self.tab_cnn.edit_acc.setText("")
        self.tab_cnn.edit_loss.setText("")
        self.tab_cnn.edit_acc.setHidden(True)
        self.tab_cnn.edit_loss.setHidden(True)

        self.draw_acc(self.gridlayout1_cnn,list(range(len(train_accuracys))),train_accuracys,val_accuracys)
        self.draw_loss(self.gridlayout2_cnn,list(range(len(train_losses))),train_losses,val_losses)
        QApplication.processEvents()
    
    def dbntrain2(self):
        self.tab_dbn.edit_acc.setHidden(False)
        self.tab_dbn.edit_loss.setHidden(False)
        self.tab_dbn.label_acc.setText('-')
        self.tab_dbn.label_recision.setText('-')
        self.tab_dbn.label_recall.setText('-')
        self.check_val_dbn()
        QApplication.processEvents()
        self.nofocus(self.tab_dbn)

        datasets,test_data_set,test_labels = get_data_dbn()
        x_dim=datasets[0].shape[1] 
        y_dim=datasets[1].shape[1] 
        tf.reset_default_graph()
        # Training

        rbm_lr,rbm_epochs,batch_size,dbn_epochs,dbn_lr = self.get_val_dbn()

        classifier = DBN(
                    hidden_act_func='sigmoid',
                    output_act_func='softmax',
                    loss_func='cross_entropy', # gauss 激活函数会自动转换为 mse 损失函数
                    struct=[x_dim, 800, 200, y_dim],
                    lr=dbn_lr,
                    momentum=0.5,
                    use_for='classification',
                    bp_algorithm='rmsp',
                    epochs=dbn_epochs,
                    batch_size=batch_size,
                    dropout=0.12,
                    units_type=['gauss','bin'],
                    rbm_lr=rbm_lr,
                    rbm_epochs=rbm_epochs,
                    cd_k=1)
        
        train_X, train_Y, test_X, test_Y = datasets   
        sess = tf.Session()
        sess.run(tf.global_variables_initializer()) # 初始化变量 

        #####################################################################
        #     开始逐层预训练 -------- < start pre-traning layer by layer>     #
        #####################################################################
        #print("Start Pre-training...")
        self.tab_dbn.edit_acc.append("Start Pre-training...")
        QApplication.processEvents()
        #pre_time_start=time.time()
        # >>> Pre-traning -> unsupervised_train_model
        #classifier.deep_feature = classifier.pt_model.train_model(train_X=train_X,train_Y=train_Y,sess=sess,summ=None)
        X = train_X 
        for i,rbm in enumerate(classifier.pt_model.pt_list):
            #print('>>> Train RBM-{}:'.format(i+1))
            self.tab_dbn.edit_acc.append('>>> Train RBM-{}:'.format(i+1))
            QApplication.processEvents()
            # 训练第i个RBM（按batch）
            #rbm.unsupervised_train_model(train_X=X,train_Y=train_Y,sess=sess,summ=None)
            labels = None
            _data=Batch(images=X,
                    labels=None,
                    batch_size=rbm.batch_size)
        
            b = int(X.shape[0]/rbm.batch_size)
        
            ########################################################
            #     开始训练 -------- < start traning for rbm/ae>     #
            ########################################################
        
            # 迭代次数
            pre_losses = []
            for ii in range(rbm.epochs):
                sum_loss=0
                for j in range(b):
                    batch_x = _data.next_batch()
                    loss,_=sess.run([rbm.loss,rbm.train_batch],feed_dict={
                            rbm.input_data: batch_x,
                            rbm.recon_data: batch_x})
                    sum_loss = sum_loss + loss
                    pre_losses.append('%.2f' % loss)
                    QApplication.processEvents()
                loss = sum_loss/b
                string = '>>> epoch = {}/{}  | 「Train」: loss = {:.4}'.format(ii+1,rbm.epochs,loss)
                #print('\r' + string)
                self.tab_dbn.edit_acc.append(string)
                QApplication.processEvents()
            self.tab_dbn.edit_acc.append(' ')
            QApplication.processEvents()
            # 得到transform值（train_X）
            X,_ = sess.run(rbm.transform(X))
        classifier.deep_feature = X

        self.draw_loss_dbn(self.gridlayout1_dbn,list(range(len(pre_losses))),pre_losses)
        QApplication.processEvents()
        self.tab_dbn.edit_acc.setText("")
        self.tab_dbn.edit_acc.setHidden(True)
        #pre_time_end=time.time()
        #classifier.pre_exp_time = pre_time_end-pre_time_start
        #print('>>> Pre-training expend time = {:.4}'.format(classifier.pre_exp_time))

        classifier.test_Y=test_Y 
        # 统计测试集各类样本总数
        classifier.stat_label_total()

        #######################################################
        #     开始微调 -------------- < start fine-tuning >    #
        #######################################################
        #print("Start Fine-tuning...")
        self.tab_dbn.edit_loss.append("Start Fine-tuning...")
        QApplication.processEvents()
        _data=Batch(images=train_X,
                    labels=train_Y,
                    batch_size=classifier.batch_size)

        b = int(train_X.shape[0]/classifier.batch_size)
        classifier.loss_and_acc=np.zeros((classifier.epochs,4))
        # 迭代次数
        time_start=time.time()
        tune_losses = []
        tune_accs = []
        tune_val_accs = []
        for i in range(classifier.epochs):
            sum_loss=0; sum_acc=0
            for j in range(b):
                batch_x, batch_y= _data.next_batch()
                loss,acc,_=sess.run([classifier.loss,classifier.accuracy,classifier.train_batch],feed_dict={
                        classifier.input_data: batch_x,
                        classifier.label_data: batch_y,
                        classifier.keep_prob: 1-classifier.dropout})
                sum_loss = sum_loss + loss; sum_acc= sum_acc +acc
                QApplication.processEvents()

            loss = sum_loss/b; acc = sum_acc/b
            tune_losses.append(loss)
            tune_accs.append(acc)

            classifier.loss_and_acc[i][0]=loss              # <0> 损失loss
            time_end=time.time()
            time_delta = time_end-time_start
            classifier.loss_and_acc[i][3]=time_delta        # <3> 耗时time
            classifier.loss_and_acc[i][1]=acc           # <1> 训练acc
            string = '>>> epoch = {}/{}  | 「Train」: loss = {:.4} , accuracy = {:.4}% '.format(i+1,classifier.epochs,loss,acc*100)

            acc=classifier.test_average_accuracy(test_X,test_Y,sess)
            tune_val_accs.append(acc)
            string = string + '  | 「Validate」: accuracy = {:.4}%'.format(acc*100)
            classifier.loss_and_acc[i][2]=acc       # <2> 测试acc

            #print('\r'+ string)
            self.tab_dbn.edit_loss.append('\r'+ string)
            QApplication.processEvents()
        
        test_acc,recalls=sess.run([classifier.accuracy,classifier.recall],feed_dict={
                                                classifier.input_data: test_data_set,
                                                classifier.label_data: test_labels,
                                                classifier.keep_prob: 1-classifier.dropout})
        #recalls:tp_op, fp_op, tn_op, fn_op
        test_recision = recalls[0]/(recalls[0]+recalls[1])
        test_recall = recalls[0]/(recalls[0]+recalls[3])
        self.tab_dbn.label_acc.setText(str('%.2f'%(test_acc*100))+'%')
        self.tab_dbn.label_recision.setText(str('%.2f'%(test_recision*100))+'%')
        self.tab_dbn.label_recall.setText(str('%.2f'%(test_recall*100))+'%')
        QApplication.processEvents()
        
        self.draw_acc_and_loss_dbn(self.gridlayout2_dbn,list(range(len(tune_accs))),tune_losses,tune_accs,tune_val_accs)
        QApplication.processEvents()

        print("-------oooo----------")
        sess.close()
        label_distribution = classifier.label_distribution

        self.tab_dbn.edit_loss.setText("")
        self.tab_dbn.edit_loss.setHidden(True)
        self.focus(self.tab_dbn)


        



def main():
    app = QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
