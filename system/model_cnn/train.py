import numpy as np
from model_cnn import Activators, CNN, DNN
'''import model_cnn.Activators  
import model_cnn.CNN   
import model_cnn.DNN ''' 

init_learning_rate = 0.0002

#一次训练
def onetrain(net,X,y):
    pre_labels=[]
    for k in range(len(X)):
        onepic = X[k]
        onepic = np.array([onepic])
        result = net.forward(onepic)
        pre_labels.append(result)
        labels = y[k].reshape(-1,1)
        net.backward(onepic,labels)
    return pre_labels

#一次测试
def onetest(net,X):
    pre_labels=[]
    for k in range(X.shape[0]):
        onepic = X[k]
        onepic = np.array([onepic])
        result = net.forward(onepic)
        pre_labels.append(result)
    return pre_labels

# 网络模型类
class CNNModel():
    # =============================构造网络结构=============================
    def __init__(self):
        # 初始化构造卷积层：输入宽度、输入高度、通道数、滤波器宽度、滤波器高度、滤波器数目、补零数目、步长、激活器、学习速率
        self.cl1 = CNN.ConvLayer(50, 50, 1, 5, 5, 6, 0, 1, Activators.ReluActivator(),init_learning_rate)  
        self.pl1 = CNN.MaxPoolingLayer(46, 46, 6, 2, 2, 2)  
        self.cl2 = CNN.ConvLayer(23, 23, 6, 5, 5, 12, 0, 1, Activators.ReluActivator(),init_learning_rate)  
        self.pl2 = CNN.MaxPoolingLayer(19, 19, 12, 2, 2, 2) 
        self.fl1 = DNN.FullConnectedLayer(972, 2, Activators.SoftmaxActivator(),init_learning_rate) 

    def update_learning_rate(self,learning_rate):
        self.cl1.learning_rate=learning_rate
        self.cl2.learning_rate=learning_rate
        self.fl1.learning_rate=learning_rate

    # 根据输入计算一次输出。因为卷积层要求的数据要求有通道数，所以onepic是一个包含深度，高度，宽度的多维矩阵
    def forward(self,onepic):  
        # print('图片：',onepic.shape)
        self.cl1.forward(onepic)      
        # print('第一层卷积结果：',self.cl1.output_array.shape)
        self.pl1.forward(self.cl1.output_array)  
        # print('第一层采样结果：',self.pl1.output_array.shape)
        self.cl2.forward(self.pl1.output_array) 
        # print('第二层卷积结果：',self.cl2.output_array.shape)
        self.pl2.forward(self.cl2.output_array)
        # print('第二层采样结果：',self.pl2.output_array.shape)
        flinput = self.pl2.output_array.flatten().reshape(-1, 1)  
        # print(flinput.shape)
        self.fl1.forward(flinput)
        # print('全连接层结果：',self.fl1.output)
        return  self.fl1.output   
  
    def backward(self,onepic,labels):
        # 计算误差
        delta =  (labels - self.fl1.output)*(1-self.fl1.output**2)

        # 反向传播
        self.fl1.backward(delta)  
        self.fl1.update()  
        # print('全连接层输入误差：', self.fl1.delta.shape)
        sensitivity_array = self.fl1.delta.reshape(self.pl2.output_array.shape)  
        self.pl2.backward(self.cl2.output_array, sensitivity_array)  
        # print('第二采样层的输入误差：', self.pl2.delta_array.shape)
        self.cl2.backward(self.pl1.output_array, self.pl2.delta_array,Activators.ReluActivator())  
        self.cl2.update()        
        # print('第二卷积层的输入误差：', self.cl2.delta_array.shape)
        self.pl1.backward(self.cl1.output_array, self.cl2.delta_array)  
        # print('第一采样层的输入误差：', self.pl1.delta_array.shape)
        self.cl1.backward(onepic, self.pl1.delta_array,Activators.ReluActivator())  
        self.cl1.update()  
        # print('第一卷积层的输入误差：', self.cl1.delta_array.shape)


   

