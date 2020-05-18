import matplotlib.pyplot as plt
import datetime
import numpy as np
import Activators  
import CNN   
import DNN  
import h5py

init_learning_rate = 0.0002
n_epochs=30
batch_size=96

learning_rate_decay_steps = 100 #学习率衰减延迟
decay_rate = 0.99 #学习率衰减率

train_accuracys = []
val_accuracys = []
train_losses = []
val_losses = []

#获取子数据
def next_batch(train_data, train_target, batch_size):  
    index = [ i for i in range(0,len(train_target)) ]  
    np.random.shuffle(index);  
    batch_data = []; 
    batch_target = [];  
    for i in range(0,batch_size):  
        batch_data.append(train_data[index[i]]);  
        batch_target.append(train_target[index[i]])  
    return batch_data, batch_target

#准确率
def accuracy(labels,pre_labels):
    n = len(labels)
    right = 0
    for i in range(n):
        real_type = labels[i].argmax()
        pre_type = pre_labels[i].argmax()
        if real_type==pre_type:
            right+=1
    acc = right/n
    return acc

#交叉熵损失
def confidence_loss(labels,pre_labels):
    labels = np.array(labels)
    pre_labels = np.array(pre_labels)
    pre_labels = np.squeeze(pre_labels)
    epsilon = 1e-10
    pre_labels[pre_labels<0.] = epsilon
    pre_labels[pre_labels>1.] = 1.
    loss = -np.mean(labels * np.log(pre_labels))
    return loss

def Exponential_Decay(INIT_LEARNING_RATE, GlOBAL_STEP, LEARNING_RATE_DECAY_STEPS, DECAY_RATE): #学习率衰减过程
    return INIT_LEARNING_RATE * (DECAY_RATE ** int(GlOBAL_STEP / LEARNING_RATE_DECAY_STEPS))

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



# 由于使用了逻辑回归函数，所以只能进行分类识别。识别ont-hot编码的结果
if __name__ == '__main__':

    # =============================加载数据集=============================
    h5f = h5py.File('./data/train.h5', 'r')
    train_data_set = h5f['X'][:,:,:,0]
    train_labels = h5f['Y']

    h5f2 = h5py.File('./data/val.h5', 'r')
    val_data_set = h5f2['X'][:,:,:,0]
    val_labels = h5f2['Y']

    h5f3 = h5py.File('./data/test.h5', 'r')
    test_data_set = h5f3['X'][:,:,:,0]
    test_labels = h5f3['Y']

    print('样本数据集的个数：%d' % len(train_data_set))
    print('验证数据集的个数：%d' % len(val_data_set))
    print('测试数据集的个数：%d' % len(test_data_set))


    # =============================构造网络结构=============================
    mynetwork =CNNModel()

    # =============================迭代训练=============================
    for epoch in range(n_epochs):  
        learning_rate = Exponential_Decay(init_learning_rate, 100 * epoch, learning_rate_decay_steps, decay_rate) #学习率衰减函数
        mynetwork.update_learning_rate(learning_rate)
        for i in range(100):
            X_batch,y_batch = next_batch(train_data_set,train_labels,batch_size)
            train_pre = onetrain(mynetwork,X_batch,y_batch)
            train_acc = accuracy(y_batch,train_pre)
            train_loss = confidence_loss(y_batch,train_pre)
            print(epoch,i,' Train Accuracy: ',train_acc,' Train Loss: ',train_loss)
            train_accuracys.append(train_acc)
            train_losses.append(train_loss)
        val_pre = onetest(mynetwork,val_data_set)
        val_acc = accuracy(val_labels,val_pre)
        val_loss = confidence_loss(val_labels,val_pre)
        print(epoch,' Validate Accuracy: ',val_acc,' Vlidate Loss: ',val_loss,'\n\n')
        val_accuracys.append(val_acc)
        val_losses.append(val_loss)
    
    print("-----------Train over-------------\n")

# =============================评估结果=============================

    test_pre = onetest(mynetwork,test_data_set)
    test_acc = accuracy(test_labels,test_pre)
    print('\nTest Accuracy: ',test_acc)
    file=open('./models_CNN/txtfiles/train_acc.txt','w') 
    file.write(str(train_accuracys)); 
    file.close()  
    file=open('./models_CNN/txtfiles/val_acc.txt','w') 
    file.write(str(val_accuracys)); 
    file.close() 
    file=open('./models_CNN/txtfiles/train_loss.txt','w') 
    file.write(str(train_losses)); 
    file.close() 
    file=open('./models_CNN/txtfiles/val_loss.txt','w') 
    file.write(str(val_losses)); 
    file.close()    

