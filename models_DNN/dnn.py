import sys ,os 
import numpy as np
import matplotlib.pyplot as plt
import h5py

batch_size = 100
n_epochs = 80 #百次
init_learning_rate = 0.0002
learning_rate_decay_steps = 80 #学习率衰减延迟
decay_rate = 1.1 #学习率衰减率

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

#交叉熵损失
def confidence_loss(labels,pre_labels):
    labels = np.array(labels)
    pre_labels = np.array(pre_labels)
    pre_labels = np.squeeze(pre_labels)
    epsilon = 1e-10
    pre_labels[pre_labels<=0.] = epsilon
    pre_labels[pre_labels>1.] = 1.
    loss = -np.mean(labels * np.log(pre_labels))
    return loss

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

def Exponential_Decay(INIT_LEARNING_RATE, GlOBAL_STEP, LEARNING_RATE_DECAY_STEPS, DECAY_RATE): #学习率衰减过程
    return INIT_LEARNING_RATE * (DECAY_RATE ** int(GlOBAL_STEP / LEARNING_RATE_DECAY_STEPS))

def sigmoid(z):
    s = 1 / (1+np.exp(-z))
    return s 

class Net(object):
 
    def __init__(self,dims,learning_rate):
        '''arguments:
            dims -- 各层节点数 如 [50*50,1024,512,256,64,2]
            learning_rate -- 学习率'''
        self.dims = dims
        self.learning_rate = learning_rate
        self.parameters = {} #W和b的dict
        self.cache = {} #A的dict 即各层输出
        self.grads = {} #DW和db的dict

        self.initilize_params()
    
    def initilize_params(self) :  #[2500,1024,512,256,64,2]
        L = len(self.dims)
        for i in range(1,L):
            self.parameters["W"+str(i)] = np.zeros((self.dims[i-1],self.dims[i]))
            self.parameters["b"+str(i)] = np.zeros((self.dims[i],1))
    
    def propagate(self,X,Y): 
        L = len(self.parameters) // 2
        m = X.shape[1]
        #前向传播 START 计算损失
        #caculate W^T*X +b
        A = X
        for i in range(1,L+1):
        
            A = np.dot(self.parameters["W"+str(i)].T,A) + self.parameters["b"+str(i)]
        
            A = sigmoid(A)
            self.cache["A"+str(i)] = A

        # 前向传播 END
    
        # 反向传播 START
        dZ = self.cache["A" + str(L)] - Y
        for i in range(1, L):
            dw = np.dot(self.cache["A" + str(L - i)],dZ.T)/m
            db = np.sum((dZ),axis = 1)/m
            db = db.reshape(self.parameters["b" + str(L - i + 1)].shape)
            dZ = (np.dot(self.parameters["W" + str(L - i + 1)], dZ)) * np.where( \
                    self.cache["A" + str(L - i)] > 0, 1, 0)
            self.grads["dW"+str(L-i)] = dw #下标从0开始
            self.grads["db"+str(L-i)] = db
        dw = np.dot(X,dZ.T)/m
        db = np.sum((dZ),axis = 1)/m
        db = db.reshape(self.parameters["b" + str(L - i)].shape)
        self.grads["dW"+str(L-i-1)] = dw #下标从0开始
        self.grads["db"+str(L-i-1)] = db

        return A
    

    # 优化
    def optimize(self,X,Y):

        outputs = self.propagate(X,Y)
        L = len(self.parameters) // 2
        for j in range(1,L+1):
            dw = self.grads["dW"+str(j-1)]
            db = self.grads["db"+str(j-1)]
            self.parameters["W"+str(j)] = self.parameters["W"+str(j)] - self.learning_rate*dw
            self.parameters["b"+str(j)] = self.parameters["b"+str(j)] - self.learning_rate*db  
        return outputs
    
    # 预测
    def predicate (self,X ):
        m = X.shape[1]
        L = len(self.parameters) // 2
        A = X
        for i in range(1,L+1):
            A = np.dot(self.parameters["W"+str(i)].T,A) + self.parameters["b"+str(i)]
            A = sigmoid(A)
        return A
    
    def fit(self,X,Y):
        self.propagate(X,Y)
        outputs = self.optimize(X,Y)
        return outputs


def main():
    h5f = h5py.File('./data/train.h5', 'r')
    train_data_set = h5f['X'][:,:,:,0]
    train_labels = h5f['Y']
    train_data_set = np.reshape(train_data_set,(train_data_set.shape[0],-1))

    h5f2 = h5py.File('./data/val.h5', 'r')
    val_data_set = h5f2['X'][:,:,:,0]
    val_labels = h5f2['Y']
    val_data_set = np.reshape(val_data_set,(val_data_set.shape[0],-1))

    h5f3 = h5py.File('./data/test.h5', 'r')
    test_data_set = h5f3['X'][0:50,:,:,0]
    test_labels = h5f3['Y'][0:50]
    test_data_set = np.reshape(test_data_set,(test_data_set.shape[0],-1))

    print('样本数据集的个数：%d' % train_data_set.shape[1])
    print('验证数据集的个数：%d' % val_data_set.shape[1])
    print('测试数据集的个数：%d\n\n' % test_data_set.shape[1])

    dims = [50*50,1024,512,256,64,2]
    mynet = Net(dims,init_learning_rate)
    
    loss_hundred = []
    for epoch in range(n_epochs):
        costs = []
        accs = []

        learning_rate = Exponential_Decay(init_learning_rate, 100 * epoch, learning_rate_decay_steps, decay_rate) #学习率衰减函数
        mynet.learning_rate = learning_rate

        for i in range(100):
            X_batch, y_batch = next_batch(train_data_set, train_labels,batch_size)
            X_batch = np.transpose(X_batch,(1,0))
            y_batch = np.transpose(y_batch,(1,0))

            outputs = mynet.fit(X_batch,y_batch)

            cost = confidence_loss(y_batch.T, outputs.T)
            acc = accuracy(y_batch.T, outputs.T)
            costs.append(cost)
            accs.append(acc)

        mean_loss = np.mean(costs)
        mean_acc = np.mean(accs)
        print("Step {} [x100]  Accuracy {} Loss {}".format(epoch + 1, mean_acc, mean_loss)) #输出每一百步的loss的平均值
        loss_hundred.append(mean_loss)

        if (epoch+1) % 10 == 0:
            outputs = mynet.predicate(np.array(val_data_set).T)
            val_acc = accuracy(val_labels,outputs.T)
            val_loss = confidence_loss(val_labels,outputs.T)
            print("Step {} [x100]  Validate Accuracy {} Validate Loss {} \n\n".format(epoch + 1, val_acc, val_loss)) #输出每一百步的loss的平均值

    plt.plot(loss_hundred)
    plt.xlabel("iterations/hundreds")
    plt.ylabel("costs")
    plt.show()

if __name__ == "__main__":
    main()