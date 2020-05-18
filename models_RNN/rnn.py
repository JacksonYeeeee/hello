import warnings
#warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt 
import h5py

TIME_STEPS = 50 #时间步长
INPUT_DIMS = 50 #输入维度
OUTPUT_DIMS = 2 #输出维度
NUM_UNITS = 128 #神经单元数目

init_learning_rate = 0.0002 #初始学习率
learning_rate_decay_steps = 100 #学习率衰减延迟
decay_rate = 0.99 #学习率衰减率
batch_size = 96 #单次训练所用的数据组数
n_epochs = 100 #训练次数（百次）

B1 = 0.9 #B1，B2，eps为Adam下降所用参数
B2 = 0.999
eps = 1e-8

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


def Relu(x): #Relu函数
    return (x > 0) * x

def Drelu(x): #Relu导函数
    return (x > 0)

def SoftMax(weighted_input):  # 前向计算，计算输出
    return np.exp(weighted_input)/(np.exp(weighted_input).sum())

def DSoftMax(output):  # 后向计算，计算导数
    return np.multiply(output, (1 - output))

class RNN(object):
    def __init__(self,time_steps, input_dims, output_dims, num_units, learning_rate):
        '''agument:
            time_steps -- 时间步长
            input_dims -- 输入维度
            output_dims -- 输出维度
            num_units -- 神经单元数目'''
        
        self.time_steps = time_steps
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_units = num_units

        self.learning_rate = learning_rate

        self.Win, self.Wrec, self.Wout, self.rbias, self.obias = self.Initializer() 
        self.mWin, self.mWrec, self.mWout, self.mrbias, self.mobias, self.vWin, self.vWrec, self.vWout, self.vrbias, self.vobias = self.Adaminitializer()
    
    def Initializer(self): #各矩阵初始化方法
        rbias = np.zeros((self.num_units))
        obias = np.full((self.output_dims), 0.1)
        Wout = np.zeros((self.num_units, self.output_dims))
        Win = np.zeros((self.input_dims, self.num_units))
        Wrec = np.zeros((self.num_units, self.num_units))
        limit = np.sqrt(6 / (self.output_dims + self.num_units)) 
        Wout += np.random.uniform(- limit, limit, (self.num_units, self.output_dims)) #这个是tensorflow内部的训练参数初始化方法，也就是一个和数组内数据各维度数有关的一个初始，可以使得下降过程更为稳定。
        limit = np.sqrt(6 / (self.input_dims + self.num_units + self.num_units))
        W = np.random.uniform(- limit, limit, (self.input_dims + self.num_units, self.num_units))
        Win += W [0 : self.input_dims , :]
        Wrec += W[self.input_dims : (self.num_units + self.input_dims), :]
        return Win, Wrec, Wout, rbias, obias
    
    def Adaminitializer(self): #Adam下降参数初始化
        mWin = np.zeros((self.input_dims, self.num_units))
        mrbias = np.zeros((self.num_units))
        mobias = 0
        mWrec = np.zeros((self.num_units, self.num_units))
        mWout = np.zeros((self.num_units, self.output_dims))
        vWin = np.zeros((self.input_dims, self.num_units))
        vrbias = np.zeros((self.num_units))
        vobias = 0
        vWrec = np.zeros((self.num_units, self.num_units))
        vWout = np.zeros((self.num_units, self.output_dims))
        return mWin, mWrec, mWout, mrbias, mobias, vWin, vWrec, vWout, vrbias, vobias
    
    def Forward_Propagation(self, inputs): #简单RNN的正向传播
        inputs = np.array(inputs)
        h = np.zeros((len(inputs), self.time_steps, self.num_units))
        outputs = np.zeros((len(inputs), self.output_dims))
        h[:,0] = Relu(np.dot(inputs[:,0],self.Win) + self.rbias)
        for t in range(self.time_steps - 1):
            h[:,t + 1] = Relu(np.dot(inputs[:,t + 1],self.Win) + np.dot(h[:,t],self.Wrec) + self.rbias) #h(t + 1) = input(t + 1) * Win + h(t) * Wrec + rbias
        outputs = np.dot(h[:,self.time_steps - 1], self.Wout) + self.obias #output = h(last) * Wout + obias
        return outputs, h
    
    def Back_Propagation(self,inputs, outputs, targets, hidden_state, batch_size): #基于MSE损失函数的梯度的BPTT反向传播
        inputs = np.array(inputs)
        dWin = np.zeros((self.input_dims, self.num_units))
        dWrec = np.zeros((self.num_units, self.num_units))
        drbias = np.zeros((self.num_units))
        dWout = np.zeros((self.num_units, self.output_dims))
        dobias = 0
        dh = np.zeros((len(inputs), self.num_units))
        dobias = np.sum(2 * (outputs - targets) / batch_size) #do = 2 * (output - target)，这个是MSE的导数，dobias = do * 1
        dWout = np.dot(hidden_state[:,self.time_steps - 1].T, 2 * (outputs - targets) / batch_size) #dWout = h(last).T * do
        dh = np.dot(2 * (outputs - targets) / batch_size, self.Wout.T) #dh = do * Wout.T
        for t in range(self.time_steps - 1):
            dh = dh * Drelu(np.dot(inputs[:, self.time_steps - 1 - t],self.Win) + np.dot(hidden_state[:, self.time_steps - 2 - t],self.Wrec) + self.rbias) #将一次循环上的偏导乘上上一次循环的激活函数的导数
            drbias += np.sum(dh, axis = 0) #drbias的BPTT
            dWrec += np.dot(hidden_state[:, self.time_steps - 2 - t].T, dh) #dWrec的BPTT
            dWin += np.dot(inputs[:, self.time_steps - 1 - t].T, dh) #dWin的BPTT
            dh = np.dot(dh, self.Wrec.T) #计算传递到上一次循环上的偏导
        drbias += np.sum(dh * Drelu(np.dot(inputs[:, 0],self.Win) + self.rbias), axis = 0) #起始循环的rbias偏导
        dWin  += np.dot(inputs[:, 0].T,dh * Drelu(np.dot(inputs[:, 0], self.Win) + self.rbias)) #起始循环的的Win偏导
        return dWin, dWrec, dWout, drbias, dobias
    
    def AdamUpdate(self, W, dW, mW, vW, GlOBAL_STEP): #Adam下降方法
        M_LEARNING_RATE = self.learning_rate * np.sqrt(1 - B2 ** (GlOBAL_STEP + 1)) / (1 - B1 ** (GlOBAL_STEP + 1))
        mW = B1 * mW + (1 - B1) * dW
        vW = B2 * vW + (1 - B2) * (dW * dW)
        W -= M_LEARNING_RATE * mW / (np.sqrt(vW) + eps) #Adam下降方法，具体方法请参见Adam下降方法的论文
        return W, mW, vW
    
    def predict(self, X):
        outputs, _= self.Forward_Propagation(X) #前传过程
        return outputs
    
    def fit(self, X, Y, batch_size, GLOBAL_STEP):
        outputs, hidden_state= self.Forward_Propagation(X) #前传过程
        self.dWin, self.dWrec, self.dWout, self.drbias, self.dobias = self.Back_Propagation(X, outputs, Y, hidden_state, batch_size) #反传过程
        self.Win, self.mWin, self.vWin = self.AdamUpdate(self.Win, self.dWin, self.mWin, self.vWin, GLOBAL_STEP) #对每个矩阵进行Adam下降
        self.Wrec, self.mWrec, self.vWrec = self.AdamUpdate(self.Wrec, self.dWrec, self.mWrec, self.vWrec, GLOBAL_STEP)
        self.Wout, self.mWout, self.vWout = self.AdamUpdate(self.Wout, self.dWout, self.mWout, self.vWout, GLOBAL_STEP)
        self.rbias, self.mrbias, self.vrbias = self.AdamUpdate(self.rbias, self.drbias, self.mrbias, self.vrbias, GLOBAL_STEP)
        self.obias, self.mobias, self.vobias = self.AdamUpdate(self.obias, self.dobias, self.mobias, self.vobias, GLOBAL_STEP)
        return outputs


def main():
    h5f = h5py.File('./data/train.h5', 'r')
    train_data_set = h5f['X'][:,:,:,0]
    train_labels = h5f['Y']

    h5f2 = h5py.File('./data/val.h5', 'r')
    val_data_set = h5f2['X'][:,:,:,0]
    val_labels = h5f2['Y']

    print('样本数据集的个数：%d' % train_data_set.shape[0])
    print('验证数据集的个数：%d' % val_data_set.shape[0])

    mynet = RNN(TIME_STEPS, INPUT_DIMS, OUTPUT_DIMS, NUM_UNITS, init_learning_rate)
    GlOBAL_STEP = 0 #全局训练次数
    loss_hundred = []
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
            acc = accuracy(y_batch, outputs)
            costs.append(cost)
            accs.append(acc)

        mean_loss = np.mean(costs)
        mean_acc = np.mean(accs)
        print("Step {} [x100]  Accuracy {} Loss {}".format(epoch + 1, mean_acc, mean_loss)) #输出每一百步的loss的平均值
        loss_hundred.append(mean_loss)

        if (epoch+1) % 10 == 0:
            outputs = mynet.predict(val_data_set)
            val_acc = accuracy(val_labels,outputs)
            val_loss = confidence_loss(val_labels,outputs)
            print("Step {} [x100]  Validate Accuracy {} Validate Loss {} \n\n".format(epoch + 1, val_acc, val_loss)) #输出每一百步的loss的平均值

    plt.plot(loss_hundred)
    plt.xlabel("iterations/hundreds")
    plt.ylabel("costs")
    plt.show()

if __name__ == "__main__":
    main()

