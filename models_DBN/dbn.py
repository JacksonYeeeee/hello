from rbm import RBM
from softmax import SoftMax
import h5py
import numpy as np
#from common import *

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

class DBN:
    def __init__(self,nlayers,hlen,ntype):
        '''
        nlayers -- 隐藏层数
        hlen -- 隐藏层结点数的向量
        ntype -- 输出维度
        '''
        self.rbm_layers = []
        self.nlayers = nlayers
        self.hlen = hlen[0]
        self.softmax_layer = None
        self.trainflag = False
        self.ntype = ntype
    def calcRBMForward(self,x):
        layerid = 1
        for rbm in self.rbm_layers:
            x = rbm.calc_forward(x)
            if layerid < self.nlayers:
                x = rbm.sample(x)
            layerid += 1
        return x
    
    
        
    def pretrainRBM(self,trainset, T, e): #T：rbm loop ; e:pre_learning_rate
        trainv = np.mat(trainset[1])   # 1x2500
        vlen = trainv.shape[1]
        trainnum = len(trainset)
        weights = []
        print("vlen = %d" %(vlen))
        print("Trainnum = %d" %(trainnum))
        for i in range(self.nlayers):   #nlayers轮预训练
            self.hlen = hlen[i]
            rbm = RBM(vlen,self.hlen)   #RBM
            if i == 0:
                traindata = trainset
            else:
                traindata = outdata
            outdata = np.zeros((trainnum,self.hlen))
            for j in range(trainnum):
                if j%50 == 0:
                    print("layer:%d CD sample %d..." %(i,j))
                trainv = np.mat(traindata[j])    #1*2500
                rbm.train_CD(trainv,T,e)
                outdata[j] = np.mat(rbm.sample(rbm.calc_forward(trainv)))   # 1xhlen
            self.rbm_layers.append(rbm)
            weights.append(rbm.W)
            vlen = self.hlen
#            hlen -= 100
        #dump_data("./models_DBN/pklfiles/dbn.pkl",weights)
        print("========= pretrainRBM complete ===========")
    
    def fineTune(self,trainset,labelset,MAXT,batch_size,landa,step=0.02):
        trainnum = len(trainset)
        '''if trainnum > 1000:
            trainnum = 1000'''
        print("Trainnum = %d" %(trainnum))
        rbm_output = np.zeros((trainnum,self.rbm_layers[-1].hsize))
        for i in range(trainnum):
            x = trainset[i]
            rbm_output[i] = self.calcRBMForward(x)   #rbm_output  0,1
        self.softmax = SoftMax(MAXT,batch_size,step,landa)
        self.softmax.process_train(rbm_output,labelset,self.ntype)
        print("======== fineTune Complete ===========")
        
    def predict(self,x):
        rbm_output = self.calcRBMForward(x)
        ptype = self.softmax.predict(rbm_output)
        return ptype
        
    def validate(self,testset,labelset):
        rate = 0
        testnum = len(testset)
        correctnum = 0
        for i in range(testnum):
            x = testset[i]
            testtype = self.predict(x)
            orgtype = labelset[i]
            print("Testype:%d\tOrgtype:%d" %(testtype,orgtype))
            if testtype == orgtype:
                correctnum += 1
        rate = float(correctnum)/testnum
        print("correctnum = %d, sumnum = %d" %(correctnum,testnum))
        print("Accuracy:%.2f" %(rate))
        return rate
        
###### main #########

ntype = 2
#nlayers = 3
#hlen = 500
hlen = [800,200,60]
nlayers = len(hlen)

pre_n_epochs = 1
pre_learning_rate = 0.01

tune_n_epochs = 800
tune_batch_size = 100
tune_learning_rate = 0.01




dbn = DBN(nlayers,hlen,ntype) 

h5f = h5py.File('./data/train.h5', 'r')
train_data_set = h5f['X'][:,:,:,0]
train_data_set = np.reshape(train_data_set,(len(train_data_set),-1))
train_labels = h5f['Y'][:,0]
train_labels = [int(x) for x in train_labels]

dbn.pretrainRBM(train_data_set,pre_n_epochs,pre_learning_rate) 


dbn.fineTune(train_data_set,train_labels,tune_n_epochs,tune_batch_size,tune_learning_rate,step=0.02) 

