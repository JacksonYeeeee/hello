import numpy as np
#from common import *
import sys
#import matplotlib.pyplot as plt

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

class SoftMax:
	def __init__(self,STOPNUM,batch_size,step,landa):   #x,y => np.mat()
		self.STOPNUM = STOPNUM
		self.batch_size = batch_size
		self.step = step
		self.landa = landa
		
	def load_theta(self,datapath):
		self.theta = cPickle.load(open(datapath,'rb'))
	
	def train(self,datapath,typenum):
		trainnum,MaxTrainNum = 0 , 2000
		train_set,valid_set,test_set = load_data(datapath)
		x,y = train_set[0],train_set[1]
		trainnum = len(x)
		if(trainnum > MaxTrainNum):
			trainnum = MaxTrainNum
		self.process_train(x[:trainnum],y[:trainnum],typenum)
	
	def process_train(self,x,y,typenum):                           # x =>(trainnum x n)     y => (trainnum x 1)
		'''xtypenum = np.zeros(typenum)'''
		costval = np.zeros(self.STOPNUM)
		'''for val in y:
			xtypenum[val]+=1
		print(xtypenum)'''
		trainnum = x.shape[0]
		'''bias = np.mat(np.ones(trainnum))'''
#		x = np.concatenate((bias.T,x),axis=1)                     # x => (trainnum x n)
		featurenum = x.shape[1]
		print("Trainnum = %d, featurenum = %d" %(trainnum,featurenum))    #featurenum = n+1(bias)
		self.theta = 0.001*np.mat(np.random.randn(typenum,featurenum))
#		self.theta[0] = np.ones(featurenum);
		lastcostJ = 1000
		for m in range(self.STOPNUM):

			x_batch, y_batch = next_batch(x, y, self.batch_size)
			trainnum = len(x_batch)

			############ Loop #############
			costs = np.zeros((typenum,trainnum))
			grads = np.zeros((typenum,featurenum))
			err = 0
			for j in range(typenum):
				jvalues = np.zeros((trainnum,featurenum))
				for i in range(trainnum):
					datas = np.zeros(typenum)
					hval = self.h(x_batch[i])
					if int(round(hval[0,0])) != y_batch[0]:
						err += 1

					ptype = hval[0,j]
					delta = -ptype

					if j == y_batch[i]:
						delta = 1-ptype
						costs[j,i] = np.log(ptype)
					jvalue = np.multiply(x_batch[i],delta)   #(1xn)
					jvalues[i] = jvalue
				grads[j] = -np.mean(jvalues,axis=0)+self.landa * self.theta[j] #gradJ => (1xn)
			
			for k in range(typenum):
				self.theta[k] = self.theta[k] - self.step*grads[k]
			costJ = -np.sum(costs)/trainnum +(self.landa/2)*np.sum(np.square(self.theta))
			costval[m] = costJ

			if(costJ > lastcostJ):
				print("costJ is increasing !!!")
				break
			print("Loop(%d) accuracy = %.6f cost = %.3f diff=%.4f" %(m, (err/2)/trainnum, costJ,costJ-lastcostJ))
			lastcostJ = costJ
		#dump_data("data/softmax.pkl",self.theta)
				
	def h(self,x):                #  x=>(1xn)
		m = np.exp(np.dot(np.mat(x),self.theta.T))   #   e(thetaT*x)      1xn * nxk  
		sump = np.sum(m)
		ret = m/sump
		return ret
		
	def predict(self,x):
		pv = self.h(np.mat(x))
		return np.argmax(pv)               # return predict type with max p(y|x)
		
	def test(self,datapath,typenum):
		train_set,valid_set,test_set = load_data(datapath)
		x,y = test_set[0],test_set[1]
		testnum = 1000
		x = x[:testnum]
		y = y[:testnum]
		#x,y = load_minst_data(datapath)
		testnum = len(x)
		bias = np.mat(np.ones(testnum))
#		x = np.concatenate((bias.T,x),axis=1)
		rightnum = 0
		corrects=np.zeros(typenum)
		print("Test sample number:%d" %(testnum))
		for i in range(testnum):
			type = softmax.predict(x[i])	
			if(y[i] == type):
				corrects[type] += 1
				rightnum += 1
		rate = float(rightnum)/testnum
		print(corrects)
		print("Accuracy rate = %.4f,rightnum = %d" %(rate,rightnum))



