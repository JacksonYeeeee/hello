import tensorflow as tf
import numpy as np
import time
np.random.seed(1337)  # for reproducibility

import os
import sys
sys.path.append("./models_DBN2/models")
sys.path.append("./models_DBN2/base")
filename = os.path.basename(__file__)

from dbn import DBN
from base_func import run_sess,tSNE_2d,Batch
import h5py

# Loading dataset

h5f = h5py.File('./data/train.h5', 'r')
train_data_set = h5f['X'][:,:,:,0]
train_data_set = np.reshape(train_data_set,(len(train_data_set),-1))
train_labels = h5f['Y']
train_labels = np.array(train_labels)
h5f2 = h5py.File('./data/val.h5', 'r')
val_data_set = h5f2['X'][:,:,:,0]
val_data_set = np.reshape(val_data_set,(len(val_data_set),-1))
val_labels = h5f2['Y']
val_labels = np.array(val_labels)

datasets = [train_data_set,train_labels,val_data_set,val_labels]

x_dim=datasets[0].shape[1] 
y_dim=datasets[1].shape[1] 
p_dim=int(np.sqrt(x_dim))

tf.reset_default_graph()
# Training
classifier = DBN(
             hidden_act_func='sigmoid',
             output_act_func='softmax',
             loss_func='cross_entropy', # gauss 激活函数会自动转换为 mse 损失函数
             struct=[x_dim, 800, 200, y_dim],
             lr=1e-3,
             momentum=0.5,
             use_for='classification',
             bp_algorithm='rmsp',
             epochs=10,
             batch_size=32,
             dropout=0.12,
             units_type=['gauss','bin'],
             rbm_lr=1e-3,
             rbm_epochs=2,
             cd_k=1)

'''run_sess(classifier,datasets,filename,load_saver='./models_DBN2/')
label_distribution = classifier.label_distribution'''

train_X, train_Y, test_X, test_Y = datasets   
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 初始化变量 

#####################################################################
#     开始逐层预训练 -------- < start pre-traning layer by layer>     #
#####################################################################
print("Start Pre-training...")
#pre_time_start=time.time()
# >>> Pre-traning -> unsupervised_train_model
#classifier.deep_feature = classifier.pt_model.train_model(train_X=train_X,train_Y=train_Y,sess=sess,summ=None)
X = train_X 
for i,rbm in enumerate(classifier.pt_model.pt_list):
    print('>>> Train RBM-{}:'.format(i+1))
    # 训练第i个RBM（按batch）
    rbm.unsupervised_train_model(train_X=X,train_Y=train_Y,sess=sess,summ=None)
    # 得到transform值（train_X）
    X,_ = sess.run(rbm.transform(X))
classifier.deep_feature = X
#pre_time_end=time.time()
#classifier.pre_exp_time = pre_time_end-pre_time_start
#print('>>> Pre-training expend time = {:.4}'.format(classifier.pre_exp_time))

classifier.test_Y=test_Y 
# 统计测试集各类样本总数
classifier.stat_label_total()

#######################################################
#     开始微调 -------------- < start fine-tuning >    #
#######################################################
print("Start Fine-tuning...")
_data=Batch(images=train_X,
            labels=train_Y,
            batch_size=classifier.batch_size)
            
b = int(train_X.shape[0]/classifier.batch_size)
classifier.loss_and_acc=np.zeros((classifier.epochs,4))
# 迭代次数
time_start=time.time()
for i in range(classifier.epochs):
    sum_loss=0; sum_acc=0
    for j in range(b):
        batch_x, batch_y= _data.next_batch()
        loss,acc,_=sess.run([classifier.loss,classifier.accuracy,classifier.train_batch],feed_dict={
                classifier.input_data: batch_x,
                classifier.label_data: batch_y,
                classifier.keep_prob: 1-classifier.dropout})
        sum_loss = sum_loss + loss; sum_acc= sum_acc +acc
    
    loss = sum_loss/b; acc = sum_acc/b
                
    classifier.loss_and_acc[i][0]=loss              # <0> 损失loss
    time_end=time.time()
    time_delta = time_end-time_start
    classifier.loss_and_acc[i][3]=time_delta        # <3> 耗时time
    classifier.loss_and_acc[i][1]=acc           # <1> 训练acc
    string = '>>> epoch = {}/{}  | 「Train」: loss = {:.4} , accuracy = {:.4}% , expend time = {:.4}'.format(i+1,classifier.epochs,loss,acc*100,time_delta)

    acc=classifier.test_average_accuracy(test_X,test_Y,sess)
    string = string + '  | 「Test」: accuracy = {:.4}%'.format(acc*100)
    classifier.loss_and_acc[i][2]=acc       # <2> 测试acc

    print('\r'+ string)
print("-------oooo----------")

sess.close()

label_distribution = classifier.label_distribution

