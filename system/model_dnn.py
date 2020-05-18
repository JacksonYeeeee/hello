import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import h5py
import numpy as np

#读取数据集
def getdata():
    h5f = h5py.File('./data/train.h5', 'r')
    X_train_images = h5f['X'][:,:,:,0]
    Y_train_labels = h5f['Y'][:,0]

    h5f2 = h5py.File('./data/val.h5', 'r')
    X_val_images = h5f2['X'][:,:,:,0]
    Y_val_labels = h5f2['Y'][:,0]

    X_train_images=np.reshape(X_train_images, (-1, n_inputs))
    X_val_images=np.reshape(X_val_images, (-1, n_inputs))

    h5f.close()
    h5f2.close()

    print('\n---------------------------')
    print('Train samples: ' + str(len(X_train_images)))
    print('Validate samples: ' + str(len(X_val_images)))
    print('----')

    return X_train_images, Y_train_labels, X_val_images, Y_val_labels

def next_batch(train_data, train_target, batch_size):  
    index = [ i for i in range(0,len(train_target)) ]  
    np.random.shuffle(index);  
    batch_data = []; 
    batch_target = [];  
    for i in range(0,batch_size):  
        batch_data.append(train_data[index[i]]);  
        batch_target.append(train_target[index[i]])  
    return batch_data, batch_target


def Exponential_Decay(INIT_LEARNING_RATE, GlOBAL_STEP, LEARNING_RATE_DECAY_STEPS, DECAY_RATE): #学习率衰减过程
    return INIT_LEARNING_RATE * (DECAY_RATE ** int(GlOBAL_STEP / LEARNING_RATE_DECAY_STEPS))

#构建图
learning_rate = 0.0002

n_inputs=50*50
n_hidden1=1024
n_hidden2=512
n_hidden3=256
n_hidden4=64
n_outputs=2

with tf.name_scope("input"):
    X=tf.compat.v1.placeholder(tf.float32,shape=(None,n_inputs),name='X')
    y=tf.compat.v1.placeholder(tf.int64,shape=(None),name='y')

with tf.name_scope("dnn"):
    #默认使用Relu函数
    hidden1=fully_connected(X,n_hidden1,scope="hidden1")
    hidden2=fully_connected(hidden1,n_hidden2,scope="hidden2")
    hidden3=fully_connected(hidden2,n_hidden3,scope="hidden3")
    hidden4=fully_connected(hidden3,n_hidden4,scope="hidden4")
    logits=fully_connected(hidden4,n_outputs,scope="outputs",activation_fn=None)


with tf.name_scope("loss"):
    #定义交叉熵损失函数，并求样本平均
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy,name="loss")


with tf.name_scope("train"):
    optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    training_op=optimizer.minimize(loss)

with tf.name_scope("val_accuracy"):
    #获取logits里面最大的哪一位并与y比较类别是否相同，返回True或False一组值
    correct=tf.nn.in_top_k(logits,y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
