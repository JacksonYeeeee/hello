import tensorflow as tf
import numpy as np
import time
np.random.seed(1337)  # for reproducibility

from model_dbn2.dbn import DBN
from model_dbn2.base_func import run_sess,tSNE_2d,Batch
import h5py

# Loading dataset


def get_data_dbn():
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
    return datasets

