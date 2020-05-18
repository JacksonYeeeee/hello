"""
Builds a HDF5 data set for test, train and validation data
Run script as python build_hdf5_datasets.py $mode
where mode can be 'test', 'train', 'val'
"""
import sys


import pickle
import h5py

import numpy as np
import pandas as pd

import os
import shutil
import cv2

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from tflearn.data_utils import build_hdf5_image_dataset

tagcsv='./data/tag.csv'
imagepath='./data/finalData/'
imsavepath='./data/'

Scale=50

def do_test_train_split(filename):
    """
    Does a test train split if not previously done

    """
    candidates = pd.read_csv(filename)

    positives = candidates[candidates['class']==1].index  
    negatives = candidates[candidates['class']==0].index

    ## Under Sample Negative Indexes
    np.random.seed(42)
    negIndexes = np.random.choice(negatives, len(positives)*10, replace = False)

    candidatesDf = candidates.iloc[list(positives)+list(negIndexes)]

    X = candidatesDf.iloc[:,:-1]
    y = candidatesDf.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y,\
     test_size = 0.20, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, \
        test_size = 0.20, random_state = 42)

    X_train.to_pickle('traindata')
    y_train.to_pickle('trainlabels')
    X_test.to_pickle('testdata')
    y_test.to_pickle('testlabels')
    X_val.to_pickle('valdata')
    y_val.to_pickle('vallabels')

def image_split(imagespath,inpfile,outDir,mode):
    
    X_data = pd.read_pickle(inpfile)
    for i in X_data.index :
        impath=imagespath+'image_'+str(i)+'.png'
        image = cv2.imread(impath,cv2.IMREAD_GRAYSCALE)
        savepath=outDir+'image_'+str(i)+'.png'
        cv2.imwrite(savepath,image)
    print('------'+mode+' image save over-------')

def biuld_hdf5(mode):
    # Read data
    X = pd.read_pickle(mode + 'data')
    y = pd.read_pickle(mode + 'labels')

    dataset_file = './data/'+mode + 'datalabels.txt'
    filenames =X.index.to_series().apply(lambda x:'./data/'+mode+ '/image_'+str(x)+'.png')
    filenames = filenames.values.astype(str)
    labels = y.values.astype(int)

    with open(dataset_file,'w') as f:
        for i in range(filenames.size):
            f.write(filenames[i]+' '+str(labels[i]))
            f.write('\n')

    output = './data/'+mode + 'dataset.h5'

    build_hdf5_image_dataset(dataset_file, image_shape = (Scale, Scale, 1), \
            mode ='file', output_path = output, categorical_labels = True, \
            normalize = True,grayscale = True)
    
    print('--------hdf5 build over-------')

    # Load HDF5 dataset
    h5f = h5py.File('./data/'+ mode+ 'dataset.h5', 'r')
    X_images = h5f['X']
    Y_labels = h5f['Y'][:]

    print(X_images.shape)
    X_images = X_images[:,:,:].reshape([-1,Scale,Scale,1])
    print(X_images.shape)
    h5f.close()

    h5f = h5py.File('./data/' + mode + '.h5', 'w')
    h5f.create_dataset('X', data=X_images)
    h5f.create_dataset('Y', data=Y_labels)
    h5f.close()
    

def main():
    # Check inputs
    if len(sys.argv) < 2:
	    raise ValueError('1 argument needed. Specify if you need to generate a train, test or val set')
    else:
	    mode = sys.argv[1]
	    if mode not in ['train', 'test', 'val']:
		    raise ValueError('Argument not recognized. Has to be train, test or val')
    
    inpfile = mode + 'data'
    outDir = './data/'+mode + '/'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    else:
        shutil.rmtree(outDir)
        os.makedirs(outDir)


    if os.path.isfile(inpfile):
        pass
    else:
        do_test_train_split(tagcsv)
    
    image_split(imagepath,inpfile,outDir,mode)

    biuld_hdf5(mode)

    
    
if __name__ == '__main__':
    main()