import h5py
import numpy as np

import tflearn
from cnn_model import CNNModel

import tensorflow as tf

from sklearn.metrics import roc_curve, auc, confusion_matrix


# 利用保存的模型预测新的值，并计算准确值acc
path = './models_DNN/'

def get_predictions(X_test_images, Y_test_labels):
    """
    Args:
    ------
    Given hdfs file of X_test_images and Y_test_labels
  
    returns:
    --------
    predictions: probability values for each class 
    label_predictions: returns predicted classes
    """

    ## Model definition
    convnet  = CNNModel()
    network = convnet.define_network(X_test_images)
    model = tflearn.DNN(network, tensorboard_verbose=0,\
  		   checkpoint_path='nodule-classifier.ckpt')
    model.load("./models_CNN2/model_conv/nodule-classifier")

    predictions = np.vstack(model.predict(X_test_images[:,:,:,:]))
    #label_predictions = np.vstack(model.predict_label(X_test_images[:,:,:,:]))
    score = model.evaluate(X_test_images, Y_test_labels)
    label_predictions = np.zeros_like(predictions)
    label_predictions[np.arange(len(predictions)), predictions.argmax(1)] = 1
    return predictions, label_predictions

def get_roc_curve(Y_test_labels, predictions):
    """
    Args:
    -------
    hdfs datasets: Y_test_labels and predictions
  
    Returns:
    --------
    fpr: false positive Rate
    tpr: true posiive Rate
    roc_auc: area under the curve value
    """
    fpr, tpr, thresholds = roc_curve(Y_test_labels[:,1], predictions[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def get_metrics(Y_test_labels, label_predictions):
    """
    Args:
    -----
    Y_test_labels, label_predictions

    Returns:
    --------
    precision, recall and specificity values and cm
    """
    cm = confusion_matrix(Y_test_labels[:,1], label_predictions[:,1])

    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]
    
    #print(TN,FP,FN,TP)

    precision = TP*1.0/(TP+FP)
    recall = TP*1.0/(TP+FN)
    specificity = TN*1.0/(TN+FP)
    acc = (TP+TN)/(TN+FP+FN+TP)

    return precision, recall, specificity, acc, cm

def main():
    # Load HDF5 dataset
    h5f2 = h5py.File('./data/test.h5', 'r')
    X = h5f2['X']
    Y = h5f2['Y']

    predictions, label_predictions = get_predictions(X,Y)
    #print(predictions)

    #fpr, tpr, roc_auc = get_roc_curve(Y,predictions)

    precision, recall, specificity, acc, cm = get_metrics(Y,label_predictions)

    print("precision", precision, "recall", recall, "specificity", specificity,"accuracy",acc)
    
main()