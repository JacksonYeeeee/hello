import numpy as np
import cv2
import csv
import os

import h5py

import tflearn
from cnn_model import CNNModel

import tensorflow as tf

def get_prediction(X_test_image):

    ## Model definition
    convnet  = CNNModel()
    network = convnet.define_network(X_test_image)
    model = tflearn.DNN(network, tensorboard_verbose=0,\
  		   checkpoint_path='nodule-classifier.ckpt')
    model.load("./models_CNN/model_conv/nodule-classifier")

    predictions = np.vstack(model.predict(X_test_image[:,:,:,:]))
    label_predictions = np.zeros_like(predictions)
    label_predictions[np.arange(len(predictions)), predictions.argmax(1)] = 1
    return predictions, label_predictions


def main():

    impath = "./data/test/image_441280.jpg"
    image = cv2.imread(impath,cv2.IMREAD_GRAYSCALE)
    image = np.array(image,dtype=np.float32) 
    image = np.reshape(image/255, (1,50, 50, 1))

    prediction, label_prediction = get_prediction(image)
    label = label_prediction[0][1].astype('int')
    if label == 1:
        print("heihei")
    else:
        print("haha")
    print(prediction,label_prediction)


main()