# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 23:37:06 2018

@author: abbas
"""
import scipy.io
import os
from seizure.tasks import *
import glob
from scipy.signal import resample
from collections import namedtuple
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras


targetFrequency = 100   #re-sample to target frequency
sampleSizeinSecond = 600


data_dir = "/home/abbas/projects/180710_EEG_DeepL/data_dir/"
#subjects = ['Dog_1','Dog_2','Dog_3','Dog_4','Dog_5','Patient_1','Patient_2']
subjects = ['Dog_1']


cursubj_data_dir = data_dir + subjects[0] + '/'

mat_f_ls = os.listdir(cursubj_data_dir)

Training_Data_Epoch = namedtuple('Training_Data_Epoch', ['signal_array', 'label'])

training_data_epoch_ls = []

for cur_f in mat_f_ls:
    if cur_f.endswith(".mat") and "test" not in cur_f:

        myfile = cursubj_data_dir +cur_f
        mydata = scipy.io.loadmat(myfile)
        ##################
        cur_label = cur_f.split('_')[2]
        #################
        cur_seg = cur_f.split('_')[4].split('.')[0]
        #######################
        cur_mainkey = [s for s in mydata.keys() if "segment" in s][0]
        mydata = mydata[cur_mainkey]
        #####################
        mydata = mydata[0][0][0]
        mydata = resample(mydata, targetFrequency*sampleSizeinSecond, axis=1)
        X_array = mydata
        mydata = None
        cur_training_data_epoch = Training_Data_Epoch(X_array,cur_label)
        training_data_epoch_ls.append(cur_training_data_epoch)

        ###################
        
############
#pouring data into linearized array for training
            
n_training_epochs  = len(training_data_epoch_ls)
n_tpoints = training_data_epoch_ls[0][0].shape[1]
n_channels = training_data_epoch_ls[0][0].shape[0]
#an array of 16channels * 60k time-points * 504 epochs
train_array = np.zeros(shape=(n_training_epochs, n_channels*n_tpoints))
train_labels = np.zeros(shape=(n_training_epochs,))
for cur_epoch in range(len(training_data_epoch_ls)):
    train_array[cur_epoch,:] = training_data_epoch_ls[cur_epoch][0].flatten()
    cur_label_val = None
    if training_data_epoch_ls[cur_epoch][1] == 'interictal':
        cur_label_val = 0
    elif training_data_epoch_ls[cur_epoch][1] == 'preictal':
        cur_label_val = 1
    
    train_labels[cur_epoch] = cur_label_val
    
    
    
    
train_array_scaled = preprocessing.scale(train_array)    
    
    
    
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(n_channels*n_tpoints,)),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

    
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
model.fit(train_array_scaled, train_labels, epochs=5)
