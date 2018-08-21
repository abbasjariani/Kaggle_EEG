# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 00:36:22 2018

@author: abbas
"""

#Training the model for one subject and checking the performance on other subjects


import csv
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
import sys
import numpy as np
import itertools
from collections import namedtuple
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import accuracy_score,roc_curve, auc
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn import linear_model
import multiprocessing
from joblib import Parallel, delayed
import multiprocessing
from copy import copy
#############################################################
#n_layers_to_search = [1,2,3,4,5]
#n_neurons_per_layer_to_search = [2,4,6,8,12,20]
n_layers_to_search = [2,3,4,5]
n_neurons_per_layer_to_search = [2,4,6,8,16,32]
n_processors = 70
#####################################
kfold_crossval = 5
epochs_nn = 200
n_channels_to_include = 6
#cur_sub = 'Dog_1'

feature_dir = "/media/abbas/Extension_3TB/180710_EEG_DeepL/feature_dir/"

log_f_path = '/media/abbas/Extension_3TB/180710_EEG_DeepL/Reinforced_learning_180818/'  + 'reinforced_learning.log'
with open(log_f_path, "a") as myfile:
    myfile.write('initially_train_sub;auc_sec_no_sec_train;auc_sec_traing_cv_mean;auc_sec_traing_cv_std\n')        

try:
    os.remove(log_f_path)
except OSError:
    pass
os.mknod(log_f_path)

############
#selected channels for each subject with greedy serach
sel_channels_dic = {'Dog_1': [1,5,14], 
                    'Dog_2': [12,5,14], 
                    'Dog_3': [14,6,7], 
                    'Dog_4': [2,12,14], 
                    'Dog_5': [12,4,7],
                    'Patient_1': [12,4,7], 
                    'Patient_2': [10,11,12], }


sub_ls = ['Dog_1','Dog_2','Dog_3','Dog_4','Patient_1','Patient_2']
for cur_ref_sub in sub_ls:
    #print cur_ref_sub
    #cur_ref_sub = 'Dog_1'
        
    cur_feat_f = feature_dir + cur_ref_sub + '_features.csv'
    cur_subj_feat_data = pd.read_csv(cur_feat_f)
    feature_names  = list(cur_subj_feat_data) 
    selected_channels = sel_channels_dic[cur_ref_sub]
    sel_features = []
    for cur_ch in selected_channels:
        cur_ch_start_string  = 'c' + str(cur_ch) + '_'
        cur_ch_features =  [i for i in feature_names if i.startswith(cur_ch_start_string)]
        sel_features = sel_features + cur_ch_features
    data_sel_features = cur_subj_feat_data[sel_features]
    Y_array  = cur_subj_feat_data.iloc[:, [1]].values
    Y_array = Y_array.flatten()
    X_array  = data_sel_features.values
    X_array_scaled = preprocessing.scale(X_array)    
    #shuffling
    s = np.arange(X_array.shape[0])
    np.random.shuffle(s)
    X_array_scaled_shuffled = X_array_scaled[s]
    Y_array_shuffled = Y_array[s]    
    y_train_inverse = np.empty(Y_array_shuffled.shape)
    y_train_inverse[Y_array_shuffled == 0.] = 1.
    y_train_inverse[Y_array_shuffled == 1.] = 0.
    #y_train_binary = np.concatenate((y_train, y_train_inverse),axis=1)
    y_train_binary = np.column_stack((Y_array_shuffled,y_train_inverse))

    model = keras.Sequential([
    #keras.layers.Dense(16, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(8, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(4, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(2, activation=tf.nn.softmax),  ])
    model.compile( optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(X_array_scaled_shuffled, y_train_binary, epochs=epochs_nn, verbose=  0)
    for sec_subj in set(sub_ls)-set([cur_ref_sub]):
        print cur_ref_sub,sec_subj
        #sec_subj = 'Dog_4'
        sec_feat_f = feature_dir + sec_subj + '_features.csv'
        sec_subj_feat_data = pd.read_csv(sec_feat_f)
        feature_names  = list(sec_subj_feat_data)        
        sec_data_sel_features = sec_subj_feat_data[sel_features]
        
        Y_array_sec  = sec_subj_feat_data.iloc[:, [1]].values
        Y_array_sec = Y_array_sec.flatten()
        X_array_sec  = sec_data_sel_features.values
        X_array_scaled_sec = preprocessing.scale(X_array_sec)    
        #shuffling
        s = np.arange(X_array_sec.shape[0])
        np.random.shuffle(s)
        X_array_scaled_shuffled_sec = X_array_scaled_sec[s]
        Y_array_shuffled_sec = Y_array_sec[s]    
        y_train_inverse_sec = np.empty(Y_array_shuffled_sec.shape)
        y_train_inverse_sec[Y_array_shuffled_sec == 0.] = 1.
        y_train_inverse_sec[Y_array_shuffled_sec == 1.] = 0.
        #y_train_binary = np.concatenate((y_train, y_train_inverse),axis=1)
        y_train_binary_sec = np.column_stack((Y_array_shuffled_sec,y_train_inverse_sec))
        pred_y_sec = model.predict(X_array_scaled_shuffled_sec)
        fpr_sec, tpr_sec, thresholds_sec = roc_curve(Y_array_shuffled_sec, pred_y_sec[:,0])
        auc_no_sec_training = auc(fpr_sec, tpr_sec)
        
        #now re-fitting on part of the data from the secondary subject
        skf = StratifiedKFold(n_splits=kfold_crossval, random_state= 42)
        auc_res_sec_training = []
        for train_index, test_index in skf.split(X_array_scaled_shuffled_sec, Y_array_shuffled_sec):
            
            model_updated_sec = copy(model)
            X_train_cv, X_test_cv = X_array_scaled_shuffled_sec[train_index], X_array_scaled_shuffled_sec[test_index]
            y_train_cv, y_test_cv = y_train_binary_sec[train_index], y_train_binary_sec[test_index]
            
            y_train_inverse_cv = np.empty(y_train_cv.shape)
            y_train_inverse_cv[y_train_cv == 0.] = 1.
            y_train_inverse_cv[y_train_cv == 1.] = 0.
            #y_train_binary = np.concatenate((y_train, y_train_inverse),axis=1)
            y_train_binary_cv = np.column_stack((y_train_cv,y_train_inverse_cv))
    
            model_updated_sec.fit(X_train_cv, y_train_cv, epochs=epochs_nn, verbose=  0)
            pred_y_test_cv = model_updated_sec.predict(X_test_cv)
            fpr, tpr, thresholds = roc_curve(y_test_cv[:,0], pred_y_test_cv[:,0])
            auc_res_sec_training.append(auc(fpr, tpr))
        auc_res_sec_training_mean = np.mean(auc_res_sec_training)
        auc_res_sec_training_std = np.std(auc_res_sec_training)
        
        cur_out_row = [cur_ref_sub,sec_subj,auc_no_sec_training, auc_res_sec_training_mean, auc_res_sec_training_std]
        with open(log_f_path, "a") as myfile:
            myfile.write(cur_out_row[0] + ';'+cur_out_row[1]+ ';' + str(auc_no_sec_training) + ';' + str(auc_res_sec_training_mean) + ';' + str(auc_res_sec_training_std) + '\n')