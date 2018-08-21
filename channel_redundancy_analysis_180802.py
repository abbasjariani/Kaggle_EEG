# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:25:20 2018

@author: abbas
"""

#export with line below, then start spyder from command line as the next command
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64

import csv
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
import sys
import numpy as np
import itertools

#from sklearn.base import BaseEstimator, ClassifierMixin
#from sklearn.exceptions import NotFittedError
#from datetime import datetime
#import os
#import inspect
##############
#path = os.path.abspath("/media/abbas/Extension_3TB/180710_EEG_DeepL/optimize_full_channel_features_180801")
#inspect.getfile("/media/abbas/Extension_3TB/180710_EEG_DeepL/optimize_full_channel_features_180801/dnn_classifier")sys.path.insert(0, <path to dirFoo>)
sys.path.insert(0, "/media/abbas/Extension_3TB/180710_EEG_DeepL/optimize_full_channel_features_180801")

n_channels_to_include = 6
from dnn_classifier import DNNClassifier
##################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import accuracy_score,roc_curve, auc
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn import linear_model
##################################
def get_auc(estimator,X_test,y_test):
    predictions_validation = estimator.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, predictions_validation)
    return auc(fpr, tpr)
############################################
def get_kerasmodel_cross_val_score(n_splits, X_array, y_array, epochs):
    #n_splits = 6
    skf = StratifiedKFold(n_splits=n_splits, random_state= 42)
    auc_res = []
    for train_index, test_index in skf.split(X_array, y_array):
        X_train, X_test = X_array[train_index], X_array[test_index]
        y_train, y_test = y_array[train_index], y_array[test_index]
        
        y_train_inverse = np.empty(y_train.shape)
        y_train_inverse[y_train == 0.] = 1.
        y_train_inverse[y_train == 1.] = 0.
        y_train_binary = np.concatenate((y_train, y_train_inverse),axis=1)
        
        model = keras.Sequential([
        #keras.layers.Dense(16, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(8, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(4, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(2, activation=tf.nn.softmax),
        ])
        model.compile(
        optimizer=tf.train.AdamOptimizer(), 
                  loss='binary_crossentropy',
        #          optimizer='sgd',
                  metrics=['accuracy'])

        model.fit(X_train, y_train_binary, epochs=epochs, verbose=  0)
        pred_y_test = model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, pred_y_test[:,0])
        auc_res.append(auc(fpr, tpr))
    return auc_res
################################################3
def get_logis_regr_cross_val_score(n_splits, X_array, y_array, epochs):
    logreg = linear_model.LogisticRegression(C=1e5)
    skf = StratifiedKFold(n_splits=n_splits, random_state= 42)
    auc_res = []
    for train_index, test_index in skf.split(X_array, y_array):
        X_train, X_test = X_array[train_index], X_array[test_index]
        y_train, y_test = y_array[train_index], y_array[test_index]
        logreg.fit(X_train, y_train)
        pred_y_test = logreg.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, pred_y_test)
        auc_res.append(auc(fpr, tpr))
    return auc_res
####################
feature_dir = "/media/abbas/Extension_3TB/180710_EEG_DeepL/feature_dir/"
progress_counter = 0
subjects = ['Dog_1','Dog_2','Dog_3','Dog_4','Dog_5','Patient_1','Patient_2']
with open('/media/abbas/Extension_3TB/180710_EEG_DeepL/channel_redundancy_analysis/channel_information_logreg_NN.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['subject','channels','auc_nn','auc_logreg'])

    for cur_sub in subjects:
        #progress_counter = progress_counter + 1
        #cur_sub = 'Dog_1'
    
        cur_feat_f = feature_dir + cur_sub + '_features.csv'
        cur_subj_feat_data = pd.read_csv(cur_feat_f)
        feature_names  = list(cur_subj_feat_data)
    
        total_channels  = None
        if cur_sub in ['Dog_1','Dog_2','Dog_3','Dog_4']:
            total_channels = 16
        if cur_sub in ['Dog_5','Patient_1']:
            total_channels = 15
        if cur_sub in ['Patient_2']:
            total_channels = 24

        for cur_n_channels_keeping in range(1,n_channels_to_include+1):
            cur_ch_subsets_set =  set(itertools.combinations(range(1,total_channels + 1), cur_n_channels_keeping))
            for cur_ch_subset in cur_ch_subsets_set:
                #cur_ch_subset_ls = []
                ch_subset_features = []
                for cur_ch in cur_ch_subset:
                    cur_ch_start_string  = 'c' + str(cur_ch) + '_'
                    cur_ch_features =  [i for i in feature_names if i.startswith(cur_ch_start_string)]
                    ch_subset_features = ch_subset_features + cur_ch_features
                    
                cur_chsubset_data_inclusive = cur_subj_feat_data[ch_subset_features]
                cur_ch_subset_exlcusive = list(set(feature_names) - set(ch_subset_features))
                cur_ch_subset_exlcusive.remove('subject')
                cur_ch_subset_exlcusive.remove('outcome_label')
                cur_chsubset_data_exclusive = cur_subj_feat_data[cur_ch_subset_exlcusive]
    
    
                ################################
                #cur_subj_feat_data = cur_chsubset_data_inclusive
                Y_array  = cur_subj_feat_data.iloc[:, [1]].values
                X_array  = cur_chsubset_data_inclusive.values
                #X_array  = cur_subj_feat_data.iloc[:, range(2,len(list(cur_subj_feat_data)))].values
                X_array_scaled = preprocessing.scale(X_array)    
                #shuffling
                s = np.arange(X_array.shape[0])
                np.random.shuffle(s)
                X_array_scaled_shuffled = X_array_scaled[s]
                Y_array_shuffled = Y_array[s]
                nn_crossval_auc = get_kerasmodel_cross_val_score(3, X_array_scaled_shuffled, Y_array_shuffled,50)
                logreg_crossval_auc = get_logis_regr_cross_val_score(3, X_array_scaled_shuffled, Y_array_shuffled, 50)
                spamwriter.writerow([cur_sub,cur_ch_subset,nn_crossval_auc,logreg_crossval_auc])
                print "-------------"
                print cur_sub
                print cur_n_channels_keeping, 'out of ', n_channels_to_include
                print 'channels: ', cur_ch_subset
                print 'nn auc: ', np.mean(nn_crossval_auc)
                print 'logreg auc: ', np.mean(logreg_crossval_auc)
                ###################################
                #cur_subj_feat_data = cur_chsubset_data_exclusive
                Y_array  = cur_subj_feat_data.iloc[:, [1]].values
                X_array  = cur_chsubset_data_exclusive.values
                X_array_scaled = preprocessing.scale(X_array)    
                #shuffling
                s = np.arange(X_array.shape[0])
                np.random.shuffle(s)
                X_array_scaled_shuffled = X_array_scaled[s]
                Y_array_shuffled = Y_array[s]
                nn_crossval_auc = get_kerasmodel_cross_val_score(3, X_array_scaled_shuffled, Y_array_shuffled,50)                
                logreg_crossval_auc = get_logis_regr_cross_val_score(3, X_array_scaled_shuffled, Y_array_shuffled,50)
                spamwriter.writerow([cur_sub,list(set(range(1,total_channels + 1)) - set(cur_ch_subset)),nn_crossval_auc,logreg_crossval_auc])    
                print "-------------"
                print 'channels: ', list(set(range(1,total_channels + 1)) - set(cur_ch_subset))
                print 'nn auc: ', np.mean(nn_crossval_auc)
                print 'logreg auc: ', np.mean(logreg_crossval_auc)

