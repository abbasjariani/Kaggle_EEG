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
from tqdm import tqdm
#############################################################
#n_layers_to_search = [1,2,3,4,5]
#n_neurons_per_layer_to_search = [2,4,6,8,12,20]
n_layers_to_search = [2,3,4,5]
n_neurons_per_layer_to_search = [2,4,8,16]
n_processors = 72
#####################################
kfold_crossval = 5
epochs_nn = 200
n_channels_to_include = 6
cur_sub = 'Dog_1'

feature_dir = "/home/abbas/projects/180710_EEG_DeepL/feature_dir/"

log_f_path = '/home/abbas/projects/180710_EEG_DeepL/ANN_structure_search/' + cur_sub + '_ANN_structure_search.log'
with open(log_f_path, "a") as myfile:
    myfile.write('subject;n_channels;NN_hidden_layers;auc_cross_val_mean;auc_cross_val_std\n')        

try:
    os.remove(log_f_path)
except OSError:
    pass
os.mknod(log_f_path)


##########################################################
# given n_layers_to_search and n_neurons_per_layer_to_search returns a list with each item as n_neuron_list
# for each of these items you should then call make_keras_model
def get_structure_combinations(n_layers_to_search, n_neurons_per_layer_to_search):
    combinations_n_neuron_list = []
    for cur_n_layer in n_layers_to_search:
        #cur_n_layer = 3
        comb_ls = [n_neurons_per_layer_to_search]*cur_n_layer
        combinations_n_neuron_list = combinations_n_neuron_list + list(itertools.product(*comb_ls))
    
    return combinations_n_neuron_list

structure_combinations = get_structure_combinations(n_layers_to_search, n_neurons_per_layer_to_search)
print len(structure_combinations),'combinations'
#############################################
#a function that retunrs a keras model given a list containing the number of 
#neurons for each layer. length of the list is the number of layers
#a final two neuron softmax layer will be added on top
def make_keras_model(n_neuron_list):
    #n_neuron_list = list(n_neuron_list)
    #n_neuron_list = [8,4]
    model = keras.Sequential()
    for cur_layer in n_neuron_list:
        model.add(keras.layers.Dense(cur_layer, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dense(2, activation=tf.nn.softmax))    
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy',metrics=['accuracy'])
    return model
############################
def get_auc_mean_std_from_structure_combinations_index(structure_combinations_index):
    #structure_combinations_index  = 30
    
    #Kfold cross validation score
    n_splits = kfold_crossval
    skf = StratifiedKFold(n_splits=n_splits, random_state= 42)
    print round(float(structure_combinations_index)/float(len(structure_combinations))*100,2), '% complete'
    auc_res = []
    for train_index, test_index in skf.split(X_array_scaled_shuffled, Y_array_shuffled):
        curmodel = make_keras_model(structure_combinations[structure_combinations_index])
        X_train, X_test = X_array_scaled_shuffled[train_index], X_array_scaled_shuffled[test_index]
        y_train, y_test = Y_array_shuffled[train_index], Y_array_shuffled[test_index]
        
        y_train_inverse = np.empty(y_train.shape)
        y_train_inverse[y_train == 0.] = 1.
        y_train_inverse[y_train == 1.] = 0.
        #y_train_binary = np.concatenate((y_train, y_train_inverse),axis=1)
        y_train_binary = np.column_stack((y_train,y_train_inverse))

        curmodel.fit(X_train, y_train_binary, epochs=epochs_nn, verbose=  0)
        pred_y_test = curmodel.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, pred_y_test[:,0])
        auc_res.append(auc(fpr, tpr))
    auc_crosval_mean = np.mean(auc_res)
    auc_crosval_std = np.std(auc_res)
    return [auc_crosval_mean,auc_crosval_std]
#########################################3

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
  
n_channels_to_search = [2,3,7]

for cur_n_ch in n_channels_to_search:
    #cur_n_ch = 7
    selected_channels = None
    if cur_n_ch == 2:
        selected_channels= [5,14]
    if cur_n_ch == 3:
        selected_channels= [5,12,14]
    if cur_n_ch == 7:
        selected_channels= [5, 6, 8, 10, 12, 14, 16]
    if cur_n_ch == 8:
        selected_channels= [5, 6, 8, 10, 12,13, 14, 16]    

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

    #get_auc_mean_std_from_structure_combinations_index(1)
    #pool = multiprocessing.Pool(n_processors)
    #zip(*pool.map(get_auc_mean_std_from_structure_combinations_index, range( len(structure_combinations))))
    results = Parallel(n_jobs=n_processors)(delayed(get_auc_mean_std_from_structure_combinations_index)(i) for i in range( len(structure_combinations)))
    #results = Parallel(n_jobs=n_processors)(delayed(get_auc_mean_std_from_structure_combinations_index)(i) for i in tqdm(range( len(structure_combinations))))
    
    #reshaping to write reports
    for i in range(len(results)):
        cur_out_row = [cur_sub,structure_combinations[i],results[i]]
        with open(log_f_path, "a") as myfile:
            myfile.write(cur_out_row[0] + ';'+str(cur_n_ch) + ';' + str(cur_out_row[1]) + ';' + str(cur_out_row[2][0]) + ';' + str(cur_out_row[2][1]) + '\n')


#################################
"""


############################################################
#
#out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))

    
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
###########################################3
#recursive function for greeddy search
def get_channel_set_performance(ch_set):
    #ch_set = cur_channel_set
    ch_set_features = []
    for cur_ch in ch_set:
        cur_ch_start_string  = 'c' + str(cur_ch) + '_'
        cur_ch_features =  [i for i in feature_names if i.startswith(cur_ch_start_string)]
        ch_set_features = ch_set_features + cur_ch_features
        
    cur_chset_data = cur_subj_feat_data[ch_set_features]
    Y_array  = cur_subj_feat_data.iloc[:, [1]].values
    X_array  = cur_chset_data.values
    #X_array  = cur_subj_feat_data.iloc[:, range(2,len(list(cur_subj_feat_data)))].values
    X_array_scaled = preprocessing.scale(X_array)    
    #shuffling
    s = np.arange(X_array.shape[0])
    np.random.shuffle(s)
    X_array_scaled_shuffled = X_array_scaled[s]
    Y_array_shuffled = Y_array[s]
    nn_crossval_auc = get_kerasmodel_cross_val_score(kfold_crossval, X_array_scaled_shuffled, Y_array_shuffled,epochs_nn)
    auc_crosval_mean = np.mean(nn_crossval_auc)
    auc_crosval_std = np.std(nn_crossval_auc)
    return auc_crosval_mean, auc_crosval_std
#########################################
#ch_set_score = namedtuple('ch_set_score', 'ch_set score')


def greedy_reduced_ch_set(ch_set):
    if len(ch_set) > 1:
        cur_best_reduced_score_mean = 0
        cur_best_reduced_score_std = None
        cur_best_reduced_chset = None
        for ch_to_drop in ch_set:
            print "-------------"
            print cur_sub, 'testing removal of channel ', ch_to_drop, ' from ', ch_set
            ch_set_reduced = list(set(ch_set) - set([ch_to_drop]))
            ch_set_reduced_score = get_channel_set_performance(ch_set_reduced)
            ch_set_reduced_score_mean = ch_set_reduced_score[0]
            print 'score: ', str(ch_set_reduced_score_mean)
            ch_set_reduced_score_std = ch_set_reduced_score[1]
            if ch_set_reduced_score_mean > cur_best_reduced_score_mean:
                cur_best_reduced_score_mean = ch_set_reduced_score_mean
                cur_best_reduced_score_std = ch_set_reduced_score_std
                cur_best_reduced_chset = ch_set_reduced
            
        with open(log_f_path, "a") as myfile:
            myfile.write(cur_sub + ';' + str(cur_best_reduced_chset) + ';' + str(cur_best_reduced_score_mean) + ';' + str(cur_best_reduced_score_std) + '\n')
        return(greedy_reduced_ch_set(cur_best_reduced_chset))
    else:
        return(0)
############################
#head of log file        
with open(log_f_path, "a") as myfile:
    myfile.write('subject;channels;auc_cross_val_mean;auc_cross_val_std\n')        
###############################
####################
progress_counter = 0

#with open('/media/abbas/Extension_3TB/180710_EEG_DeepL/channel_redundancy_analysis/channel_information_logreg_NN_2.csv', 'wb') as csvfile:
    #spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #spamwriter.writerow(['subject','channels','auc_nn','auc_logreg'])

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
        
    ########################
    #greedy search for reducing channel number
    #start from full channels
    #find the channel that if you drop it, the minimum
    continue_search = True
    #starting the search with full channel set
    full_channel_set  = range(1,total_channels + 1)
    fulchset_score = get_channel_set_performance(full_channel_set)
    fulchset_score_mean  = fulchset_score[0]
    fulchset_score_std  = fulchset_score[1]
    #writing full channe model peroframnce
    with open(log_f_path, "a") as myfile:
        myfile.write(cur_sub + ';' + str(full_channel_set) + ';' + str(fulchset_score_mean) + ';' + str(fulchset_score_std) + '\n')
    
    greedy_reduced_ch_set(full_channel_set)
            
"""
