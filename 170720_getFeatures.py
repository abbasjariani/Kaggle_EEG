# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 23:37:06 2018

@author: abbas
"""
import scipy.io
import os
#from seizure.tasks import *
import glob
from scipy.signal import resample
from collections import namedtuple
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv
from scipy.fftpack import rfft, irfft, fftfreq,fft,ifft



#targetFrequency = 100   #re-sample to target frequency
targetFrequency = 400   #re-sample to target frequency
sampleSizeinSecond = 600
bufferSizeinSeconds = 60

#data_dir = "/home/abbas/Desktop/prjects/180720_EEG/data_dir/"
data_dir = "/home/abbas/projects/180710_EEG_DeepL/data_dir/"
feature_dir = "/home/abbas/projects/180710_EEG_DeepL/feature_dir/"
subjects = ['Dog_1','Dog_2','Dog_3','Dog_4','Dog_5','Patient_1','Patient_2']
#subjects = ['Patient_1','Patient_1']

for cur_subj in subjects:
    print 'importing data for subject : '+ cur_subj
    #cur_subj = subjects[5]
    cursubj_data_dir = data_dir + cur_subj + '/'
    
    cur_feature_f = feature_dir + cur_subj+'_features.csv'
    mat_f_ls = os.listdir(cursubj_data_dir)
    
    Training_Data_Epoch = namedtuple('Training_Data_Epoch', ['signal_array', 'label'])
    
    training_data_epoch_ls = []
    
    for cur_f in mat_f_ls:
        if cur_f.endswith(".mat") and "test" not in cur_f:
            #cur_f = mat_f_ls[20]
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
            sampleFrequency = mydata[0][0][2][0][0]
            mydata = mydata[0][0][0]
            print cur_f, mydata.shape
            """
            mydata = resample(mydata, targetFrequency*sampleSizeinSecond, axis=1)
            X_array = mydata
            #buffering: get the last 60s (eg) segment
            seg_to_buffer = range(X_array.shape[1] - targetFrequency*bufferSizeinSeconds , X_array.shape[1])
            X_array_buffered = X_array[:,seg_to_buffer]
            mydata = None
            cur_training_data_epoch = Training_Data_Epoch(X_array_buffered,cur_label)
            training_data_epoch_ls.append(cur_training_data_epoch)
            """
    
            ###################
    #creating feature vec header
    feature_names = ['min_sig_vals','max_sig_vals','mean_sig_vals','var_sig_vals','std_sig_vals','rootmeansquare_sig_vals','hjorth_mobility_sig_vals','hjorth_complexity_sig_vals','kurtosis_sig_vals','power_delta','power_theta','power_alpha','power_beta','power_lowgamma','power_highgamma']
    feature_names = feature_names + ['min_x_ifft_theta',' max_x_ifft_theta',' var_x_ifft_theta',' std_x_ifft_theta',' rootmeansquare_x_ifft_theta',' hjorth_mobility_x_ifft_theta',' hjorth_complexity_x_ifft_theta',' kurtosis_x_ifft_theta']
    feature_names = feature_names + ['min_x_ifft_ph_gamma','max_x_ifft_ph_gamma','var_x_ifft_ph_gamma','std_x_ifft_ph_gamma','rootmeansquare_x_ifft_ph_gamma','hjorth_mobility_x_ifft_ph_gamma','hjorth_complexity_x_ifft_ph_gamma','kurtosis_x_ifft_ph_gamma']
    channel_strings = []
    for c in range(1,17):
        channel_strings.append('c'+str(c)+'_')
    feature_channel_name = ['subject','outcome_label']
    for f in feature_names:
        for c in channel_strings:
            feature_channel_name.append(c+f)
    with open(cur_feature_f, 'wb') as csvfile:
        mycsvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        mycsvwriter.writerow(feature_channel_name)
    ####################################
        print 'Extracting features for subject : ' + cur_feature_f

    ##################################333
        #Feature extraction
        for cur_epoch in range(len(training_data_epoch_ls)):
            #cur_epoch = 0
            Y_label = training_data_epoch_ls[cur_epoch][1]
            Y_label_num = None
            if Y_label == 'interictal':
                Y_label_num = 0
            if Y_label == 'preictal':
                Y_label_num = 1
            x_array = training_data_epoch_ls[cur_epoch][0]
            min_sig_vals = x_array.min(axis=1)
            max_sig_vals = x_array.max(axis=1)
            mean_sig_vals = x_array.mean(axis=1)
            var_sig_vals = np.var(x_array, axis=1)
            std_sig_vals = np.std(x_array, axis=1)
            rootmeansquare_sig_vals = np.sqrt(np.sum(np.square(x_array), axis= 1))
            #    Hjorth mobility and complexity
            #https://wikivisually.com/wiki/Hjorth_parameters
            diff = np.diff(x_array)
            hjorth_mobility_sig_vals = np.sqrt(np.var(diff, axis=1) / np.var(x_array, axis=1))
            diff2 = np.diff(diff)
            mobility_d = np.sqrt(np.var(diff2, axis=1) / np.var(diff, axis=1))
            hjorth_complexity_sig_vals = mobility_d / hjorth_mobility_sig_vals
            kurtosis_sig_vals = scipy.stats.kurtosis(x_array,axis= 1)
            ############
            ############
            #Getting freq bands power
            #https://stackoverflow.com/questions/15382076/plotting-power-spectrum-in-python
            ps = np.abs(np.fft.fft(x_array))**2
            fft_freq = np.fft.fftfreq(x_array.shape[1], 1.0/targetFrequency)
            #idx = np.argsort(fft_freq)
            idx_delta = np.concatenate(np.argwhere(( (fft_freq > 0.1) & (fft_freq <  4)) | ((fft_freq > -4) & (fft_freq <  -0.1) ) ))
            idx_theta = np.concatenate(np.argwhere(( (fft_freq > 4) & (fft_freq <  8)) | ((fft_freq > -8) & (fft_freq <  -4) ) ))
            idx_alpha = np.concatenate(np.argwhere(( (fft_freq > 8) & (fft_freq <  12)) | ((fft_freq > -12) & (fft_freq <  -8) ) ))
            idx_beta = np.concatenate(np.argwhere(( (fft_freq > 12) & (fft_freq <  30)) | ((fft_freq > -30) & (fft_freq <  -12) ) ))
            idx_lowgamma = np.concatenate(np.argwhere(( (fft_freq > 30) & (fft_freq <  70)) | ((fft_freq > -70) & (fft_freq <  -30) ) ))
            idx_highgamma = np.concatenate(np.argwhere(( (fft_freq > 70) & (fft_freq <  180)) | ((fft_freq > -180) & (fft_freq <  -70) ) ))
            power_total = ps.sum(axis=1)
            power_delta = ps[:,idx_delta].sum(axis=1)
            power_delta = power_delta/power_total
            power_theta = ps[:,idx_theta].sum(axis=1)
            power_theta = power_theta/power_total
            power_alpha = ps[:,idx_alpha].sum(axis=1)
            power_alpha = power_alpha/power_total
            power_beta = ps[:,idx_beta].sum(axis=1)
            power_beta = power_beta/power_total
            power_lowgamma = ps[:,idx_lowgamma].sum(axis=1)
            power_lowgamma = power_lowgamma/power_total
            power_highgamma = ps[:,idx_highgamma].sum(axis=1)
            power_highgamma = power_highgamma/power_total
            ########################
            #inverse fft of 4-8Hz and 75-97 Hz
            #https://stackoverflow.com/questions/16715641/inverse-of-fft-not-the-same-as-original-function
            myfft = np.fft.fft(x_array)
            #fft_freq = np.fft.fftfreq(x_array.shape[1], 1.0/targetFrequency)
            #idx_theta calculated already
            idx_partial_high_gamma = np.concatenate(np.argwhere( ((fft_freq > 75) & (fft_freq <  97) ) | ((fft_freq > -97) & (fft_freq <  -75) )))
            myfft_theta = myfft[:,idx_theta]
            myfft_partial_high_gamma = myfft[:,idx_partial_high_gamma]
            x_ifft_theta  = ifft(myfft_theta, axis = 1).real
            x_ifft_ph_gamma  = ifft(myfft_partial_high_gamma, axis = 1).real
            
            min_x_ifft_theta = x_ifft_theta.min(axis=1)
            max_x_ifft_theta = x_ifft_theta.max(axis=1)
            var_x_ifft_theta = np.var(x_ifft_theta, axis=1)
            std_x_ifft_theta = np.std(x_ifft_theta, axis=1)
            rootmeansquare_x_ifft_theta = np.sqrt(np.sum(np.square(x_ifft_theta), axis= 1))
            diff = np.diff(x_ifft_theta)
            hjorth_mobility_x_ifft_theta = np.sqrt(np.var(diff, axis=1) / np.var(x_ifft_theta, axis=1))
            diff2 = np.diff(diff)
            mobility_d = np.sqrt(np.var(diff2, axis=1) / np.var(diff, axis=1))
            hjorth_complexity_x_ifft_theta = mobility_d / hjorth_mobility_x_ifft_theta
            kurtosis_x_ifft_theta = scipy.stats.kurtosis(x_ifft_theta,axis= 1)
            
            min_x_ifft_ph_gamma = x_ifft_ph_gamma.min(axis=1)
            max_x_ifft_ph_gamma = x_ifft_ph_gamma.max(axis=1)
            var_x_ifft_ph_gamma = np.var(x_ifft_ph_gamma, axis=1)
            std_x_ifft_ph_gamma = np.std(x_ifft_ph_gamma, axis=1)
            rootmeansquare_x_ifft_ph_gamma = np.sqrt(np.sum(np.square(x_ifft_ph_gamma), axis= 1))
            diff = np.diff(x_ifft_ph_gamma)
            hjorth_mobility_x_ifft_ph_gamma = np.sqrt(np.var(diff, axis=1) / np.var(x_ifft_ph_gamma, axis=1))
            diff2 = np.diff(diff)
            mobility_d = np.sqrt(np.var(diff2, axis=1) / np.var(diff, axis=1))
            hjorth_complexity_x_ifft_ph_gamma = mobility_d / hjorth_mobility_x_ifft_ph_gamma
            kurtosis_x_ifft_ph_gamma = scipy.stats.kurtosis(x_ifft_ph_gamma,axis= 1)
            ########################################3

            #################################3
            feature_vec_all_channels = np.concatenate((min_sig_vals,max_sig_vals,mean_sig_vals,var_sig_vals,std_sig_vals,rootmeansquare_sig_vals,hjorth_mobility_sig_vals,hjorth_complexity_sig_vals,kurtosis_sig_vals,power_delta,power_theta,power_alpha,power_beta,power_lowgamma,power_highgamma,min_x_ifft_theta, max_x_ifft_theta, var_x_ifft_theta, std_x_ifft_theta, rootmeansquare_x_ifft_theta, hjorth_mobility_x_ifft_theta, hjorth_complexity_x_ifft_theta, kurtosis_x_ifft_theta,min_x_ifft_ph_gamma, max_x_ifft_ph_gamma, var_x_ifft_ph_gamma, std_x_ifft_ph_gamma, rootmeansquare_x_ifft_ph_gamma, hjorth_mobility_x_ifft_ph_gamma, hjorth_complexity_x_ifft_ph_gamma, kurtosis_x_ifft_ph_gamma),axis=0)
            #feature_vec_all_channels_2 = np.concatenate((min_x_ifft_theta, max_x_ifft_theta, var_x_ifft_theta, std_x_ifft_theta, rootmeansquare_x_ifft_theta, hjorth_mobility_x_ifft_theta, hjorth_complexity_x_ifft_theta, kurtosis_x_ifft_theta),axis=0)
            #feature_vec_all_channels_3 = np.concatenate((min_x_ifft_ph_gamma, max_x_ifft_ph_gamma, var_x_ifft_ph_gamma, std_x_ifft_ph_gamma, rootmeansquare_x_ifft_ph_gamma, hjorth_mobility_x_ifft_ph_gamma, hjorth_complexity_x_ifft_ph_gamma, kurtosis_x_ifft_ph_gamma),axis=0)
            #feature_vec_all_channels = np.concatenate((feature_vec_all_channels_1,feature_vec_all_channels_2,feature_vec_all_channels_3),axis= 0)
            #############################################
            label_feature_vec = np.concatenate((np.array([Y_label_num]),feature_vec_all_channels))
            label_feature_vec = np.concatenate((np.array([cur_subj]),label_feature_vec))
            
            mycsvwriter.writerow(list(label_feature_vec))
        #################

"""
    ##############################################
########################################

n_training_epochs  = len(training_data_epoch_ls)
n_tpoints = training_data_epoch_ls[0][0].shape[1]
n_channels = training_data_epoch_ls[0][0].shape[0]
############################
#Feature extraction:


##########################################333
#pouring data into linearized array for training
#an array of 16channels * 60k time-points * 504 epochs
#train_array = np.zeros(shape=(n_training_epochs, n_channels*n_tpoints))
#train_labels = np.zeros(shape=(n_training_epochs,))
#for cur_epoch in range(len(training_data_epoch_ls)):
#    train_array[cur_epoch,:] = training_data_epoch_ls[cur_epoch][0].flatten()
#    cur_label_val = None
#    if training_data_epoch_ls[cur_epoch][1] == 'interictal':
#        cur_label_val = 0
#    elif training_data_epoch_ls[cur_epoch][1] == 'preictal':
#        cur_label_val = 1
    
#    train_labels[cur_epoch] = cur_label_val
###################################
    
    
    
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
"""