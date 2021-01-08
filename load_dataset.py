# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:08:37 2020

@author: andrea
"""
# this file loads the dataset and normalizes it
import os 
import numpy as np

def load_dataset(save_path):

    x_train_reduced=np.load(os.path.join(save_path,'NI_x_train_reduced.npy'))
    x_test_reduced=np.load(os.path.join(save_path,'NI_x_test_reduced.npy'))
    x_val_reduced=np.load(os.path.join(save_path,'NI_x_val_reduced.npy'))
    
    y_train_reduced=np.load(os.path.join(save_path,'NI_y_train_reduced.npy'))
    y_test_reduced=np.load(os.path.join(save_path,'NI_y_test_reduced.npy'))
    y_val_reduced=np.load(os.path.join(save_path,'NI_y_val_reduced.npy'))
    
    mean0=0.5037397340299304
    mean1=0.46401040012766465
    mean2=0.4230448464587425
    std0=0.28759323355072713
    std1=0.2789484134306188
    std2=0.2929629313131349
    
    x_train_reduced[:,:,:,0]=(x_train_reduced[:,:,:,0]-mean0)/std0
    x_train_reduced[:,:,:,1]=(x_train_reduced[:,:,:,1]-mean1)/std1
    x_train_reduced[:,:,:,2]=(x_train_reduced[:,:,:,2]-mean2)/std2
    
    x_test_reduced[:,:,:,0]=(x_test_reduced[:,:,:,0]-mean0)/std0
    x_test_reduced[:,:,:,1]=(x_test_reduced[:,:,:,1]-mean1)/std1
    x_test_reduced[:,:,:,2]=(x_test_reduced[:,:,:,2]-mean2)/std2
    
    x_val_reduced[:,:,:,0]=(x_val_reduced[:,:,:,0]-mean0)/std0
    x_val_reduced[:,:,:,1]=(x_val_reduced[:,:,:,1]-mean1)/std1
    x_val_reduced[:,:,:,2]=(x_val_reduced[:,:,:,2]-mean2)/std2
    return x_train_reduced,x_test_reduced,x_val_reduced,y_train_reduced,y_test_reduced,y_val_reduced

