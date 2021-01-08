# -*- coding: utf-8 -*-
"""

@author: andrea
"""

from __future__ import print_function
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import load_model 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping #, TensorBoard
import tensorflow as tf
import os 
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from NIVGG import VGG_class_NI
from load_dataset import *


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4), # fraction of gpu memory used
    device_count = {'GPU': 1}
)
save_path=os.path.join(os.getcwd() ,'data_NI') #dataseet folder
operation_path=os.getcwd() # TMP model save folder
model_path=os.path.join(save_path,'vgg16NI_reduced.h5')#  saved model 


parser = argparse.ArgumentParser() # get arguments
parser.add_argument('--P',nargs="+", default=[0.1],help="Pruning rate e.g 0.1=10% of convolutional filters removed") # pruning rate
parser.add_argument('--criteria', type=str,default='random',help="Criteria used to evaluate fileters. Options: 'l1','apoz','fm_c','random' ") # criteria used to prune e.g l1 norm
parser.add_argument('--globale', type=int,default=False,help='1 if fiters are evaluated on the whole network, 0 if the filters are avaluated layer by layer') # criteria used to select the convolutional filters
parser.add_argument("--B", nargs="+",help="used for the fm_c method, percentages of best filters to keep e.g [0.1,0.2] keeps 10% in the first pass and 20% on the second")
parser.add_argument("--M", nargs="+",help="used for the fm_c method, percentages of worst filters to keep e.g [0.5,0.6] tries to delete 50% in the first pass and 60% on the second")
args = parser.parse_args()



PP=args.P
BB=args.B
MM=args.M
criteria=args.criteria
globale=args.globale
P=[float(i) for i in PP]

if BB !=None and MM!=None:  # check if feature maps per class criteria is being used
    B=[float(i) for i in BB]
    M=[float(i) for i in MM]
    assert len(B)==len(M), "Error: len of B and M must be equal"
    


#debug
console=1
if console==0:
    B=[0.1]
    M=[0.5]
    P=[0.3,0.3]
    criteria='fm_c'
    globale=1
# #debug
print(criteria)
print(globale)
print(P)
try: 
    print(M)
    print(B)
except:
    print('no fm_c')



x_train_reduced,x_test_reduced,x_val_reduced,y_train_reduced,y_test_reduced,y_val_reduced=load_dataset(save_path)

#Load pre trained model if exists
if os.path.exists(model_path):
    model= VGG_class_NI(model_path) 
else:
    model= VGG_class_NI() 
model.summary()


classes=[]
if criteria=='l1':
    for p in P:
        rank=model.rank(x_val_reduced,y_val_reduced,'l1')
        model.pruning(rank,[p for i in range(13)],globale,x_val_reduced,y_val_reduced) 
        score_val,score_test=model.train(x_train_reduced,y_train_reduced,x_val_reduced,y_val_reduced,x_test_reduced,y_test_reduced,operation_path)
    
    flops,n_par,mem=model.check_pruning()
    print('accuracy=',score_test) 
    print('Flops=',sum(flops))
    print('Memory (MB)=', sum(mem)/2**20)
    print('# Parameters (K)=', sum(n_par)/1000)
    np.save(os.path.join(operation_path,'score_test.npy'),score_test)
    np.save(os.path.join(operation_path,'flops.npy'),flops)
    np.save(os.path.join(operation_path,'n_par.npy'),n_par)
    np.save(os.path.join(operation_path,'mem.npy'),mem)
    model.save(os.path.join(os.getcwd() ,'Pruned_model.h5') )

elif criteria=='apoz':
    for p in P:
        reverse_rank=model.rank(x_val_reduced,y_val_reduced,'apoz')
        rank=[]
        for i in reverse_rank:
            rank.append([-k/np.mean(i) for k in i])
        model.pruning(rank,[p for i in range(13)],globale,x_val_reduced,y_val_reduced) 
        score_val,score_test=model.train(x_train_reduced,y_train_reduced,x_val_reduced,y_val_reduced,x_test_reduced,y_test_reduced,operation_path)
        
    flops,n_par,mem=model.check_pruning()
    print('accuracy=',score_test)    
    print('Flops=',sum(flops))
    print('Memory (MB)=', sum(mem)/2**20)
    print('# Parameters (K)=', sum(n_par)/1000)
    
    np.save(os.path.join(operation_path,'score_test.npy'),score_test)
    np.save(os.path.join(operation_path,'flops.npy'),flops)
    np.save(os.path.join(operation_path,'n_par.npy'),n_par)
    np.save(os.path.join(operation_path,'mem.npy'),mem)    
    model.save(os.path.join(os.getcwd() ,'Pruned_model.h5') )

    
elif criteria=='random':
    for p in P:
        rank=[]
        n_filters=model.check_nfilters()# nflters on the vgg16
        for i in n_filters:
            rank.append(np.random.permutation(range(0,i)))   
        model.pruning(rank,[p for i in range(13)],globale,x_val_reduced,y_val_reduced) 
        score_val,score_test=model.train(x_train_reduced,y_train_reduced,x_val_reduced,y_val_reduced,x_test_reduced,y_test_reduced,operation_path)
    
    flops,n_par,mem=model.check_pruning()
    print('accuracy=',score_test)    
    print('Flops=',sum(flops))
    print('Memory (MB)=', sum(mem)/2**20)
    print('# Parameters (K)=', sum(n_par)/1000)
    
    np.save(os.path.join(operation_path,'score_test.npy'),score_test)
    np.save(os.path.join(operation_path,'flops.npy'),flops)
    np.save(os.path.join(operation_path,'n_par.npy'),n_par)
    np.save(os.path.join(operation_path,'mem.npy'),mem)    
    model.save(os.path.join(os.getcwd() ,'Pruned_model.h5') )

elif criteria=='fm_c':
    for b,m in zip(B,M):
        rank=model.rank(x_val_reduced,y_val_reduced,'fm_c')
        model.pruning(rank,[0 for i in range(13)],globale,x_val_reduced,y_val_reduced,classi=[b,m]) 
        score_val,score_test=model.train(x_train_reduced,y_train_reduced,x_val_reduced,y_val_reduced,x_test_reduced,y_test_reduced,operation_path)
        print('test score:',score_test)
    flops,n_par,mem=model.check_pruning()
    print('accuracy=',score_test)    
    print('Flops=',sum(flops))
    print('Memory (MB)=', sum(mem)/2**20)
    print('# Parameters (K)=', sum(n_par)/1000)
    
    np.save(os.path.join(operation_path,'score_test.npy'),score_test)
    np.save(os.path.join(operation_path,'flops.npy'),flops)
    np.save(os.path.join(operation_path,'n_par.npy'),n_par)
    np.save(os.path.join(operation_path,'mem.npy'),mem)
    model.save(os.path.join(os.getcwd() ,'Pruned_model.h5') )
        

del model
K.clear_session()
tf.reset_default_graph()



#comment main library





    






