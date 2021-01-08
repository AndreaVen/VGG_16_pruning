# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 16:18:03 2020

@author: andrea
"""
from __future__ import print_function
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import load_model 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping 
from keras.regularizers import l1
from kerassurgeon import Surgeon # k
import tensorflow as tf
from keras import optimizers
import numpy as np
from keras import backend as K
from keras import regularizers
from tqdm import tqdm
import os 
import numpy as np







config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4),
    device_count = {'GPU': 1}
)




#this class accept the path to a trained VGG 16 model and can prune it with different methods
class VGG_class_NI:
    def __init__(self,train_path):
        self.name='' 
        self.model=load_model(train_path)
        sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
        for layer in self.model.layers: # set all layer to be trainable 
            if 'conv' in layer.name:
                layer.trainable=True
                layer.kernel_regularizer=l1(0.001)
       
        
     #predict output
    def predict(self,x_test):
        x_test = x_test.astype('float32')
        y_pred=self.model.predict(x_test)
        return y_pred
    #summary of the network
    def summary(self):
        return self.model.summary()
    
    
    #change the "head" of the VGG16, d1 and d2 are the new fully connected layers and d3 is the output layer with d3=number of classes to classify.
    
    def reduce(self,d1=512,d2=512,d3=3): 
        model=self.model
        print(model.layers[-4])
        new_model=Model(input=[model.input],output=[Dense(d1,activation="relu")(self.model.layers[-4].output)])
        new_model=Model(input=[self.model.input],output=[Dense(d2,activation="relu")(new_model.output)])
        new_model=Model(input=[self.model.input],output=[Dense(d3,activation="softmax")(new_model.output)])
        self.model=new_model
        # self.model.layers.pop()

        
        
        
    #evaluate test set
    def evaluate(self,x_test,y_test):
        learning_rate = 0.001
        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
        my_score=self.model.evaluate(x_test,y_test)
        return my_score

    #save the model 
    def save(self,path):
         self.model.save(path)
    #load the model
    def load(self,path):
        self.model=load_model(path)
    #Random prunint on P% of filters. Every layer gets P% of filters removed, the indexes of the filters removed are returned as mega_list
    def random_pruning(self,P):
        from kerassurgeon import Surgeon 
        surgeon = Surgeon(self.model)
        layer_list=self.model.layers # list of layers in the model
        to_prune_list=[]
        for i in range(len(layer_list)):
            if 'conv' in str(layer_list[i]): #select convolutional layers only
                to_prune_list.append(i) # gets their indexes
        print(to_prune_list)
        n_filters=[]
        for i in to_prune_list:
             # gets the number of convolutional filters in every layer
            n_filters.append(self.model.get_layer(index=i).get_config()['filters'])

        mega_list=[]  #mega list is a list of list where every element are the index of the filter removed in that layer
        print(n_filters)
        for kk in range(len(to_prune_list)): # for every conv layer
            num_filter_layer=n_filters[kk]# number of filters in that layer
            random_index=[] # index of the filters to be pruned 
            for jj in range(int(np.ceil(P*num_filter_layer))): # gets random indexes
                tmp=np.random.randint(0,num_filter_layer)
                while tmp in random_index:
                    tmp=np.random.randint(0,num_filter_layer) # if the filter has alread benn choosen, try again 
                random_index.append(tmp)
               
            mega_list.append(random_index) 
            layerr=self.model.layers[to_prune_list[kk]]
            surgeon.add_job('delete_channels', layerr, channels=random_index) 
            
        self.model = surgeon.operate()
    
        return mega_list
    
    # This method takes a vector percentage as input where every element in the vector is 
    #the percentage of random pruning of the relative layer. 
    def percentage_pruning(self,percentage): 
        from kerassurgeon import Surgeon # k
        surgeon = Surgeon(self.model)
        layer_list=self.model.layers # list of layers in the model 
        to_prune_list=[]
        for i in range(len(layer_list)):
            if 'conv' in str(layer_list[i]):
                to_prune_list.append(i) # check the indexes of the convolutional layers in the model
        print(to_prune_list)
        n_filters=[]
        for i in to_prune_list:
            # gets the number of filter in each convolutional layer
            n_filters.append(self.model.get_layer(index=i).get_config()['filters'])
        mega_list=[]   
        print(n_filters)
        for kk,_ in enumerate(to_prune_list): # iterate on each convolutional layer 
            num_filter_layer=n_filters[kk]# number of filters in the current layer, which is to_prune_list[kk]
            random_index=[] # index of the filters to be pruned 
            for jj in range(int(np.ceil(percentage[kk]*num_filter_layer))): # P% of the filters 
                tmp=np.random.randint(0,num_filter_layer)
                while tmp in random_index:
                    tmp=np.random.randint(0,num_filter_layer) # if the filter has alread benn choosen, try again 
                random_index.append(tmp)
               
            mega_list.append(random_index) # contain a list of all the random index of that layer, the length of the mega list is thus the length 
            # of the convolutional filters 
            layerr=self.model.layers[to_prune_list[kk]]
            surgeon.add_job('delete_channels', layerr, channels=random_index) 
        self.model = surgeon.operate()
        return mega_list


    def input(self):
            return self.model.input
    def output(self):
        return self.model.output
    
    
    
    #This method uses a binary mask to remove the filters from the network. The length of the vector Vec is equal to
    #the number of conv filters in the network. In this vecto the elemen marked 1 is removed, the 0 are left as they are.
    def mask_pruning(self,Vec):
        vec=[]
        for idx,i in enumerate(Vec):
            if i==1:
                vec.append(idx) # appendo nel vettore vec gli indici dei filtri da eliminare 
        
        #this function takes a vector of indexes, if there is a 1 in the j-th position the j-th filter of the network will be pruned.
        from kerassurgeon import Surgeon # k
        surgeon = Surgeon(self.model)
        layer_list=self.model.layers # list of layers in the model e.g [<keras.engine.input_layer.InputLayer at 0x2068f6c2048>,...
        to_prune_list=[]
        for i in range(len(layer_list)):
            if 'conv' in str(layer_list[i]):
                to_prune_list.append(i) # check the indexes of the convolutional layers in the model
        n_filters=[]
        # print(to_prune_list)
        for i in to_prune_list:
            n_filters.append(self.model.get_layer(index=i).get_config()['filters']) # gets the number of filter in each convolutional layer
        # print('num,er of filters=',n_filters)
        # return n_filters
        
        
        sumlist=[0 for i in n_filters] # vettore lungo quanto il numero di strati 
        for idx in range(len(n_filters)):
            sumlist[idx]=sum(n_filters[0:idx+1])    #[64,64,128]->[64,128,256]
            
        # print(sumlist)
        # print('vec=',vec)
        mega_list=[[],[],[],[],[],[],[],[],[],[],[],[],[]]
        for num in vec:
            for idx,i in enumerate(sumlist):
                if num-i<0:
                    strato=idx
                    if idx==0:
                        numero=num
                    else:
                        numero=num-sumlist[idx-1]
                    # print('strato=',idx,'numero=',numero)
                    mega_list[idx].append(numero)
         
                    break
                   
            # print('dimensione di mega list=',len(mega_list))
        for kk in range(len(to_prune_list)): # iterate on each convolutional layer 
            # of the convolutional filters 
            layerr=self.model.layers[to_prune_list[kk]]
            # print('len di mega_list[kk]=',len(mega_list[kk]))
            surgeon.add_job('delete_channels', layerr, channels=mega_list[kk]) 
            
        self.model = surgeon.operate()
      
        
        return 0
       
  
            
     #This method uses the binary vector trainable_index to enable or disable the training of a specified layer.
     # the length of the vector is the same as the number of layer, 1 means the layers is trainable and 0 means it will not be trained.
    def set_trainable_layers(self,trainable_index):
        vec=[l for l in self.model.layers if 'conv' in l.name or 'dense' in l.name]
        print('len(vec)',len(vec))
        for idx,l in enumerate(vec):
            if trainable_index[idx]==1:
                vec[idx].trainable=True
            else:
                vec[idx].trainable=False
        self.model.summary()
    #check trainable layers 
    def is_trainable(self):
        for l in self.model.layers:
            print(l.name, l.trainable)
    # Option to give an unique code to the network  
    def give_name(self,name):
        self.name=name
        self.model.name=name
    #Return unique name of the network
    def return_name(self):
        return self.model.name
    
    
    
   
            
    
    def check_pruning(self):
        #This method returns the statistic of the current architecture: flops, feature maps memory, number of parameters.
        # the method return these statistics for every layer of the network
        #the formula used is:
        #FLOPS calculation wout*Hout*k**2*cin*cout (convolution)
        #FLOPS calculation Wout*Hout*k**2*cin (pooling layer)
        #FLOPS calculation Cin*cout (dense layer)
        #MEM cout*Hout*Wout*4/1024 (mem of feature maps from conv layers and pool layers)
        #MEM C*4/1024 # dense memory
        #N PAR conv layer= Cin*cout*k**2 + cout 
        #N PAR Poolinglayer =0
        #N PAR Dense layer= Cin*cout+cout
        # flops=0
        #
        n=[3]
        for layer in self.model.layers:
            if 'conv' in layer.name:
                tmp=np.array(layer.get_weights()[0]).squeeze().shape[-1]
                n.append(tmp)
    
        dimension=[224,224,112,112,56,56,56,28,28,28,14,14,14]  # number of filters in vgg16, to be modified for differnt architectures
        

        flops=[n[i]*n[i+1]*dimension[i]**2*9 for i in range(len(n)-1)]
        pooling=9*[n[3]*112**2,n[5]*56**2,n[8]*28**2,n[11]*14**2,n[-1]*7**2] # pooling layers
        new_flops=flops[0:2]
        new_flops.append(pooling[0])
        new_flops.extend(flops[2:4])
        new_flops.append(pooling[1])
        new_flops.extend(flops[4:7])
        new_flops.append(pooling[2])
        new_flops.extend(flops[7:10])
        new_flops.append(pooling[3])
        new_flops.extend(flops[10:13])
        new_flops.append(pooling[4])
        dense=[]
        for layer in self.model.layers:
           if 'dense' in layer.name:
               tmp=np.array(layer.get_weights()[0]).squeeze().shape[-1]
               dense.append(tmp)
        new_flops.append(n[-1]*49*dense[0])
        new_flops.append(dense[1]*dense[0])
        new_flops.append(dense[1]*3)
        num_par=[n[i]*n[i+1]*9+n[i+1] for i in range(len(n)-1)]
        tb=[2,5,9,13,17]
        for i in tb:
            num_par.insert(i,0)
        num_par.append(n[-1]*49*dense[0]+dense[0])
        num_par.append(dense[1]*dense[0]+dense[1])
        num_par.append(dense[1]*3+3) # output layer
        mem=[]
        for i in range(len(n)-1):
            mem.append(n[i+1]*dimension[i]**2*4/1024)
        
        tb=[2,5,9,13,17]
        tt=[112**2*64,56**2*128,28**2*256,14**2*512,49*512]
        for idx,i in enumerate(tb):
            mem.insert(i,tt[idx]*4/1024)
            
        mem.append(dense[0]*4/1024)
        mem.append(dense[1]*4/1024)
        mem.append(3*4/1024)
          
        return new_flops,num_par,mem
    
    
    def return_model(self):
        return self.model
    
        
            
    #training of the neural network. The train set is used for training, the validation set is needed for the 
    #early stopping criteria. If a folder is given the method saves the temporary model there, otherwise it saves it in the current folder.
    
    def train(self,x_train,y_train,x_val,y_val,x_test,y_test,path_check=os.getcwd()):
        batch_size = 16
        maxepoches = 20
        learning_rate = 0.001
        sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
        early = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=3, verbose=1, mode='auto')
        check_name=os.path.join(path_check,'NI_best_TMP.h5') # name of the temporary model
        checkpoint = ModelCheckpoint(check_name,monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        historytemp = self.model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=maxepoches, validation_data=(x_val, y_val),callbacks=[early,checkpoint],
                           verbose=1)

        self.model.load_weights(check_name)#load the best model obtained 
        score_val=self.model.evaluate(x_val,y_val)  #validation accuracy
        score_test=self.model.evaluate(x_test,y_test) #test accuracy
        
        print('score_val={},score_test={}'.format(score_val,score_test)) 
        return score_val[1],score_test[1] 

    #this method returns the numer of filters for every layer 
    def check_nfilters(self):
        n=[]
        for layer in self.model.layers:
            if 'conv' in layer.name:
                tmp=np.array(layer.get_weights()[0]).squeeze().shape[-1]
                n.append(tmp)
        return n        
    
    
    

    
    
   
  
    # this method is used to rank the filters in the network. method is a string that can be 'l1','fm_c','apz'
    #the fm_c work with 3 output classes 
    def rank(self,x_,y,metodo):
        if metodo=='fm_c':
            inp = self.model.input     # input of the nwtwork
            outputs = [layer.output for layer in self.model.layers if 'conv' in layer.name ] #output of the network
            functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
            mean_layer=[[],[],[]] #rank for the 3 classes
            num=0
            def l1_image(strato):
                norm=[]
                for iix in range(strato.shape[-1]):
                    norm.append(np.sum(abs(strato.squeeze()[:,:,iix])))      
                norm=norm/np.mean(norm)
                return norm
            
            for idx,i in tqdm(enumerate(x_)):
                classe=np.argmax(y[idx]) # class of the given image
                num+=1
                x=np.reshape(i,[1,224,224,3])
                layer_outs = [func([x, 1.]) for func in functors] # list of layers output
                if len(mean_layer[classe])==0:# first image
                    mean_layer[classe]=[l1_image(g[0]) for g in layer_outs ] 
                else:
                    mean_layer[classe]=np.array(mean_layer[classe]).squeeze()+np.array([l1_image(g[0]) for g in layer_outs ])
                #mean layer is a vector that contains the mean output (of all the images) of all the layers
                
            idx=0
            started=0
            layer=[lay for lay in self.model.layers if 'conv' in lay.name]
            mega_norm0=[]
            mega_norm1=[]
            mega_norm2=[]
            n_filters=[64,64,128,128,256,256,256,512,512,512,512,512,512]
            for idx,i in enumerate(layer):
                # for every layer and for every image
                norm0=mean_layer[0][idx].squeeze()  # prendo tutte le fm di quello strato
                norm0=norm0*3/(len(x_))
                norm1=mean_layer[1][idx].squeeze()  # prendo tutte le fm di quello strato
                norm1=norm1*3/(len(x_))
                norm2=mean_layer[2][idx].squeeze()  # prendo tutte le fm di quello strato
                norm2=norm2*3/(len(x_))
                mega_norm0.append(norm0)
                mega_norm1.append(norm1)
                mega_norm2.append(norm2)
            mega_norm=[mega_norm0,mega_norm1,mega_norm2]
            return mega_norm
        elif metodo=='l1':
            pesi=[]
            def sum_weights(pesi):
                #given a layer pesi in input the function return the l1 norm of that layer
                norm=[]
                for i in range(pesi.shape[-1]):
                    tmp=np.sum(abs(pesi[:,:,:,i]))/pesi.shape[0]**2
                    norm.append(tmp)
                return norm 
            started=0
            idx=0
            to_remove_channel=[]
            global_arr=[]
            for layer in self.model.layers:
                if 'conv' in layer.name:
                   
                    pesi.append(layer.get_weights())
                    tmp=layer.get_weights()[0]# taking kernel weights and not their bias
                    if started: 
                        tmp=np.delete(tmp,to_remove_channel,2)
                    arr=sum_weights((tmp))
                    arr=arr/np.mean(arr)   
                    global_arr.append(arr) 
            return global_arr
        elif metodo=='apoz':
            def apz_image(strato):
               norm=[]
               for iix in range(strato.shape[-1]):
                   # print('numero di zeri=',np.sum(strato.squeeze()[:,:,iix]==0))
                   norm.append(np.sum(strato.squeeze()[:,:,iix]==0))
               norm=np.array(norm)/(strato.shape[-2]**2)
               return norm
            inp = self.model.input                                        # input placeholder
            outputs = [layer.output for layer in self.model.layers if 'conv' in layer.name ]
            functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
            norm=[]
            mean_layer=[]
            num=0
            for idx,i in tqdm(enumerate(x_)):
                classe=np.argmax(y[idx])
                num+=1
                x=np.reshape(i,[1,224,224,3])
                layer_outs = [func([x, 1.]) for func in functors]
                if len(mean_layer)==0:# prima immagine
                    mean_layer=[apz_image(g[0]) for g in layer_outs ]
                else: 
                    mean_layer=np.array(mean_layer).squeeze()+np.array([apz_image(g[0]) for g in layer_outs ])
        
            idx=0
            started=0
            layer=[lay for lay in self.model.layers if 'conv' in lay.name]
            mega_arr=[] 
            for k in mean_layer:
                arr=k/len(x)
                mega_arr.append(arr)
            return mega_arr  

    # this method execute the pruning of the network given the ranked filters list in input rank. vec is a vector that specifies the pruning rate
    # for every layer. meodo is a flag that specifies how the rank list is used. 
    #x_ and y is the validation set or a subset of images that might be needed to train the network after the pruning 
    #classi is a list only used if the rank has been made with the fm_c criteria. the list contains the best percentage of filters to keep B 
    #and the worst percentage to remove M
    #method==0: the worst filters from rank are evaluated layer by layer. Once the worst filters are decided the whole network is pruned,
    #the train of the network must be made outside this method.
    #method==1: the worst filters from rank are evaluated globally on the whole network. Once the worst filters are decided the whole network is pruned,
    #the train of the network must be made outside this method.
    #method==2 the worst filters from rank are evaluated layer by layer. The worst filters are removed one layer at time,
    #after a layer is pruned the whole network is trained on the valisation set x_,y.
    def pruning(self,rank,vec,metodo,x_,y,classi=[]): 
        lunghezza=len(rank)
        if type(vec)==list: # vec can contain either the percentage of filters to remove or their exact number
            if type(vec[0])==int:
                type_flag=0 
            elif type(vec[0])==float: 
                type_flag=1 
        else:
            type_flag=2 
        if lunghezza==3: # cm_c criterion is used for the rank
            mega_norm0=rank[0]     
            mega_norm1=rank[1]   
            mega_norm2=rank[2] 
            n_filters=[64,64,128,128,256,256,256,512,512,512,512,512,512] # number of filters in vgg 16
       

            un_arr0=[oo for o in mega_norm0 for oo in o]
            un_arr1=[oo for o in mega_norm1 for oo in o]
            un_arr2=[oo for o in mega_norm2 for oo in o]
            if metodo==0:# rank is used layer by layer with no training in between
                surgeon = Surgeon(self.model)   
                B=classi[0]
                M=classi[1]
                layer=[lay for lay in self.model.layers if 'conv' in lay.name]
                for idx,i in enumerate(layer): #get the best and worst filters for every class.
                    best0=list(np.argsort(mega_norm0[idx])[-int(B*len(mega_norm0[idx]))::]) 
                    best1=list(np.argsort(mega_norm1[idx])[-int(B*len(mega_norm0[idx]))::])
                    best2=list(np.argsort(mega_norm2[idx])[-int(B*len(mega_norm0[idx]))::])
                    worst0=list(np.argsort(mega_norm0[idx])[0:int(M*len(mega_norm0[idx]))])
                    worst1=list(np.argsort(mega_norm1[idx])[0:int(M*len(mega_norm0[idx]))])
                    worst2=list(np.argsort(mega_norm2[idx])[0:int(M*len(mega_norm0[idx]))])
                    to_prune=[]
                    for iii in  worst0:                        
                        if (iii not in best1) and (iii not in best2):
                            if iii not in to_prune:
                                to_prune.append(iii)        
                    for iii in worst1:
                        if (iii not in best0) and (iii not in best2):
                           if iii not in to_prune:
                                to_prune.append(iii)
                    for iii in worst2:
                        if (iii not in best0) and (iii not in best1):
                            if iii not in to_prune:
                                to_prune.append(iii)
                    print('number of filters to remove=',len(to_prune))
                    surgeon.add_job('delete_channels', layer[idx], channels=to_prune )    
                self.model=surgeon.operate()
            elif metodo==1: # rank is used globally with no training in between                
                surgeon = Surgeon(self.model)   
                B=classi[0]
                M=classi[1]
                best0=list(np.argsort(un_arr0)[-int(B*sum(self.check_nfilters()))::])
                best1=list(np.argsort(un_arr1)[-int(B*sum(self.check_nfilters()))::])
                best2=list(np.argsort(un_arr2)[-int(B*sum(self.check_nfilters()))::])
                worst0=list(np.argsort(un_arr0)[0:int(M*sum(self.check_nfilters()))])
                worst1=list(np.argsort(un_arr1)[0:int(M*sum(self.check_nfilters()))])
                worst2=list(np.argsort(un_arr2)[0:int(M*sum(self.check_nfilters()))])      
                lay=[i for i in self.model.layers if 'conv' in i.name]
                
                for indice,hj in enumerate(lay):
                    to_prune=[]
                    for iii in  worst0:
                        if (iii not in best1) and (iii not in best2):
                            if un_arr0[iii] in mega_norm0[indice]:
                                tmp=np.argwhere(mega_norm0[indice]==un_arr0[iii])
                                for gt in tmp:
                                     if gt not in to_prune and len(to_prune)<int(n_filters[indice]*0.95): 
                                         # used to prevent a whole network to be pruned
                                         to_prune.append(int(gt))
                    
                    for iii in  worst1:
                        if (iii not in best0) and (iii not in best2):
                            if un_arr1[iii] in mega_norm1[indice]:
                                tmp=np.argwhere(mega_norm1[indice]==un_arr1[iii])
                                for gt in tmp:
                                    if gt not in to_prune and len(to_prune)<int(n_filters[indice]*0.95):                  
                                        to_prune.append(int(gt))
              
                    for iii in  worst2:
                        if (iii not in best0) and (iii not in best1):
                            if un_arr2[iii] in mega_norm2[indice]:
                                tmp=np.argwhere(mega_norm2[indice]==un_arr2[iii])
                                for gt in tmp:
                                    if gt not in to_prune and len(to_prune)<int(n_filters[indice]*0.95):  
                                        to_prune.append(int(gt))
               
                    surgeon.add_job('delete_channels', hj, channels=to_prune )  
                    print('number of filters to remove=',len(to_prune))
                self.model=surgeon.operate()
   
            elif metodo==2:# rank is used layer by layer but there is a training for every pruning step
                surgeon = Surgeon(self.model)   
                B=classi[0]
                M=classi[1]
                save_path=os.path.join(os.getcwd(),'saved_model_tmp.h5')
                self.model.save(save_path) # save 
                for idx in range(12,-1,-1):
                    self.model=load_model(save_path)
                    trainable_lay=[0 for i in range(16)] # all layers are trainable
                    trainable_lay[idx]=1
                    self.set_trainable_layers(trainable_lay)
                    surgeon = Surgeon(self.model)
                    layer=[jj for jj in self.model.layers if 'conv' in jj.name]

                    best0=list(np.argsort(mega_norm0[idx])[-int(B*len(mega_norm0[idx]))::])
                    best1=list(np.argsort(mega_norm1[idx])[-int(B*len(mega_norm0[idx]))::])
                    best2=list(np.argsort(mega_norm2[idx])[-int(B*len(mega_norm0[idx]))::])
                    worst0=list(np.argsort(mega_norm0[idx])[0:int(M*len(mega_norm0[idx]))])
                    worst1=list(np.argsort(mega_norm1[idx])[0:int(M*len(mega_norm0[idx]))])
                    worst2=list(np.argsort(mega_norm2[idx])[0:int(M*len(mega_norm0[idx]))])
                    to_prune=[]
                    for iii in  worst0:                        
                        if (iii not in best1) and (iii not in best2):
                            if iii not in to_prune:
                                to_prune.append(iii)        
                    for iii in worst1:
                        if (iii not in best0) and (iii not in best2):
                           if iii not in to_prune:
                                to_prune.append(iii)
                    for iii in worst2:
                        if (iii not in best0) and (iii not in best1):
                            if iii not in to_prune:
                                to_prune.append(iii)
                    print('number of filters to remove=',len(to_prune))
                    surgeon.add_job('delete_channels', layer[idx], channels=to_prune )  
                    self.model=surgeon.operate() 
                    sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
                    self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
                    for i in range(5): # 
                        self.model.fit(x_,y,epochs=3,batch_size=8)
                        performance=self.model.evaluate(x_,y)
                        if performance[1]>0.9:
                            break
                    self.model.save(save_path)
                    del self.model
                    K.clear_session()
                    tf.reset_default_graph()
                self.model=load_model(save_path)  
        elif lunghezza !=3:# l1 and apz criteria
            if metodo==0: 
                surgeon = Surgeon(self.model)   
                layer=[jj for jj in self.model.layers if 'conv' in jj.name]
                un_arr0=[oo for o in rank for oo in o]
                for idx in range(13):
                    norm=rank[idx]
                    if type_flag==0 :
                        indici=np.array(norm).argsort()[0:vec[idx]]                        
                    elif type_flag==1: 
                        indici=np.array(norm).argsort()[0:int(vec[idx]*len(rank[idx]))]                        

                    to_remove_channel=indici.tolist()
                    surgeon.add_job('delete_channels', layer[idx], channels=indici )
                self.model=surgeon.operate()                    
            elif metodo==1:
                surgeon = Surgeon(self.model)   
                un_arr=[oo for o in rank for oo in o]
                layer=[jj for jj in self.model.layers if 'conv' in jj.name]
                if type_flag==0:
                    norm=np.argsort(un_arr)[0:sum(vec)]# ordered vector
                elif type_flag==1:# percentage is used
                    norm=np.argsort(un_arr)[0:int(np.sum(self.check_nfilters())*vec[0])]
                    print('len norm=',len(norm))
                
                for idx,jj in enumerate(layer):
                    to_prune=[]
                    for kk in norm:
                        if un_arr[kk] in rank[idx]:
                            tmp=np.argwhere(rank[idx]==un_arr[kk]).squeeze()
                            try: 
                                if int(tmp) not in to_prune and len(to_prune)<int(n_filters[idx]*0.95):
                                    to_prune.append(int(tmp))
                            except:
                                for ij in tmp:
                                    if ij not in to_prune and len(to_prune)<int(n_filters[idx]*0.95):
                                        to_prune.append(ij)
                    print('len(to_prune)',len(to_prune),idx)           
                    surgeon.add_job('delete_channels', layer[idx], channels=to_prune )
    
                self.model=surgeon.operate()
            elif metodo==2: # faccio un train per ogni strato 
                surgeon = Surgeon(self.model)   
                self.model.save('R:\\saved_model_tmp.h5')
                for idx in range(12,-1,-1):
                    self.model=load_model('R:\\saved_model_tmp.h5')
                    # trainable_lay=[0 for i in range(16)]
                    # trainable_lay[idx]=1
                    # self.set_trainable_layers(trainable_lay)
                    surgeon = Surgeon(self.model)
                    layer=[jj for jj in self.model.layers if 'conv' in jj.name]
                    norm=rank[idx]
                    if type_flag==0 : #lavoro direttamente con il numero di filtri 
                        indici=np.array(norm).argsort()[0:vec[idx]]                        
                        print('len(indici)=',len(indici))
                    elif type_flag==1: # specifico solo la percentuale
                        indici=np.array(norm).argsort()[0:int(vec[idx]*len(rank[idx]))]                        
                        print('len(indici)=',len(indici))
                    elif type_flag==2:
                        #specifico la distanza da una sigma 
                        std=np.std(norm)
                        index=(norm-np.man(norm))<-std
                        indici=np.argwhere(index==1)
                    
                    
                    print('number of filters to remove=',len(indici))
                    surgeon.add_job('delete_channels', layer[idx], channels=indici)  
                    self.model=surgeon.operate() 
                    sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
                    self.model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
   
                    for i in range(5):
                        self.model.fit(x_,y,epochs=3,batch_size=8)
                        performance=self.model.evaluate(x_,y)
                        if performance[1]>0.9:
                            break
                    save_path=os.path.join(os.getcwd(),'saved_model_tmp.h5')
                    self.model.save(save_path)
                    del self.model
                    K.clear_session()
                    tf.reset_default_graph()
                self.model=load_model(save_path)  
    
    #this method returns the feature maps of a single image passed in input.
    def show_fm(self,x_,y):
        from tqdm import tqdm
        inp = self.model.input                                     
        outputs = [layer.output for layer in self.model.layers if 'conv' in layer.name ]
        functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
        layer_outs = [func([x_, 1.]) for func in functors] 
            
        return layer_outs  

        