import numpy as np
import pandas as pd
import math

class baseModel(object):
    '''
    this is base model which predicts the mean value for a given item_type and outlet_type
    '''
    def __init__(self):
        print('base model object instantiated')
        
    def fit(self,data):
        self.train_ds = data
        
    def score_reg(self,data):
        data_arr = data.values
        feature = data_arr[:,0:-1]
        labels = data_arr[:,-1].reshape(-1,1)
        n = data_arr.shape[0]
        pred = np.array([np.mean(labels,axis=0)]*n).reshape(-1,1)
        #print(pred)
        rmse = np.sqrt((1/n)*np.sum(np.square(pred-labels),axis=0))
        return rmse
    
class KNearestNeighbor(object):
    '''
    implement K Nearest Neighbor alogrithm usin numpy
    '''
    def __init__(self,k):
        self.train_data = []
        self.K = k
        print('KNN object instantiated')
        
    def fit(self,data):
        '''
        Since KNN is a lazy algorithm it doesn't do much during training time except for memorizing all training example.
        '''
        self.train_data = data
    
    def get_neighbor_labels(self,x,dist_type):
        x = x.reshape(1,-1)
        train_arr = self.train_data.values
        features = train_arr[:,0:-1]
        labels = train_arr[:,-1]
        labels = labels.reshape(-1,1)
        if dist_type == 'euclidean':
            dist = np.sqrt(np.sum(np.square(features-x),axis=1)).reshape(-1,1)
        if dist_type =='manhattan':
            dist = np.sum(np.absolute(features-x),axis=1).reshape(-1,1)
        dist = np.hstack((dist,labels))
        dist = dist[np.argsort(dist[:,0])]
        lables = dist[:,-1].reshape(-1,1).astype(int)
        return lables
    
    def score_class(self,data,dist_type='euclidean'):
        data_arr = data.values
        feature = data_arr[:,0:-1]
        labels = data_arr[:,-1]
        pred = np.zeros((labels.shape))
        for i in range(data_arr.shape[0]):
            pred[i] = self.predict_class(feature[i,:],dist_type)
        accuracy = np.mean([pred==labels],dtype=np.float32)
        return accuracy
                                
    def predict_class(self,x,dist_type='euclidean'):
        neighbor_labels = self.get_neighbor_labels(x,dist_type)
        label_counts = np.bincount(neighbor_labels[0:self.K,0])
        return np.argmax(label_counts)
    
    def score_reg(self,data,dist_type='euclidean'):
        data_arr = data.values
        feature = data_arr[:,0:-1]
        labels = data_arr[:,-1].reshape(-1,1)
        n = data_arr.shape[0]
        pred = np.zeros((labels.shape))
        for i in range(n):
            pred[i] = self.predict_reg(feature[i,:],dist_type)
        rmse = np.sqrt((1/n)*np.sum(np.square(pred-labels),axis=0))
        return rmse
                                   
    def predict_reg(self,x,dist_type='euclidean'):
        neighbor_labels = self.get_neighbor_labels(x,dist_type)
        label_avg = np.mean(neighbor_labels[0:self.K,0],axis=0)
        return label_avg
    
class weightedKNN(object):
    '''
    in this class we will build the weighted KNN alogrithm
    '''
    def __init__(self,k):
        self.k = k
        self.train_data = []
        print('weighted KNN object instantiated')
        
    def fit(self,data):
        '''
        Since KNN is a lazy algorithm it doesn't do much during training time except for memorizing all training example.
        '''
        self.train_data = data
        
    def get_weighted_labels(self,x,kernal):
        x = x.reshape(1,-1)
        train_arr = self.train_data.values
        features = train_arr[:,0:-1]
        labels = train_arr[:,-1]
        labels = labels.reshape(-1,1)
        dist = np.sqrt(np.sum(np.square(features-x),axis=1)).reshape(-1,1)
        weighted_lables = np.hstack((dist,labels))
        #print('weighted_lables',weighted_lables)
        weighted_lables = weighted_lables[np.argsort(weighted_lables[:,0])]
        #print('weighted_lables',weighted_lables)
        k_weighted_lables = weighted_lables[0:self.k,:]
        #print('k_weighted_lables',k_weighted_lables)
        dist_k_plus1 = weighted_lables[self.k,0]
        #print('dist_k_plus1+1e-20',dist_k_plus1++1e-20)
        k_weighted_lables[:,0] = k_weighted_lables[:,0]/(dist_k_plus1+1e-20)
        #print('k_weighted_lables',k_weighted_lables)
        if kernal == 'gauss':
            k_weighted_lables[:,0] = (1/math.sqrt(2*math.pi)*np.exp(-np.square(k_weighted_lables[:,0])/2))
        if kernal == 'inversion':
            k_weighted_lables[:,0] = 1/(np.abs(k_weighted_lables[:,0])+1e-20)
        #print('k_weighted_lables',k_weighted_lables)
        return k_weighted_lables
    
    def score_class(self,data,kernal='gauss'):
        data_arr = data.values
        feature = data_arr[:,0:-1]
        labels = data_arr[:,-1]
        pred = np.zeros((labels.shape))
        for i in range(data_arr.shape[0]):
            #print('feature[i,:]',feature[i,:])
            pred[i] = self.predict_class(feature[i,:],kernal)
            #print('pred[i]',pred[i])
        accuracy = np.mean([pred==labels],dtype=np.float32)
        return accuracy
    
    def predict_class(self,x,kernal):
        weighted_lables = self.get_weighted_labels(x,kernal)
        #print('weighted_lables',weighted_lables)
        avg_weight_lables = []
        #print('np.unique(weighted_lables[:,1])',np.unique(weighted_lables[:,1]))
        for x in sorted(np.unique(weighted_lables[:,1])):
            avg_weight_lable = [np.sum(weighted_lables[np.where(weighted_lables[:,1]==x)]),x]
            avg_weight_lables.append(avg_weight_lable)
        avg_weight_lables = np.array(avg_weight_lables)
        #print('avg_weight_lables',avg_weight_lables.shape)
        predicted_class = avg_weight_lables[np.argmax(avg_weight_lables[:,0])][1]
        return predicted_class
        