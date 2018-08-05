from builtins import range
from builtins import object
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedKFold

class logisticRegression(object):
    '''
    implents logistic regression using numpy
    '''
    def __init__(self):
        self.loss_hist = []
        self.acc_hist = []
        self.class_w = []
        
    def costfunction(self,x,h,y):
            m = y.shape[0]
            loss = np.asscalar((-1/m)*(np.dot(np.transpose(y),np.log(h)) + np.dot(np.transpose(1-y),np.log(1-h))))
            grad = np.dot(np.transpose(x),(h - y))*(1/m)
            return loss, grad 
    
    def score(self,X,y,class_lst):
        class_pred = []
        self.class_w = np.array(self.class_w)
        for i in range(self.class_w.shape[0]):
            z_pred = np.array(np.dot(X,self.class_w[i]),dtype=np.float32)
            pred = 1/(1 + np.exp(-z_pred))
            class_pred.append(pred)
        class_pred = np.array(class_pred).reshape(self.class_w.shape[0],X.shape[0])
        final_pred = np.argmax(class_pred,axis=0)+1
        cls_int = 1
        for cls in class_lst:
            y[np.where(y == cls)] = cls_int
            cls_int +=1
        accuracy = np.mean([final_pred == y],dtype=np.float32)
        return accuracy
    
    def fit(self,train_ds,class_lst,batch_size=120,num_iter=100,kfold=10,learning_rate=0.1,print_every=10):
        print('training for lr:',learning_rate)
        train_batch = train_ds.sample(n=batch_size)
        train_arr = train_batch.values
        train_x = np.hstack((np.ones((train_arr.shape[0],1)),train_arr[:,0:4]))
        train_y = train_arr[:,-1]
        skf = StratifiedKFold(n_splits=kfold)
        skf.get_n_splits(train_x,train_y)
        fold_count = 0
        accuracy = 0.0
        for train_index, test_index in skf.split(train_x,train_y):
            self.class_w = []
            fold_count +=1
            print('training fold:',fold_count)
            for cls in class_lst:
                print('training for class:',cls)
                train_split_x, test_split_x = train_x[train_index], train_x[test_index]
                train_split_y, test_split_y = train_y[train_index], train_y[test_index]
                train_split_y = np.array([train_split_y == cls],dtype=np.int32)
                train_split_y = train_split_y.reshape(train_split_x.shape[0],-1)
                loss = 0.0
                self.loss_hist = []
                w = np.zeros((train_ds.shape[1],1))
                for i in range(num_iter):
                    z = np.array(np.dot(train_split_x,w),dtype=np.float32)
                    h = 1/(1 + np.exp(-z))
                    loss,grad = self.costfunction(train_split_x,h,train_split_y)
                    w = w - learning_rate*grad
                    self.loss_hist.append(loss)
                self.class_w.append(w)
            accuracy = self.score(test_split_x,test_split_y,class_lst)
            print('overall accuracy for lr: ',learning_rate,' is :',accuracy)