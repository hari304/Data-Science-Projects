from builtins import range
from builtins import object
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
#import tqdm

class logisticRegression(object):
    '''
    implents logistic regression using numpy
    '''
    def __init__(self):
        self.loss_hist = []
        self.acc_hist = []
           
    def costfunction(self,x,h,y):
            y = y.reshape(x.shape[0],1)
            m = y.shape[0]
            loss = np.asscalar((-1/m)*(np.dot(np.transpose(y),np.log(h)) + np.dot(np.transpose(1-y),np.log(1-h))))
            grad = np.dot(np.transpose(x),(h - y))*(1/m)
            return loss, grad 
    '''    
    def score(self,X,y,w,show_cm=False):
        z_pred = np.array(np.dot(X,w),dtype=np.float32)
        pred = 1/(1 + np.exp(-z_pred)) 
        pred =np.array([pred>=0.5],dtype=np.int32)
        pred = pred.flatten()
        true_positive = np.count_nonzero(pred[(pred==1) & (pred == y)])
        false_positive = np.count_nonzero(pred[(pred==1) & (pred != y)])
        true_negative = np.count_nonzero(pred[(pred==0) & (pred == y)]==0)
        false_nagative = np.count_nonzero(pred[(pred == 0) & (pred !=y)]==0)
        if show_cm:
            print('true_positive',true_positive)
            print('false_positive',false_positive)
            print('true_negative',true_negative)
            print('false_nagative',false_nagative)
            print('precision',(true_positive/(true_positive+false_positive)))
            print('sensitivity or recall',(true_positive/(true_positive+false_nagative)))
            print('specificity',(true_negative/(true_negative+false_positive)))
        accuracy = (true_positive + true_negative)/(true_positive + false_positive + true_negative + false_nagative)
        return accuracy
    '''
    def score(self,X,y,w,show_cm=False):
        z_pred = np.array(np.dot(X,w),dtype=np.float32)
        pred_prob = 1/(1 + np.exp(-z_pred))
        rand_num = 100
        cut_offs = np.random.rand(rand_num)
        sensitivity = np.zeros((rand_num))
        one_minus_specificity =np.zeros((rand_num))
        for i in range(len(cut_offs)):  
            pred =np.array([pred_prob>=cut_offs[i]],dtype=np.int32)
            pred = pred.flatten()
            true_positive = np.count_nonzero(pred[(pred==1) & (pred == y)])
            false_positive = np.count_nonzero(pred[(pred==1) & (pred != y)])
            true_negative = np.count_nonzero(pred[(pred==0) & (pred == y)]==0)
            false_nagative = np.count_nonzero(pred[(pred == 0) & (pred !=y)]==0)
            sensitivity[i] = true_positive/(true_positive+false_nagative)
            one_minus_specificity[i] = 1 - (true_negative/(true_negative+false_positive))
        sensitivity = np.sort(sensitivity)
        one_minus_specificity = np.sort(one_minus_specificity)
        if show_cm:
            plt.plot(one_minus_specificity,sensitivity)
            plt.xlabel('1-specificity')
            plt.ylabel('sensitivity')
            plt.title('ROC')
            plt.show()
        auc = metrics.auc(one_minus_specificity,sensitivity,reorder=True)
        #print('Area Under the Curve',auc)
        return auc
        
    def fit(self,train_ds,num_iter=100,kfold=10,learning_rate=0.1,print_every=10,verbose=True):
        train_arr = train_ds.values
        train_x = np.hstack((np.ones((train_arr.shape[0],1)),train_arr[:,0:-1]))
        train_y = train_arr[:,-1]
        skf = StratifiedKFold(n_splits=kfold)
        skf.get_n_splits(train_x,train_y)
        fold_count = 0
        best_val_auc = 0.0
        best_trian_auc = 0.0
        best_w = np.zeros((train_ds.shape[1],1))
        if verbose == False:
            self.printProgressBar(0, kfold,'Progress:','Complete', length = 50)
        for train_index, test_index in skf.split(train_x,train_y):
            auc_val = 0.0
            auc_trian = 0.0
            fold_count +=1
            if verbose == True:
                print('training fold:',fold_count)
            train_split_x, test_split_x = train_x[train_index], train_x[test_index]
            train_split_y, test_split_y = train_y[train_index], train_y[test_index]
            loss = 0.0
            self.loss_hist = []
            #w = np.random.rand(train_ds.shape[1],1)
            w = np.zeros((train_ds.shape[1],1))
            #for i in tqdm.tqdm(range(num_iter)):
            for i in range(num_iter):
                z = np.array(np.dot(train_split_x,w),dtype=np.float32)
                h = 1/(1 + np.exp(-z))
                loss,grad = self.costfunction(train_split_x,h,train_split_y)
                w = w - learning_rate*grad
                self.loss_hist.append(loss)
                if verbose == True and i % print_every == 0:
                    print('(Iteration %d / %d) loss: %f'  % (i + 1, num_iter,loss))
            if verbose == True:
                # Plot the training losses
                plt.plot(self.loss_hist)
                plt.xlabel('Iteration')
                plt.ylabel('loss')
                plt.title('Training loss history')
                plt.show()
            auc_val = self.score(test_split_x,test_split_y,w)
            auc_trian = self.score(train_split_x,train_split_y,w)
            if auc_trian > best_trian_auc:
                best_trian_auc = auc_trian
            if auc_val > best_val_auc:
                best_val_auc = auc_val
                best_w = w
            if verbose == False:
                self.printProgressBar(fold_count, kfold, prefix = 'Progress:', suffix = 'Complete', length = 50)
        print('For lr:',learning_rate,' training auc:',best_trian_auc,' validation auc:',best_val_auc)
        return best_w,best_val_auc
    
    # Print iterations progress
    def printProgressBar(self,iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        # Print New Line on Complete
        if iteration == total: 
            print()

class DecisionTree(object):
    '''
    implement Decision Tree using Numpy
    '''
    def __init__(self):
        self.Dtree = np.array([])
        self.train_arr = np.array([])
        
    def entropy(self,feature_lables):
        #feature_lables.flatten()
        #print('feature_lables',feature_lables)
        categories = np.unique(feature_lables)
        entropy = 0.0
        for cat in categories:
            c = np.count_nonzero(feature_lables == cat)/len(feature_lables)
            entropy -= (c)*np.log2(c)
        return entropy
    
    def get_split_node(self,train_arr):
        data_set_entropy = self.entropy(train_arr[:,-1])
        #print('data_set_entropy',data_set_entropy)
        num_record = self.train_arr.shape[0]
        feature_info_gain = np.zeros((train_arr.shape[1]-1))
        for i in range(self.train_arr.shape[1]-1):
            train_feat_arr = train_arr[:,[i,-1]]
            feature_entropy = 0.0
            categories = np.unique(train_feat_arr[:,0])
            for cat in categories:
                cat_lables = train_feat_arr[train_feat_arr[:,0] == cat][:,-1]
                cat_count = train_feat_arr[train_feat_arr[:,0] == cat].shape[0]
                feature_entropy += (cat_count/num_record)*self.entropy(cat_lables)
            feature_info_gain[i] = data_set_entropy - feature_entropy
        #print('feature_info_gain',feature_info_gain)
        return np.argmax(feature_info_gain)
    
    def predict(self,test_x):
        pred = []
        for x in test_x:
            #print(x)
            decision_node = self.Dtree[0,0]
            filter_train = self.train_arr
            #print('self.Dtree',self.Dtree)
            #print('filter_train.shape',filter_train.shape)
            while decision_node is not -1:
                #print('self.Dtree[:,0]',self.Dtree[:,0])
                #print('decision_node',decision_node)
                split_cat = x[decision_node]
                #print('self.Dtree[:,1]',self.Dtree[:,1])
                #print('split_cat',split_cat)
                #print('np.array_str(self.Dtree[:,1]) == str(split_cat)',self.Dtree[:,1].astype(str) == str(split_cat))
                filter_train = filter_train[filter_train[:,decision_node].astype(str) == str(split_cat)]
                #print('filter_train.shape',filter_train.shape)
                #print(self.Dtree[(self.Dtree[:,0] == decision_node) & (self.Dtree[:,1].astype(str) == str(split_cat))])
                decision_node = np.asscalar(self.Dtree[(self.Dtree[:,0] == decision_node) 
                                                       & (self.Dtree[:,1].astype(str) == str(split_cat))][:,-1])
            yesprop = np.count_nonzero(filter_train[:,-1] =='Y')/filter_train.shape[0]
            pred.append(yesprop)
        pred = np.array(pred)
        return pred
        
    def score(self,X,y,show_cm=False):
        pred_prob = self.predict(X)
        rand_num = 10000
        epsilon = 1e-20
        np.place(y,y=='Y',[1])
        np.place(y,y=='N',[0])
        cut_offs = np.random.rand(rand_num)
        #cut_offs = [0.5]
        sensitivity = np.zeros((rand_num))
        one_minus_specificity =np.zeros((rand_num))
        for i in range(len(cut_offs)):
            #print('i',i)
            #print('cut_offs[i]',cut_offs[i])
            #print('pred_prob',pred_prob)
            pred =np.array([pred_prob>=cut_offs[i]],dtype=np.int32)
            pred = pred.flatten()
            #print('pred',pred)
            true_positive = np.count_nonzero(pred[(pred==1) & (pred == y)])
            #print('true_positive',true_positive)
            false_positive = np.count_nonzero(pred[(pred==1) & (pred != y)])
            #print('false_positive',false_positive)
            true_negative = np.count_nonzero(pred[(pred==0) & (pred == y)]==0)
            #print('true_negative',true_negative)
            false_nagative = np.count_nonzero(pred[(pred == 0) & (pred !=y)]==0)
            #print('false_nagative',false_nagative)
            sensitivity[i] = true_positive/(true_positive+false_nagative+epsilon)
            #print('sensitivity[i]',sensitivity[i])
            one_minus_specificity[i] = 1 - (true_negative/(true_negative+false_positive+epsilon))
            #print('one_minus_specificity[i]',one_minus_specificity[i])
        sensitivity = np.sort(sensitivity)
        one_minus_specificity = np.sort(one_minus_specificity)
        if show_cm:
            plt.plot(one_minus_specificity,sensitivity)
            plt.xlabel('1-specificity')
            plt.ylabel('sensitivity')
            plt.title('ROC')
            plt.show()
        auc = metrics.auc(one_minus_specificity,sensitivity,reorder=True)
        #print('Area Under the Curve',auc)
        return auc
    
        
    def fit(self,train_ds,max_node_points=2,verbose=False):
        self.train_arr = train_ds.values
        split_node = self.get_split_node(self.train_arr)
        split_node_cat = np.unique(self.train_arr[:,split_node])
        self.Dtree = np.transpose(np.vstack(([split_node]*len(split_node_cat),split_node_cat,[-1]*len(split_node_cat))))
        #print('Dtree',Dtree)
        #print('Dtree.dtype',Dtree.shape)
        for dtree_rec in self.Dtree:
            #print('dtree_rec',dtree_rec)
            train_cat_arr = self.train_arr[self.train_arr[:,dtree_rec[0]] == dtree_rec[1]]
            #print('train_cat_arr',train_cat_arr)
            yesprop = np.count_nonzero(train_cat_arr[:,-1]=='Y')/train_cat_arr.shape[0]
            #print('yesprop',yesprop)
            noprop = np.count_nonzero(train_cat_arr[:,-1]=='N')/train_cat_arr.shape[0] 
            #print('noprop',noprop)
            if (train_cat_arr.shape[0] <= max_node_points) or (yesprop >=0.9 or noprop>=0.9):
                continue
            else:
                split_node = self.get_split_node(train_cat_arr)
                self.Dtree[(self.Dtree[:,0]==dtree_rec[0]) & (self.Dtree[:,1]==dtree_rec[1])] = [dtree_rec[0],dtree_rec[1],split_node]
                split_node_cat = np.unique(train_cat_arr[:,split_node])
                subnode_rec = np.transpose(np.vstack(([split_node]*len(split_node_cat),split_node_cat,[-1]*len(split_node_cat))))
                #print('subnode_rec',subnode_rec)
                self.Dtree = np.vstack((self.Dtree,subnode_rec))
        print('Dtree',self.Dtree)     
            