import numpy as np
import pandas as pd
from builtins import object
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from scipy.stats import f
from scipy import stats

class baseModel(object):
    '''
    this is base model which predicts the mean value for a given item_type and outlet_type
    '''
    def __init__(self,train_ds):
        self.train_ds = train_ds

    def predict(self,x):
        n = x.shape[0]
        prediction = np.zeros((x.shape[0],1))
        for i in range(n):
            prediction[i] = self.train_ds[(self.train_ds['item_type_Baking Goods'] == x[i,3])&
                                       (self.train_ds['item_type_Breads'] == x[i,4])&
                                       (self.train_ds['item_type_Breakfast']==x[i,5])&
                                       (self.train_ds['item_type_Canned']==x[i,6])&
                                       (self.train_ds['item_type_Dairy']==x[i,7])&
                                       (self.train_ds['item_type_Frozen Foods']==x[i,8])&
                                       (self.train_ds['item_type_Fruits and Vegetables']==x[i,9])&
                                       (self.train_ds['item_type_Hard Drinks']==x[i,10])&
                                       (self.train_ds['item_type_Health and Hygiene']==x[i,11])&
                                       (self.train_ds['item_type_Household']==x[i,12])&
                                       (self.train_ds['item_type_Meat']==x[i,13])&
                                       (self.train_ds['item_type_Others']==x[i,14])&
                                       (self.train_ds['item_type_Seafood']==x[i,15])&
                                       (self.train_ds['item_type_Snack Foods']==x[i,16])&
                                       (self.train_ds['item_type_Soft Drinks']==x[i,17])&
                                       (self.train_ds['item_type_Starchy Foods']==x[i,18])&
                                       (self.train_ds['Outlet_Type_Grocery Store']==x[i,27])&
                                       (self.train_ds['Outlet_Type_Supermarket Type1']==x[i,28])&
                                       (self.train_ds['Outlet_Type_Supermarket Type2']==x[i,29])&
                                       (self.train_ds['Outlet_Type_Supermarket Type3']==x[i,30])]['Item_Outlet_Sales'].mean()
            
        return prediction
    
    def score(self,ds):
        arr = ds.values
        X = arr[:,0:-1]
        y = arr[:,-1]
        h = self.predict(X)
        n = X.shape[0]
        y = y.reshape((n,1))
        h = h.reshape((n,1))
        rmse = np.sqrt((1/n)*np.sum(np.square(h-y),axis=0))
        return rmse
        
        
class LinearRegression(object):
    '''
       implement Linear Regression using Numpy
    '''
    def __init__(self):
        self.loss_hist = []
    
    def costfunction(self,h,X,y):
        n = X.shape[0]
        y = y.reshape((n,1))
        h = h.reshape((n,1))
        loss = (1/(2*n))*np.sum(np.square(h-y),axis=0) 
        grad = (1/n)*np.dot(X.transpose(),(h-y))
        return loss,grad
    
    def predict(self,X,w):
        prediction = np.dot(X,w)
        return prediction
        
    def score(self,ds,w):
        arr = ds.values
        X = np.hstack((np.ones((arr.shape[0],1)),arr[:,0:-1]))
        y = arr[:,-1]
        h = self.predict(X,w)
        n = X.shape[0]
        y = y.reshape((n,1))
        h = h.reshape((n,1))
        rmse = np.sqrt((1/n)*np.sum(np.square(h-y),axis=0))
        return rmse    
    
    def fit(self,train_ds,num_iter,learning_rate=0.01,print_every=100,verbose=False):
        train_arr = train_ds.values
        train_x = np.hstack((np.ones((train_arr.shape[0],1)),train_arr[:,0:-1]))
        train_y = train_arr[:,-1]
        train_rmse = 0.0
        val_rmse = 0.0
        loss = 0.0
        self.loss_hist = []
        w = np.random.rand(train_x.shape[1],1)
        #w = np.zeros((train_x.shape[1],1))    
        for i in range(num_iter):
            h = np.dot(train_x,w)
            loss,grad = self.costfunction(h,train_x,train_y)
            w = w - learning_rate*grad
            self.loss_hist.append(loss)
            if verbose == True and i % print_every == 0:
                print('iteration ',i,'/',num_iter,' loss:',loss)
                #print('w',w)
        if verbose == True:
            plt.plot(self.loss_hist)
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.title('training loss history')
            plt.show()
        return w
    
class ClassicLinearRegression(object):
    '''
    for the lack of better work I am naming this calss as Classic regression.
    We will try to simulate the regression function in MS Excel
    '''
    def __init__(self):
        print('Classic Linear Regression Object created')
     
    def fit(self,train_ds):
        train_arr = train_ds.values 
        train_x = np.hstack((np.ones((train_arr.shape[0],1)),train_arr[:,0:-1]))
        train_y = train_arr[:,-1]
        w = np.zeros((train_x.shape[1],1))
        xTx = np.dot(train_x.transpose(),train_x)
        xTxInv = np.linalg.pinv(xTx) #D N dot N D = D D
        xTy = np.dot(train_x.transpose(),train_y) #D N dot N 1 = D 1
        b_hat = np.dot(xTxInv,xTy) #D D dot D 1
        return b_hat
    
    def predict(self,x,b_hat):
        x = np.hstack((np.ones((x.shape[0],1)),x))
        y_hat = np.dot(x,b_hat)
        return y_hat
    
    def regressionOutput(self,ds,b_hat):
        arr = ds.values
        x = arr[:,0:-1]
        y = arr[:,-1]
        n = x.shape[0]
        k = x.shape[1]
        y_hat = self.predict(x,b_hat)
        sum_square_error = np.sum(np.square(y_hat-y),axis=0)
        simple_avg_sqr_error = sum_square_error/n
        r_sqr = 1-(simple_avg_sqr_error/np.var(y))
        multiple_r = np.sqrt(r_sqr)
        adjusted_r_sqr = 1-(n-1)/(n-k-1)*(1-r_sqr)
        std_error_regression = np.sqrt(1-adjusted_r_sqr)*np.std(y,ddof=1)
        
        print('----------------------')
        print('Regression Statistics')
        print('----------------------')
        print('Multiple R:',round(multiple_r,4))
        print('R Square:',round(r_sqr,4))
        print('Adjusted R Sqaure:',round(adjusted_r_sqr,4))
        print('Standard Error of Regression:',round(std_error_regression,4))
        print('observations:',n)
        print('----------------------')
        
        y_bar = np.mean(y)
        ss_reg = np.sum(np.square(y_hat-y_bar),axis=0)
        ms_reg = ss_reg/k
        ss_res = np.sum(np.square(y_hat-y),axis=0)
        ms_res = ss_res/(n-k)
        F = ms_reg/ms_res
        p_value = 2*f.sf(np.absolute(F),k,(n-k))
        
        anova_dict = {' ':['Regression','Residual','Total'],
                      'DF':[k,n-k,n],
                      'SS':[round(ss_reg,4),round(ss_res,4),round(ss_reg+ss_res,4)],
                      'MS':[round(ms_reg,4),round(ms_res,4),' '],
                      'F':[round(F,4),' ',' '],
                      'Significance F':[p_value,' ',' ']}
        anova_ds = pd.DataFrame(anova_dict,columns=[' ','DF','SS','MS','F','Significance F'])
        anova_ds = anova_ds.set_index(' ')
        print('---------------------------------------------------------------------------')
        print('ANOVA')
        print('---------------------------------------------------------------------------')
        print(anova_ds)
        print('---------------------------------------------------------------------------')
        
        mean_sqr_error = np.square(std_error_regression)
        x = np.hstack((np.ones((x.shape[0],1)),x))
        xTx = np.dot(x.transpose(),x)
        xTxInv = np.linalg.pinv(xTx)
        #print('xTxInv',xTxInv[5,5])
        #print('mean_sqr_error',mean_sqr_error)
        cov_matrix = xTxInv*mean_sqr_error
        np.set_printoptions(precision=4)
        #np.set_printoptions(suppress=True)
        print('----------------------')
        print('Covariance Matrix')
        print('----------------------')
        print(cov_matrix)
        print('----------------------')
        
        head = list(ds)
        head.insert(0,'Intercept')
        head.remove('Item_Outlet_Sales')
        coff = b_hat 
        std_err = np.sqrt(cov_matrix.diagonal()) 
        #print('cov_matrix.diagonal()',cov_matrix.diagonal().shape)
        #print('std_err',std_err)
        t_stat = coff/std_err
        p_values = 2*(1-stats.t.cdf(np.absolute(t_stat),(n-k-1)))
        lower95 = b_hat - stats.t.isf(0.025,(n-k-1))*std_err
        upper95 = b_hat + stats.t.isf(0.025,(n-k-1))*std_err
        reg_output_dict = {'headers': head,
                          'Coefficients':coff,
                          'Standard Error':std_err,
                          't-Stat':t_stat,
                          'p-value':p_values,
                          'Lower 95%':lower95,
                          'Upper 95%':upper95}
        reg_output_ds = pd.DataFrame(reg_output_dict,columns=['headers','Coefficients','Standard Error','t-Stat',
                                                              'p-value','Lower 95%','Upper 95%'])
        #reg_output_ds.set_index(' ')
        print('---------------------------------------------------------------------------------')
        print('Regression Output')
        print('---------------------------------------------------------------------------------')
        print(reg_output_ds.to_string(index=False))
        print('---------------------------------------------------------------------------------')
