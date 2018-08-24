import numpy as np
from builtins import object
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from scipy.stats import f
from scipy import stats

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
        loss = (1/(2*n))*np.square(np.sum((h-y),axis=0)) 
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
        rmse = np.sqrt((1/(n))*np.square(np.sum((h-y),axis=0)))
        return rmse
    
    def fit(self,train_ds,num_iter,learning_rate=0.01,print_every=100,verbose=False):
        train_arr = train_ds.values
        train_x = np.hstack((np.ones((train_arr.shape[0],1)),train_arr[:,0:-1]))
        train_y = train_arr[:,-1]
        train_rmse = 0.0
        val_rmse = 0.0
        loss = 0.0
        self.loss_hist = []
        w = np.zeros((train_x.shape[1],1))
            
        for i in range(num_iter):
            h = np.dot(train_x,w)
            loss,grad = self.costfunction(h,train_x,train_y)
            w = w - learning_rate*grad
            self.loss_hist.append(loss)
            if verbose == True and i % print_every == 0:
                print('iteration ',i,'/',num_iter,' loss:',loss)
               
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
        xTxInv = np.linalg.inv(xTx) #D N dot N D = D D
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
        adjusted_r_sqr = 1-(n-1)/(n-k-1)*(1-r_sqr)
        std_error_regression = np.sqrt(1-adjusted_r_sqr)*np.std(y,ddof=1)
        
        print('----------------------')
        print('Regression Statistics')
        print('----------------------')
        print('Multiple R:',np.sqrt(r_sqr))
        print('R Square:',r_sqr)
        print('Adjusted R Sqaure:',adjusted_r_sqr)
        print('Standard Error of Regression:',std_error_regression)
        print('observations:',n)
        print('----------------------')
        
        y_bar = np.mean(y)
        ss_reg = np.sum(np.square(y_hat-y_bar),axis=0)
        ms_reg = ss_reg/k
        ss_res = np.sum(np.square(y_hat-y),axis=0)
        ms_res = ss_res/(n-k)
        F = ms_reg/ms_res
        p_value = 1-f.cdf(F,k,(n-k))
        
        print('---------------------------------------------------------------------------')
        print('ANOVA')
        print('---------------------------------------------------------------------------')
        print('             DF                 SS        MS         F       Significance F')
        print('---------------------------------------------------------------------------')
        print('Regression  ',k,'   ',ss_reg,'  ',ms_reg,'  ',F,'  ',p_value)
        print('Residual    ',n-k,'  ',ss_res,'  ',ms_res)
        print('Total       ',n,'  ',ss_reg+ss_res)
        print('--------------------------------------------------------------------------')

        mean_sqr_error = np.square(std_error_regression)
        x = np.hstack((np.ones((x.shape[0],1)),x))
        xTx = np.dot(x.transpose(),x)
        xTxInv = np.linalg.inv(xTx)
        cov_matrix = xTxInv*mean_sqr_error
        print('----------------------')
        print('Covariance Matrix')
        print('----------------------')
        print(cov_matrix)
        print('----------------------')
        
        coff = b_hat.reshape(-1,1)
        std_err = np.sqrt(cov_matrix.diagonal()).reshape(-1,1)
        t_stat = coff/std_err
        p_values = (1-stats.t.cdf(t_stat,2*(n-k))).reshape(-1,1)
        reg_output = np.hstack((coff,std_err,t_stat,p_values))
        print('---------------------------------------------------------------------------------')
        print('Regression Output')
        print('---------------------------------------------------------------------------------')
        print ('Coefficients      Standard Error     t-Stat    p-value    lower 95%    upper 95%')
        print('---------------------------------------------------------------------------------')
        print(reg_output)
        print('---------------------------------------------------------------------------------')
    