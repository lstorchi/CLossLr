from scipy.optimize import minimize
import numpy as np

########################################################################################

def mean_absolute_percentage_error(y_pred, y_true):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)
    
    if np.any(y_true==0):
        print("Found zeroes in y_true. MAPE undefined. Removing from set...")
        idx = np.where(y_true==0)
        y_true = np.delete(y_true, idx)
        y_pred = np.delete(y_pred, idx)
        
    return(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

########################################################################################

def mean_average_error(y_pred, y_true):
       
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred) 
    
    return np.mean(np.abs(y_pred - y_true))
    
########################################################################################

def mean_absolute_error(y_pred, y_true):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)
    
    return np.mean(np.abs(y_true - y_pred))

########################################################################################

def residual_sum_square(y_pred, y_true):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)
    
    return np.sum((y_true - y_pred)**2)

########################################################################################

class custom_loss_lr:

    def __init__(self, loss):
        self.__loss__ = loss

    def fit(self, X, y):
       
       def objective_function(beta, X, Y):
        
        error = self.__loss__ (np.matmul(X, beta), Y)
        
        return(error) 
       