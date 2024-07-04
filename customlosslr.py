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

    def __init__(self, loss, normalize=False, \
                  l2regular=0.0, met='BFGS', maxiter=1000):

        self.__loss__ = loss
        self.__met__ = met
        self.__maxiter__ = maxiter
        self.__l2regular__ = l2regular
        self.__normalize__ = normalize

        self.__beta_hat__ = None
        self.__results__ = None


    def fit(self, X, y):

        if type(X) is not np.ndarray:
            X = np.array(X)

        if type(y) is not np.ndarray:
            y = np.array(y)

        if len(y.shape) != 1:
            raise Exception("y must be a 1D numpy array.")
        
        if len(X.shape) != 2:
            raise Exception("X must be a 2D numpy array.")
        
        if X.shape[0] != y.shape[0]:
            raise Exception("X and y must have the same number of observations.")
        
        if self.__normalize__:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
       
        def objective_function(beta, X, Y):
           error = self.__loss__ (np.matmul(X, beta), Y) + \
            self.__l2regular__ * np.sum(beta**2)

           return(error) 
       
        Xn = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        beta_init = np.array([1]*Xn.shape[1])
        self.__results__ = minimize(objective_function, 
                    beta_init, args=(Xn,y), method=self.__met__, \
                    options={'maxiter': self.__maxiter__})
        self.__beta_hat__ = self.__results__.x
        
        optlf = self.__loss__(np.matmul(Xn, self.__beta_hat__), y)

        return optlf
    

    def predict(self, X):

        if type(X) is not np.ndarray:
            X = np.array(X)

        if self.__beta_hat__ is None:
            raise Exception("Model not trained yet. Call fit method first.")
        
        if X.shape[1] != self.__beta_hat__.shape[0]-1:
            raise Exception("Number of features in X does not match the number of features in the model.")
        
        if self.__normalize__:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        Xn = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        
        return np.matmul(Xn, self.__beta_hat__)  


    def get_beta(self):
        return self.__beta_hat__    


    def get_intecept(self):
        return self.__beta_hat__[0]


    def get_coefficients(self):
        return self.__beta_hat__[1:]

######################################################################################## 