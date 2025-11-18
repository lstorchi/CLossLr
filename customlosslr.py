from scipy.optimize import minimize
from sklearn.model_selection import KFold
import numpy as np

########################################################################################

def mean_absolute_percentage_error(y_pred, y_true):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)

    epsilon = 0.0    
    if np.any(y_true==0):
        #print("Found zeroes in y_true. MAPE undefined. Removing from set...")
        #idx = np.where(y_true==0)
        #y_true = np.delete(y_true, idx)
        #y_pred = np.delete(y_pred, idx)
        print("Found zeroes in y_true. MAPE undefined. Using modified MAPE...") 
        epsilon = 1e-8 

    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

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

    def __init__(self, loss, normalize=False, xmean=None, xstd=None, \
                  l2regular=0.0, met='BFGS', maxiter=10000, \
                    supress_warnings=False, jumpconvcheck=False, \
                    fit_intercept=True, usedifferential=False,
                    bounds=None): 
        
        #method = 'Nelder-Mead' seems better in convergence
        #method = 'BFGS'
        self.__loss__ = loss
        self.__met__ = met
        self.__maxiter__ = maxiter
        self.__l2regular__ = l2regular
        self.__supress_warnings__ = supress_warnings
        self.__jump_convcheck__ = jumpconvcheck
        self.__fit_intercept__ = fit_intercept 

        self.__usedifferential__ = usedifferential
        if bounds is None:
            self.__bounds__ = [-100, 100]
        else:
            self.__bounds__ = bounds

        # if normalize is True, the model will normalize the features
        # before fitting the model, and in precition time, it will
        # normalize the features using the same mean and standard deviation
        # used in the training set.of not specified the mean and standard
        self.__normalize__ = normalize
        self.__mean__ = None  
        self.__std__ = None
        if self.__normalize__:
            if xmean is not None:
                self.__mean__ = xmean
            if xstd is not None:
                self.__std__ = xstd

        self.__beta_hat__ = None
        self.__results__ = None


    def set_maxiter(self, maxiter):
        self.__maxiter__ = maxiter 


    def set_supress_warnings(self, supress_warnings):
        self.__supress_warnings__ = supress_warnings

    
    def set_solver(self, met): 
        self.__met__ = met


    def set_l2regular(self, l2regular):
        self.__l2regular__ = l2regular

    def fit(self, X, y, beta_init_values=None):

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
            if self.__mean__ is None:
                self.__mean__ = np.mean(X, axis=0)
            if self.__std__ is None:
                self.__std__ = np.std(X, axis=0)

            X = (X - self.__mean__) / self.__std__
       
        def objective_function(beta, X, Y):
           error = self.__loss__ (np.matmul(X, beta), Y) + \
            self.__l2regular__ * np.sum(beta**2)

           return(error) 
       
        if self.__fit_intercept__:
            Xn = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        else:
            Xn = X

        beta_init = np.array([1]*Xn.shape[1])
        if beta_init_values is not None:
            beta_init = beta_init_values
        else:
            try:
                # This solves (X^T X)^-1 X^T y
                beta_init = np.linalg.lstsq(Xn, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                # Fallback if matrix is singular
                beta_init = np.zeros(Xn.shape[1])  

        if beta_init.shape[0] != Xn.shape[1]:
            if self.__fit_intercept__:
                exp_shape = X.shape[1] + 1
            else:
                exp_shape = X.shape[1]
            raise Exception("Number of features in beta_init (%d) does not match X (%d)." % (beta_init.shape[0], exp_shape))

        beta_init_d = np.array(beta_init, dtype=np.float64)
        Xn_d = np.array(Xn, dtype=np.float64)
        y_d = np.array(y, dtype=np.float64)
        
        if self.__usedifferential__:
            boundsmin = self.__bounds__[0]
            boundsmax = self.__bounds__[1]
            bounds = [(boundsmin, boundsmax) for _ in range(Xn.shape[1])]
            self.__results__ = differential_evolution(  
                objective_function, 
                bounds, 
                args=(Xn, y),
                maxiter=self.__maxiter__ // 10 # It needs fewer iters than BFGS
                )
            self.__beta_hat__ = self.__results__.x
            optlf = self.__loss__(np.matmul(Xn, self.__beta_hat__), y)
        else:
            self.__results__ = minimize(objective_function, 
                    beta_init_d, args=(Xn_d,y_d), method=self.__met__, \
                    options={'maxiter': self.__maxiter__})
            self.__beta_hat__ = self.__results__.x
            optlf = self.__loss__(np.matmul(Xn, self.__beta_hat__), y)

        alldiffs = []
        for idx, v in enumerate(self.__beta_hat__):
            startv = beta_init[idx]
            alldiffs.append(abs(v-startv))
        avgdiff = np.mean(alldiffs)

        if not self.__jump_convcheck__:
            if self.__results__.success is False:
                msg = "Optimization did not converge. Try increasing maxiter." + \
                    self.__results__.message
                raise Exception(msg)
            
        if not self.__supress_warnings__:
            if avgdiff < 1e-9:
                msg = "Optimization problem." + \
                    "Average difference between initial and final values is too small." \
                    "Try different solver or initial values." \
                    " [%14.5e]"%(avgdiff)   
                raise Warning(msg)

        return optlf
    

    def set_beta_no_fit(self, beta):
        self.__beta_hat__ = beta
    
    def predict(self, X):

        if type(X) is not np.ndarray:
            X = np.array(X)

        if self.__beta_hat__ is None:
            raise Exception("Model not trained yet. Call fit method first.")
        
        expected_features = 0
        if self.__fit_intercept__:
            expected_features = self.__beta_hat__.shape[0] - 1
        else:
            expected_features = self.__beta_hat__.shape[0]

        if X.shape[1] != expected_features:
            raise Exception("Number of features in X (%d) does not match the number of features in the model (%d)." % (X.shape[1], expected_features))
        
        if self.__normalize__:
            X = (X - self.__mean__) / self.__std__

        if self.__fit_intercept__:
            Xn = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        else:
            Xn = X
        
        return np.matmul(Xn, self.__beta_hat__)  

    
    def get_beta(self):
        return self.__beta_hat__    

    def get_intercept(self):
        if self.__fit_intercept__:
            if self.__beta_hat__ is None:
                return None
            return self.__beta_hat__[0]
        else:
            return 0.0

    def get_coefficients(self):
        if self.__beta_hat__ is None:
            return None
        
        if self.__fit_intercept__:
            return self.__beta_hat__[1:]
        else:
            return self.__beta_hat__

######################################################################################## 

def get_optimal_regularization_lambda(X, Y, lambdas, lossfunction, numfolds=10, \
                           normalize=False, met='BFGS', maxiter=1000, debug=False, \
                           fit_intercept=True): 
    
    if type(X) is not np.ndarray:
        X = np.array(X)
    
    if type(Y) is not np.ndarray:
        Y = np.array(Y)

    if len(Y.shape) != 1:
        raise Exception("Y must be a 1D numpy array.")
    
    if len(X.shape) != 2:
        raise Exception("X must be a 2D numpy array.")
    
    if X.shape[0] != Y.shape[0]:
        raise Exception("X and Y must have the same number of observations.")

    if len(lambdas) == 0:
        raise Exception("Lambda must be a list of values.")

    if numfolds < 2:
        raise Exception("Number of folds must be at least 2.")                           
    
    cv_scores = []
    for l in lambdas:
        if l < 0:
            raise Exception("Lambda must be a positive value.")
    
        kf = KFold(n_splits=numfolds, shuffle=True)
        kf.get_n_splits(X)
            
        k_fold_scores = []
            
        f = 1
        if debug:
            print("Lambda: %14.5e"%(l))
        for train_index, test_index in kf.split(X):
            CV_X = X[train_index,:]
            CV_Y = Y[train_index]
                
            holdout_X = X[test_index,:]
            holdout_Y = Y[test_index]
                
            lambda_fold_model = custom_loss_lr(\
                loss=lossfunction, \
                normalize=normalize, \
                l2regular=l, \
                met=met, \
                maxiter=maxiter, \
                fit_intercept=fit_intercept) # <--- PASS PARAMETER
            lambda_fold_model.fit(CV_X, CV_Y)

            fold_preds = lambda_fold_model.predict(holdout_X)
            fold_score = lossfunction(fold_preds, holdout_Y)
            k_fold_scores.append(fold_score)
            if debug:
                print("  Fold %4d Score %14.5e"%(f, fold_score))
            f += 1

        lambda_scores = np.mean(k_fold_scores)
        if debug:
            print("Lambda: %14.5e Score %14.5e"%(l, lambda_scores))
            print()
        cv_scores.append(lambda_scores)

    best_lambda = lambdas[np.argmin(cv_scores)]
    
    return best_lambda

########################################################################################
