import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
 
from customlosslr import custom_loss_lr
 
 
class CustomLossRegressor(RegressorMixin, BaseEstimator):
    """scikit-learn wrapper around custom_loss_lr, usable inside RANSACRegressor.
    """
 
    def __init__(self, loss, normalize=False, xmean=None, xstd=None,
                l2regular=0.0, met="BFGS", maxiter=10000,
                supress_warnings=True, jumpconvcheck=True,
                fit_intercept=False, usedifferential=False,
                bounds=None, beta_init_values=None):
        # scikit-learn needs every parameter stored unchanged, under its own name
        self.loss = loss
        self.normalize = normalize
        self.xmean = xmean
        self.xstd = xstd
        self.l2regular = l2regular
        self.met = met
        self.maxiter = maxiter
        self.supress_warnings = supress_warnings
        self.jumpconvcheck = jumpconvcheck
        self.fit_intercept = fit_intercept
        self.usedifferential = usedifferential
        self.bounds = bounds
        self.beta_init_values = beta_init_values
 
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
 
        xmean, xstd = self.xmean, self.xstd
        if self.normalize:
            # Recomputed on every fit; a constant column in a small subset
            # would otherwise give std = 0 and NaNs.
            if xmean is None:
                xmean = X.mean(axis=0)
            if xstd is None:
                xstd = X.std(axis=0)
                xstd = np.where(xstd == 0, 1.0, xstd)
 
        # A fresh model per fit, so no state carries over between RANSAC subsets
        self.model_ = custom_loss_lr(
            loss=self.loss, normalize=self.normalize, xmean=xmean, xstd=xstd,
            l2regular=self.l2regular, met=self.met, maxiter=self.maxiter,
            supress_warnings=self.supress_warnings,
            jumpconvcheck=self.jumpconvcheck,
            fit_intercept=self.fit_intercept,
            usedifferential=self.usedifferential, bounds=self.bounds,
            beta_init_values=self.beta_init_values
        )
        self.loss_value_ = self.model_.fit(X, y)
        self.opt_result_ = self.model_.__results__
        self.success_ = bool(self.opt_result_.success)
        self.coef_ = np.asarray(self.model_.get_beta())
        self.intercept_ = self.coef_[0] if self.fit_intercept else 0.0
        self.n_features_in_ = X.shape[1]
        return self
 
    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(np.asarray(X, dtype=np.float64))