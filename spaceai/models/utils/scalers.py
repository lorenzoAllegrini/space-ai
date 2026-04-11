import numpy as np
import pandas as pd
from scipy.stats import iqr as scipy_iqr
from sklearn.base import BaseEstimator, TransformerMixin
import time

class RollingRobustScalerWithPrior(BaseEstimator, TransformerMixin):
    """
    Rolling robust scaler that uses a historical IQR prior to stabilize 
    scaling in flat/low-noise regions.
    
    Moved to spaceai framework to ensure stable Pickle serialization for SML.
    """

    def __init__(self, window: int = 10):
        self.window = window
        self.prior_iqr_ = None

    def fit(self, X, y=None):
        """Calculate global IQR on training data as a prior."""
        t0 = time.time()
        self.prior_iqr_ = scipy_iqr(X, axis=0) + 1e-6
        return self

    def transform(self, X):
        """Scale X using rolling median/IQR with a historical prior constraint."""
        if self.prior_iqr_ is None:
            raise ValueError("Scaler must be fitted before transform.")

        t0 = time.time()
        df = pd.DataFrame(X)
        roll = df.rolling(window=self.window, min_periods=1)

        rolling_median = roll.median()

        rolling_q75 = roll.quantile(0.75)
        rolling_q25 = roll.quantile(0.25)
        rolling_iqr = rolling_q75 - rolling_q25

        denominator = np.maximum(rolling_iqr.values, self.prior_iqr_)

        X_scaled = (df - rolling_median) / denominator
        res = X_scaled.values
        return res
