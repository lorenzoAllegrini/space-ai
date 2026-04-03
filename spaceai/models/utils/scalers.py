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
        # Aumentiamo l'epsilon per stabilità numerica (da 1e-12 a 1e-6).
        # Se i dati sono quasi costanti, non vogliamo dividere per numeri minuscoli.
        self.prior_iqr_ = scipy_iqr(X, axis=0) + 1e-6
        # print(f"[DEBUG] RollingRobustScalerWithPrior.fit took {time.time() - t0:.2f}s")
        return self

    def transform(self, X):
        """Scale X using rolling median/IQR with a historical prior constraint."""
        if self.prior_iqr_ is None:
            raise ValueError("Scaler must be fitted before transform.")

        t0 = time.time()
        df = pd.DataFrame(X)
        roll = df.rolling(window=self.window, min_periods=1)

        # La mediana mobile segue la nuova baseline (cancella il concept drift)
        rolling_median = roll.median()

        # L'IQR mobile stima la variazione locale
        rolling_q75 = roll.quantile(0.75)
        rolling_q25 = roll.quantile(0.25)
        rolling_iqr = rolling_q75 - rolling_q25

        # Se il segnale è "piatto", usiamo l'IQR storico per schiacciare i valori
        denominator = np.maximum(rolling_iqr.values, self.prior_iqr_)

        X_scaled = (df - rolling_median) / denominator
        res = X_scaled.values
        # print(f"[DEBUG] RollingRobustScalerWithPrior.transform took {time.time() - t0:.2f}s for {len(X)} samples")
        return res
