"""Abstract base class for anomaly classifiers."""

from abc import abstractmethod
from typing import Optional

import numpy as np
class AnomalyClassifier:
    """
    Abstract base for time-series wrappers: defines common interface and input preparation.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Fit the model on time-series data X, optionally with labels y.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on time-series data X, returning a numpy array of outputs.
        """

    @staticmethod
    def _prepare_input(X: np.ndarray) -> np.ndarray:
        """
        Ensure X is 3D with shape (n_samples, n_channels=1, n_timestamps).
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input X must be 2D (n_samples, n_timestamps)")
        return X.reshape(X.shape[0], 1, X.shape[1])
