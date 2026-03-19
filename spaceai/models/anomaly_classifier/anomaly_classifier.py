"""Abstract base class for anomaly classifiers."""

from abc import abstractmethod
from typing import Optional, List, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch

class AnomalyClassifier:
    """
    Abstract base for time-series wrappers: defines common interface and input preparation.
    """

    @abstractmethod
    def fit(  # pylint: disable=invalid-name
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit the model on time-series data X, optionally with labels y.
        """

    @abstractmethod
    def predict(self, X: Any) -> Tuple[np.ndarray, Dict[str, Any]]:  # pylint: disable=invalid-name
        """
        Predict on time-series data X, returning a tuple of (predictions, metrics).
        """

    def fit_predict(self, X: Any, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Convenience method for continual learning. Predicts on X, then fits on (X, y).
        Streaming wrappers can override this to optimize the roundtrip.
        """
        preds, metrics = self.predict(X)
        self.fit(X, y)
        return preds, metrics

    def map_to_timestamps(
        self, channel_data: Any, anomalies: List[Tuple[int, int]]
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Map a list of predicted or ground-truth anomaly indices to global timestamps.
        Must be implemented by child classes according to their prediction domains (window vs sample level).
        """
        return []

    @abstractmethod
    def prepare_labels(channel_labels):
        """ Prepare the ground trutg to uniform with the predicted labels"""

    def save(self, path: str) -> None:
        """Save the classifier to disk."""
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> "AnomalyClassifier":
        """Load a classifier from disk."""
        return torch.load(path, weights_only=False)

    @staticmethod
    def _prepare_input(X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Ensure X is 3D with shape (n_samples, n_channels=1, n_timestamps).
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input X must be 2D (n_samples, n_timestamps)")
        return X.reshape(X.shape[0], 1, X.shape[1])

