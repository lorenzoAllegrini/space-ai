from __future__ import annotations
"""Abstract base class for anomaly classifiers."""

from abc import abstractmethod
from typing import Optional

import numpy as np
import torch

from spaceai.benchmark.callbacks import CallbackHandler
from contextlib import contextmanager

class AnomalyClassifier:
    """
    Abstract base for time-series wrappers: defines common interface and input preparation.
    """
    def __init__(self, callback_handler: Optional[CallbackHandler] = None):
        self.callback_handler = callback_handler
        
    @contextmanager
    def _callback_context(self, name: str, results: dict):
        if self.callback_handler:
            self.callback_handler.start()
            yield
            self.callback_handler.stop()
            results.update({f"{name}_{k}": v for k, v in self.callback_handler.collect(reset=True).items()})
        else:
            yield

    @abstractmethod
    def fit(  # pylint: disable=invalid-name
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit the model on time-series data X, optionally with labels y.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Predict on time-series data X, returning a numpy array of outputs.
        """

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
