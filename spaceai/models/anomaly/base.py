from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, TYPE_CHECKING
import numpy as np
from spaceai.benchmark.callbacks.mixin import CallbackMixin

from spaceai.benchmark.callbacks.handler import CallbackHandler

from sklearn.base import BaseEstimator

class BaseClassifier(BaseEstimator, CallbackMixin, ABC):
    """
    Abstract base class for underlying anomaly detection algorithms.
    Provides standard callback support for fitting and prediction phases.
    """
    
    def __init__(
        self, 
        callback_handler: Optional[CallbackHandler] = None, 
        **kwargs
    ) -> None:
        super().__init__(callback_handler=callback_handler, **kwargs)
        self.is_fitted_ = False

    @abstractmethod
    def fit(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Fit the model on data X."""

    @abstractmethod
    def predict(
        self, 
        X: np.ndarray, 
        results: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Predict anomaly scores for data X."""

    def fit_predict(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        results: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Sequential fit and predict on the same data."""
        self.fit(X, y, results=results)
        return self.predict(X, results=results)
