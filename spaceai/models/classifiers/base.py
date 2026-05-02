from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import numpy as np

from spaceai.benchmark.callbacks.mixin import CallbackMixin
from spaceai.benchmark.callbacks.handler import CallbackHandler


class BaseClassifier(CallbackMixin, ABC):
    """
    Abstract base class for underlying anomaly detection algorithms.
    Thanks to CallbackMixin, it only needs to implement pure array-to-array methods.
    """
    
    def __init__(
        self, 
        callback_handler: Optional[CallbackHandler] = None, 
        **kwargs
    ) -> None:
        super().__init__(callback_handler=callback_handler, **kwargs)
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, results: Optional[Dict[str, Any]] = None, **kwargs) -> "BaseClassifier":
        """Fit the model on raw numpy arrays."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, results: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
        """Predict or score raw numpy arrays."""
        pass

    def fit_predict(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        results: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Sequential fit and predict on the same data."""
        self.fit(X, y, results=results)
        return self.predict(X, results=results)


class SklearnClassifier(BaseClassifier):
    """Clean wrapper for sklearn/PyOD models to fit the BaseClassifier interface."""
    
    def __init__(
        self, 
        model: Any, 
        supervised: bool = False, 
        return_labels: bool = False, 
        return_proba: bool = False,
        callback_handler: Optional[CallbackHandler] = None
    ):
        super().__init__(callback_handler=callback_handler)
        self.model = model
        self.supervised = supervised
        self.return_labels = return_labels
        self.return_proba = return_proba

    def fit(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        results: Optional[Dict[str, Any]] = None, 
        **kwargs
    ) -> "SklearnClassifier":
        with self._callback_context("classifier_fit", results):
            X_arr = np.asarray(X)
            if self.supervised and y is not None:
                self.model.fit(X_arr, y)
            else:
                self.model.fit(X_arr)
        self.is_fitted_ = True
        return self

    def predict(
        self, 
        X: np.ndarray, 
        results: Optional[Dict[str, Any]] = None, 
        **kwargs
    ) -> np.ndarray:
        with self._callback_context("classifier_predict", results):
            X_arr = np.asarray(X)
            
            if self.return_labels:
                res = self.model.predict(X_arr)
            elif self.return_proba and hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_arr)
                if probs.ndim > 1 and probs.shape[1] == 2:
                    res = probs[:, 1]
                else:
                    res = probs
            elif hasattr(self.model, "decision_function"):
                res = self.model.decision_function(X_arr)
            elif hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_arr)
                if probs.ndim > 1 and probs.shape[1] == 2:
                    res = probs[:, 1]
                else:
                    res = probs
            else:
                res = self.model.predict(X_arr)
            
            return np.asarray(res)