from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, TYPE_CHECKING
import numpy as np
from spaceai.benchmark.callbacks.mixin import CallbackMixin

from spaceai.benchmark.callbacks.handler import CallbackHandler

class BaseClassifier(CallbackMixin, ABC):
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
        self.role = "classifier"

    def fit(self, *args, **kwargs) -> None:
        """Public entry point for fitting. Handles both raw data and PipelineMessage."""
        if len(args) > 0 and hasattr(args[0], "split_label"):
             msg = args[0]
             return self.fit(msg.data, y=msg.labels, results=msg.results)
        
        X = args[0] if len(args) > 0 else None
        y = kwargs.get("y", args[1] if len(args) > 1 else None)
        results = kwargs.get("results", args[2] if len(args) > 2 else None)
        return self._fit(X, y=y, results=results)

    @abstractmethod
    def _fit(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Internal fit implementation for raw data."""

    def predict(
        self, 
        X: Union[np.ndarray, "PipelineMessage"], 
        results: Optional[Dict[str, Any]] = None
    ) -> Union[np.ndarray, "PipelineMessage"]:
        """Public entry point for prediction. Handles both raw data and PipelineMessage."""
        if hasattr(X, "split_label"):
             msg = X
             scores = self.predict(msg.data, results=msg.results)
             msg.data = scores
             return msg
             
        return self._predict(X, results=results)

    @abstractmethod
    def _predict(
        self, 
        X: np.ndarray, 
        results: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Internal prediction implementation for raw data."""

    def fit_predict(
        self, 
        X: Union[np.ndarray, "PipelineMessage"], 
        y: Optional[np.ndarray] = None, 
        results: Optional[Dict[str, Any]] = None
    ) -> Union[np.ndarray, "PipelineMessage"]:
        """Sequential fit and predict on the same data."""
        self.fit(X, y, results=results)
        return self.predict(X, results=results)


class SklearnClassifier(BaseClassifier):
    """Generic wrapper for sklearn/PyOD models to fit the BaseClassifier interface."""
    
    def __init__(self, model: Any, supervised: bool = False, callback_handler: Optional[CallbackHandler] = None):
        super().__init__(callback_handler=callback_handler)
        self.model = model
        self.supervised = supervised

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, results: Optional[Dict[str, Any]] = None) -> None:
        if self.supervised:
            self.model.fit(X, y)
        else:
            self.model.fit(X)

    def _predict(self, X: np.ndarray, results: Optional[Dict[str, Any]] = None) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            if probs.ndim > 1 and probs.shape[1] == 2:
                return probs[:, 1]
            return probs
        return self.model.predict(X)
