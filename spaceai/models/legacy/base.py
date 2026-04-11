from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Dict, TYPE_CHECKING
import numpy as np
import pandas as pd
import joblib
from spaceai.benchmark.callbacks.mixin import CallbackMixin
from spaceai.benchmark.callbacks.handler import CallbackHandler

if TYPE_CHECKING:
    from spaceai.data import AnomalyDataset

class AnomalyClassifier(CallbackMixin, ABC):
    """
    Abstract base for legacy monolithic time-series classifiers.
    Defines common interface and input preparation.
    """
    
    def __init__(self, callback_handler: Optional[CallbackHandler] = None, **kwargs):
        super().__init__(callback_handler=callback_handler, **kwargs)

    @abstractmethod
    def fit(self, X: Any, y: Optional[np.ndarray] = None, results: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Fit the model on data X."""

    @abstractmethod
    def predict(self, X: Any, results: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict on data X, returning (predictions, metrics)."""

    def fit_predict(self, X: Any, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        preds, metrics = self.predict(X)
        self.fit(X, y)
        return preds, metrics

    def map_to_timestamps(
        self, channel_data: Any, anomalies: List[Tuple[int, int]]
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        return []

    @abstractmethod
    def prepare_labels(self, channel_labels: Any) -> List[Tuple[int, int]]:
        """Prepare ground truth."""

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "AnomalyClassifier":
        return joblib.load(path)

    @staticmethod
    def _prepare_input(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input X must be 2D (n_samples, n_timestamps)")
        return X.reshape(X.shape[0], 1, X.shape[1])
