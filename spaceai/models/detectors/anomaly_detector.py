"""Anomaly detector module."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    Literal,
    Optional,
    Union,
    Any,
)

import numpy as np
from spaceai.benchmark.callbacks.mixin import CallbackMixin
from sklearn.base import BaseEstimator

from spaceai.models.predictors.seq_model import SequenceModel

class AnomalyDetector(BaseEstimator, CallbackMixin):
    """Base class for anomaly detectors."""

    requires_calibration = True

    def __init__(self, callback_handler: Optional[Any] = None, filter_valid: bool = False):
        super().__init__(callback_handler=callback_handler)
        self.is_fitted_ = False
        self._predictor: Optional[SequenceModel] = None
        self.ignore_first_n_factor: float = 0
        self.filter_valid = filter_valid

    def __call__(
        self, input_data: np.ndarray, y_true: np.ndarray, results: Optional[Dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        """Detect anomalies in the input data.

        Args:
            input_data (np.ndarray): Input data
            y_true (np.ndarray): True values
            results (Optional[Dict[str, Any]]): Results dictionary for metrics
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Detected anomalies
        """
        y_hat = self.predict_values(input_data)
        return self.detect(y_hat, results=results, **kwargs)

    def detect(self, y_pred: np.ndarray, results: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
        """Standard method for detection from scores."""
        raise NotImplementedError

    def bind_predictor(
        self,
        predictor: SequenceModel,
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        """Bind a predictor to the framework.

        If the predictor is a torch model, move it to the specified device.

        Args:
            predictor (Callable[..., np.ndarray]): Predictor function
            device (Literal["cpu", "cuda"], optional): Device to move the predictor to.
                Defaults to "cpu".
        """
        self._predictor = predictor
        if isinstance(self._predictor, torch.nn.Module):
            self._predictor.to(device)

    def predict_values(self, input_data: np.ndarray) -> np.ndarray:
        """Predict values using the bound predictor.

        Args:
            input_data (np.ndarray): Input data to predict on

        Returns:
            np.ndarray: Predicted values
        """
        if self._predictor is None:
            raise ValueError("Predictor must be bound before calling predict_values.")
        return self._predictor(input_data).detach().cpu().numpy()

    def detect_anomalies(
        self, y_pred: np.ndarray, y_true: np.ndarray, results: Optional[Dict[str, Any]] = None, **kwargs
    ) -> np.ndarray:
        """Detect anomalies in the prediction data."""
        return self.detect(y_pred, results=results, **kwargs)

    def flush_detector(self) -> Optional[np.ndarray]:
        """Flush the detector state."""
        raise NotImplementedError

    def evaluate_anomalies(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs
    ) -> Dict[str, Union[int, float]]:
        """Evaluate anomaly detection performance."""
        raise NotImplementedError

    def fit(self, *args, results: Optional[Dict[str, Any]] = None, **kwargs):
        """Fit the detector."""


__all__ = ["AnomalyDetector"]
