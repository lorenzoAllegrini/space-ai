"""Anomaly detector module."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    Literal,
    Optional,
    Union,
)

import numpy as np
from spaceai.benchmark.callbacks.mixin import CallbackMixin
from sklearn.base import BaseEstimator

class AnomalyDetector(BaseEstimator, CallbackMixin):
    """Base class for anomaly detectors."""

    requires_calibration = True

    def __init__(self, callback_handler: Optional[Any] = None, filter_valid: bool = False):
        super().__init__(callback_handler=callback_handler)
        self.is_fitted_ = False
        self._predictor: Optional[SequenceModel] = None
        self.ignore_first_n_factor: float = 0
        self.filter_valid = filter_valid

    def pipeline_step(self, state: "PipelineState", is_fit: bool = False, **kwargs) -> "PipelineState":
        """
        Execute a modular pipeline step for detection.
        
        If is_fit=True, it calibrates the detector using state.data and state.labels.
        If self.filter_valid=True, it removes anomalous samples from the calibration set.
        """
        if is_fit:
            scores = state.data
            y = state.labels

            # Filtering: only keep normal samples if filter_valid is True
            if self.filter_valid and y is not None:
                mask = (y == 0)
                if hasattr(scores, "iloc"): # DataFrame
                    scores_fit = scores[mask]
                else: # Numpy
                    scores_fit = scores[mask]
                
                y_fit = y[mask]
                
                
                self.fit(scores_fit, y=y_fit, results=state.metrics, **kwargs)
            else:
                self.fit(scores, y=y, results=state.metrics, **kwargs)
            
            self.is_fitted_ = True
            return state # Detector fit doesn't modify data, just updates internal threshold
        
        # Inference step
        state.data = self.detect(state.data, results=state.metrics, **kwargs)
        return state

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
