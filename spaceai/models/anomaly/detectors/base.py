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
import torch
from spaceai.benchmark.callbacks.mixin import CallbackMixin

if TYPE_CHECKING:
    from spaceai.models.predictors import SequenceModel

from abc import abstractmethod
class AnomalyDetector(CallbackMixin):
    """Base class for anomaly detectors."""

    def __init__(self, callback_handler: Optional[Any] = None):
        super().__init__(callback_handler=callback_handler)
        self._predictor: Optional[SequenceModel] = None
        self.ignore_first_n_factor: float = 0
        self.role = "detector"

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

    def transform(self, message: "PipelineMessage") -> "PipelineMessage":
        """
        Process the scores in the message, return binary detections and populate intervals.
        """
        y_hat = message.data
        detections = self.detect(y_hat, results=message.results)
        message.data = detections
        
        # Extract intervals if they don't exist yet
        if hasattr(message, "pred_intervals") and (message.pred_intervals is None or len(message.pred_intervals) == 0):
            from spaceai.benchmark.benchmark import Benchmark
            # Convert binary array to list of (start, end) indices
            intervals = Benchmark.process_pred_anomalies(detections)
            message.pred_intervals = intervals
            if len(intervals) > 0:
                print(f"DEBUG: Detector ({self.__class__.__name__}) - Extracted {len(intervals)} pred_intervals.", flush=True)
            
        return message

    def fit(self, *messages: "PipelineMessage", **kwargs):
        """Fit the detector using data in the message(s).
        
        Prefer validation message for calibration if multiple messages are provided.
        """
        if not messages:
            return
        target = next((m for m in messages if m.split_label == "val"), messages[0])
        return self._fit(np.asarray(target.data), labels=target.labels, results=target.results)

    @abstractmethod
    def _fit(self, scores: np.ndarray, labels: Optional[np.ndarray] = None, results: Optional[Dict[str, Any]] = None) -> None:
        """Internal fit implementation for calibration data."""
        pass


__all__ = ["AnomalyDetector"]
