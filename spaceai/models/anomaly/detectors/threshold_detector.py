from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, TYPE_CHECKING
from .base import AnomalyDetector

if TYPE_CHECKING:
    from spaceai.models.anomaly_classifier.anomaly_classifier import PipelineMessage


class ThresholdDetector(AnomalyDetector):
    """Simple threshold-based anomaly detector.

    Returns 1 (anomaly) if the score is greater than a given threshold,
    0 (normal) otherwise.

    Args:
        threshold (float): The threshold value. Defaults to 0.5.
    """

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__()
        self.threshold = threshold

    def _fit(self, scores: np.ndarray, results: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Simple threshold detector does not require calibration."""
        pass

    def detect(self, scores: np.ndarray, results: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
        """Apply threshold to continuous scores and return binary labels.

        Args:
            scores (np.ndarray): Continuous anomaly scores.
            results (Optional[Dict[str, Any]]): Optional results dictionary.

        Returns:
            np.ndarray: Binary anomaly labels (1 = anomaly, 0 = normal).
        """
        # Ensure scores is numpy array for comparison
        scores = np.asarray(scores)
        return (scores > self.threshold).astype(int)

    def detect_anomalies(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Detect anomalies in the prediction data.
        
        This satisfies the AnomalyDetector abstract interface.
        """
        return self.detect(y_pred)

    def flush_detector(self) -> Optional[np.ndarray]:
        """Flush the detector state."""
        return None

    def evaluate_anomalies(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs
    ) -> dict[str, int | float]:
        """Evaluate anomaly detection performance."""
        # This would require ground truth, typically handled by Benchmark
        return {}


class QuantileThresholdDetector(AnomalyDetector):
    """Threshold detector based on anomaly score quantiles.

    Calculates the threshold as the given quantile of the scores seen during fit.

    Args:
        quantile (float): The quantile to use (e.g., 0.95). 
            Values between 0 and 1.
    """

    def __init__(self, quantile: float = 0.95, **kwargs):
        super().__init__()
        self.quantile = quantile
        self.threshold: Optional[float] = None

    def _fit(self, scores: np.ndarray, results: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Calibrate the threshold using the given scores."""
        if scores.size == 0:
            self.threshold = 0.5
            return

        self.threshold = float(np.quantile(scores, self.quantile))
        if results is not None:
            results["calibrated_quantile_threshold"] = self.threshold

    def detect(self, scores: np.ndarray, results: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
        """Apply the calibrated threshold to new scores.

        Args:
            scores (np.ndarray): Continuous anomaly scores.
            results (Optional[Dict[str, Any]]): Optional results dictionary.

        Returns:
            np.ndarray: Binary anomaly labels.
        """
        if self.threshold is None:
            # Fallback if fit was not called
            return (np.asarray(scores) > 0.5).astype(int)
        
        return (np.asarray(scores) > self.threshold).astype(int)

    def detect_anomalies(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs
    ) -> np.ndarray:
        return self.detect(y_pred)

    def flush_detector(self) -> Optional[np.ndarray]:
        return None

    def evaluate_anomalies(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs
    ) -> dict[str, int | float]:
        return {}


__all__ = ["ThresholdDetector", "QuantileThresholdDetector"]
