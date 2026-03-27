from __future__ import annotations

import numpy as np
from .base import AnomalyDetector


class ThresholdDetector(AnomalyDetector):
    """Simple threshold-based anomaly detector.

    Returns 1 (anomaly) if the score is greater than a given threshold,
    0 (normal) otherwise.

    Args:
        threshold (float): The threshold value. Defaults to 0.5.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def detect(self, scores: np.ndarray) -> np.ndarray:
        """Apply threshold to continuous scores and return binary labels.

        Args:
            scores (np.ndarray): Continuous anomaly scores.

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


__all__ = ["ThresholdDetector"]
