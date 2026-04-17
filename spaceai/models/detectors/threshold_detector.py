from typing import Optional, Any, Dict
import numpy as np
import pandas as pd
from .anomaly_detector import AnomalyDetector
import logging

logging.basicConfig(level=logging.INFO)

class ThresholdDetector(AnomalyDetector):
    """Simple threshold-based anomaly detector with EWMA smoothing.

    Returns 1 (anomaly) if the score is greater than a given threshold,
    0 (normal) otherwise. Supports automatic calibration via quantiles.

    Args:
        threshold (float): The threshold value. Defaults to 0.5.
        quantile (Optional[float]): If provided, the threshold is calibrated 
            using this quantile during fit().
        smoothing_alpha (float): Alpha factor for EWMA smoothing (0 < alpha <= 1).
            1.0 means no smoothing. Defaults to 1.0.
        callback_handler (Optional[Any]): Handler for simulation/benchmark callbacks.
    """

    def __init__(
        self, 
        quantile: float = 0.95,
        threshold: Optional[float] = None, 
        smoothing_alpha: float = 1.0,
        threshold_alpha: float = 1.0,
        filter_valid: bool = False,
        callback_handler: Optional[Any] = None
    ):
        super().__init__(callback_handler=callback_handler, filter_valid=filter_valid)
        self.threshold = threshold
        self.quantile = quantile
        self.smoothing_alpha = smoothing_alpha
        self.threshold_alpha = threshold_alpha
        self.filter_supervised = True
        self.mode = "upper" 
        self._last_ewma: Optional[float] = None



    def fit(self, scores: np.ndarray, y: Optional[np.ndarray] = None, results: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Calibrate the threshold based on the distribution of scores."""
        if scores is None or len(scores) == 0:
            return
            
        # Handle cases with multiple columns (take the first if ambiguous)
        if len(scores.shape) > 1 and scores.shape[1] > 1:
            if hasattr(scores, "iloc"):
                scores = scores.iloc[:, 0]
            else:
                scores = scores[:, 0]
            
        
        with self._callback_context("detector_fit", results):
            # Apply EWMA smoothing if enabled
            if self.smoothing_alpha < 1.0:
                scores_smoothed = pd.Series(scores).ewm(alpha=self.smoothing_alpha, adjust=False).mean().values
                self._last_ewma = scores_smoothed[-1]
                scores = scores_smoothed
            
            if self.quantile is not None:
                mask = np.isfinite(scores)
                if not np.any(mask):
                    logging.warning("[THRESHOLD WARNING] All scores are NaN/Inf, skipping calibration.")
                    return
                
                valid_scores = scores[mask]
                
                q_calc = min(max(self.quantile, 0.0), 1.0)
                new_threshold = np.nanquantile(valid_scores, q_calc)
                
                if self.quantile > 1.0:
                    new_threshold *= self.quantile
            
            if self.threshold is not None:
                self.threshold = self.threshold_alpha * new_threshold + (1-self.threshold_alpha) * self.threshold
            else:
                self.threshold = new_threshold
            
            q5, q50, q95 = np.quantile(scores, [0.05, 0.5, 0.95])

            logging.info(f"[THRESHOLD] Detector calibrated with threshold: {self.threshold:.6f}")

        return self

    def detect(self, scores: np.ndarray, results: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
        """Apply threshold to continuous scores and return binary labels."""
        if scores is None or len(scores) == 0:
            return np.array([])
            
        if len(scores.shape) > 1 and scores.shape[1] == 0:
            return np.zeros(len(scores)).astype(int)

        with self._callback_context("detector_detect", results):
            # Handle cases with multiple columns (take the first if ambiguous)
            if len(scores.shape) > 1 and scores.shape[1] > 0:
                if hasattr(scores, "iloc"):
                    scores = scores.iloc[:, 0].values
                else:
                    scores = scores[:, 0]
                    
            scores = np.asarray(scores).flatten()
            
            # Apply EWMA smoothing if enabled
            if self.smoothing_alpha < 1.0:
                if self._last_ewma is not None:
                    # Prepending last EWMA value to maintain continuity across segments
                    scores_combined = np.concatenate(([self._last_ewma], scores))
                    smoothed_combined = pd.Series(scores_combined).ewm(alpha=self.smoothing_alpha, adjust=False).mean().values
                    scores = smoothed_combined[1:]
                else:
                    scores = pd.Series(scores).ewm(alpha=self.smoothing_alpha, adjust=False).mean().values
                
                self._last_ewma = scores[-1]
            return (scores > self.threshold).astype(int) if self.threshold is not None else np.zeros(len(scores), dtype=int)


    def filter(self, X: np.ndarray, y: Optional[np.ndarray] = None, results: Optional[Dict[str, Any]] = None, **kwargs) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Filter out anomalies from the data."""
        with self._callback_context("detector_filter", results):
            if self.threshold is None and y is None:
                return X, y, np.ones(len(X), dtype=bool)
            
            # Determine mask
            if self.filter_supervised and y is not None:
                # SUPERVISED FILTERING (Ground Truth)
                mask = (np.array(y).flatten() == 0)
            else:
                # UNSUPERVISED FILTERING (Pseudo-labels)
                pseudo_labels = self.detect(X)
                mask = (pseudo_labels == 0)
            
            # Apply mask to data
            X_filtered = X[mask]
            
            # Apply same mask to labels if present to maintain alignment
            y_filtered = None
            if y is not None:
                y_filtered = y[mask]
            
        return X_filtered, y_filtered, mask

    def detect_anomalies(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Detect anomalies in the prediction data."""
        return self.detect(y_pred, **kwargs)

    def flush_detector(self) -> Optional[np.ndarray]:
        """Resets the temporal memory (EWMA) for a new sequence analysis."""
        self._last_ewma = None
        return None

    def evaluate_anomalies(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs
    ) -> dict[str, int | float]:
        """Evaluate anomaly detection performance."""
        return {}


__all__ = ["ThresholdDetector"]
