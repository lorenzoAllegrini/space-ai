from typing import Optional, Any, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from .base import AnomalyDetector


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
        threshold: float = 0.5, 
        quantile: Optional[float] = None,
        smoothing_alpha: float = 1.0,
        filter_valid: bool = False,
        callback_handler: Optional[Any] = None
    ):
        super().__init__(callback_handler=callback_handler, filter_valid=filter_valid)
        self.threshold = threshold
        self.quantile = quantile
        self.smoothing_alpha = smoothing_alpha
        self._last_ewma: Optional[float] = None


    def fit(self, scores: np.ndarray, y: Optional[np.ndarray] = None, results: Optional[Dict[str, Any]] = None) -> None:
        """Calibrate the threshold based on the distribution of scores.
        
        If `y` is provided, calibrates smartly to minimize false positives.
        Otherwise, falls back to the preset quantile.
        """
        with self._callback_context("detector_fit", results):
            # Apply EWMA smoothing if enabled
            if self.smoothing_alpha < 1.0:
                scores_smoothed = pd.Series(scores).ewm(alpha=self.smoothing_alpha, adjust=False).mean().values
                self._last_ewma = scores_smoothed[-1]
                scores = scores_smoothed
            
            # --- MODALITÀ INTELLIGENTE (Supervisionata) ---
            if y is not None and len(np.unique(y)) > 1:
                # Sanitize scores to avoid NaN values crashing precision_recall_curve
                scores_clean = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Calcola tutte le possibili soglie e le relative metriche
                precisions, recalls, thresholds = precision_recall_curve(y, scores_clean)
                
                # Usiamo F0.5 per dare molto più peso alla Precision (minimizzare i Falsi Positivi)
                # rispetto alla Recall. Se vuoi essere ancora più severo sui FP, usa beta_sq = 0.1**2
                beta_sq = 1 
                
                denominator = (beta_sq * precisions) + recalls
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    f_beta_scores = (1 + beta_sq) * (precisions * recalls) / denominator
                    # Pulisce eventuali divisioni per zero
                    f_beta_scores[denominator == 0] = 0.0
                
                # Ignoriamo l'ultimo elemento di precision/recall perché non ha una soglia corrispondente
                best_idx = np.argmax(f_beta_scores[:-1])
                
                best_idx = np.argmax(f_beta_scores[:-1])
                
                # 1. TROVA LA SOGLIA DI SCORE ASSOLUTA
                self.threshold = thresholds[best_idx]
                
                # 2. CALCOLA IL QUANTILE CORRISPONDENTE A RITROSO
                # Conta quanti score sono minori della soglia e divide per il totale
                self.quantile = (scores_clean < self.threshold).mean()
                
                print(f"Soglia ottimale: {self.threshold:.4f} (Quantile: {self.quantile:.4f})")
                
                if results is not None:
                    results["calibrated_threshold"] = self.threshold
                    results["calibrated_quantile"] = self.quantile # Ora lo salvi nei log!
            
            # --- MODALITÀ STANDARD (Non supervisionata) ---
            elif self.quantile is not None:
                # Clip to [0, 1] for numpy and keep track if it was > 1.0
                q_calc = min(max(self.quantile, 0.0), 1.0)
                self.threshold = np.quantile(scores, q_calc)
                
                # If quantile was > 1.0, we treat the excess as a multiplier/margin
                if self.quantile > 1.0:
                    self.threshold *= self.quantile
            
            import logging
            logging.info(f"[THRESHOLD] Detector calibrated with threshold: {self.threshold:.6f}")

    def detect(self, scores: np.ndarray, results: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
        """Apply threshold to continuous scores and return binary labels.
        
        Args:
            scores (np.ndarray): Continuous anomaly scores.

        Returns:
            np.ndarray: Binary anomaly labels (1 = anomaly, 0 = normal).
        """
        with self._callback_context("detector_detect", results):
            scores = np.asarray(scores)
            
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

            return (scores > self.threshold).astype(int)

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
