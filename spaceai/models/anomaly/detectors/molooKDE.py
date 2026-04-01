from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union
import warnings

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import RobustScaler

try:
    from ripser import ripser
except ImportError:
    warnings.warn("Libreria 'ripser' non trovata. Esegui: pip install ripser")

# Assumendo che AnomalyDetector sia nel tuo framework
from .base import AnomalyDetector


class MoLooKDEDetector(AnomalyDetector):
    """
    Morphological Leave-One-Out Kernel Density Estimator with adaptive GPD threshold.
    Implementation based on the algorithm presented at the ML4ITS workshop (ESA-ADB).
    """

    def __init__(
        self,
        alpha: float = 0.001,
        p: float = 3.0,
        unitize: bool = True,
        pot_percentile: float = 99.0,
        smoothing_alpha: float = 0.4,
        min_allowed_ll: float = -200.0,
        callback_handler: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(callback_handler=callback_handler)
        """
        Args:
            alpha (float): Confidence level for anomaly alarm (e.g., 0.01 = 1% probability of false positive).
            p (float): Exponent for topological bandwidth calculation.
            unitize (bool): If True, scale data to [0, 1] as per the paper.
            pot_percentile (float): Percentile used for Peaks Over Threshold (POT) extreme score isolation.
            smoothing_alpha (float): Alpha parameter for EWMA smoothing.
            min_allowed_ll (float): Minimum log-likelihood to clip values before processing.
        """
        self.alpha = alpha
        self.p = p
        self.unitize = unitize
        self.pot_percentile = pot_percentile
        self.min_allowed_ll = min_allowed_ll
        self.scaler = RobustScaler() if unitize else None
        self.kde: Optional[KernelDensity] = None
        self.gpd_params: Optional[tuple] = None
        self.pot_threshold: float = 0.0
        self.smoothing_alpha = smoothing_alpha
        self._last_ewma: Optional[float] = None

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        """Ensures the array is 2D for compatibility with sklearn and ripser."""
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            return X_arr.reshape(-1, 1)
        return X_arr

    def _fit(self, X: np.ndarray, labels: Optional[np.ndarray] = None, results: Optional[Dict[str, Any]] = None, **kwargs) -> MoLooKDEDetector:
        """Trains the spatial KDE and calculates GPD parameters for adaptive thresholding."""
        with self._callback_context("detector_fit", results):
            # 1. Anomaly Filtering (Pre-Smoothing)
            if labels is not None and len(labels) == len(X):
                mask = (labels == 0)
                n_anomalies = np.sum(labels == 1)
                if n_anomalies > 0:
                    # Safety check: ensure at least 20% of data remains
                    if np.sum(mask) >= len(X) * 0.2:
                        X = X[mask]
                        print(f"DEBUG: Detector ({self.__class__.__name__}) - Filtered {n_anomalies} anomalies before fit.", flush=True)
                    else:
                        print(f"WARNING: Detector ({self.__class__.__name__}) - Too many anomalies ({n_anomalies}/{len(X)}). Skipping filtering.", flush=True)

            # 2. Preprocessing & Smoothing (EWMA) - Stability Core
            X_clipped = np.clip(X, a_min=self.min_allowed_ll, a_max=None)
            X_smoothed = pd.Series(X_clipped).ewm(alpha=self.smoothing_alpha, adjust=False).mean().values
            self._last_ewma = X_smoothed[-1]
            
            # 3. Scaling (Internal RobustScaler)
            X_2d = self._ensure_2d(X_smoothed)
            if self.unitize:
                self.scaler = RobustScaler()
                X_scaled = self.scaler.fit_transform(X_2d)
                # Store cap for detection phase
                self.score_cap = np.percentile(X_smoothed, 99.5) * 1.1
            else:
                X_scaled = X_2d
                self.score_cap = np.inf

            # 4. Topological Analysis (Persistence Homology) for spatial scale definition
            # Always use X_scaled (smoothed) for topology
            if len(X_scaled) > 5000:
                indices = np.random.choice(len(X_scaled), 5000, replace=False)
                X_topo = X_scaled[indices]
            else:
                X_topo = X_scaled

            diagrams = ripser(X_topo)['dgms']
            h0 = diagrams[0]  # Connected components (0-dimensional homology)
            h0_finite = h0[h0[:, 1] != np.inf]
            d_star = np.max(h0_finite[:, 1] - h0_finite[:, 0]) if len(h0_finite) > 0 else 0.1

            h = max((d_star) ** (2.0 / self.p), 1e-4)
            self.kde = KernelDensity(kernel='epanechnikov', bandwidth=h)
            
            # 5. KDE Fit & Scoring (on smoothed data)
            if len(X_scaled) > 20000:
                indices = np.random.choice(len(X_scaled), 20000, replace=False)
                self.kde.fit(X_scaled[indices])
                log_y = self.kde.score_samples(X_scaled[indices])
            else:
                self.kde.fit(X_scaled)
                log_y = self.kde.score_samples(X_scaled)
            
            scores = -log_y
            
            # 6. GPD Fit for adaptive threshold
            self.pot_threshold = np.percentile(scores, self.pot_percentile)
            extreme_scores = scores[scores > self.pot_threshold]

            if len(extreme_scores) > 0:
                self.gpd_params = genpareto.fit(extreme_scores, floc=self.pot_threshold)
            else:
                self.gpd_params = (0.1, self.pot_threshold, 1.0)
            
            scores = -log_y
            self.pot_threshold = np.percentile(scores, self.pot_percentile)
            
            extreme_scores = scores[scores > self.pot_threshold]

            if len(extreme_scores) > 0:
                self.gpd_params = genpareto.fit(extreme_scores, floc=self.pot_threshold)
            else:
                self.gpd_params = (0.1, self.pot_threshold, 1.0)
            
            if results is not None:
                results["gpd_params"] = [float(p) for p in self.gpd_params]
                results["pot_threshold"] = float(self.pot_threshold)
            
        # If validation data is provided, evaluate it as well
        msg_val = kwargs.get("msg_val")
        if msg_val is not None and msg_val.data is not None:
            val_probs = self.detect(msg_val.data, return_probs=True, results=results)
        
        return self

    def detect(self, X: np.ndarray, return_probs: bool = False, results: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Detect anomalies using the fitted KDE and GPD threshold."""
        # Clip scores using the fitted cap
        X = np.clip(X, None, getattr(self, "score_cap", np.inf))
        
        # Scale new data
        X_scaled = self.scaler.transform(X.reshape(-1, 1)).flatten()
        
        """
        Calculates Leave-One-Out probability for new points and classifies them.
        Returns a binary array (1 = anomaly, 0 = normal).
        """
        with self._callback_context("detector_detect", results):
            if self.kde is None or self.gpd_params is None:
                raise RuntimeError("Detector must be fitted before calling detect()")

            X_clipped = np.clip(X, a_min=self.min_allowed_ll, a_max=None)
            
            if self._last_ewma is not None:
                X_combined = np.concatenate(([self._last_ewma], X_clipped))
                smoothed_combined = pd.Series(X_combined).ewm(alpha=self.smoothing_alpha, adjust=False).mean().values
                X_smoothed = smoothed_combined[1:]
            else:
                X_smoothed = pd.Series(X_clipped).ewm(alpha=self.smoothing_alpha, adjust=False).mean().values
            self._last_ewma = X_smoothed[-1]

            X_2d = self._ensure_2d(X_smoothed)
            X_scaled = self.scaler.transform(X_2d) if self.unitize else X_2d

            # Score calculation and probability estimation via Extreme Value Theory
            log_y_new = self.kde.score_samples(X_scaled)
            scores_new = -log_y_new

            c, loc, scale = self.gpd_params
            probs = genpareto.sf(scores_new, c, loc=loc, scale=scale)
            probs[scores_new <= self.pot_threshold] = 1.0

            if return_probs:
                return probs

            anomalies = (probs < self.alpha).astype(int)
            print(f"DEBUG: Detector ({self.__class__.__name__}) - Prob Stats: Min={probs.min():.4f}, Max={probs.max():.4f}, Mean={probs.mean():.4f}", flush=True)
            print(f"DEBUG: Detector ({self.__class__.__name__}) - Found {np.sum(anomalies)} anomalous samples (Alpha: {self.alpha}).", flush=True)
            return anomalies

    def detect_anomalies(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Alias for compatibility with the AnomalyDetector interface."""
        return self.detect(y_pred, **kwargs)

    def evaluate_anomalies(
        self, y_pred: np.ndarray, y_true: np.ndarray, **kwargs
    ) -> dict[str, int | float]:
        """Placeholder for performance evaluation."""
        return {}

    def flush_detector(self) -> Optional[np.ndarray]:
        """Resets the temporal memory (EWMA) for a new sequence analysis."""
        self._last_ewma = None
        return None

    def __call__(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.detect(X, **kwargs)

__all__ = ["MoLooKDEDetector"]