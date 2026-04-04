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

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, results: Optional[Dict[str, Any]] = None) -> MoLooKDEDetector:
        """Trains the spatial KDE and calculates GPD parameters for adaptive thresholding."""
        print(f"[DEBUG] MoLooKDE.fit received X of length: {len(X)}")
        with self._callback_context("detector_fit", results):
            # Preprocessing: clipping and EWMA smoothing
            X_clipped = np.clip(X, a_min=self.min_allowed_ll, a_max=None)
            X_smoothed = pd.Series(X_clipped).ewm(alpha=self.smoothing_alpha, adjust=False).mean().values
            self._last_ewma = X_smoothed[-1]
            
            X_2d = self._ensure_2d(X_smoothed)
            X_scaled = self.scaler.fit_transform(X_2d) if self.unitize else X_2d

            # Topological Analysis (Persistence Homology) for spatial scale definition
            topo_limit = 2000
            print(f"Starting topological analysis on {min(len(X_scaled), topo_limit)} points...")
            if len(X_scaled) > topo_limit:
                # Force local seed for reproducibility in bandwidth calculation
                rng = np.random.default_rng(42)
                indices = rng.choice(len(X_scaled), topo_limit, replace=False)
                X_topo = X_scaled[indices]
            else:
                X_topo = X_scaled

            diagrams = ripser(X_topo)['dgms']
            h0 = diagrams[0]  # Connected components (0-dimensional homology)
            print("Topological analysis completed.")
            
            h0_finite = h0[h0[:, 1] != np.inf]
            d_star = np.max(h0_finite[:, 1] - h0_finite[:, 0]) if len(h0_finite) > 0 else 0.1

            h = max((d_star) ** (2.0 / self.p), 1e-4)
            print(f"Calculated bandwidth: {h:.6f}. Fitting KDE...")
            self.kde = KernelDensity(kernel='epanechnikov', bandwidth=h)
            
            # Support set reduction for memory efficiency
            kde_limit = 1000
            print(f"Calculating KDE scores for {min(len(X_scaled), kde_limit)} points...")
            if len(X_scaled) > kde_limit:
                rng = np.random.default_rng(42)
                indices = rng.choice(len(X_scaled), kde_limit, replace=False)
                self.kde.fit(X_scaled[indices])
                log_y = self.kde.score_samples(X_scaled[indices])
            else:
                self.kde.fit(X_scaled)
                log_y = self.kde.score_samples(X_scaled)
            scores = -log_y
            print("Fitting GPD for adaptive threshold...")
            self.pot_threshold = np.percentile(scores, self.pot_percentile)
            extreme_scores = scores[scores > self.pot_threshold]

            if len(extreme_scores) > 0:
                self.gpd_params = genpareto.fit(extreme_scores, floc=self.pot_threshold)
            else:
                self.gpd_params = (0.1, self.pot_threshold, 1.0)
            print("Detector fitted successfully.")

            return self

    def detect(self, X: np.ndarray, return_probs: bool = False, results: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
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
            
            # DIAGNOSTIC LOG (internal)
            # print(f"[DEBUG-DETECTOR] Input scores (clipped) mean: {np.mean(X_clipped):.4f}")
            # print(f"[DEBUG-DETECTOR] Smoothed scores mean: {np.mean(X_smoothed):.4f}")

            c, loc, scale = self.gpd_params
            probs = genpareto.sf(scores_new, c, loc=loc, scale=scale)
            probs[scores_new <= self.pot_threshold] = 1.0

            anomalies = (probs < self.alpha).astype(int)
            


            if return_probs:
                return probs
            
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