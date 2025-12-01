"""Rocket feature extractor module."""

from typing import Optional
import numpy as np
from sktime.transformations.panel.rocket import Rocket  # type: ignore

class RocketFeatureExtractor:
    """
    Wrapper for Rocket to handle 2D input (n_samples, window_size) 
    and convert it to 3D (n_samples, 1, window_size) for sktime.
    """

    def __init__(self, num_kernels: int = 100):
        self.num_kernels = num_kernels
        self.rocket = Rocket(num_kernels=num_kernels, n_jobs=1)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform the data."""
        X = self._prepare_input(X)
        return self.rocket.fit_transform(X).values

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        X = self._prepare_input(X)
        return self.rocket.transform(X).values

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """Ensure X is 3D with shape (n_samples, n_channels=1, n_timestamps)."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input X must be 2D (n_samples, n_timestamps)")
        return X.reshape(X.shape[0], 1, X.shape[1])
