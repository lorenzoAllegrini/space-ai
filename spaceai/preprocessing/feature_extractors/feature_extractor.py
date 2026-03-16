"""Abstract base class for feature extractors."""

from abc import abstractmethod
from typing import Optional

import numpy as np
import torch


class FeatureExtractor:
    """
    Abstract base for feature extractors: defines common interface.
    """

    def __init__(self, window_size: int, stride: int) -> None:
        self._window_size = window_size
        self._stride = stride

    @property
    def window_size(self) -> int:
        """Size of the sliding window used for segmentation."""
        return self._window_size

    @property
    def stride(self) -> int:
        """Step size between consecutive windows."""
        return self._stride

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the extracted features."""

    @abstractmethod
    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "FeatureExtractor":
        """
        Fit the feature extractor on data X.
        """

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data X into extracted features.
        """

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit to data, then transform it.
        """
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """Save the feature extractor to disk."""
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> "FeatureExtractor":
        """Load a feature extractor from disk."""
        return torch.load(path, weights_only=False)
