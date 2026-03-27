from __future__ import annotations
"""Abstract base class for feature extractors."""

from typing import Optional, Dict, Any, TYPE_CHECKING, Union
import numpy as np
import torch
from spaceai.benchmark.callbacks.mixin import CallbackMixin

from abc import abstractmethod
if TYPE_CHECKING:
    from spaceai.benchmark.callbacks.handler import CallbackHandler
    from spaceai.data.anomaly_dataset import AnomalyDataset

class FeatureExtractor(CallbackMixin):
    """
    Abstract base for feature extractors: defines common interface.
    """

    def __init__(
        self, 
        window_size: int, 
        stride: int, 
        callback_handler: Optional[CallbackHandler] = None,
        **kwargs
    ) -> None:
        self._window_size = window_size
        self._stride = stride
        self._current_dataset: Optional[AnomalyDataset] = None
        self._current_indices: Optional[np.ndarray] = None
        # Do not pass extra kwargs to CallbackMixin to avoid TypeError: object.__init__()
        super().__init__(callback_handler=callback_handler)

    @property
    def window_size(self) -> int:
        """Size of the sliding window used for segmentation."""
        return self._window_size

    @window_size.setter
    def window_size(self, value: int) -> None:
        self._window_size = value

    @property
    def stride(self) -> int:
        """Step size between consecutive windows."""
        return self._stride

    @stride.setter
    def stride(self, value: int) -> None:
        self._stride = value

    def set_context(self, dataset: Optional[AnomalyDataset] = None, indices: Optional[np.ndarray] = None) -> None:
        """Set the global context for feature extraction."""
        self._current_dataset = dataset
        self._current_indices = indices

    def clear_context(self) -> None:
        """Clear the global context."""
        self._current_dataset = None
        self._current_indices = None


    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the extracted features."""

    @abstractmethod
    def fit(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        results: Optional[Dict[str, Any]] = None
    ) -> "FeatureExtractor":
        """
        Fit the feature extractor on data X.
        """

    @abstractmethod
    def transform(
        self, 
        X: Union[np.ndarray, Any],
        results: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
        suffix: str = ""
    ) -> Union[np.ndarray, Any]:
        """
        Transform data X (segments) into extracted features.
        Supports PipelineMessage.
        """

    def fit_transform(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        results: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
        suffix: str = ""
    ) -> np.ndarray:
        """
        Fit to data, then transform it.
        """
        return self.fit(X, y, results=results).transform(X, results=results, save_dir=save_dir, suffix=suffix)

    def save(self, path: str) -> None:
        """Save the feature extractor to disk."""
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> "FeatureExtractor":
        """Load a feature extractor from disk."""
        return torch.load(path, weights_only=False)
