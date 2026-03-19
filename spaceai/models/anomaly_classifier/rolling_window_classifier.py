from __future__ import annotations

"""Abstract base class for anomaly classifiers."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Any, List, Union

if TYPE_CHECKING:
    from spaceai.data import AnomalyDataset

import numpy as np
import torch

from spaceai.preprocessing.ts_splitter import TimeSeriesSplitter
from spaceai.models.anomaly_classifier.anomaly_classifier import AnomalyClassifier
from spaceai.preprocessing.feature_extractors.feature_extractor import FeatureExtractor
from spaceai.benchmark.callbacks import CallbackHandler
from spaceai.models.anomaly import AnomalyDetector

class RollingWindowClassifier(AnomalyClassifier):
    """
    Abstract base for time-series wrappers: defines common interface and input preparation.
    """
    def __init__(self,
                ts_splitter: TimeSeriesSplitter,
                base_classifier: Any,
                supervised_classifier: bool = False,
                feature_extractor: Optional[FeatureExtractor]= None,
                callback_handler: Optional[CallbackHandler] = None,
                detector: Optional[AnomalyDetector] = None,
                ):
        super().__init__(callback_handler=callback_handler)
        self.ts_splitter = ts_splitter
        self.feature_extractor = feature_extractor
        self.base_classifier = base_classifier
        self.supervised_classifier = supervised_classifier
        self.detector = detector
    
    def fit( 
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        channel_labels: Optional[np.array] = None, 
    ) -> Dict[str, Any]:
        """
        Fit the model on time-series data X, optionally with labels y.
        """
        results = {}

        channel_data, channel_labels = self._prepare_input(channel_data, channel_labels)

        if self.feature_extractor is not None:
            with self._callback_context("feature_extraction", results):
                channel_data = self.feature_extractor.fit_transform(channel_data)

        with self._callback_context("fitting", results):
            if self.supervised_classifier:
                self.base_classifier.fit(channel_data, channel_labels)
            else:
                self.base_classifier.fit(channel_data)

        return results

    def predict(self, channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset]) -> np.ndarray:
        """
        Predict on time-series data X, returning a numpy array of outputs.
        """
        results = {}

        channel_data, _ = self._prepare_input(channel_data)

        if self.feature_extractor is not None:
            with self._callback_context("feature_extraction_predict", results):
                channel_data = self.feature_extractor.transform(channel_data)

        with self._callback_context("prediction", results):
            residuals = self.base_classifier.predict(channel_data)

        if self.detector is not None:
            with self._callback_context("detection", results):
                y_pred = self.detector.detect(residuals)

        return y_pred, results

    def prepare_labels(self, channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset]) -> List[Tuple[int, int]]:
        """
        Prepare labels for training.
        """
        if isinstance(channel_data, AnomalyDataset):
            splitted_channel = self.ts_splitter.segment_dataset(channel_data, mode="anomaly")
            return splitted_channel.intervals
        elif isinstance(channel_data, np.ndarray):
            window_labels = self.ts_splitter.split_labels(channel_data)
            indices = np.where(window_labels == 1)[0]
            
            if indices.size == 0:
                return []
            
            groups = [list(group) for group in mit.consecutive_groups(indices)]
            anomalies_intervals = [[group[0], group[-1]] for group in groups]
            
            return anomalies_intervals
            
        return []

    def map_to_timestamps(self, channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset], anomalies: List[Tuple[int, int]]) -> List[Tuple[Any, Any]]:
        if isinstance(channel_data, AnomalyDataset):
            return self.ts_splitter.get_timestamp_intervals(channel_data, anomalies)
            
        offset = getattr(channel_data, "start_idx", 0)
        time_intervals = []
        for w_start, w_end in anomalies:
            s_idx = w_start * self.ts_splitter.step_size + offset
            e_idx = w_end * self.ts_splitter.step_size + self.ts_splitter.window_size - 1 + offset
            time_intervals.append((s_idx, e_idx))
            
        return time_intervals

    def save(self, path: str) -> None:
        """Save the classifier to disk."""
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> "AnomalyClassifier":
        """Load a classifier from disk."""
        return torch.load(path, weights_only=False)

    @staticmethod
    def _prepare_input(
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        channel_labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:  # pylint: disable=invalid-name
        """
        Ensure X is 3D with shape (n_samples, n_channels=1, n_timestamps).
        """
        if isinstance(channel_data, AnomalyDataset):
            splitted_channel = self.ts_splitter.segment_dataset(channel_data)
            channel_data = splitted_channel.segments
            if channel_labels is not None:
                channel_labels = splitted_channel.labels
        
        if isinstance(channel_data, np.ndarray):
            if channel_labels is not None and len(channel_labels) >= len(channel_data):
                channel_labels = self.ts_splitter.split_labels(channel_labels)
            channel_data = self.ts_splitter.split(channel_data)

        return channel_data, channel_labels
    