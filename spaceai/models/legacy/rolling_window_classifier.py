from __future__ import annotations

"""Abstract base class for anomaly classifiers."""

import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Any, List, Tuple, Union

if TYPE_CHECKING:
    from spaceai.data import AnomalyDataset

import numpy as np
import torch
import more_itertools as mit

from spaceai.preprocessing.ts_splitter import TimeSeriesSplitter
from .anomaly_classifier import AnomalyClassifier
from spaceai.preprocessing.feature_extractors.feature_extractor import FeatureExtractor
from spaceai.benchmark.callbacks import CallbackHandler
from spaceai.models.detectors import AnomalyDetector
from spaceai.data import AnomalyDataset

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
                eval_perc: Optional[float] = None,
                filter_valid_for_detector: bool = False,
                ):
        super().__init__(callback_handler=callback_handler)
        self.ts_splitter = ts_splitter
        self.feature_extractor = feature_extractor
        self.base_classifier = base_classifier
        self.supervised_classifier = supervised_classifier
        self.detector = detector
        self.eval_perc = eval_perc
        self.filter_valid_for_detector = filter_valid_for_detector
    
    def fit( 
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        channel_labels: Optional[np.ndarray] = None, 
        results_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fit the model on time-series data X, optionally with labels y.
        """
        results = {}
        t0 = time.time()
        X, y, indices, dataset = self._prepare_input(
            channel_data, channel_labels, results=results, save_dir=results_dir, suffix="train"
        )
        results['window_size'] = getattr(self.ts_splitter, "window_size", None)

        if self.detector is not None and hasattr(self.detector, 'fit') and self.eval_perc is not None and 0.0 < self.eval_perc < 1.0:
            split_idx = int(len(X) * (1 - self.eval_perc))
            X_train = X[:split_idx]
            X_val = X[split_idx:]
            idx_train = indices[:split_idx] if indices is not None else None
            idx_val = indices[split_idx:] if indices is not None else None
            y_train = y[:split_idx] if y is not None else None
            y_val = y[split_idx:] if y is not None else None
        else:
            X_train = X
            X_val = None
            idx_train = indices
            idx_val = None
            y_train = y
            y_val = None

        if self.feature_extractor is not None:
            with self._callback_context("feature_extraction", results):
                self.feature_extractor.set_context(dataset=dataset, indices=idx_train)
                t0 = time.time()
                X_train = self.feature_extractor.fit_transform(
                    X_train, y_train, results=results, save_dir=results_dir, suffix="train"
                )
                if X_val is not None:
                    self.feature_extractor.set_context(dataset=dataset, indices=idx_val)
                    t0 = time.time()
                    X_val = self.feature_extractor.transform(
                        X_val, results=results, save_dir=results_dir, suffix="val"
                    )
            # Clear context after use
            self.feature_extractor.clear_context()

        # Short-circuit if no features were selected
        # if self.feature_extractor is not None and getattr(self.feature_extractor, "kill_switch_active", False):
        #     return results

        with self._callback_context("training", results):
            t0 = time.time()
            if self.supervised_classifier:
                self.base_classifier.fit(X_train, y_train)
            else:
                self.base_classifier.fit(X_train)
            
            # Extract num_epochs if available
            epochs = getattr(self.base_classifier, "epochs_count", getattr(self.base_classifier, "n_iter_", None))
            if epochs:
                results["num_epochs"] = epochs
        
        if self.detector is not None and hasattr(self.detector, 'fit'):
            with self._callback_context("detector_calibration", results):
                if X_val is not None:
                    val_scores = self.base_classifier.predict(X_val)
                    if self.filter_valid_for_detector and y_val is not None:
                        val_scores_filtered = val_scores[y_val == 0]
                        self.detector.fit(val_scores_filtered, results=results)
                    else:
                        self.detector.fit(val_scores, y=y_val, results=results)
                else:
                    # Fallback sui dati di train (attenzione all'overfitting delle soglie)
                    train_scores = self.base_classifier.predict(X_train)
                    if self.filter_valid_for_detector and y_train is not None:
                        train_scores_filtered = train_scores[y_train == 0]
                        self.detector.fit(train_scores_filtered, results=results)
                    else:
                        self.detector.fit(train_scores, y=y_train, results=results)

        return results

    def predict(
        self, 
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        results_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict on time-series data X, returning a numpy array of outputs.
        """
        results = {}
        t0 = time.time()
        X, _, indices, dataset = self._prepare_input(channel_data, results=results, save_dir=results_dir, suffix="test")

        if self.feature_extractor is not None:
            with self._callback_context("test_feature_extraction", results):
                self.feature_extractor.set_context(dataset=dataset, indices=indices)
                t0 = time.time()
                channel_data = self.feature_extractor.transform(
                    X, results=results, save_dir=results_dir, suffix="test"
                )
                self.feature_extractor.clear_context()
            
            # Opzione A: Kill-switch Short-circuit
            # if getattr(self.feature_extractor, "kill_switch_active", False):
            #     # Restituisce un array di zeri (nessuna anomalia)
            #     # Dobbiamo assicurarci che channel_data abbia la lunghezza corretta
            #     return np.zeros(len(X)), results
        else:
            channel_data = X

        with self._callback_context("prediction", results):
            t0 = time.time()
            if hasattr(self.base_classifier, "predict_proba"):
                y_pred = self.base_classifier.predict_proba(channel_data)
                if y_pred.ndim > 1 and y_pred.shape[1] == 2:
                    y_pred = y_pred[:, 1]
            else:
                y_pred = self.base_classifier.predict(channel_data)

        if self.detector is not None:
            with self._callback_context("detection", results):
                t0 = time.time()
                y_pred = self.detector.detect(y_pred)

        return y_pred, results

    def prepare_labels(
        self, 
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        results: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, int]]:
        """
        Prepare labels for training.
        """
        if isinstance(channel_data, AnomalyDataset):
            splitted_channel = self.ts_splitter.segment_dataset(channel_data, results=results)
            return splitted_channel.intervals
        elif isinstance(channel_data, np.ndarray):
            window_labels = self.ts_splitter.split_labels(channel_data, results=results)
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
        #torch.save(self, path)

    @staticmethod
    def load(path: str) -> "AnomalyClassifier":
        """Load a classifier from disk."""
        return torch.load(path, weights_only=False)

    def _get_timesteps(self, channel_data: Any, results: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
        """Extract timestamps for the current segments if data is an AnomalyDataset."""
        if isinstance(channel_data, AnomalyDataset):
            splitted_channel = self.ts_splitter.segment_dataset(channel_data, results=results)
            # Use segment_indices to get timestamps for all segments, not just anomalies
            if len(splitted_channel.segment_indices) > 0:
                end_indices = splitted_channel.segment_indices[:, 1].astype(int)
                tt = channel_data.timestamps[end_indices]
                return getattr(tt, "values", tt)
        return None

    def _prepare_input(
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        channel_labels: Optional[np.ndarray] = None,
        results: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
        suffix: str = ""
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[AnomalyDataset]]:  # pylint: disable=invalid-name
        """
        Ensure X is 3D with shape (n_samples, n_channels=1, n_timestamps).
        Returns (segments, labels, indices, original_dataset).
        """
        indices = None
        dataset = None
        if isinstance(channel_data, AnomalyDataset):
            dataset = channel_data
            splitted_channel = self.ts_splitter.segment_dataset(
                channel_data, results=results, save_dir=save_dir, suffix=suffix
            )
            
            # Propagate found window size/stride to feature extractor (important if dynamic)
            if self.feature_extractor is not None:
                if hasattr(self.feature_extractor, "window_size"):
                    self.feature_extractor.window_size = self.ts_splitter.window_size
                if hasattr(self.feature_extractor, "stride"):
                    self.feature_extractor.stride = self.ts_splitter.step_size

            channel_data = splitted_channel.segments
            indices = splitted_channel.segment_indices
            if channel_labels is None:
                channel_labels = splitted_channel.labels
            else:
                channel_labels = self.ts_splitter.split_labels(channel_labels, results=results)
        
        elif isinstance(channel_data, np.ndarray):
            if channel_labels is not None and len(channel_labels) >= len(channel_data):
                channel_labels = self.ts_splitter.split_labels(channel_labels, results=results)
            channel_data = self.ts_splitter.split(channel_data, results=results)

        return channel_data, channel_labels, indices, dataset



