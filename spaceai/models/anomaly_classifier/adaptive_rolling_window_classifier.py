"""Adaptive rolling window classifier with online drift detection."""

from __future__ import annotations

import collections
from collections import deque
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from spaceai.benchmark.callbacks import CallbackHandler
from spaceai.data import AnomalyDataset
from spaceai.models.anomaly import AnomalyDetector
from spaceai.models.drift_detectors.drift_detector import DriftDetector
from spaceai.models.drift_detectors.utils import ReplayBuffer
from spaceai.preprocessing.feature_extractors.feature_extractor import FeatureExtractor
from spaceai.preprocessing.ts_splitter import TimeSeriesSplitter

from .rolling_window_classifier import RollingWindowClassifier
import time 


class AdaptiveRollingWindowClassifier(RollingWindowClassifier):
    """Adaptive rolling-window classifier that monitors for concept drift and automatically retrains on a fixed-size buffer.

    Args:
        drift_detector (DriftDetector): An instantiated drift-detector object (e.g., ``ADWINDetector``).
        buffer_size (Union[int, str, pd.Timedelta]): Maximum number of segments to retain in the replay buffer. Defaults to 1000.
            Ignored if a custom `replay_buffer` is provided.
        replay_buffer (Optional[ReplayBuffer]): An optional custom replay buffer.
            If not provided, a standard :class:`TimeDecayReplayBuffer` is used.
        *args: Positional arguments forwarded to :class:`RollingWindowClassifier`.
        **kwargs: Keyword arguments forwarded to :class:`RollingWindowClassifier`.
    """

    def __init__(
        self,
        drift_detector: Optional[DriftDetector] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        *,
        ts_splitter: TimeSeriesSplitter,
        base_classifier: Any,
        supervised_classifier: bool = False,
        feature_extractor: Optional[FeatureExtractor] = None,
        callback_handler: Optional[CallbackHandler] = None,
        detector: Optional[AnomalyDetector] = None,
    ) -> None:
        super().__init__(
            ts_splitter=ts_splitter,
            base_classifier=base_classifier,
            supervised_classifier=supervised_classifier,
            feature_extractor=feature_extractor,
            callback_handler=callback_handler,
            detector=detector,
        )
        self.drift_detector = drift_detector

        self.short_term_buffer = deque()
        self.replay_buffer = replay_buffer

        self._is_fitted: bool = False
        self.global_steps = 0

    def step(
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        channel_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict on new data and update the model in a single pass.

        Args:
            channel_data: Raw channel data (array, list, or AnomalyDataset).
            channel_labels: Optional pointwise labels.

        Returns:
            Tuple of (predictions array, metrics dict).
        """

        if not self._is_fitted:
            raise ValueError("Model is not fitted. Please fit the model before using it.")

        self.global_steps += 1
        results: Dict[str, Any] = {}

        prepared_data, prepared_labels = self._prepare_input(
            channel_data, channel_labels
        )
            
        timestamps = np.asarray(self._get_timesteps(channel_data)).ravel()

        if self.feature_extractor is not None:
            with self._callback_context("feature_extraction", results):
                prepared_data = self.feature_extractor.transform(prepared_data)

        self.short_term_buffer.append((prepared_data, prepared_labels, timestamps))

        with self._callback_context("prediction", results):
            y_pred = self.base_classifier.predict(prepared_data)
        y_pred = np.array(y_pred)
        
        print(f"y_pred mean={np.mean(y_pred):.4f} max={np.max(y_pred):.4f}")

        drift_detected = False
        if self.drift_detector is not None:
            with self._callback_context("drift_detection", results):
                drift_detected = self.drift_detector.process(y_pred)
            width = self.drift_detector.current_width
        else:
            drift_detected = True
            width = 1

        if drift_detected:
            retrain_data, retrain_y = self._prepare_retraining_data(width)

            if self.supervised_classifier:
                self.base_classifier.fit(retrain_data, retrain_y)
            else:
                self.base_classifier.fit(retrain_data)
            print(f"\n Retraining complete!")
            time.sleep(1)

        if self.detector is not None:
            with self._callback_context("detection", results):
                y_pred = self.detector.detect(y_pred)

        return y_pred, results

    def _prepare_retraining_data(self, width: int) -> Tuple[np.ndarray, np.ndarray]:
        """Consolidate the short-term buffer, extract replay data, and build the retraining set."""
    
        old_items = list(self.short_term_buffer)[:-width]
        if old_items:
            old_X = np.concatenate([it[0] for it in old_items], axis=0)
            old_y = np.concatenate([it[1] for it in old_items], axis=0)
            old_ts = None
            if old_items[0][2] is not None:
                old_ts = np.concatenate([it[2] for it in old_items], axis=0)
            self.replay_buffer.add(old_X, old_y, timestamps=old_ts)
        
        recent_items = list(self.short_term_buffer)[-width:]
        recent_samples_X = np.concatenate([it[0] for it in recent_items], axis=0)
        recent_samples_y = np.concatenate([it[1] for it in recent_items], axis=0)
        
        replay_data_X, replay_data_y = self.replay_buffer.sample(sample_size=min(len(recent_samples_X)*2, 10000))

        self.short_term_buffer.clear()
        self.short_term_buffer.extend(recent_items)
        if self.drift_detector is not None:    
            self.drift_detector.reset()

        replay_len = len(replay_data_X) if replay_data_X else 0

        if replay_data_X:
            replay_X = np.vstack(replay_data_X)
            replay_y = np.concatenate([np.atleast_1d(y) for y in replay_data_y]) if replay_data_y else None
            retrain_data = np.concatenate((recent_samples_X, replay_X), axis=0)
            retrain_y = np.concatenate((recent_samples_y, replay_y), axis=0) if replay_y is not None else recent_samples_y
        else:
            retrain_data = recent_samples_X
            retrain_y = recent_samples_y
            
        return retrain_data, retrain_y

    def fit(
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        channel_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Initial training using the parent's classical fit path.

        Calls ``RollingWindowClassifier.fit()`` to properly train the
        base classifier (including any Pipeline steps like scalers),
        then populates the replay buffer with preprocessed features.
        """

        results = super().fit(channel_data, channel_labels)
        self._is_fitted = True

        prepared_data, prepared_labels = self._prepare_input(channel_data, channel_labels)

        timestamps = self._get_timesteps(channel_data)

        # Store preprocessed features (not raw) in replay buffer
        if self.feature_extractor is not None:
            prepared_data = self.feature_extractor.transform(prepared_data)

        # Ensure timestamps are 1D (flattened) before adding to buffer
        if timestamps is not None:
            timestamps = np.asarray(timestamps).ravel()

        self.replay_buffer.add(prepared_data, prepared_labels, timestamps=timestamps)

        return results