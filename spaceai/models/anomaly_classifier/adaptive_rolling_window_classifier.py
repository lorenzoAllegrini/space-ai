"""Adaptive rolling window classifier with online drift detection."""

from __future__ import annotations

import collections
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from spaceai.benchmark.callbacks import CallbackHandler
from spaceai.data import AnomalyDataset
from spaceai.models.anomaly import AnomalyDetector
from spaceai.models.drift_detectors.drift_detector import DriftDetector
from spaceai.preprocessing.feature_extractors.feature_extractor import FeatureExtractor
from spaceai.preprocessing.ts_splitter import TimeSeriesSplitter

from .rolling_window_classifier import RollingWindowClassifier


class AdaptiveRollingWindowClassifier(RollingWindowClassifier):
    """Adaptive rolling-window classifier that monitors for concept drift and automatically retrains on a fixed-size buffer.

    Args:
        drift_detector (DriftDetector): An instantiated drift-detector object (e.g., ``ADWINDetector``).
        buffer_size (Union[int, str, pd.Timedelta]): Maximum number of segments to retain in the replay buffer. Defaults to 1000.
        *args: Positional arguments forwarded to :class:`RollingWindowClassifier`.
        **kwargs: Keyword arguments forwarded to :class:`RollingWindowClassifier`.
    """

    def __init__(
        self,
        drift_detector: DriftDetector,
        buffer_size: Union[int, str, pd.Timedelta] = 1000,
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
        self.buffer_size_raw = buffer_size
        self._buffer_size_resolved: Optional[int] = None
        self.data_buffer: Optional[collections.deque] = None
        self.label_buffer: Optional[collections.deque] = None
        self._is_fitted: bool = False

        if isinstance(buffer_size, int):
            self._init_buffers(buffer_size)


    def _init_buffers(self, maxlen: int) -> None:
        """Initialize or re-initialize deques with a fixed maxlen."""
        self._buffer_size_resolved = maxlen
        self.data_buffer = collections.deque(maxlen=maxlen)
        self.label_buffer = collections.deque(maxlen=maxlen)

    def _ensure_buffers(self, sampling_period: Optional[float] = None) -> None:
        """Lazily resolve a time-based buffer_size and create deques."""
        if self.data_buffer is not None:
            return
        if isinstance(self.buffer_size_raw, int):
            self._init_buffers(self.buffer_size_raw)
        else:
            total_samples = self.ts_splitter._resolve_samples(
                self.buffer_size_raw, sampling_period
            )
            step_samples = self.ts_splitter._resolve_samples(
                self.ts_splitter.step_size_raw, sampling_period
            )
            maxlen = max(1, total_samples // step_samples)
            logging.info(
                "Resolved buffer duration %s to %d segments.",
                self.buffer_size_raw,
                maxlen,
            )
            self._init_buffers(maxlen)

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
        results: Dict[str, Any] = {}
        sampling_period = getattr(channel_data, "sampling_period", None)

        # Lazy buffer initialisation for time-based sizes
        self._ensure_buffers(sampling_period)
        assert self.data_buffer is not None
        assert self.label_buffer is not None

        prepared_data, prepared_labels = self._prepare_input(
            channel_data, channel_labels
        )

        self.data_buffer.extend(prepared_data)
        if prepared_labels is not None:
            self.label_buffer.extend(prepared_labels)
        if not self._is_fitted:
            buffer_X = list(self.data_buffer)
            buffer_y = (
                list(self.label_buffer) if len(self.label_buffer) > 0 else None
            )
            fit_results = super().fit(buffer_X, buffer_y)
            results.update(fit_results)
            self._is_fitted = True
            results["initial_fit"] = True


        if self.feature_extractor is not None:
            with self._callback_context("feature_extraction", results):
                features = self.feature_extractor.transform(prepared_data)
        else:
            features = prepared_data


        with self._callback_context("prediction", results):
            y_pred = self.base_classifier.predict(features)

        # --- drift detection & conditional retraining ---
        if self.drift_detector is not None:
            with self._callback_context("drift_detection", results):
                drift_detected = self.drift_detector.update_batch(y_pred)

            if drift_detected:
                drift_window_size = self.drift_detector.current_width
                self.drift_detector.reset()

                new_concept_X = list(self.data_buffer)[-drift_window_size:]
                new_concept_y = (
                    list(self.label_buffer)[-drift_window_size:]
                    if len(self.label_buffer) > 0
                    else None
                )

                with self._callback_context("drift_retraining", results):
                    super().fit(new_concept_X, new_concept_y)

                results["drift_detected"] = True
                results["drift_window_size"] = drift_window_size
            else:
                results["drift_detected"] = False
                results["drift_window_size"] = 0
        else:
            results["drift_detected"] = False
            results["drift_window_size"] = 0

        # --- optional post-prediction anomaly detector ---
        if self.detector is not None:
            with self._callback_context("detection", results):
                y_pred = self.detector.detect(y_pred)

        return y_pred, results


    def fit(
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        channel_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Wrapper around ``step`` for training-API compatibility.

        Executes the online predict-then-train cycle and discards the
        predictions, returning only the metrics dictionary.
        """
        _, results = self.step(channel_data, channel_labels)
        return results