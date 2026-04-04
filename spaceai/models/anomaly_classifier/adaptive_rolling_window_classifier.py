"""Adaptive rolling window classifier with online drift detection."""

from __future__ import annotations

import collections
from collections import deque
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import numpy as np
import pandas as pd
import torch

from spaceai.benchmark.callbacks import CallbackHandler
from spaceai.data import AnomalyDataset
from spaceai.models.anomaly import AnomalyDetector
from spaceai.models.drift_detectors.drift_detector import DriftDetector
from spaceai.models.drift_detectors.utils import ReplayBuffer
from spaceai.preprocessing.feature_extractors.feature_extractor import FeatureExtractor
from spaceai.preprocessing.ts_splitter import TimeSeriesSplitter

from .rolling_window_classifier import RollingWindowClassifier


class AdaptiveRollingWindowClassifier(RollingWindowClassifier):
    """Adaptive rolling-window classifier that monitors for concept drift and automatically retrains on a fixed-size buffer.

    Args:
        drift_detector (DriftDetector): An instantiated drift-detector object (e.g., ``ADWINDetector``).
        eval_perc (Optional[float]): Percentage of data to use for detector calibration.
        ts_splitter (TimeSeriesSplitter): Buffer/Window manager.
        base_classifier (Any): The underlying ML model.
        supervised_classifier (bool): Whether it requires labels for training.
        feature_extractor (Optional[FeatureExtractor]): Optional processing pipeline.
        callback_handler (Optional[CallbackHandler]): Performance monitoring.
        detector (Optional[AnomalyDetector]): The anomaly scoring refinement (e.g. MoLooKDE).
        alpha_buffer (float): Probability threshold for adding samples to the short-term buffer (denoising).
    """

    def __init__(
        self,
        drift_detector: Optional[DriftDetector] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        eval_perc: Optional[float] = 0.85,  
        *,
        ts_splitter: TimeSeriesSplitter,
        base_classifier: Any,
        supervised_classifier: bool = False,
        feature_extractor: Optional[FeatureExtractor] = None,
        callback_handler: Optional[CallbackHandler] = None,
        detector: Optional[AnomalyDetector] = None,
        alpha_buffer: float = 0.25,
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
        self.eval_perc = eval_perc 

        self.short_term_buffer = deque()
        self.replay_buffer = replay_buffer

        self._is_fitted: bool = False
        self.global_steps = 0
        self.initial_train_size = 0
        self.alpha_buffer = alpha_buffer

        # Routing the callback handler to sub-components
        if self.drift_detector is not None and getattr(self.drift_detector, "callback_handler", None) is None:
            self.drift_detector.callback_handler = callback_handler
        if self.replay_buffer is not None and getattr(self.replay_buffer, "callback_handler", None) is None:
            self.replay_buffer.callback_handler = callback_handler


    def fit(
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        channel_labels: Optional[np.ndarray] = None,
        results_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initial training with optional chronological validation hold-out for the detector."""

        results: Dict[str, Any] = {}

        prepared_data, prepared_labels, indices, _ = self._prepare_input(
            channel_data, channel_labels, results=results, save_dir=results_dir, suffix="train"
        )
        timestamps = self._get_timesteps(channel_data, results=results)
        
        if timestamps is not None:
            timestamps = np.asarray(timestamps).ravel()

        if self.detector is not None and hasattr(self.detector, 'fit') and self.eval_perc is not None and 0.0 < self.eval_perc < 1.0:
            split_idx = int(len(prepared_data) * (1 - self.eval_perc))
            X_train_raw = prepared_data[:split_idx]
            X_val_raw = prepared_data[split_idx:]
            y_train = prepared_labels[:split_idx] if prepared_labels is not None else None

            if self.feature_extractor is not None:
                X_train = self.feature_extractor.fit_transform(X_train_raw, y_train, results=results, save_dir=results_dir, suffix="train")
                X_val = self.feature_extractor.transform(X_val_raw, results=results, save_dir=results_dir, suffix="val")
                buffer_data = np.vstack((X_train, X_val))
            else:
                X_train, X_val = X_train_raw, X_val_raw
                buffer_data = prepared_data

            self._run_training_cycle(
                data=X_train, 
                labels=y_train,
                results=results, 
                X_val=X_val, 
                is_retraining=False
            )
        else:
            # Fallback to standard fit from parent class
            res_parent = super().fit(channel_data, channel_labels, results_dir=results_dir)
            results.update(res_parent)
            
            if self.feature_extractor is not None:
                buffer_data = self.feature_extractor.transform(prepared_data, results=results, save_dir=results_dir, suffix="train")
            else:
                buffer_data = prepared_data
            
            if self.detector is not None and hasattr(self.detector, 'fit'):
                logging.warning("No eval_perc defined! Fitting detector on training data (high overfitting risk).")
                initial_preds = self.base_classifier.predict(buffer_data)
                self.detector.fit(initial_preds)

        self.replay_buffer.add(buffer_data, prepared_labels, timestamps=timestamps, results=results)
        self.initial_train_size = len(buffer_data)
        self._is_fitted = True
        return results

    def step(
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        channel_labels: Optional[np.ndarray] = None,
        results_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict on new data and update the model incrementally if drift is detected."""
        
        # --- LAZY FIT ---
        if not self._is_fitted:
            logging.info("Lazy fitting AdaptiveRollingWindowClassifier on the first experience.")
            metrics = self.fit(channel_data, channel_labels, results_dir=results_dir)
            predictions, _ = self.predict(channel_data)
            return predictions, metrics

        # --- STANDARD STEP FOR FITTED MODELS ---
        results: Dict[str, Any] = {}

        prepared_data, prepared_labels, indices, _ = self._prepare_input(
            channel_data, channel_labels, results=results, save_dir=results_dir, suffix=f"step_{self.global_steps}"
        )
            
        timestamps = np.asarray(self._get_timesteps(channel_data, results=results)).ravel()

        if self.feature_extractor is not None:
            prepared_data = self.feature_extractor.transform(
                prepared_data, results=results, save_dir=results_dir, suffix=f"step_{self.global_steps}"
            )

        # 1. Prediction (Raw Scores)
        with self._callback_context("prediction", results):
            y_pred = self.base_classifier.predict(prepared_data)
        
        # 2. Buffer Management (Denoising)
        # We only add to buffer points that are likely "normal" according to the current detector
        if self.detector is not None:
            probs = self.detector.detect(y_pred, return_probs=True)
            mask = (probs >= self.alpha_buffer)
        else:
            mask = np.ones(len(y_pred), dtype=bool)
        
        if np.any(mask):
            data_to_buffer = prepared_data[mask]
            labels_to_buffer = prepared_labels[mask] if prepared_labels is not None else None
            timestamps_to_buffer = timestamps[mask] if (timestamps is not None and len(timestamps) == len(mask)) else None
            self.short_term_buffer.append((data_to_buffer, labels_to_buffer, timestamps_to_buffer))
        else:
            logging.warning("All samples in this step were filtered out (detected as anomalies). Buffer not updated.")

        # 3. Drift Detection
        drift_detected = False
        if self.drift_detector is not None:
            drift_detected = self.drift_detector.process(np.mean(y_pred), results=results)
            width = self.drift_detector.current_width
        else:
            # If no drift detector is present, we might want to force update periodically (optional)
            drift_detected = True
            width = 1

        # 4. Incremental Retraining
        if drift_detected:
            print("drift detected")
            self.global_steps += 1
            logging.info("--- Drift Detected! Starting incremental update [Step %d] ---", self.global_steps)

            retrain_data, calib_X = self._prepare_retraining_data(width, results=results)
            
            if retrain_data is not None:
                self._run_training_cycle(
                    data=retrain_data, 
                    results=results, 
                    X_val=calib_X, 
                    is_retraining=True
                )
                logging.info("Incremental update complete.")
            else:
                logging.warning("Skipping update: No valid data in buffer after filtering.")
            
            if self.drift_detector is not None:
                self.drift_detector.reset()

        # 5. Final Detection (Binary Anomaly Classification)
        if self.detector is not None:
            with self._callback_context("detection", results):
                y_pred = self.detector.detect(y_pred)

        return y_pred, results

    def _prepare_retraining_data(self, width: int, results: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Consolidates short-term buffer, adds older items to replay, and constructs retraining set."""
    
        # Items to be moved to permanent Replay Buffer
        old_items = list(self.short_term_buffer)[:-width]
        if old_items:
            old_X = np.concatenate([it[0] for it in old_items], axis=0)
            old_ts = np.concatenate([it[2] for it in old_items], axis=0) if old_items[0][2] is not None else None
            self.replay_buffer.add(old_X, None, timestamps=old_ts, results=results)
        
        # Recent items (since last drift)
        recent_items = list(self.short_term_buffer)[-width:]
        if not recent_items:
            return None, None
            
        recent_samples_X = np.concatenate([it[0] for it in recent_items], axis=0)

        # Sample from history to prevent forgetting
        replay_data_X, _ = self.replay_buffer.sample(sample_size=min(len(recent_samples_X)*2, 100000), results=results)
        replay_X = np.vstack(replay_data_X) if replay_data_X else np.empty((0, recent_samples_X.shape[1]))

        calib_X = None
        
        # Calibration split for the detector
        if self.detector is not None and hasattr(self.detector, 'fit') and self.eval_perc is not None and 0.0 < self.eval_perc < 1.0:
            split_idx_recent = int(len(recent_samples_X) * (1 - self.eval_perc))
            recent_train_X = recent_samples_X[:split_idx_recent]
            recent_calib_X = recent_samples_X[split_idx_recent:]
            
            if len(replay_X) > 0:
                split_idx_replay = int(len(replay_X) * (1 - self.eval_perc))
                replay_train_X = replay_X[:split_idx_replay]
                replay_calib_X = replay_X[split_idx_replay:]
                
                retrain_data = np.concatenate((recent_train_X, replay_train_X), axis=0)
                calib_X = np.concatenate((recent_calib_X, replay_calib_X), axis=0)
            else:
                retrain_data = recent_train_X
                calib_X = recent_calib_X
        else:
            retrain_data = np.concatenate((recent_samples_X, replay_X), axis=0) if len(replay_X) > 0 else recent_samples_X
            calib_X = None

        # Clear short-term buffer after consumption
        self.short_term_buffer.clear()
        return retrain_data, calib_X

    def _run_training_cycle(
        self,
        data: np.ndarray,
        results: Dict[str, Any],
        labels: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        is_retraining: bool = False
    ) -> None:
        """Core logic for fitting base classifier and calibrating the detector."""
        X_train, y_train, X_calib = data, labels, X_val

        with self._callback_context("training" if not is_retraining else "retraining", results):
            if self.supervised_classifier:
                try:
                    self.base_classifier.fit(X_train, y_train, results=results)
                except (TypeError, ValueError):
                    self.base_classifier.fit(X_train, y_train)
            else:
                try:
                    self.base_classifier.fit(X_train, results=results)
                except (TypeError, ValueError):
                    self.base_classifier.fit(X_train)

        # Calibrate/Update the anomaly detector thresholds
        if X_calib is not None and self.detector is not None and hasattr(self.detector, 'fit'):
            with self._callback_context("validation_prediction", results):
                try:
                    y_pred_val = self.base_classifier.predict(X_calib, results=results)
                except (TypeError, ValueError):
                    y_pred_val = self.base_classifier.predict(X_calib)
            
            logging.info("Recalibrating Anomaly Detector on %d validation predictions...", len(y_pred_val))
            self.detector.fit(y_pred_val)
            
        elif self.detector is not None and hasattr(self.detector, 'fit') and is_retraining:
            logging.warning("Calibration fallback: Using training data for detector refinement.")
            y_pred_train = self.base_classifier.predict(X_train)
            self.detector.fit(y_pred_train)