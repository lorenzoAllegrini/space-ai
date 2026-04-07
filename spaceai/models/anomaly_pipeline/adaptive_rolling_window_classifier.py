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
from spaceai.models.detectors import AnomalyDetector
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
        eval_perc: Optional[float] = 0.85,  
        *,
        ts_splitter: TimeSeriesSplitter,
        base_classifier: Any,
        supervised_classifier: bool = False,
        feature_extractor: Optional[FeatureExtractor] = None,
        callback_handler: Optional[CallbackHandler] = None,
        detector: Optional[AnomalyDetector] = None,
        alpha_buffer: float = 0.25,
        filter_valid_for_detector: bool = False,
    ) -> None:
        super().__init__(
            ts_splitter=ts_splitter,
            base_classifier=base_classifier,
            supervised_classifier=supervised_classifier,
            feature_extractor=feature_extractor,
            callback_handler=callback_handler,
            detector=detector,
            eval_perc=eval_perc,
            filter_valid_for_detector=filter_valid_for_detector,
        )
        self.drift_detector = drift_detector
        self.eval_perc = eval_perc 

        self.short_term_buffer = deque()
        self.replay_buffer = replay_buffer

        self._is_fitted: bool = False
        self.global_steps = 0
        self.initial_train_size = 0
        self.alpha_buffer = alpha_buffer

        # Dirottiamo il callback handler
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

        self._is_fitted = True
        results: Dict[str, Any] = {}

        prepared_data, prepared_labels = self._prepare_input(
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
            y_val = prepared_labels[split_idx:] if prepared_labels is not None else None

            if self.feature_extractor is not None:
                with self._callback_context("feature_extraction", results):
                    X_train = self.feature_extractor.fit_transform(X_train_raw, results=results, save_dir=results_dir, suffix="train")
                    X_val = self.feature_extractor.transform(X_val_raw, results=results, save_dir=results_dir, suffix="val")
                buffer_data = np.vstack((X_train, X_val))
                
                if getattr(self.feature_extractor, "kill_switch_active", False):
                    print("[DEBUG] AdaptiveRollingWindowClassifier fitting short-circuit activated: no useful features extracted.")
                    self.replay_buffer.add(buffer_data, prepared_labels, timestamps=timestamps, results=results)
                    self.initial_train_size = len(buffer_data)
                    return results
            else:
                X_train, X_val = X_train_raw, X_val_raw
                buffer_data = prepared_data

            self._run_training_cycle(
                data=X_train, 
                labels=y_train,
                results=results, 
                X_val=X_val,
                y_val=y_val, 
                is_retraining=False
            )
        else:
            results = super().fit(channel_data, channel_labels, results_dir=results_dir)
            if self.feature_extractor is not None:
                # results is already updated by super().fit
                buffer_data = self.feature_extractor.transform(prepared_data, results=results, save_dir=results_dir, suffix="train")
                if getattr(self.feature_extractor, "kill_switch_active", False):
                    print("[DEBUG] AdaptiveRollingWindowClassifier fitting short-circuit activated (fallback): no useful features extracted.")
                    self.replay_buffer.add(buffer_data, prepared_labels, timestamps=timestamps, results=results)
                    self.initial_train_size = len(buffer_data)
                    return results
            else:
                buffer_data = prepared_data
            
            if self.detector is not None and hasattr(self.detector, 'fit'):
                logging.warning("Nessun eval_perc definito! Il detector verrà fittato sui dati di training (rischio overfitting).")
                initial_preds = self.base_classifier.predict(buffer_data)
                self.detector.fit(initial_preds)

        self.replay_buffer.add(buffer_data, prepared_labels, timestamps=timestamps, results=results)
        self.initial_train_size = len(buffer_data)
        return results

    def step(
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        channel_labels: Optional[np.ndarray] = None,
        results_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict on new data and update the model in a single pass.

        Args:
            channel_data: Raw channel data (array, list, or AnomalyDataset).
            channel_labels: Optional pointwise labels.
            results_dir: Optional directory for saving plots.

        Returns:
            Tuple of (predictions array, metrics dict).
        """

        if not self._is_fitted:
            raise ValueError("Model is not fitted. Please fit the model before using it.")


        results: Dict[str, Any] = {}

        prepared_data, prepared_labels = self._prepare_input(
            channel_data, channel_labels, results=results, save_dir=results_dir, suffix=f"step_{self.global_steps}"
        )
            
        timestamps = np.asarray(self._get_timesteps(channel_data, results=results)).ravel()

        if self.feature_extractor is not None:
            with self._callback_context("feature_extraction", results):
                prepared_data = self.feature_extractor.transform(
                    prepared_data, results=results, save_dir=results_dir, suffix=f"step_{self.global_steps}"
                )
            if getattr(self.feature_extractor, "kill_switch_active", False):
                return np.zeros(len(prepared_data)), results

        with self._callback_context("prediction", results):
            y_pred = self.base_classifier.predict(prepared_data)
        
        probs = self.detector.detect(y_pred, return_probs=True)
        mask = (probs >= self.alpha_buffer)
        
        if np.any(mask):
            if isinstance(prepared_data, (np.ndarray, pd.DataFrame)):
                data_to_buffer = prepared_data[mask]
            else:
                data_to_buffer = [prepared_data[i] for i, m in enumerate(mask) if m]
                
            labels_to_buffer = prepared_labels[mask]

            timestamps_to_buffer = None
            if timestamps is not None and len(timestamps) == len(mask):
                timestamps_to_buffer = timestamps[mask]
            
            self.short_term_buffer.append((data_to_buffer, labels_to_buffer, timestamps_to_buffer))
        else:
            logging.warning("All samples in this step were filtered out (all detected as anomalies).")

        drift_detected = False
        if self.drift_detector is not None:
            drift_detected = self.drift_detector.process(np.mean(y_pred), results=results)
            
            width = self.drift_detector.current_width
        else:
            drift_detected = True
            width = 1

        if drift_detected:
            self.global_steps += 1
            print("-------------------------")
            print(f"\n total_steps: {self.global_steps}")

            retrain_data, retrain_labels, calib_X, calib_y = self._prepare_retraining_data(width, results=results)
            
            if retrain_data is None:
                print("Skipping retraining: No valid data in buffer (too many anomalies filtered?).")
            else:
                self._run_training_cycle(
                    data=retrain_data, 
                    labels=retrain_labels,
                    results=results, 
                    X_val=calib_X, 
                    y_val=calib_y,
                    is_retraining=True
                )

            print(f"\n Retraining complete!")
            time.sleep(1)

        if self.detector is not None:
            with self._callback_context("detection", results):
                y_pred = self.detector.detect(y_pred)

        return y_pred, results

    def _prepare_retraining_data(self, width: int, results: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Consolida il buffer a breve termine, estrae i dati di replay e costruisce il set di retraining."""
    
        old_items = list(self.short_term_buffer)[:-width]
        if old_items:
            old_X = np.concatenate([it[0] for it in old_items], axis=0)
            old_y = np.concatenate([it[1] for it in old_items], axis=0) if old_items[0][1] is not None else None
            old_ts = np.concatenate([it[2] for it in old_items], axis=0) if old_items[0][2] is not None else None
            self.replay_buffer.add(old_X, old_y, timestamps=old_ts, results=results)
        
        recent_items = list(self.short_term_buffer)[-width:]
        if not recent_items:
            return None, None, None, None
            
        recent_samples_X = np.concatenate([it[0] for it in recent_items], axis=0)
        recent_samples_y = np.concatenate([it[1] for it in recent_items], axis=0) if recent_items[0][1] is not None else None

        # Campioniamo i dati storici PRIMA dello split
        replay_data_X, replay_data_Y = self.replay_buffer.sample(sample_size=min(len(recent_samples_X)*2, 100000), results=results)
        replay_X = np.vstack(replay_data_X) if replay_data_X else np.empty((0, recent_samples_X.shape[1]))
        replay_y = np.concatenate(replay_data_Y) if replay_data_Y and replay_data_Y[0] is not None else None

        calib_X = None
        calib_y = None
        retrain_labels = None
        
        # Split per calibrazione detector se richiesto e se il detector può essere fittato
        if self.detector is not None and hasattr(self.detector, 'fit') and self.eval_perc is not None and 0.0 < self.eval_perc < 1.0:
            # 1. Split dei dati recenti
            split_idx_recent = int(len(recent_samples_X) * (1 - self.eval_perc))
            recent_train_X = recent_samples_X[:split_idx_recent]
            recent_calib_X = recent_samples_X[split_idx_recent:]
            recent_train_y = recent_samples_y[:split_idx_recent] if recent_samples_y is not None else None
            recent_calib_y = recent_samples_y[split_idx_recent:] if recent_samples_y is not None else None
            
            # 2. Split dei dati storici (Cruciale per fermare il collasso della bandwidth!)
            if len(replay_X) > 0:
                split_idx_replay = int(len(replay_X) * (1 - self.eval_perc))
                replay_train_X = replay_X[:split_idx_replay]
                replay_calib_X = replay_X[split_idx_replay:]
                replay_train_y = replay_y[:split_idx_replay] if replay_y is not None else None
                replay_calib_y = replay_y[split_idx_replay:] if replay_y is not None else None
                
                # Uniamo i pezzi
                retrain_data = np.concatenate((recent_train_X, replay_train_X), axis=0)
                calib_X = np.concatenate((recent_calib_X, replay_calib_X), axis=0)
                if recent_train_y is not None and replay_train_y is not None:
                    retrain_labels = np.concatenate((recent_train_y, replay_train_y), axis=0)
                else:
                    retrain_labels = recent_train_y if recent_train_y is not None else replay_train_y
                if recent_calib_y is not None and replay_calib_y is not None:
                    calib_y = np.concatenate((recent_calib_y, replay_calib_y), axis=0)
                else:
                    calib_y = recent_calib_y if recent_calib_y is not None else replay_calib_y
            else:
                retrain_data = recent_train_X
                calib_X = recent_calib_X
                retrain_labels = recent_train_y
                calib_y = recent_calib_y
        else:
            retrain_data = np.concatenate((recent_samples_X, replay_X), axis=0) if len(replay_X) > 0 else recent_samples_X
            calib_X = None
            if recent_samples_y is not None and replay_y is not None:
                retrain_labels = np.concatenate((recent_samples_y, replay_y), axis=0)
            else:
                retrain_labels = recent_samples_y if recent_samples_y is not None else replay_y
            calib_y = None

        self.short_term_buffer.clear()
        if self.drift_detector is not None:    
            self.drift_detector.reset()
            
        return retrain_data, retrain_labels, calib_X, calib_y

    def _run_training_cycle(
        self,
        data: np.ndarray,
        results: Dict[str, Any],
        labels: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        is_retraining: bool = False
    ) -> None:
        """Core logic for fitting base classifier and calibrating the detector."""
        X_train, y_train, X_calib, y_calib = data, labels, X_val, y_val

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
            
            # Extract num_epochs if available
            epochs = getattr(self.base_classifier, "epochs_count", getattr(self.base_classifier, "n_iter_", None))
            if epochs:
                results["num_epochs"] = epochs

        if X_calib is not None and self.detector is not None and hasattr(self.detector, 'fit'):
            with self._callback_context("validation_prediction", results):
                try:
                    y_pred_val = self.base_classifier.predict(X_calib, results=results)
                except (TypeError, ValueError):
                    y_pred_val = self.base_classifier.predict(X_calib)
            
            if self.filter_valid_for_detector and y_calib is not None:
                y_pred_val = y_pred_val[y_calib == 0]

            msg = "Ricalibrazione dinamica del Detector post-drift..." if is_retraining else \
                  f"Calibrazione Anomaly Detector su {len(y_pred_val)} predizioni di validation..."
            print(msg)
            with self._callback_context("detector_calibration", results):
                self.detector.fit(y_pred_val)
            
        elif self.detector is not None and hasattr(self.detector, 'fit') and is_retraining:
            # Fallback di emergenza se eval_perc è None
            logging.warning("Calibrazione su dati di addestramento! Rischio code piatte nella GPD.")
            y_pred_train = self.base_classifier.predict(X_train)
            if self.filter_valid_for_detector and y_train is not None:
                y_pred_train = y_pred_train[y_train == 0]
            self.detector.fit(y_pred_train)