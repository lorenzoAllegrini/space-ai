from __future__ import annotations

"""Abstract base class for anomaly classifiers."""

import time
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
                ):
        super().__init__(callback_handler=callback_handler)
        self.ts_splitter = ts_splitter
        self.feature_extractor = feature_extractor
        self.base_classifier = base_classifier
        self.supervised_classifier = supervised_classifier
        self.detector = detector
        self.eval_perc = eval_perc
    
    def fit( 
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        channel_labels: Optional[np.ndarray] = None, 
        results_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fit the model on time-series data, propagating metadata through PipelineMessage.
        """
        results = {}
        t0 = time.time()
        
        message = self._prepare_input(
            channel_data, channel_labels, results=results, save_dir=results_dir, split_label="train"
        )
        # Segmentation: ensure the message is segmented
        message = self.ts_splitter.fit_transform(message)
        
        print(f"[DEBUG] _prepare_input (train) took {time.time() - t0:.2f}s. Segments: {len(message.data)}")
        results['window_size'] = getattr(self.ts_splitter, "window_size", None)

        X = message.data
        y = message.labels
        indices = message.original_indices
        dataset = getattr(message, "original_dataset", None)

        if self.detector is not None and hasattr(self.detector, 'fit') and self.eval_perc is not None and 0.0 < self.eval_perc < 1.0:
            split_idx = int(len(X) * (1 - self.eval_perc))
            X_train = X[:split_idx]
            X_val = X[split_idx:]
            idx_train = indices[:split_idx] if indices is not None else None
            idx_val = indices[split_idx:] if indices is not None else None
            y_train = y[:split_idx] if y is not None else None
            
            # Create sub-messages for Train/Val if needed by components
            train_msg = PipelineMessage(
                data=X_train, labels=y_train, original_indices=idx_train, results=results, save_dir=results_dir, split_label="train"
            )
            val_msg = PipelineMessage(
                data=X_val, labels=y[split_idx:] if y is not None else None, original_indices=idx_val, results=results, save_dir=results_dir, split_label="val"
            )
        else:
            X_train = X
            X_val = None
            idx_train = indices
            idx_val = None
            y_train = y
            train_msg = message
            train_msg.split_label = "train"
            val_msg = None

        if self.feature_extractor is not None:
            self.feature_extractor.set_context(dataset=dataset, indices=idx_train)
            t0 = time.time()
            # Pass both train + val so fit can use val for feature selection
            if val_msg is not None:
                train_msg = self.feature_extractor.fit_transform(train_msg, val_msg)
            else:
                train_msg = self.feature_extractor.fit_transform(train_msg)
            X_train = train_msg.data
            print(f"[DEBUG] feature_extractor.fit_transform took {time.time() - t0:.2f}s")
            
            if val_msg is not None:
                self.feature_extractor.set_context(dataset=dataset, indices=idx_val)
                t0 = time.time()
                val_msg = self.feature_extractor.transform(val_msg)
                X_val = val_msg.data
                print(f"[DEBUG] feature_extractor.transform (val) took {time.time() - t0:.2f}s")
                
        # Clear context after use
        if self.feature_extractor is not None:
            self.feature_extractor.clear_context()

        # Short-circuit if no features were selected
        if self.feature_extractor is not None and self.feature_extractor.kill_switch_active:
            print(f"[DEBUG] RollingWindowClassifier short-circuit in fit (0 features).")
            return results

        with self._callback_context("fitting", results):
            t0 = time.time()
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
            print(f"[DEBUG] base_classifier.fit took {time.time() - t0:.2f}s")
        
        if self.detector is not None and hasattr(self.detector, 'fit'):
            with self._callback_context("detector_calibration", results):
                if X_val is not None:
                    print(f"[DEBUG] Fitting detector on X_val (length: {len(X_val)})")
                    t0 = time.time()
                    val_scores = self.base_classifier.predict(X_val)
                    self.detector.fit(val_scores)
                    print(f"[DEBUG] detector calibration (val) took {time.time() - t0:.2f}s")
                else:
                    # Fallback sui dati di train (attenzione all'overfitting delle soglie)
                    print(f"[DEBUG] X_val is None. Falling back to X_train (length: {len(X_train)})")
                    t0 = time.time()
                    train_scores = self.base_classifier.predict(X_train)
                    print(f"[DEBUG] base_classifier.predict (train scores) took {time.time() - t0:.2f}s")
                    self.detector.fit(train_scores)
                    print(f"[DEBUG] detector calibration (train) took {time.time() - t0:.2f}s")

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
        
        # 1. Prepare Message
        message = self._prepare_input(channel_data, results=results, save_dir=results_dir, split_label="test")
        
        # 2. Segmentation
        message = self.ts_splitter.transform(message)
        print(f"[DEBUG] _prepare_input (test) took {time.time() - t0:.2f}s. Segments: {len(message.data)}")

        # 3. Feature Extraction
        if self.feature_extractor is not None:
            self.feature_extractor.set_context(dataset=getattr(message, "original_dataset", None), indices=message.original_indices)
            t0 = time.time()
            message = self.feature_extractor.transform(message)
            print(f"[DEBUG] feature_extractor.transform (test) took {time.time() - t0:.2f}s")
            self.feature_extractor.clear_context()
            
            if getattr(self.feature_extractor, "kill_switch_active", False):
                print("[DEBUG] Feature selection kill-switch is ACTIVE. Forcing zero anomalies.")
                return np.zeros(len(message.data)), results
        
        channel_data_proc = message.data

        with self._callback_context("prediction", results):
            t0 = time.time()
            try:
                if hasattr(self.base_classifier, "predict_proba"):
                    y_pred = self.base_classifier.predict_proba(channel_data_proc, results=results)
                    if y_pred.ndim > 1 and y_pred.shape[1] == 2:
                        y_pred = y_pred[:, 1]
                else:
                    y_pred = self.base_classifier.predict(channel_data_proc, results=results)
            except (TypeError, ValueError):
                if hasattr(self.base_classifier, "predict_proba"):
                    y_pred = self.base_classifier.predict_proba(channel_data_proc)
                    if y_pred.ndim > 1 and y_pred.shape[1] == 2:
                        y_pred = y_pred[:, 1]
                else:
                    y_pred = self.base_classifier.predict(channel_data_proc)
            
            print(f"[DEBUG] base_classifier.predict took {time.time() - t0:.2f}s")
            print(f"[DEBUG] y_pred stats - min: {y_pred.min():.4f}, max: {y_pred.max():.4f}, mean: {y_pred.mean():.4f}")
            print(f"[DEBUG] y_pred counts - >0: {np.sum(y_pred > 0)}, >0.5: {np.sum(y_pred > 0.5)}, ==1: {np.sum(y_pred == 1)}")

        if self.detector is not None:
            with self._callback_context("detection", results):
                t0 = time.time()
                y_pred = self.detector.detect(y_pred)
                print(f"[DEBUG] detector.detect took {time.time() - t0:.2f}s")

        return y_pred, results

    def _prepare_input(
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset],
        channel_labels: Optional[np.ndarray] = None,
        results: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
        split_label: str = "train"
    ) -> PipelineMessage:
        """
        Prepare the raw data into a PipelineMessage.
        """
        from spaceai.models.anomaly_classifier.anomaly_classifier import PipelineMessage
        
        msg = PipelineMessage(
            data=channel_data,
            labels=channel_labels,
            results=results or {},
            save_dir=save_dir,
            split_label=split_label
        )
        
        if isinstance(channel_data, AnomalyDataset):
            setattr(msg, "original_dataset", channel_data)
            
        return msg



