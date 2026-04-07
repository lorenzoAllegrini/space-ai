from __future__ import annotations
"""Abstract base class for anomaly classifiers."""

from abc import abstractmethod
from typing import Optional, List, Tuple, Any, Union, Dict, TYPE_CHECKING
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from spaceai.benchmark.callbacks.mixin import CallbackMixin

from spaceai.benchmark.callbacks.handler import CallbackHandler
from spaceai.data import AnomalyDataset, AnomalyDatasetSubset
import joblib

class AnomalyClassifier(CallbackMixin):
    """
    Abstract base for time-series wrappers: defines common interface and input preparation.
    """
    
    def __init__(self, callback_handler: Optional[CallbackHandler] = None, **kwargs):
        super().__init__(callback_handler=callback_handler, **kwargs)

    @abstractmethod
    def fit(  # pylint: disable=invalid-name
        self, X: np.ndarray, y: Optional[np.ndarray] = None, results: Optional[Dict[str, Any]] = None, **kwargs
    ) -> None:
        """
        Fit the model on time-series data X, optionally with labels y.
        """

    @abstractmethod
    def predict(self, X: Any, results: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:  # pylint: disable=invalid-name
        """
        Predict on time-series data X, returning a tuple of (predictions, metrics).
        """

    def fit_predict(self, X: Any, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Convenience method for continual learning. Predicts on X, then fits on (X, y).
        Streaming wrappers can override this to optimize the roundtrip.
        """
        preds, metrics = self.predict(X)
        self.fit(X, y)
        return preds, metrics

    def map_to_timestamps(
        self, channel_data: Any, anomalies: List[Tuple[int, int]]
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Map a list of predicted or ground-truth anomaly indices to global timestamps.
        Must be implemented by child classes according to their prediction domains (window vs sample level).
        """
        return []

    @abstractmethod
    def prepare_labels(self, channel_labels: Any) -> List[Tuple[int, int]]:
        """ Prepare the ground truth to uniform with the predicted labels"""

    def save(self, path: str) -> None:
        """Save the classifier to disk."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "AnomalyClassifier":
        """Load a classifier from disk."""
        return joblib.load(path)

    @staticmethod
    def _prepare_input(X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """
        Ensure X is 3D with shape (n_samples, n_channels=1, n_timestamps).
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input X must be 2D (n_samples, n_timestamps)")
        return X.reshape(X.shape[0], 1, X.shape[1])



@dataclass
class PipelineState:
    data: Union[np.ndarray, List[Any], Any]
    labels: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None
    intervals: Optional[List[List[int]]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AnomalyDetectionPipeline:
    """
    Abstract base for time-series wrappers: defines common interface and input preparation.
    """
    def __init__(self,
                steps: List[Tuple[str, Union[Any, Any]]],
                callback_handler: Optional[CallbackHandler] = None,
                eval_perc: Optional[float] = None,
                ):
        self.callback_handler = callback_handler
        self.steps = steps
        self.named_steps = dict(steps)
        self.eval_perc = eval_perc
        
    def _prepare_flows(
        self, 
        channel_data: Any, 
        channel_labels: Optional[Any], 
        results_dir: Optional[str]
    ) -> Tuple[list, list, list, bool]:
        """
        Prepara gli stati (PipelineState) e divide gli step della pipeline 
        tra quelli di base e quelli che richiedono calibrazione su validazione.
        """
        needs_calib = bool(self.eval_perc and self.eval_perc > 0.0 and len(self.steps) > 1)
        
        base_steps = []
        calib_steps = []
        
        for name, processor in self.steps:
            if not hasattr(processor, "pipeline_step"):
                continue
            if getattr(processor, "requires_calibration", False) and needs_calib:
                calib_steps.append(processor)
            else:
                base_steps.append(processor)

        flows = []
        if needs_calib:
            d_tr, d_vl, l_tr, l_vl = self._split_timeseries(channel_data, channel_labels, self.eval_perc)
            flows.append((PipelineState(data=d_tr, labels=l_tr, metadata={"save_dir": results_dir}), True))
            flows.append((PipelineState(data=d_vl, labels=l_vl, metadata={"save_dir": results_dir}), False))
        else:
            flows.append((PipelineState(data=channel_data, labels=channel_labels, metadata={"save_dir": results_dir}), True))

        return base_steps, calib_steps, flows, needs_calib


    def fit(
        self, 
        channel_data: Any, 
        channel_labels: Optional[Any] = None, 
        results_dir: Optional[str] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Esegue l'addestramento della pipeline, gestendo dinamicamente i flussi 
        di Train ed eventuale Validazione (per calibrazione nodi finali).
        """
        # 1. Setup delegato al metodo helper
        base_steps, calib_steps, flows, needs_calib = self._prepare_flows(
            channel_data, channel_labels, results_dir
        )

        # 2. Addestramento Base (es. Splitter, Extractor, Classifier...)
        for processor in base_steps:
            # Check for short-circuit (kill-switch activated in previous steps)
            if any(state.metadata.get("kill_switch_active", False) for state, _ in flows):
                continue

            flows = [
                (processor.pipeline_step(state, is_fit=is_fit, **kwargs), is_fit) 
                for state, is_fit in flows
            ]

        # 3. Addestramento Nodi di Calibrazione (es. ThresholdDetector...)
        if needs_calib:
            for processor in calib_steps:
                # Si addestrano SOLO sull'ultimo flusso (il Validation Set)
                val_state = flows[-1][0]
                val_state = processor.pipeline_step(val_state, is_fit=True, **kwargs)
                flows[-1] = (val_state, False)

        # 4. Raccolta metriche
        metrics = {}
        for state, _ in flows:
            metrics.update(state.metrics)
            
        return metrics
        
    def predict(
        self, 
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset, Any],
        results_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict via the modular pipeline.
        """
        state = PipelineState(data=channel_data, metadata={"save_dir": results_dir})

        for name, processor in self.steps:
            if hasattr(processor, "pipeline_step"):
                state = processor.pipeline_step(state, is_fit=False)
            
        return state.data, state.metrics

    
    def prepare_labels(
        self, 
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset, Any],
        results: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, int]]:
        """
        Prepare labels by running them through the splitters in the pipeline.
        """
        state = PipelineState(data=channel_data)
        
        from spaceai.preprocessing.ts_splitter import TimeSeriesSplitter
        for _, processor in self.steps:
            if isinstance(processor, TimeSeriesSplitter):
                if hasattr(processor, "pipeline_step"):
                    state = processor.pipeline_step(state, is_fit=False)
                return state.intervals if state.intervals is not None else []
                
        return []

    def _split_timeseries(
        self, 
        X: Any, 
        y: Optional[np.ndarray], 
        eval_perc: float
    ) -> Tuple[Any, Any, Optional[np.ndarray], Optional[np.ndarray]]:
        if eval_perc <= 0.0 or eval_perc >= 1.0:
            return X, None, y, None

        if hasattr(X, "__len__"):
            n_samples = len(X)
            split_idx = int(n_samples * (1 - eval_perc))

            if isinstance(X, AnomalyDataset):
                X_tr = AnomalyDatasetSubset(parent=X, start_idx=0, end_idx=split_idx)
                X_vl = AnomalyDatasetSubset(parent=X, start_idx=split_idx, end_idx=n_samples)
            else:
                X_tr, X_vl = X[:split_idx], X[split_idx:]

            y_tr = y[:split_idx] if y is not None else None
            y_vl = y[split_idx:] if y is not None else None
            
            return X_tr, X_vl, y_tr, y_vl
        
        return X, None, y, None

    def save(self, path: str) -> None:
        """Save the classifier to disk."""
        pass

    @staticmethod
    def load(path: str) -> "AnomalyDetectionPipeline":
        """Load a classifier from disk."""
        return joblib.load(path)

    def map_to_timestamps(
        self, 
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset, Any], 
        anomalies: List[Tuple[int, int]],
        results: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Any, Any]]:
        """
        Map window-level anomalies back to timestamps using message metadata.
        """
        state = PipelineState(data=channel_data)
        
        for _, processor in self.steps:
            from spaceai.preprocessing.ts_splitter import TimeSeriesSplitter
            if isinstance(processor, TimeSeriesSplitter):
                if hasattr(processor, "pipeline_step"):
                    state = processor.pipeline_step(state, is_fit=False)
                break
        
        if state.indices is None:
            return []

        # Map using timestamps if available
        timestamps = getattr(channel_data, "timestamps", None)
        offset = getattr(channel_data, "start_idx", 0)
        
        time_intervals = []
        for ws, we in anomalies:
            if ws >= len(state.indices):
                continue
            s_idx = int(state.indices[ws][0])
            e_idx = int(state.indices[min(we, len(state.indices)-1)][1])
            
            if timestamps is not None:
                time_intervals.append((timestamps[s_idx], timestamps[e_idx]))
            else:
                time_intervals.append((s_idx + offset, e_idx + offset))
                
        return time_intervals
    