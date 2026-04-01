from __future__ import annotations
"""Abstract base class for anomaly classifiers."""

from abc import abstractmethod
from typing import Optional, List, Tuple, Any, Union, Dict, TYPE_CHECKING
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
import logging

from spaceai.benchmark.callbacks.mixin import CallbackMixin
from spaceai.benchmark.callbacks.handler import CallbackHandler
from spaceai.data import AnomalyDataset

class AnomalyClassifier(CallbackMixin):
    """
    Abstract base for time-series wrappers: defines common interface and input preparation.
    """
    
    def __init__(self, callback_handler: Optional[CallbackHandler] = None, **kwargs):
        super().__init__(callback_handler=callback_handler, **kwargs)

    @abstractmethod
    def fit(  # pylint: disable=invalid-name
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit the model on time-series data X, optionally with labels y.
        """

    @abstractmethod
    def predict(self, X: Any) -> Tuple[np.ndarray, Dict[str, Any]]:  # pylint: disable=invalid-name
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
    def prepare_labels(channel_labels):
        """ Prepare the ground trutg to uniform with the predicted labels"""

    def save(self, path: str) -> None:
        """Save the classifier to disk."""
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> "AnomalyClassifier":
        """Load a classifier from disk."""
        return torch.load(path, weights_only=False)

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
class PipelineMessage:
    data: Union[np.ndarray, List[Any]]
    original_indices: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    pred_intervals: Optional[List[Tuple[int, int]]] = None
    true_intervals: Optional[List[Tuple[int, int]]] = None
    results: Dict[str, Any] = field(default_factory=dict)
    save_dir: Optional[str] = None
    split_label: str = "train"  
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def replace(self, **kwargs) -> "PipelineMessage":
        """Returns a new PipelineMessage with updated fields."""
        import dataclasses
        return dataclasses.replace(self, **kwargs)

class AnomalyDetectionPipeline(AnomalyClassifier):
    """Modular anomaly detection pipeline with role-aware fit logic.
    
    Steps are native components (splitters, extractors, classifiers, detectors)
    that implement the PipelineMessage contract.
    
    Args:
        steps: List of (name, processor) tuples.
        callback_handler: Optional callback handler.
        eval_perc: Fraction of data to hold out for validation/calibration.
    """
    def __init__(self,
                steps: List[Tuple[str, Union[CallbackMixin, Any, None]]],
                callback_handler: Optional[CallbackHandler] = None,
                eval_perc: Optional[float] = None,
                ):
        super().__init__(callback_handler=callback_handler)
        self.steps = [(name, proc) for name, proc in steps if proc is not None]
        self.named_steps = dict(self.steps)
        self.eval_perc = eval_perc
   
    def fit( 
        self,
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset, Any],
        channel_labels: Optional[np.ndarray] = None, 
        results_dir: Optional[str] = None,
        return_message: bool = False
    ) -> Dict[str, Any]:
        """
        Fit the model on time-series data, propagating metadata through PipelineMessage.
        Each processor is fitted sequentially and then transforms the message for the next stage.
        """
        msg = self._prepare_message(channel_data, channel_labels, save_dir=results_dir)
        msgs = self._split_data(msg)
        
        for name, processor in self.steps:
            print(f"DEBUG: Pipeline FIT - Processing step: {name} ({processor.__class__.__name__})", flush=True)
            if hasattr(processor, "fit"):
                processor.fit(*msgs)

            if getattr(processor, "kill_switch_active", False) or \
               any(m.metadata.get("kill_switch_active", False) for m in msgs):
                logging.warning("Pipeline short-circuit at %s (%s). No features selected or empty data.", name, processor.__class__.__name__)
                for m in msgs:
                    m.metadata["kill_switch_active"] = True
                self.kill_switch_active = True
                for m in msgs:
                    m.data = np.zeros(m.data.shape[0])
                break

            if hasattr(processor, "transform"):
                msgs = [processor.transform(m) for m in msgs]
                for i, m in enumerate(msgs):
                    shape = m.data.shape if hasattr(m.data, 'shape') else len(m.data) if hasattr(m.data, '__len__') else 'N/A'
                    print(f"DEBUG: Pipeline FIT - Step {name} output {i} shape: {shape}", flush=True)
            
        # Store validation results as attribute (sklearn convention)
        if len(msgs) > 1:
            val_msg = msgs[1]
            self.val_results_ = {
                "y_true": val_msg.labels,
                "y_pred": val_msg.data,
                "true_intervals": val_msg.true_intervals,
                "original_indices": val_msg.original_indices,
            }
        else:
            self.val_results_ = None

        if return_message:
            return msgs[0] if len(msgs) == 1 else msgs
        return {**msgs[0].results, **msgs[0].metadata}

    def predict(
        self, 
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset, Any],
        results_dir: Optional[str] = None,
        return_message: bool = False 
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict via the modular pipeline.
        """
        msg = self._prepare_message(channel_data, save_dir=results_dir, split_label="test")

        if self.kill_switch_active:
            logging.warning("Pipeline prediction short-circuit (kill switch activated during fit). Returning all zeros.")
            n_samples = len(msg.data) if hasattr(msg.data, "__len__") else 0
            return np.zeros(n_samples), {**msg.results, **msg.metadata}

        for name, processor in self.steps:
            print(f"DEBUG: Pipeline PREDICT - Processing step: {name} ({processor.__class__.__name__})", flush=True)
            if getattr(processor, "kill_switch_active", False) or msg.metadata.get("kill_switch_active", False):
                logging.warning("Pipeline prediction short-circuit at %s (%s). Returning all zeros.", name, processor.__class__.__name__)
                n_samples = len(msg.data) if hasattr(msg.data, "__len__") else 0
                return np.zeros(n_samples), {**msg.results, **msg.metadata}

            if hasattr(processor, "predict"):
                msg = processor.predict(msg)
            elif hasattr(processor, "transform"):
                msg = processor.transform(msg)
            
            shape = msg.data.shape if hasattr(msg.data, 'shape') else len(msg.data) if hasattr(msg.data, '__len__') else 'N/A'
            print(f"DEBUG: Pipeline PREDICT - Step {name} output shape: {shape}", flush=True)
            if hasattr(msg, 'pred_intervals') and msg.pred_intervals:
                print(f"DEBUG: Pipeline PREDICT - Step {name} generated {len(msg.pred_intervals)} intervals.", flush=True)
        if msg.original_indices is not None:
            msg.results["original_indices"] = msg.original_indices

        if return_message:
            return msg
        return msg.data, msg.results

    @staticmethod   
    def _prepare_message(
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset, Any],
        channel_labels: Optional[np.ndarray] = None,
        save_dir: Optional[str] = None,
        split_label: str = "train"
    ) -> PipelineMessage:
        """Initialize the PipelineMessage with data and initial original_indices."""
        msg = PipelineMessage(
            data=channel_data,
            original_indices=np.arange(len(channel_data)) if hasattr(channel_data, "__len__") else None,
            labels=channel_labels,
            save_dir=save_dir,
            split_label=split_label
        )
        if isinstance(channel_data, AnomalyDataset):
            setattr(msg, "original_dataset", channel_data)
        return msg

    def _split_data(self, msg: PipelineMessage) -> List[PipelineMessage]:
        """Splits the message data for internal evaluation (e.g. calibration)."""
        if self.eval_perc is None or self.eval_perc <= 0.0 or msg.data is None:
            return [msg]
            
        n_samples = len(msg.data)
        split_idx = int(n_samples * (1 - self.eval_perc))
        
        if isinstance(msg.data, AnomalyDataset):
            from spaceai.data.anomaly_dataset import AnomalyDatasetSubset
            data_train = AnomalyDatasetSubset(msg.data, 0, split_idx - 1)
            data_val = AnomalyDatasetSubset(msg.data, split_idx, n_samples - 1)
        else:
            data_train = msg.data[:split_idx]
            data_val = msg.data[split_idx:]

        train_labels = msg.labels[:split_idx] if msg.labels is not None else None
        val_labels = msg.labels[split_idx:] if msg.labels is not None else None
        
        train_indices = msg.original_indices[:split_idx] if msg.original_indices is not None else None
        val_indices = msg.original_indices[split_idx:] if msg.original_indices is not None else None

        msg_train = PipelineMessage(
            data=data_train, labels=train_labels, original_indices=train_indices,
            results=msg.results.copy(), save_dir=msg.save_dir, split_label="train"
        )
        msg_val = PipelineMessage(
            data=data_val, labels=val_labels, original_indices=val_indices,
            results=msg.results.copy(), save_dir=msg.save_dir, split_label="val"
        )
        
        return [msg_train, msg_val]

    def prepare_labels(
        self, 
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset, Any],
        results: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, int]]:
        """Prepare labels by running them through the splitters in the pipeline."""
        msg = self._prepare_message(channel_data)
        
        from spaceai.preprocessing.ts_splitter import TimeSeriesSplitter
        for _, processor in self.steps:
            wrapped = getattr(processor, "model", processor)
            if isinstance(wrapped, TimeSeriesSplitter):
                msg = wrapped.transform(msg, mode="anomaly", results=results)
                return msg.true_intervals if msg.true_intervals is not None else []
                
        return []

     
    def save(self, path: str):
        """Save pipeline to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def map_to_timestamps(
        self, 
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset, Any], 
        anomalies: List[Tuple[int, int]],
        results: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Any, Any]]:
        """Map window-level anomalies back to timestamps using message metadata."""
        msg = self._prepare_message(channel_data)
        
        from spaceai.preprocessing.ts_splitter import TimeSeriesSplitter
        for _, processor in self.steps:
            wrapped = getattr(processor, "model", processor)
            if isinstance(wrapped, TimeSeriesSplitter):
                msg = wrapped.transform(msg, results=results)
                break
        
        if msg.original_indices is None:
            return []

        timestamps = getattr(channel_data, "timestamps", None)
        offset = getattr(channel_data, "start_idx", 0)
        
        time_intervals = []
        for ws, we in anomalies:
            if ws >= len(msg.original_indices):
                continue
            s_idx = int(msg.original_indices[ws][0])
            e_idx = int(msg.original_indices[min(we, len(msg.original_indices)-1)][1])
            
            if timestamps is not None:
                time_intervals.append((timestamps[s_idx], timestamps[e_idx]))
            else:
                time_intervals.append((s_idx + offset, e_idx + offset))
                
        return time_intervals