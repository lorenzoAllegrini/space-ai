from __future__ import annotations

from typing import Optional, List, Tuple, Any, Union, Dict, TYPE_CHECKING
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from spaceai.benchmark.callbacks.mixin import CallbackMixin

from spaceai.benchmark.callbacks.handler import CallbackHandler
from spaceai.data import AnomalyDataset, AnomalyDatasetSubset

@dataclass
class PipelineState:
    data: Union[np.ndarray, List[Any], Any]
    labels: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None
    intervals: Optional[List[List[int]]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def segments(self) -> Union[np.ndarray, List[Any], Any]:
        """Backward compatibility alias for data."""
        return self.data

    @property
    def segment_indices(self) -> Optional[np.ndarray]:
        """Backward compatibility alias for indices."""
        return self.indices

@dataclass
class PhaseConfig:
    method: str
    supervised: bool = False

@dataclass
class PipelineStep:
    name: str
    processor: CallbackMixin    
    phases: Dict[str, Union[str, PhaseConfig]] = field(
        default_factory=lambda: {"train": None, "val": None, "predict": None}
    )

class AnomalyDetectionPipeline:
    """
    Abstract base for time-series wrappers: defines common interface and input preparation.
    """
    def __init__(self,
                steps: List[PipelineStep],
                eval_perc: Optional[float] = None,
                phase_map: Optional[Dict[str, List[str]]] = None
                ):
        self.steps = steps
        #self.eval_perc = eval_perc
        
        # Default phase map if none provided
        self.phase_map = phase_map or {
            "fit": ["train", "val"],
            "predict": ["predict"]
        }

    def _get_steps_for_phase(self, phase: str) -> List[Tuple[str, Any, Union[str, PhaseConfig]]]:
        """Filter and return: (name, processor, phase_config)"""
        phase_steps = []
        for step in self.steps:
            if hasattr(step, "phases"):
                if phase in step.phases:
                    config = step.phases[phase]
                    phase_steps.append((step.name, step.processor, config))
                
        return phase_steps
        
    def fit(
        self, 
        channel_data: Any, 
        channel_labels: Optional[Any] = None, 
        results_dir: Optional[str] = None, 
        phase: str = "fit",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute training by separating execution into phases defined in phase_map[phase].
        """

        metrics = {}
        fit_phases = self.phase_map.get(phase, ["train", "val"])
        
        for i, p_name in enumerate(fit_phases):
    
            if i == 0:
                state = PipelineState(data=channel_data, labels=channel_labels, metadata={"save_dir": results_dir})
            else:
                state = PipelineState(data=None, labels=None, metadata={"save_dir": results_dir})
            
            for name, processor, config in self._get_steps_for_phase(p_name):
                if hasattr(processor, "pipeline_step"):
                    is_node_fit = True if i == 0 else getattr(processor, "requires_calibration", False)
                    
                    state = self._run_step(
                        processor, state, is_fit=is_node_fit, results=metrics, phase_config=config, **kwargs
                    )

        metrics["metadata"] = state.metadata
        return metrics
        
    def _map_state_to_points(self, state: PipelineState, channel_data: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        total_len = len(channel_data) if hasattr(channel_data, "__len__") else 0
        if total_len == 0:
            return state.data, state.labels
            
        point_data = np.zeros(total_len)
        point_labels = np.zeros(total_len, dtype=int) if state.labels is not None else None
        data_arr = np.asarray(state.data)
        labels_arr = np.asarray(state.labels) if state.labels is not None else None
        
        for i, (start, end) in enumerate(state.indices):
            val = data_arr[i]
            # Handle 2D data (e.g. from models returning probabilities for each class)
            if hasattr(val, "__len__") and len(val) > 0:
                val = val[0]
                
            point_data[int(start):int(end)+1] = np.maximum(point_data[int(start):int(end)+1], val)
            if point_labels is not None:
                point_labels[int(start):int(end)+1] = np.maximum(point_labels[int(start):int(end)+1], labels_arr[i])
                
        return point_data, point_labels

    def predict(
        self, 
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset, Any],
        y: Optional[np.ndarray] = None,
        results_dir: Optional[str] = None,
        phase: str = "predict",
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Global predict. Executes the phases defined in phase_map[phase].
        """
        state = PipelineState(data=channel_data, labels=y, metadata={"save_dir": results_dir})
        metrics = {}
        
        predict_phases = self.phase_map.get(phase, ["predict"])
        
        for p_pred in predict_phases:
            for name, processor, config in self._get_steps_for_phase(p_pred):
                if hasattr(processor, "pipeline_step"):
                    state = self._run_step(
                        processor, state, is_fit=False, results=metrics, phase_config=config, **kwargs
                    )
            
        metrics["metadata"] = state.metadata
        
        if state.indices is not None:
            state.data, state.labels = self._map_state_to_points(state, channel_data)

        return state.data, state.labels, state.metrics

    def _run_step(self, processor, state, is_fit, results, phase_config, **kwargs):
        """Execute a step, managing supervision and injecting the correct method."""
        
        if isinstance(phase_config, PhaseConfig):
            method_name = phase_config.method
            supervised = phase_config.supervised
        else:
            method_name = phase_config
            supervised = False
            
        effective_state = state
        if not supervised and state.labels is not None:
            effective_state = PipelineState(
                data=state.data,
                labels=None,
                indices=state.indices,
                intervals=state.intervals,
                metrics=state.metrics,
                metadata=state.metadata
            )

        step_kwargs = kwargs.copy()
        step_kwargs.pop("method_name", None)
        
        new_state = processor.pipeline_step(effective_state, is_fit=is_fit, results=results, method_name=method_name, **step_kwargs)
        
        if not supervised and state.labels is not None and new_state.labels is None:
            new_state.labels = state.labels

        return new_state

    def _get_shape(self, data):
        if hasattr(data, "shape"):
            return data.shape
        if isinstance(data, list):
            return f"list(len={len(data)})"
        return "scalar/unknown"
    
    def prepare_labels(
        self, 
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset, Any],
        results: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, int]]:
        """
        Prepare absolute sample-level intervals from the dataset.
        """
        state = PipelineState(data=channel_data)
        
        from spaceai.preprocessing.ts_splitter import TimeSeriesSplitter
        found_splitter = False
        for step in self.steps:
            if isinstance(step.processor, TimeSeriesSplitter):
                if hasattr(step.processor, "pipeline_step"):
                    state = step.processor.pipeline_step(state, is_fit=False, results=results)
                    found_splitter = True
                break
        
        if not found_splitter or state.intervals is None or state.indices is None:
            anoms = getattr(channel_data, "anomalies", [])
            logging.debug("AnomalyDetectionPipeline: Splitter not found or no intervals. Using raw dataset anomalies: %d", len(anoms))
            return [[int(s), int(e)] for s, e in anoms]

        sample_intervals = []
        for ws, we in state.intervals:
            s_idx = int(state.indices[max(0, ws)][0])
            e_idx = int(state.indices[min(we, len(state.indices)-1)][1])
            sample_intervals.append((s_idx, e_idx))
        
        logging.debug("AnomalyDetectionPipeline: Prepared %d sample intervals from %d segments", len(sample_intervals), len(state.intervals))
        return sample_intervals

    def _split_timeseries(
        self, 
        X: Any, 
        y: Optional[np.ndarray], 
        eval_perc: Optional[float]
    ) -> Tuple[Any, Any, Optional[np.ndarray], Optional[np.ndarray]]:
        if eval_perc is None or eval_perc <= 0.0 or eval_perc >= 1.0:
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

    def map_to_timestamps(
        self, 
        channel_data: Union[np.ndarray, List[np.ndarray], AnomalyDataset, Any], 
        anomalies: List[Tuple[int, int]],
        results: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Any, Any]]:
        if not anomalies:
            return []

        timestamps = getattr(channel_data, "timestamps", None)
        offset = getattr(channel_data, "start_idx", 0)
        
        time_intervals = []
        for s_idx, e_idx in anomalies:
            if timestamps is not None:
                s_safe = max(0, min(len(timestamps)-1, int(s_idx)))
                e_safe = max(0, min(len(timestamps)-1, int(e_idx)))
                time_intervals.append((timestamps[s_safe], timestamps[e_safe]))
            else:
                time_intervals.append((int(s_idx) + offset, int(e_idx) + offset))
                
        return time_intervals

    def prepare_labels(self, channel_labels: Any) -> List[Tuple[int, int]]:
        """Prepare ground truth labels as interval tuples."""
        if hasattr(channel_labels, 'anomaly_sequences'):
            return channel_labels.anomaly_sequences
        return []

    def save(self, path: str) -> None:
        """Save the classifier to disk."""
        pass

    @staticmethod
    def load(path: str) -> "AnomalyDetectionPipeline":
        """Load a classifier from disk."""
        return joblib.load(path)