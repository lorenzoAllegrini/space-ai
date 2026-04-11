import numpy as np
from typing import Optional, Dict, Any
from spaceai.benchmark.callbacks.mixin import CallbackMixin
from spaceai.models.detectors import AnomalyDetector

class BufferHandler(CallbackMixin):
    """
    Centralized and declarative Active Memory manager.
    Replaces Sampler, Collector, and Stasher in a single cohesive class.
    Its methods are dynamically invoked by AnomalyDetectionPipeline.
    """
    requires_calibration = True
    
    def __init__(
        self, 
        buffer: Any, # Use Any to avoid circular imports with ReplayBuffer
        replay_detector: Optional[AnomalyDetector] = None, 
        **kwargs
    ):
        self.buffer = buffer
        self.replay_detector = replay_detector
        self.temporary_memory = None
        self.temporary_labels = None
        super().__init__(**kwargs)

    def collect(
        self, 
        X: np.ndarray,
        y: Optional[np.ndarray] = None, 
        results: Optional[Dict[str, Any]] = None, 
        **kwargs
    ) -> tuple[np.ndarray, Optional[np.ndarray]]: 
        """
        To be placed AFTER the feature extractor and BEFORE the model.
        - In Train: Samples historical data and joins it with the current batch.
        - In Predict/Val: Saves a copy of the features in metadata.
        """
        self.temporary_memory = X
        self.temporary_labels = y
        
        return X, y

    def threshold(
        self, 
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        results: Optional[Dict[str, Any]] = None, 
        is_fit: bool = False,
        **kwargs
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        To be placed AFTER the model (DPMM) and BEFORE the production detector.
        - In Val (is_fit=True): Calibrates the internal strict detector.
        - In Predict: Evaluates probabilities, filters features and saves to memory.
        """
        with self._callback_context("buffer_handler_threshold", results):
            
            if self.replay_detector is None:
                return X, y

            if is_fit:
                self.replay_detector.fit(X, y=y, results=results)
                return X, y

            retrain_mask = self.replay_detector.detect(X)
            nominal_idx = (np.array(retrain_mask).flatten() == 0)
            
            if self.temporary_memory is None or len(self.temporary_memory) != len(retrain_mask):
                raise ValueError("Error in BufferHandler Thresholding: Size mismatch.")
            
            retrain_data = self.temporary_memory[nominal_idx]
            retrain_labels = self.temporary_labels[nominal_idx] if self.temporary_labels is not None else None
            
            avg_all = np.mean(X)
            min_all, max_all = np.min(X), np.max(X)
            q5, q95 = np.quantile(X, 0.05), np.quantile(X, 0.95)
            
            avg_buffered = np.mean(X[nominal_idx]) if any(nominal_idx) else 0
            

            if len(retrain_data) > 0:
                self.buffer.add(retrain_data, y=retrain_labels)