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
                print(f"[BUFFER] Adding {len(retrain_data)} nominal samples to replay buffer (Buffer size: {len(self.buffer)}/{self.buffer.max_size})")
                self.buffer.add(retrain_data, y=retrain_labels)
            else:
                print("[BUFFER] Warning: No nominal samples found to add to buffer.")
        return X, y

    def sample(
        self, 
        X: np.ndarray,
        y: Optional[np.ndarray] = None, 
        results: Optional[Dict[str, Any]] = None, 
        **kwargs
    ) -> tuple[np.ndarray, Optional[np.ndarray]]: 
        """
        Samples from the replay buffer and combines with X.
        Intended to be used only in the training phase.
        """
        with self._callback_context("buffer_handler_sample", results):
            buffer_size = len(self.buffer) if self.buffer is not None else 0
            if buffer_size == 0:
                print("[BUFFER] Replay buffer is empty, skipping sampling.")
                return X, y
                
            sample_size = getattr(self.buffer, 'sample_size', 1000)
            sampled_X, sampled_y = self.buffer.sample(sample_size=sample_size, results=results)
            
            if len(sampled_X) > 0:
                print(f"[BUFFER] Sampling {len(sampled_X)} historical samples and mixing with {len(X)} new samples")
                if hasattr(self.buffer, '_concatenate'):
                    X = self.buffer._concatenate(sampled_X, X)
                else:
                    X = np.concatenate([np.array(sampled_X), X], axis=0)
                    
                if y is not None and sampled_y is not None:
                    y = np.concatenate([np.array(sampled_y), y], axis=0)
                    
        return X, y