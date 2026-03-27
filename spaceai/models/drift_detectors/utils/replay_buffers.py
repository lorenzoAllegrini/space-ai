from __future__ import annotations
"""Replay buffers with optional delayed processing functionality."""

import collections
import random
from typing import Any, List, Optional, Tuple, Union, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd
from spaceai.benchmark.callbacks.mixin import CallbackMixin

if TYPE_CHECKING:
    from spaceai.benchmark.callbacks.handler import CallbackHandler


class ReplayBuffer(CallbackMixin):
    """Abstract interface for a replay buffer.
    
    Allows storing and retrieving historical data for model retraining.
    """
    
    def add(self, X: Any, y: Optional[Any] = None, score: float = 0.0, results: Optional[Dict[str, Any]] = None) -> None:
        """Add a new observation to the buffer.
        
        Args:
            X: The input features/segment.
            y: The target label (if applicable).
            score: An optional anomaly/drift score associated with the observation.
            results (Optional[Dict[str, Any]]): Dictionary to update with metrics.
        """
        raise NotImplementedError

    def get_all(self, results: Optional[Dict[str, Any]] = None) -> Tuple[List[Any], Optional[List[Any]]]:
        """Retrieve all currently stored observations.
        
        Returns:
            Tuple containing the list of inputs and the list of labels.
            If labels are not stored, the second element is None.
        """
        raise NotImplementedError
        
    def reset(self) -> None:
        """Clear the buffer."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError

class TimeDecayReplayBuffer(ReplayBuffer):
    """Replay Buffer with Soft Forgetting via exponential decay probabilities.
    """
    
    def __init__(
        self, 
        max_size: int = 100000, 
        half_life_segments: Union[int, str, pd.Timedelta] = "180D", 
        min_prob: float = 1e-6, 
        callback_handler: Optional[CallbackHandler] = None,
        **kwargs
    ):
        self.max_size = max_size
        self.min_prob = min_prob
        
        # Check if half_life specifies a time duration
        if isinstance(half_life_segments, pd.Timedelta) or (isinstance(half_life_segments, str) and not str(half_life_segments).isdigit()):
            self.half_life_time = pd.Timedelta(half_life_segments)
            self.half_life_segments = None
        else:
            self.half_life_segments = int(half_life_segments)
            self.half_life_time = None
        
        self.data_buffer = collections.deque(maxlen=max_size)
        self.label_buffer = collections.deque(maxlen=max_size)
        self.time_buffer = collections.deque(maxlen=max_size)
        self._uses_timestamps = None
        super().__init__(callback_handler=callback_handler, **kwargs)
        
    def add(
        self, 
        X: Any, 
        y: Optional[Any] = None, 
        score: float = 0.0, 
        timestamps: Optional[Any] = None,
        results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert data into the fixed-size history."""
        with self._callback_context("replay_buffer_add", results):
            has_timestamps = timestamps is not None
            if self._uses_timestamps is None:
                self._uses_timestamps = has_timestamps
            elif self._uses_timestamps and not has_timestamps:
                 # If we started with timestamps but got None, we treat them as None (soft failure)
                 pass
            elif not self._uses_timestamps and has_timestamps:
                 # If we started without but got some, we can still use them (upgrading)
                 self._uses_timestamps = True
                
            is_batch = False
            if isinstance(X, list) and (len(X) == 0 or isinstance(X[0], (np.ndarray, list))):
                is_batch = True
            elif hasattr(X, "ndim") and X.ndim >= 2 and len(X) > 0:
                is_batch = True
            elif isinstance(X, (pd.DataFrame, pd.Series)):
                is_batch = True
                
            if is_batch:
                self.data_buffer.extend(X.values if hasattr(X, "values") else X)
                if y is not None:
                    self.label_buffer.extend(y.values if hasattr(y, "values") else y)
                if timestamps is not None:
                    # Ensure timestamps is 1D and flattened
                    ts_array = np.asarray(timestamps).ravel()
                    self.time_buffer.extend(ts_array)
                elif self.half_life_time is not None:
                    self.time_buffer.extend([None] * len(X))
            else:
                self.data_buffer.append(X)
                if y is not None:
                    self.label_buffer.append(y)
                if timestamps is not None:
                    # Still ensure scalar or 1st element if it's a single item passed as array
                    ts_val = np.asarray(timestamps).item() if hasattr(timestamps, "item") else timestamps
                    self.time_buffer.append(ts_val)
                elif self.half_life_time is not None:
                    self.time_buffer.append(None)
            
    def sample(self, sample_size: int, results: Optional[Dict[str, Any]] = None) -> Tuple[List[Any], Optional[List[Any]]]:
        """Estrae un campione usando probabilità a decadimento esponenziale (temporale o spaziale)."""
        with self._callback_context("replay_buffer_sample", results):
            n = len(self.data_buffer)
        
        if n == 0 or sample_size <= 0:
            return [], [] if len(self.label_buffer) > 0 else None
            
        actual_sample_size = min(sample_size, n)
        
        # Calcolo pesi ed estrazione
        decay_lambda = np.log(2)
        
        # Verifichiamo se usare il Time-based decay (e se time_buffer è allineato a data_buffer)
        if getattr(self, "half_life_time", None) is not None and hasattr(self, "time_buffer") and len(self.time_buffer) == n:
            batch_latest_times = []
            for t in self.time_buffer:
                # Se il timestamp è un array/lista di date del segmento, prendiamo l'ultima
                if isinstance(t, (list, np.ndarray, pd.Series, pd.DatetimeIndex)):
                    batch_latest_times.append(t[-1])
                else:
                    batch_latest_times.append(t)
                    
            timestamps_arr = pd.to_datetime(batch_latest_times)
            current_time = timestamps_arr.max()
            
            time_diffs = current_time - timestamps_arr
            
            # Forziamo a un array NumPy 1D per evitare conflitti di shape
            distances = np.asarray(time_diffs.total_seconds()).ravel() / self.half_life_time.total_seconds()
            weights = np.exp(-decay_lambda * distances) + self.min_prob
            
        else:
            # Segment-based decay (usato come fallback)
            hl = getattr(self, "half_life_segments", None) or 500000
            distances = np.arange(n - 1, -1, -1)
            weights = np.exp(-decay_lambda * (distances / hl)) + self.min_prob

        probabilities = np.asarray(weights) / np.sum(weights)
        
        probabilities[-1] = 1.0 - np.sum(probabilities[:-1])
        
        # Controllo anti-crash finale per garantire che n e p siano identici
        if len(probabilities) != n:
            raise ValueError(f"CRITICO: Dimensione dati n={n}, ma probabilità len={len(probabilities)}")
        
        # Estrazione degli indici basata sulle probabilità calcolate
        sampled_indices = np.random.choice(n, size=actual_sample_size, replace=False, p=probabilities)
        
        # Accesso rapido convertendo le deque in liste (O(1) lookup)
        data_list = list(self.data_buffer)
        sampled_X = [data_list[i] for i in sampled_indices]
        
        if len(self.label_buffer) > 0:
            label_list = list(self.label_buffer)
            sampled_y = [label_list[i] for i in sampled_indices]
        else:
            sampled_y = None
            
        return sampled_X, sampled_y
        
    def get_all(self) -> Tuple[List[Any], Optional[List[Any]]]:
        labels = list(self.label_buffer) if len(self.label_buffer) > 0 else None
        return list(self.data_buffer), labels
        
    def reset(self) -> None:
        self.data_buffer.clear()
        self.label_buffer.clear()
        if hasattr(self, "time_buffer"):
            self.time_buffer.clear()
        self._uses_timestamps = None
        
    def __len__(self) -> int:
        return len(self.data_buffer)

