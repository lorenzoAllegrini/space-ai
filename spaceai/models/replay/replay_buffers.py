from __future__ import annotations
import collections
from typing import Any, List, Optional, Tuple, Dict, TYPE_CHECKING
import numpy as np
import pandas as pd
from spaceai.benchmark.callbacks.mixin import CallbackMixin

if TYPE_CHECKING:
    from spaceai.benchmark.callbacks.handler import CallbackHandler

class ReplayBuffer(CallbackMixin):
    """Abstract base class for Replay Buffers.
    Handles standard pipeline routing, concatenation, and basic deque operations.
    """
    
    def __init__(
        self, 
        max_size: int = 100000, 
        sample_size: int = 10000,
        callback_handler: Optional[CallbackHandler] = None,
        **kwargs
    ):
        self.max_size = max_size
        self.sample_size = sample_size
        self.data_buffer = collections.deque(maxlen=max_size)
        self.label_buffer = collections.deque(maxlen=max_size)
        super().__init__(callback_handler=callback_handler, **kwargs)

    def fit_transform(
        self, 
        X: Any, 
        y: Optional[Any] = None,  
        results: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[Any, Optional[Any]]:
        """Universal pipeline logic: samples, concatenates, and then adds new data."""
        with self._callback_context("replay_buffer_fit_transform", results):
            sampled_X, sampled_y = self.sample(sample_size=self.sample_size, results=results)
            
            combined_X = self._concatenate(sampled_X, X)
            if y is None:
                filler_y = [None] * len(X)
                combined_y = self._concatenate(sampled_y, filler_y)
            else:
                combined_y = self._concatenate(sampled_y, y)
            
            self.add(X, y, results=results)
            return combined_X, combined_y

    def transform(
        self, 
        X: Any, 
        y: Optional[Any] = None,  
        results: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[Any, Optional[Any]]:
        """Pass-through during inference phase."""
        with self._callback_context("replay_buffer_transform", results):
            return X, y

    def _concatenate(self, sampled_data: List[Any], current_data: Any) -> Any:
        """Robust utility to concatenate mixed data types."""
        if not sampled_data:
            return current_data
            
        if isinstance(current_data, pd.DataFrame):
            sampled_df = pd.DataFrame(sampled_data, columns=current_data.columns)
            return pd.concat([sampled_df, current_data], ignore_index=True)
        elif isinstance(current_data, pd.Series):
            sampled_series = pd.Series(sampled_data, name=current_data.name)
            return pd.concat([sampled_series, current_data], ignore_index=True)
        elif isinstance(current_data, np.ndarray):
            return np.concatenate([np.array(sampled_data), current_data], axis=0)
        elif isinstance(current_data, list):
            return sampled_data + current_data
        else:
            return sampled_data + [current_data]

    def add(
        self, 
        X: Any, 
        y: Optional[Any] = None,  
        results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Inserts data into the buffer based on type (batch or single)."""
        with self._callback_context("replay_buffer_add", results):
            is_batch = False
            if isinstance(X, list) and (len(X) == 0 or isinstance(X[0], (np.ndarray, list))):
                is_batch = True
            elif hasattr(X, "ndim") and X.ndim >= 2 and len(X) > 0:
                is_batch = True
            elif isinstance(X, (pd.DataFrame, pd.Series)):
                is_batch = True
                
            if is_batch:
                batch_len = len(X.values if hasattr(X, "values") else X)
                self.data_buffer.extend(X.values if hasattr(X, "values") else X)
                if y is not None:
                    self.label_buffer.extend(y.values if hasattr(y, "values") else y)
                else:
                    self.label_buffer.extend([None] * batch_len)
            else:
                self.data_buffer.append(X)
                if y is not None:
                    self.label_buffer.append(y)
                else:
                    self.label_buffer.append(None)

    def sample(self, sample_size: int, results: Optional[Dict[str, Any]] = None) -> Tuple[List[Any], Optional[List[Any]]]:
        """ABSTRACT METHOD: child classes must define how to sample."""
        raise NotImplementedError("Child classes must implement sample() method")

    def get_all(self, results: Optional[Dict[str, Any]] = None) -> Tuple[List[Any], Optional[List[Any]]]:
        labels = list(self.label_buffer) if len(self.label_buffer) > 0 else None
        return list(self.data_buffer), labels
        
    def reset(self) -> None:
        self.data_buffer.clear()
        self.label_buffer.clear()
        
    def __len__(self) -> int:
        return len(self.data_buffer)