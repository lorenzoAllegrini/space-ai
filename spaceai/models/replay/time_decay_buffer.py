
from __future__ import annotations
import collections
from typing import Any, List, Optional, Tuple, Dict, TYPE_CHECKING
import numpy as np
import pandas as pd
from .replay_buffers import ReplayBuffer

if TYPE_CHECKING:
    from spaceai.benchmark.callbacks.handler import CallbackHandler

class TimeDecayReplayBuffer(ReplayBuffer):
    """
    Replay Buffer with Soft Forgetting via index-based exponential decay.
    Inherits pipeline, concatenation, and storage logic from ReplayBuffer.
    """
    
    def __init__(
        self, 
        max_size: int = 100000, 
        half_life_segments: int = 5000, 
        min_prob: float = 1e-6, 
        callback_handler: Optional[CallbackHandler] = None,
        **kwargs
    ):
        super().__init__(max_size=max_size, callback_handler=callback_handler, **kwargs)
        
        self.half_life_segments = int(half_life_segments)
        self.min_prob = min_prob
            
    def sample(self, sample_size: int, results: Optional[Dict[str, Any]] = None) -> Tuple[List[Any], Optional[List[Any]]]:
        """Samples elements with higher probability for recently inserted items."""
        with self._callback_context("replay_buffer_sample", results):
            n = len(self.data_buffer)
        
        if n == 0 or sample_size <= 0:
            return [], [] if len(self.label_buffer) > 0 else None
            
        actual_sample_size = min(sample_size, n)
        decay_lambda = np.log(2)
        
        distances = np.arange(n - 1, -1, -1)
        weights = np.exp(-decay_lambda * (distances / self.half_life_segments)) + self.min_prob
 
        probabilities = np.asarray(weights) / np.sum(weights)
        probabilities[-1] = 1.0 - np.sum(probabilities[:-1])
        
        sampled_indices = np.random.choice(n, size=actual_sample_size, replace=False, p=probabilities)
        
        data_list = list(self.data_buffer)
        sampled_X = [data_list[i] for i in sampled_indices]
        
        if len(self.label_buffer) > 0:
            label_list = list(self.label_buffer)
            sampled_y = [label_list[i] for i in sampled_indices]
            n_pos = np.sum([y for y in sampled_y if y is not None]) if sampled_y else 0
        else:
            sampled_y = None
            
        return sampled_X, sampled_y