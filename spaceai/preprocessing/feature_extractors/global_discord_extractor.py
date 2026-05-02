"""Global Matrix Profile (Discord and Motif) feature extractor."""

from __future__ import annotations
import os
from typing import Optional, Dict, Any, TYPE_CHECKING, List
import numpy as np
import pandas as pd
import stumpy
from .feature_extractor import FeatureExtractor

if TYPE_CHECKING:
    from spaceai.data.anomaly_dataset import AnomalyDataset

class GlobalDiscordExtractor(FeatureExtractor):
    """
    Feature extractor that computes Global Matrix Profile statistics 
    (Discords, Motifs, etc.) using the full signal context.
    """
    def __init__(
        self, 
        window_size: int, 
        stride: int, 
        m: int = 15,
        history_len: Optional[int] = None,
        mp_features: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(window_size, stride, **kwargs)
        self.m = m
        self.history_len = history_len
        self.mp_features = mp_features or ["max"]
        self._cached_mp = None
        self._context_offset = 0

    @property
    def output_dim(self) -> int:
        return len(self.mp_features)

    def set_context(self, dataset: Optional[AnomalyDataset] = None, indices: Optional[np.ndarray] = None) -> None:
        """
        Compute the Global Matrix Profile once when the context is set.
        """
        super().set_context(dataset, indices)
        if dataset is None or not hasattr(dataset, "data"):
            self._cached_mp = None
            self._context_offset = 0
            return

        raw_signal = dataset.data
        if hasattr(raw_signal, "values"):
            raw_signal = raw_signal.values
        
        if raw_signal.ndim > 1:
            raw_signal = raw_signal[:, 0]
        
        if not isinstance(raw_signal, np.ndarray):
            raw_signal = np.array(raw_signal)

        if indices is not None and len(indices) > 0:
            last_idx = int(np.max(indices[:, 1]))
            first_idx = int(np.min(indices[:, 0]))
            
            if self.history_len:
                start_idx = min(max(0, last_idx - self.history_len), first_idx)
                self._context_offset = start_idx
                raw_signal = raw_signal[start_idx : last_idx + 1]
            else:
                self._context_offset = 0
                raw_signal = raw_signal[: last_idx + 1]
        else:
            if self.history_len and len(raw_signal) > self.history_len:
                self._context_offset = len(raw_signal) - self.history_len
                raw_signal = raw_signal[self._context_offset:]
            else:
                self._context_offset = 0
        
        current_m = self.window_size if self.m in [None, "window_size"] else max(3, self.m)
        raw_signal = raw_signal.astype(np.float64)

        if len(raw_signal) > current_m:
            self._cached_mp = stumpy.stump(raw_signal, current_m)[:, 0]
            self._resolved_m = current_m
        else:
            self._cached_mp = None

    def fit(self, X, y=None, results=None):
        return self

    def transform(
        self, 
        X: Union[np.ndarray, Any],
        results: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
        suffix: str = ""
    ) -> Union[pd.DataFrame, Any]:
        """
        Extract multiple Matrix Profile statistics for each window.
        Supports PipelineMessage.
        """
        if hasattr(X, "data") and not isinstance(X, (np.ndarray, pd.DataFrame)):
            msg = X
            msg.data = self.transform(msg.data, results=results, save_dir=save_dir, suffix=suffix)
            return msg

        with self._callback_context("global_feature_extraction", results):
            if self._cached_mp is None or self._current_indices is None:
                discord_features = np.zeros((len(X), len(self.mp_features)), dtype=np.float32)
            else:
                m = getattr(self, "_resolved_m", self.m) or self.window_size
                stats_map = {"max": np.max, "min": np.min, "mean": np.mean, "std": np.std, "median": np.median}
                
                rows = []
                for s, e in self._current_indices:
                    mp_start = int(s - self._context_offset)
                    mp_end = int(e - m + 2 - self._context_offset)
                    
                    if 0 <= mp_start < len(self._cached_mp) and mp_end > mp_start:
                        mp_slice = self._cached_mp[mp_start:mp_end]
                        rows.append([stats_map[fea](mp_slice) for fea in self.mp_features])
                    else:
                        rows.append([0.0] * len(self.mp_features))
                discord_features = np.array(rows, dtype=np.float32)

            columns = [f"global_mp_{fea}" for fea in self.mp_features]
            df = pd.DataFrame(discord_features, columns=columns)
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filename = f"global_mp_features{'_'+suffix if suffix else ''}.csv"
                df.to_csv(os.path.join(save_dir, filename), index=False)
                
            return df
